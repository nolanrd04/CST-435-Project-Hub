"""
GAN Trainer
Implements the training loop following the architecture flow:
2 -> 3: Generator generates fake images from noise
1 and 3 -> 4: Label images as real or fake
4 -> 5: Train discriminator
5 -> 6: Use discriminator to train generator
6 -> 7 and 8: Generator generates new fake images, discriminator evaluates
7 -> 8: Discriminator continues to evaluate
8 -> 7 and 9: If discriminator fooled, continue or end training
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from pathlib import Path
import json
import os
from datetime import datetime
import time
from typing import Dict, Optional


class TrainingTimeEstimator:
    """
    Hybrid approach for estimating remaining training time.
    Uses EMA for batch timing initially, then switches to actual epoch times after warmup.
    """
    
    def __init__(self, total_epochs: int, warmup_epochs: int = 3):
        self.total_epochs = total_epochs
        self.warmup_epochs = warmup_epochs
        self.epoch_times = []
        self.batch_times = []
        self.ema_batch_time = None
        self.ema_alpha = 0.15  # EMA smoothing factor
        
    def record_batch_time(self, batch_time: float):
        """Record a batch execution time and update EMA"""
        self.batch_times.append(batch_time)
        
        if self.ema_batch_time is None:
            self.ema_batch_time = batch_time
        else:
            # Exponential Moving Average: weight recent values more heavily
            self.ema_batch_time = (self.ema_alpha * batch_time) + ((1 - self.ema_alpha) * self.ema_batch_time)
    
    def record_epoch_time(self, epoch_time: float):
        """Record full epoch time"""
        self.epoch_times.append(epoch_time)
    
    def estimate_remaining_time(self, current_epoch: int, total_batches: int) -> Dict[str, str]:
        """
        Estimate remaining training time using hybrid approach.
        
        Args:
            current_epoch: Current epoch number (0-indexed)
            total_batches: Total batches per epoch
            
        Returns:
            Dictionary with estimates
        """
        remaining_epochs = self.total_epochs - current_epoch - 1
        
        if remaining_epochs < 0:
            remaining_epochs = 0
        
        # Strategy based on training progress
        if current_epoch < self.warmup_epochs or len(self.epoch_times) == 0:
            # Warmup phase: use batch-level EMA
            if self.ema_batch_time is None:
                return {'optimistic': 'N/A', 'realistic': 'N/A', 'pessimistic': 'N/A', 'remaining_epochs': remaining_epochs, 'in_warmup': True}
            
            # Estimate this epoch and remaining epochs
            remaining_time = self.ema_batch_time * total_batches * remaining_epochs
            
        else:
            # Post-warmup: use actual epoch times
            avg_epoch_time = sum(self.epoch_times) / len(self.epoch_times)
            
            # Detect if we're slowing down (learning rate decrease effects)
            if len(self.epoch_times) >= 2:
                recent_avg = sum(self.epoch_times[-2:]) / 2
                trend_factor = recent_avg / avg_epoch_time if avg_epoch_time > 0 else 1.0
            else:
                trend_factor = 1.0
            
            # Remaining epochs with trend adjustment
            remaining_time = avg_epoch_time * trend_factor * remaining_epochs
        
        # Generate estimates with confidence intervals
        optimistic_time = remaining_time * 0.8   # 20% faster
        realistic_time = remaining_time
        pessimistic_time = remaining_time * 1.2  # 20% slower
        
        return {
            'optimistic': self._format_time(optimistic_time),
            'realistic': self._format_time(realistic_time),
            'pessimistic': self._format_time(pessimistic_time),
            'remaining_epochs': remaining_epochs,
            'in_warmup': current_epoch < self.warmup_epochs
        }
    
    @staticmethod
    def _format_time(seconds: float) -> str:
        """Format seconds into human-readable format"""
        if seconds < 60:
            return f"{seconds:.0f}s"
        elif seconds < 3600:
            minutes = seconds / 60
            return f"{minutes:.1f}m"
        else:
            hours = seconds / 3600
            return f"{hours:.2f}h"


class GANTrainer:
    """
    Trainer for Generative Adversarial Networks
    """
    
    def __init__(self, generator, discriminator, gen_optimizer, disc_optimizer, 
                 device='cpu', latent_dim=100):
        """
        Initialize GAN Trainer
        
        Args:
            generator (nn.Module): Generator network
            discriminator (nn.Module): Discriminator network
            gen_optimizer (Optimizer): Generator optimizer
            disc_optimizer (Optimizer): Discriminator optimizer
            device (str): Device to train on
            latent_dim (int): Dimension of latent noise vector
        """
        self.generator = generator
        self.discriminator = discriminator
        self.gen_optimizer = gen_optimizer
        self.disc_optimizer = disc_optimizer
        self.device = device
        self.latent_dim = latent_dim
        
        # Loss function - Binary Cross Entropy with logits
        self.criterion = nn.BCELoss()
        
        # Labels for real and fake images
        self.real_label = 1.0
        self.fake_label = 0.0
        
        # Training history
        self.history = {
            'gen_loss': [],
            'disc_loss': [],
            'disc_real_acc': [],
            'disc_fake_acc': [],
            'epoch_gen_loss': [],
            'epoch_disc_loss': [],
            'epoch': []
        }
        
        # Generated images for tracking progress
        self.generated_images_history = {}
    
    def generate_noise(self, batch_size):
        """
        Generate random noise tensor
        
        Args:
            batch_size (int): Number of noise samples
            
        Returns:
            torch.Tensor: Noise tensor shape (batch_size, latent_dim)
        """
        return torch.randn(batch_size, self.latent_dim, device=self.device)
    
    def train_discriminator_batch(self, real_images, batch_size):
        """
        Train discriminator on one batch
        
        Step 4: Label images as real or fake
        Step 5: Train discriminator
        
        Args:
            real_images (torch.Tensor): Batch of real images
            batch_size (int): Batch size
            
        Returns:
            dict: Loss and accuracy metrics
        """
        self.discriminator.zero_grad()
        
        # Real images (Step 1: Dataset)
        real_images = real_images.to(self.device)
        real_labels = torch.full((batch_size, 1), self.real_label, device=self.device)
        
        # Forward pass on real images (Step 4a: Label real images)
        real_output = self.discriminator(real_images)
        disc_loss_real = self.criterion(real_output, real_labels)
        
        # Backward pass for real images
        disc_loss_real.backward()
        
        # Calculate accuracy on real images
        real_accuracy = (real_output > 0.5).float().mean().item()
        
        # Fake images (Step 2-3: Generator generates from noise)
        noise = self.generate_noise(batch_size)
        fake_images = self.generator(noise).detach()  # Detach to not update generator
        fake_labels = torch.full((batch_size, 1), self.fake_label, device=self.device)
        
        # Forward pass on fake images (Step 4b: Label fake images)
        fake_output = self.discriminator(fake_images)
        disc_loss_fake = self.criterion(fake_output, fake_labels)
        
        # Backward pass for fake images
        disc_loss_fake.backward()
        
        # Calculate accuracy on fake images
        fake_accuracy = (fake_output < 0.5).float().mean().item()
        
        # Update discriminator
        self.disc_optimizer.step()
        
        # Total discriminator loss
        disc_loss_total = disc_loss_real + disc_loss_fake
        
        return {
            'disc_loss': disc_loss_total.item(),
            'disc_loss_real': disc_loss_real.item(),
            'disc_loss_fake': disc_loss_fake.item(),
            'real_accuracy': real_accuracy,
            'fake_accuracy': fake_accuracy
        }
    
    def train_generator_batch(self, batch_size):
        """
        Train generator on one batch
        
        Step 5: Use discriminator to train generator
        Step 6-8: Generator generates fake images that fool discriminator
        
        Args:
            batch_size (int): Batch size
            
        Returns:
            dict: Loss metrics
        """
        self.generator.zero_grad()
        
        # Generate fake images (Step 6-7: Generator generates fake images)
        noise = self.generate_noise(batch_size)
        fake_images = self.generator(noise)
        
        # Labels for generator training (we want discriminator to think they're real)
        fake_labels = torch.full((batch_size, 1), self.real_label, device=self.device)
        
        # Forward pass (Step 8: Discriminator evaluates fake images)
        fake_output = self.discriminator(fake_images)
        gen_loss = self.criterion(fake_output, fake_labels)
        
        # Backward pass
        gen_loss.backward()
        
        # Update generator
        self.gen_optimizer.step()
        
        return {
            'gen_loss': gen_loss.item(),
            'discriminator_fooled': (fake_output > 0.5).float().mean().item()
        }
    
    def train_epoch(self, dataloader, epoch_num=0):
        """
        Train for one epoch
        
        Args:
            dataloader (DataLoader): Training data loader
            epoch_num (int): Current epoch number
            
        Returns:
            dict: Epoch statistics
        """
        self.generator.train()
        self.discriminator.train()
        
        epoch_gen_loss = 0.0
        epoch_disc_loss = 0.0
        epoch_real_acc = 0.0
        epoch_fake_acc = 0.0
        epoch_fooled = 0.0
        num_batches = 0
        
        for batch_idx, real_images in enumerate(dataloader):
            batch_size = real_images.size(0)
            
            # Step 2-5: Train discriminator
            disc_metrics = self.train_discriminator_batch(real_images, batch_size)
            
            # Step 5-8: Train generator
            gen_metrics = self.train_generator_batch(batch_size)
            
            # Accumulate metrics
            epoch_disc_loss += disc_metrics['disc_loss']
            epoch_gen_loss += gen_metrics['gen_loss']
            epoch_real_acc += disc_metrics['real_accuracy']
            epoch_fake_acc += disc_metrics['fake_accuracy']
            epoch_fooled += gen_metrics['discriminator_fooled']
            
            self.history['gen_loss'].append(gen_metrics['gen_loss'])
            self.history['disc_loss'].append(disc_metrics['disc_loss'])
            self.history['disc_real_acc'].append(disc_metrics['real_accuracy'])
            self.history['disc_fake_acc'].append(disc_metrics['fake_accuracy'])
            
            num_batches += 1
        
        # Average metrics
        avg_gen_loss = epoch_gen_loss / num_batches
        avg_disc_loss = epoch_disc_loss / num_batches
        avg_real_acc = epoch_real_acc / num_batches
        avg_fake_acc = epoch_fake_acc / num_batches
        avg_fooled = epoch_fooled / num_batches
        
        self.history['epoch_gen_loss'].append(avg_gen_loss)
        self.history['epoch_disc_loss'].append(avg_disc_loss)
        self.history['epoch'].append(epoch_num)
        
        return {
            'gen_loss': avg_gen_loss,
            'disc_loss': avg_disc_loss,
            'real_accuracy': avg_real_acc,
            'fake_accuracy': avg_fake_acc,
            'discriminator_fooled': avg_fooled
        }
    
    def train(self, dataloader, num_epochs=400, save_dir='./models', version='v1', 
              save_interval_count=5):
        """
        Train the GAN
        
        Args:
            dataloader (DataLoader): Training data loader
            num_epochs (int): Number of epochs to train
            save_dir (str): Base directory to save checkpoints (models/)
            version (str): Dataset version (e.g., 'v1')
            save_interval_count (int): Number of times to save images during training (default 5)
        """
        # Create directory structure - NEW: model_v1 instead of discriminator_v1, with info/ subfolder
        model_dir = Path(save_dir) / f'model_{version}'
        model_info_dir = model_dir / 'info'  # NEW: Nested under model_v1
        generated_images_dir = model_dir / 'generated_epoch_images'
        
        model_dir.mkdir(parents=True, exist_ok=True)
        model_info_dir.mkdir(parents=True, exist_ok=True)
        generated_images_dir.mkdir(parents=True, exist_ok=True)
        
        # Calculate save interval
        save_interval = max(1, num_epochs // save_interval_count)
        
        # Initialize time estimator
        time_estimator = TrainingTimeEstimator(total_epochs=num_epochs)
        training_start_time = time.time()
        
        print("\n" + "="*70)
        print("GAN TRAINING STARTED")
        print("="*70)
        print(f"Total Epochs: {num_epochs}")
        print(f"Batch Size: {dataloader.batch_size}")
        print(f"Batches per Epoch: {len(dataloader)}")
        print(f"Total Batches: {len(dataloader) * num_epochs}")
        print(f"Device: {self.device}")
        print(f"Model Directory: {model_dir}")
        print(f"Model Info Directory: {model_info_dir}")
        print(f"Saving images every {save_interval} epochs ({save_interval_count} times total)")
        print("="*70)
        
        for epoch in range(num_epochs):
            epoch_start_time = time.time()
            epoch_metrics = self.train_epoch(dataloader, epoch + 1)
            epoch_time = time.time() - epoch_start_time
            
            # Record epoch time for estimation
            time_estimator.record_epoch_time(epoch_time)
            
            # Print progress for EVERY epoch
            print(f"\nEpoch [{epoch + 1:4d}/{num_epochs}] ({epoch_time:.2f}s)")
            print(f"  Gen Loss: {epoch_metrics['gen_loss']:.4f}")
            print(f"  Disc Loss: {epoch_metrics['disc_loss']:.4f}")
            print(f"  Real Accuracy: {epoch_metrics['real_accuracy']:.4f}")
            print(f"  Fake Accuracy (Discriminator): {epoch_metrics['fake_accuracy']:.4f}")
            print(f"  Discriminator Fooled Rate: {epoch_metrics['discriminator_fooled']:.4f}")
            
            # Get time estimates
            estimates = time_estimator.estimate_remaining_time(epoch, len(dataloader))
            if estimates['optimistic'] != 'N/A':
                print(f"\n  ⏱️  Estimated Remaining Time:")
                if estimates.get('in_warmup'):
                    print(f"     (Warmup phase - estimates less accurate)")
                print(f"     Optimistic:  {estimates['optimistic']}")
                print(f"     Realistic:   {estimates['realistic']}")
                print(f"     Pessimistic: {estimates['pessimistic']}")
            
            # Generate and save images at intervals
            if (epoch + 1) % save_interval == 0 or epoch == 0:
                print(f"  -> Saving images at epoch {epoch + 1}")
                self.save_generated_images(epoch + 1, generated_images_dir)
        
        total_training_time = time.time() - training_start_time
        
        print("\n" + "="*70)
        print("GAN TRAINING COMPLETED")
        print("="*70)
        print(f"Total Training Time: {self._format_time(total_training_time)}")
        
        # Save final models
        self.save_models(model_dir, version)
        
        # Save training history
        self.save_history(model_info_dir, version)
    
    def generate_images(self, num_images=16):
        """
        Generate images without gradient computation
        
        Args:
            num_images (int): Number of images to generate
            
        Returns:
            torch.Tensor: Generated images
        """
        self.generator.eval()
        
        with torch.no_grad():
            noise = self.generate_noise(num_images)
            images = self.generator(noise)
        
        return images
    
    @staticmethod
    def _format_time(seconds: float) -> str:
        """Format seconds into human-readable format"""
        if seconds < 60:
            return f"{seconds:.0f}s"
        elif seconds < 3600:
            minutes = seconds / 60
            return f"{minutes:.1f}m"
        else:
            hours = seconds / 3600
            return f"{hours:.2f}h"
    
    def save_generated_images(self, epoch, save_dir, num_images=16):
        """
        Generate and save images as a grid
        
        Args:
            epoch (int): Current epoch
            save_dir (str or Path): Directory to save images
            num_images (int): Number of images to generate
        """
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        images = self.generate_images(num_images)
        images = images.cpu().detach()
        
        # Denormalize from [-1, 1] to [0, 1]
        images = (images + 1) / 2
        
        # Create grid
        fig, axes = plt.subplots(4, 4, figsize=(8, 8))
        
        for idx, ax in enumerate(axes.flatten()):
            if idx < num_images:
                img = images[idx]
                # Handle both grayscale and RGB
                if img.shape[0] == 1:
                    img = img.squeeze(0)  # Remove channel dimension for grayscale
                    ax.imshow(img, cmap='gray')
                else:
                    img = img.permute(1, 2, 0).numpy()
                    img = np.clip(img, 0, 1)
                    ax.imshow(img)
            ax.axis('off')
        
        plt.suptitle(f'Generated Images - Epoch {epoch}', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        # Save image
        image_path = save_dir / f'epoch_{epoch:04d}.png'
        plt.savefig(image_path, dpi=100, bbox_inches='tight')
        plt.close()
        
        self.generated_images_history[epoch] = str(image_path)
    
    def save_models(self, save_dir, version):
        """
        Save generator and discriminator models
        
        Args:
            save_dir (str or Path): Directory to save models
            version (str): Dataset version
        """
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        gen_path = save_dir / f'generator_{version}.pt'
        disc_path = save_dir / f'discriminator_{version}.pt'
        
        torch.save(self.generator.state_dict(), gen_path)
        torch.save(self.discriminator.state_dict(), disc_path)
        
        print(f"\nModels saved:")
        print(f"  Generator: {gen_path}")
        print(f"  Discriminator: {disc_path}")
    
    def save_history(self, save_dir, version):
        """
        Save training history
        
        Args:
            save_dir (str or Path): Directory to save history
            version (str): Dataset version
        """
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # Convert to serializable format
        history_json = {
            'epoch_gen_loss': self.history['epoch_gen_loss'],
            'epoch_disc_loss': self.history['epoch_disc_loss'],
            'epoch': self.history['epoch'],
            'generated_images_milestones': self.generated_images_history
        }
        
        history_path = save_dir / f'training_history_{version}.json'
        with open(history_path, 'w') as f:
            json.dump(history_json, f, indent=2)
        
        print(f"  Training History: {history_path}")
    
    def plot_training_history(self, save_dir=None, version='v1'):
        """
        Plot training losses and accuracies
        
        Args:
            save_dir (str or Path): Directory to save plot
            version (str): Dataset version
        """
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        
        # Generator and Discriminator Loss
        axes[0, 0].plot(self.history['epoch_gen_loss'], label='Generator Loss', linewidth=2)
        axes[0, 0].plot(self.history['epoch_disc_loss'], label='Discriminator Loss', linewidth=2)
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].set_title('Loss vs Epoch')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Generator Loss Only
        axes[0, 1].plot(self.history['epoch_gen_loss'], label='Generator Loss', color='green', linewidth=2)
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].set_title('Generator Loss vs Epoch')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Discriminator Loss Only
        axes[1, 0].plot(self.history['epoch_disc_loss'], label='Discriminator Loss', color='red', linewidth=2)
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Loss')
        axes[1, 0].set_title('Discriminator Loss vs Epoch')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Accuracy metrics
        axes[1, 1].plot(self.history['disc_real_acc'], label='Real Accuracy', linewidth=2)
        axes[1, 1].plot(self.history['disc_fake_acc'], label='Fake Accuracy', linewidth=2)
        axes[1, 1].set_xlabel('Batch')
        axes[1, 1].set_ylabel('Accuracy')
        axes[1, 1].set_title('Discriminator Accuracy vs Batch')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_dir:
            save_dir = Path(save_dir)
            save_dir.mkdir(parents=True, exist_ok=True)
            plot_path = save_dir / f'training_history_{version}.png'
            plt.savefig(plot_path, dpi=150, bbox_inches='tight')
            print(f"\nTraining history plot saved: {plot_path}")
        
        plt.show()


class MultiFruitGANTrainer:
    """
    Trainer for multi-fruit GAN training.
    Trains separate Generator/Discriminator pairs for each fruit in a single session.
    """
    
    def __init__(self, device='cpu', latent_dim=100):
        """
        Initialize Multi-Fruit GAN Trainer
        
        Args:
            device (str): Device to train on
            latent_dim (int): Dimension of latent noise vector
        """
        self.device = device
        self.latent_dim = latent_dim
        self.fruit_trainers = {}  # Dict to store trainers per fruit
        self.training_history = {}  # Store history for all fruits
        
    def add_fruit_trainer(self, fruit_name, generator, discriminator, 
                         gen_optimizer, disc_optimizer):
        """
        Add a trainer for a specific fruit
        
        Args:
            fruit_name (str): Name of the fruit
            generator (nn.Module): Generator network
            discriminator (nn.Module): Discriminator network
            gen_optimizer (Optimizer): Generator optimizer
            disc_optimizer (Optimizer): Discriminator optimizer
        """
        trainer = GANTrainer(
            generator=generator,
            discriminator=discriminator,
            gen_optimizer=gen_optimizer,
            disc_optimizer=disc_optimizer,
            device=self.device,
            latent_dim=self.latent_dim
        )
        self.fruit_trainers[fruit_name] = trainer
        self.training_history[fruit_name] = {
            'epoch_gen_loss': [],
            'epoch_disc_loss': [],
            'epoch': []
        }
    
    def train_all_fruits(self, dataloaders, num_epochs=400, save_dir='./models', 
                        model_name='v1', save_interval_count=5):
        """
        Train all fruit-specific models
        
        Args:
            dataloaders (dict): Dictionary of dataloaders per fruit {fruit_name: dataloader}
            num_epochs (int): Number of epochs per fruit
            save_dir (str): Base directory to save checkpoints (models/)
            model_name (str): Name of the model folder (e.g., 'v1')
            save_interval_count (int): Number of times to save images during training
        """
        # Create base model directory
        model_dir = Path(save_dir) / f'model_{model_name}'
        model_info_dir = model_dir / 'info'
        model_dir.mkdir(parents=True, exist_ok=True)
        model_info_dir.mkdir(parents=True, exist_ok=True)
        
        print("\n" + "="*70)
        print("MULTI-FRUIT GAN TRAINING STARTED")
        print("="*70)
        print(f"Model Name: {model_name}")
        print(f"Number of Fruits: {len(self.fruit_trainers)}")
        print(f"Fruits: {', '.join(self.fruit_trainers.keys())}")
        print(f"Total Epochs per Fruit: {num_epochs}")
        print(f"Device: {self.device}")
        print(f"Model Directory: {model_dir}")
        print("="*70)
        
        total_fruits = len(self.fruit_trainers)
        global_training_start = time.time()
        
        for fruit_idx, (fruit_name, trainer) in enumerate(self.fruit_trainers.items(), 1):
            print(f"\n{'='*70}")
            print(f"TRAINING FRUIT {fruit_idx}/{total_fruits}: {fruit_name.upper()}")
            print("="*70)
            
            if fruit_name not in dataloaders:
                print(f"⚠️  No dataloader found for {fruit_name}, skipping...")
                continue
            
            dataloader = dataloaders[fruit_name]
            
            # Create fruit-specific directories
            fruit_generated_dir = model_dir / f'generated_epoch_images_{fruit_name}'
            fruit_generated_dir.mkdir(parents=True, exist_ok=True)
            
            # Calculate save interval
            save_interval = max(1, num_epochs // save_interval_count)
            
            # Initialize time estimator
            time_estimator = TrainingTimeEstimator(total_epochs=num_epochs)
            fruit_training_start = time.time()
            
            print(f"Batch Size: {dataloader.batch_size}")
            print(f"Batches per Epoch: {len(dataloader)}")
            print(f"Dataset Size: {len(dataloader.dataset)}")
            
            # Train for this fruit
            for epoch in range(num_epochs):
                epoch_start_time = time.time()
                epoch_metrics = trainer.train_epoch(dataloader, epoch + 1)
                epoch_time = time.time() - epoch_start_time
                
                # Record epoch time for estimation
                time_estimator.record_epoch_time(epoch_time)
                
                # Store history
                self.training_history[fruit_name]['epoch_gen_loss'].append(epoch_metrics['gen_loss'])
                self.training_history[fruit_name]['epoch_disc_loss'].append(epoch_metrics['disc_loss'])
                self.training_history[fruit_name]['epoch'].append(epoch + 1)
                
                # Print progress for EVERY epoch
                print(f"\nEpoch [{epoch + 1:4d}/{num_epochs}] ({epoch_time:.2f}s)")
                print(f"  Gen Loss: {epoch_metrics['gen_loss']:.4f}")
                print(f"  Disc Loss: {epoch_metrics['disc_loss']:.4f}")
                print(f"  Real Accuracy: {epoch_metrics['real_accuracy']:.4f}")
                print(f"  Fake Accuracy (Discriminator): {epoch_metrics['fake_accuracy']:.4f}")
                print(f"  Discriminator Fooled Rate: {epoch_metrics['discriminator_fooled']:.4f}")
                
                # Training diagnostics
                self._print_training_diagnostics(epoch_metrics, epoch + 1)
                
                # Get time estimates
                estimates = time_estimator.estimate_remaining_time(epoch, len(dataloader))
                if estimates['optimistic'] != 'N/A':
                    print(f"\n  ⏱️  Time for {fruit_name}:")
                    if estimates.get('in_warmup'):
                        print(f"     (Warmup phase - estimates less accurate)")
                    print(f"     Optimistic:  {estimates['optimistic']}")
                    print(f"     Realistic:   {estimates['realistic']}")
                    print(f"     Pessimistic: {estimates['pessimistic']}")
                    
                    # Calculate and show total time for all remaining fruits
                    remaining_fruits = total_fruits - fruit_idx
                    if remaining_fruits > 0:
                        realistic_per_fruit = self._parse_time_to_seconds(estimates['realistic'])
                        total_remaining = (remaining_fruits * realistic_per_fruit) + (realistic_per_fruit * estimates.get('remaining_epochs', 0))
                        print(f"\n  🕐 Total Time for All Remaining Fruits:")
                        print(f"     {self._format_time(total_remaining)} (est.)")
                
                # Generate and save images at intervals
                if (epoch + 1) % save_interval == 0 or epoch == 0:
                    print(f"  -> Saving images at epoch {epoch + 1}")
                    trainer.save_generated_images(epoch + 1, fruit_generated_dir)
            
            fruit_training_time = time.time() - fruit_training_start
            
            print(f"\n✓ {fruit_name.upper()} Training Complete ({self._format_time(fruit_training_time)})")
            
            # Save fruit-specific models
            self._save_fruit_models(fruit_name, trainer, model_dir)
        
        # Save combined training history and parameters
        self._save_all_histories(model_info_dir, model_name)
        
        total_training_time = time.time() - global_training_start
        
        print("\n" + "="*70)
        print("MULTI-FRUIT GAN TRAINING COMPLETED")
        print("="*70)
        print(f"Total Training Time: {self._format_time(total_training_time)}")
        print(f"All models saved to: {model_dir}")
        print(f"Model info saved to: {model_info_dir}")
        print("="*70)
    
    def _save_fruit_models(self, fruit_name, trainer, model_dir):
        """Save generator and discriminator for specific fruit"""
        gen_path = model_dir / f'generator_{fruit_name}.pt'
        disc_path = model_dir / f'discriminator_{fruit_name}.pt'
        
        torch.save(trainer.generator.state_dict(), gen_path)
        torch.save(trainer.discriminator.state_dict(), disc_path)
        
        print(f"  ✓ Models saved for {fruit_name}:")
        print(f"    - {gen_path.name}")
        print(f"    - {disc_path.name}")
    
    @staticmethod
    def _print_training_diagnostics(metrics, epoch):
        """Print training diagnostics and warnings"""
        fool_rate = metrics['discriminator_fooled']
        gen_loss = metrics['gen_loss']
        disc_loss = metrics['disc_loss']
        
        # Check for training issues
        warnings = []
        
        if fool_rate < 0.15:
            warnings.append(f"⚠️  DISCRIMINATOR TOO STRONG: Fool rate {fool_rate:.2%} (ideal: 20-50%)")
            warnings.append("    → Generator learning is blocked!")
            warnings.append("    → Try: Lower learning rate, reduce discriminator capacity, or train longer")
        elif fool_rate > 0.85:
            warnings.append(f"⚠️  GENERATOR TOO STRONG: Fool rate {fool_rate:.2%} (ideal: 20-50%)")
            warnings.append("    → Discriminator can't detect fake images")
            warnings.append("    → Try: Increase discriminator capacity or increase learning rate")
        
        if gen_loss > 5.0:
            warnings.append(f"⚠️  GEN LOSS EXPLODING: {gen_loss:.4f} (should be < 2.0)")
            warnings.append("    → Try: Lower learning rate")
        
        if disc_loss > 5.0:
            warnings.append(f"⚠️  DISC LOSS EXPLODING: {disc_loss:.4f} (should be < 2.0)")
            warnings.append("    → Try: Lower learning rate")
        
        if warnings:
            print("\n  🔧 TRAINING DIAGNOSTICS:")
            for warning in warnings:
                print(f"  {warning}")
    
    @staticmethod
    def _parse_time_to_seconds(time_str):
        """Convert formatted time string back to seconds"""
        if 's' in time_str:
            return float(time_str.replace('s', ''))
        elif 'm' in time_str:
            return float(time_str.replace('m', '')) * 60
        elif 'h' in time_str:
            return float(time_str.replace('h', '')) * 3600
        return 0
    
    def _save_all_histories(self, save_dir, model_name):
        """Save training histories for all fruits"""
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # Save individual fruit histories
        for fruit_name, history in self.training_history.items():
            history_path = save_dir / f'training_history_{fruit_name}.json'
            with open(history_path, 'w') as f:
                json.dump(history, f, indent=2)
        
        # Save summary
        summary = {
            'model_name': model_name,
            'fruits_trained': list(self.training_history.keys()),
            'training_timestamp': datetime.now().isoformat(),
            'history_files': {
                fruit: f'training_history_{fruit}.json' 
                for fruit in self.training_history.keys()
            }
        }
        
        summary_path = save_dir / f'training_summary_{model_name}.json'
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"\n  ✓ Training histories saved to: {save_dir}")
    
    @staticmethod
    def _format_time(seconds: float) -> str:
        """Format seconds into human-readable format"""
        if seconds < 60:
            return f"{seconds:.0f}s"
        elif seconds < 3600:
            minutes = seconds / 60
            return f"{minutes:.1f}m"
        else:
            hours = seconds / 3600
            return f"{hours:.2f}h"


if __name__ == "__main__":
    print("GAN Trainer module loaded successfully")
