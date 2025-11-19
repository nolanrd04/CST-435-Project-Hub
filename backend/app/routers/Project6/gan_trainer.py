"""
GAN Trainer Module
Handles training logic for GAN models with multi-fruit support
"""

import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
import json
import time
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from gan_model import Generator, Discriminator, initialize_weights
from data_loader import load_dataset_for_version, create_data_loader


class MultiFruitGANTrainer:
    """
    Trainer class for training multiple fruit-specific GANs
    Each fruit gets its own Generator/Discriminator pair
    """

    def __init__(self, model_name, data_version, config):
        """
        Initialize the MultiFruitGANTrainer

        Args:
            model_name (str): Name of the model (e.g., 'v1', 'attempt_2')
            data_version (str): Version of the dataset to use (e.g., 'v1')
            config (dict): Training configuration
        """
        self.model_name = model_name
        self.data_version = data_version
        self.config = config

        # Setup directories
        self.script_dir = Path(__file__).parent
        self.model_dir = self.script_dir / 'models' / f'model_{model_name}'
        self.model_dir.mkdir(parents=True, exist_ok=True)

        self.info_dir = self.model_dir / 'info'
        self.info_dir.mkdir(exist_ok=True)

        # Device configuration
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")

        # Training history for all fruits
        self.all_histories = {}

        # Save configuration
        self.save_config()

    def save_config(self):
        """Save training configuration to JSON"""
        config_path = self.info_dir / f'training_config_{self.model_name}.json'
        config_data = {
            'model_name': self.model_name,
            'data_version': self.data_version,
            'timestamp': datetime.now().isoformat(),
            'device': str(self.device),
            'config': self.config
        }

        with open(config_path, 'w') as f:
            json.dump(config_data, f, indent=2)

        print(f"Configuration saved to: {config_path}")

    def train_single_fruit(self, fruit_name):
        """
        Train a GAN for a single fruit type

        Args:
            fruit_name (str): Name of the fruit to train on

        Returns:
            dict: Training history
        """
        print(f"\n{'='*60}")
        print(f"Training GAN for: {fruit_name.upper()}")
        print(f"{'='*60}")

        start_time = time.time()

        # Load data for this specific fruit
        dataset, _, channels = load_dataset_for_version(
            self.data_version,
            selected_fruits=[fruit_name]
        )

        dataloader = create_data_loader(
            dataset,
            batch_size=self.config['batch_size'],
            shuffle=True
        )

        print(f"Dataset loaded: {len(dataset)} images")
        print(f"Channels: {channels}")

        # Get image size from config (auto-detected from data)
        img_size = self.config.get('img_size', 28)  # Default to 28 if not specified

        # Initialize models
        generator = Generator(
            latent_dim=self.config['latent_dim'],
            channels=channels,
            img_size=img_size
        ).to(self.device)

        discriminator = Discriminator(
            channels=channels,
            img_size=img_size
        ).to(self.device)

        # Initialize weights
        initialize_weights(generator)
        initialize_weights(discriminator)

        # Loss function
        adversarial_loss = nn.BCELoss()

        # Optimizers
        optimizer_G = optim.Adam(
            generator.parameters(),
            lr=self.config['learning_rate'],
            betas=(self.config['beta1'], 0.999)
        )

        optimizer_D = optim.Adam(
            discriminator.parameters(),
            lr=self.config['learning_rate'],
            betas=(self.config['beta1'], 0.999)
        )

        # Create directory for epoch images
        epoch_images_dir = self.model_dir / f'generated_epoch_images_{fruit_name}'
        epoch_images_dir.mkdir(exist_ok=True)

        # Training history
        history = {
            'fruit': fruit_name,
            'g_losses': [],
            'd_losses': [],
            'epochs': [],
            'timestamps': []
        }

        # Fixed noise for consistent visualization
        fixed_noise = torch.randn(16, self.config['latent_dim']).to(self.device)

        # Training loop
        epochs = self.config['epochs']
        print(f"\nStarting training for {epochs} epochs...")

        # Calculate milestone epochs (5 evenly spaced checkpoints)
        milestone_epochs = []
        if epochs >= 5:
            for i in range(1, 6):
                milestone = int(epochs * i / 5)
                milestone_epochs.append(milestone)
        else:
            milestone_epochs = list(range(1, epochs + 1))

        print(f"Will save images at epochs: {milestone_epochs}")

        # Time tracking for estimates
        epoch_times = []

        for epoch in range(1, epochs + 1):
            epoch_start = time.time()

            # Training metrics for this epoch
            g_losses = []
            d_losses = []

            # Progress bar
            pbar = tqdm(dataloader, desc=f'Epoch {epoch}/{epochs}')

            for i, real_imgs in enumerate(pbar):
                batch_size = real_imgs.size(0)
                real_imgs = real_imgs.to(self.device)

                # Adversarial ground truths
                valid = torch.ones(batch_size, 1).to(self.device)
                fake = torch.zeros(batch_size, 1).to(self.device)

                # -----------------
                # Train Generator
                # -----------------
                optimizer_G.zero_grad()

                # Sample noise
                z = torch.randn(batch_size, self.config['latent_dim']).to(self.device)

                # Generate fake images
                gen_imgs = generator(z)

                # Loss measures generator's ability to fool the discriminator
                g_loss = adversarial_loss(discriminator(gen_imgs), valid)

                g_loss.backward()
                optimizer_G.step()

                # ---------------------
                # Train Discriminator
                # ---------------------
                optimizer_D.zero_grad()

                # Measure discriminator's ability to classify real from generated samples
                real_loss = adversarial_loss(discriminator(real_imgs), valid)
                fake_loss = adversarial_loss(discriminator(gen_imgs.detach()), fake)
                d_loss = (real_loss + fake_loss) / 2

                d_loss.backward()
                optimizer_D.step()

                # Record losses
                g_losses.append(g_loss.item())
                d_losses.append(d_loss.item())

                # Update progress bar
                pbar.set_postfix({
                    'D_loss': f'{d_loss.item():.4f}',
                    'G_loss': f'{g_loss.item():.4f}'
                })

            # Epoch statistics
            avg_g_loss = np.mean(g_losses)
            avg_d_loss = np.mean(d_losses)
            epoch_time = time.time() - epoch_start
            epoch_times.append(epoch_time)

            # Save history
            history['epochs'].append(epoch)
            history['g_losses'].append(avg_g_loss)
            history['d_losses'].append(avg_d_loss)
            history['timestamps'].append(datetime.now().isoformat())

            # Calculate time estimate
            avg_epoch_time = np.mean(epoch_times)
            remaining_epochs = epochs - epoch
            est_time_remaining = avg_epoch_time * remaining_epochs

            # Format time remaining
            if est_time_remaining >= 60:
                time_str = f"{est_time_remaining/60:.1f}min"
            else:
                time_str = f"{est_time_remaining:.0f}s"

            print(f"Epoch {epoch}/{epochs} - "
                  f"D_loss: {avg_d_loss:.4f}, "
                  f"G_loss: {avg_g_loss:.4f}, "
                  f"Time: {epoch_time:.2f}s, "
                  f"ETA: {time_str}")

            # Save sample images at milestone epochs
            if epoch in milestone_epochs:
                self.save_sample_images(
                    generator,
                    fixed_noise,
                    epoch,
                    epoch_images_dir,
                    fruit_name
                )

        # Save final models
        self.save_models(generator, discriminator, fruit_name)

        # Save training history
        self.save_history(history, fruit_name)

        total_time = time.time() - start_time
        print(f"\nTraining completed for {fruit_name} in {total_time/60:.2f} minutes")

        return history

    def save_sample_images(self, generator, noise, epoch, save_dir, fruit_name):
        """
        Generate and save sample images

        Args:
            generator (Generator): The generator model
            noise (torch.Tensor): Fixed noise for generation
            epoch (int): Current epoch number
            save_dir (Path): Directory to save images
            fruit_name (str): Name of the fruit
        """
        generator.eval()

        with torch.no_grad():
            gen_imgs = generator(noise).cpu()

            # Denormalize from [-1, 1] to [0, 1]
            gen_imgs = (gen_imgs + 1) / 2.0

            # Create grid
            fig, axes = plt.subplots(4, 4, figsize=(8, 8))
            fig.suptitle(f'{fruit_name.capitalize()} - Epoch {epoch}', fontsize=16)

            for idx, ax in enumerate(axes.flat):
                img = gen_imgs[idx].squeeze().numpy()

                if gen_imgs.shape[1] == 1:  # Grayscale
                    ax.imshow(img, cmap='gray')
                else:  # RGB
                    img = np.transpose(img, (1, 2, 0))
                    ax.imshow(img)

                ax.axis('off')

            plt.tight_layout()

            # Save image
            save_path = save_dir / f'epoch_{epoch:04d}.png'
            plt.savefig(save_path, dpi=100, bbox_inches='tight')
            plt.close()

        generator.train()

    def save_models(self, generator, discriminator, fruit_name):
        """
        Save trained models

        Args:
            generator (Generator): The generator model
            discriminator (Discriminator): The discriminator model
            fruit_name (str): Name of the fruit
        """
        gen_path = self.model_dir / f'generator_{fruit_name}.pt'
        disc_path = self.model_dir / f'discriminator_{fruit_name}.pt'

        torch.save(generator.state_dict(), gen_path)
        torch.save(discriminator.state_dict(), disc_path)

        print(f"Models saved:")
        print(f"  Generator: {gen_path}")
        print(f"  Discriminator: {disc_path}")

    def save_history(self, history, fruit_name):
        """
        Save training history to JSON

        Args:
            history (dict): Training history
            fruit_name (str): Name of the fruit
        """
        history_path = self.info_dir / f'training_history_{fruit_name}.json'

        with open(history_path, 'w') as f:
            json.dump(history, f, indent=2)

        self.all_histories[fruit_name] = history
        print(f"Training history saved: {history_path}")

    def save_summary(self, fruits_trained):
        """
        Save overall training summary

        Args:
            fruits_trained (list): List of fruits that were trained
        """
        summary = {
            'model_name': self.model_name,
            'data_version': self.data_version,
            'fruits_trained': fruits_trained,
            'total_fruits': len(fruits_trained),
            'timestamp': datetime.now().isoformat(),
            'config': self.config
        }

        summary_path = self.info_dir / f'training_summary_{self.model_name}.json'

        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)

        print(f"\nTraining summary saved: {summary_path}")

    def train_all_fruits(self, fruits):
        """
        Train GANs for all specified fruits

        Args:
            fruits (list): List of fruit names to train

        Returns:
            dict: All training histories
        """
        print(f"\n{'='*60}")
        print(f"Multi-Fruit GAN Training")
        print(f"Model: {self.model_name}")
        print(f"Data Version: {self.data_version}")
        print(f"Fruits to train: {', '.join(fruits)} ({len(fruits)} fruits)")
        print(f"{'='*60}")

        total_start = time.time()
        fruit_times = []

        for idx, fruit in enumerate(fruits, 1):
            print(f"\n{'='*60}")
            print(f"Training fruit {idx}/{len(fruits)}: {fruit}")

            # Show estimated total time remaining based on completed fruits
            if fruit_times:
                avg_fruit_time = np.mean(fruit_times)
                remaining_fruits = len(fruits) - idx + 1
                est_total_remaining = avg_fruit_time * remaining_fruits

                if est_total_remaining >= 3600:
                    eta_str = f"{est_total_remaining/3600:.1f}h"
                elif est_total_remaining >= 60:
                    eta_str = f"{est_total_remaining/60:.1f}min"
                else:
                    eta_str = f"{est_total_remaining:.0f}s"

                print(f"Overall Progress: {idx-1}/{len(fruits)} complete, ETA for remaining: {eta_str}")

            print(f"{'='*60}")

            fruit_start = time.time()
            self.train_single_fruit(fruit)
            fruit_time = time.time() - fruit_start
            fruit_times.append(fruit_time)

        total_time = time.time() - total_start

        # Save summary
        self.save_summary(fruits)

        print(f"\n{'='*60}")
        print(f"ALL TRAINING COMPLETED!")
        print(f"{'='*60}")
        print(f"Total fruits trained: {len(fruits)}")
        print(f"Total training time: {total_time/3600:.2f} hours ({total_time/60:.1f} minutes)")
        print(f"Average time per fruit: {np.mean(fruit_times)/60:.1f} minutes")
        print(f"{'='*60}")

        return self.all_histories


if __name__ == "__main__":
    # Test the trainer
    print("Testing GAN Trainer...")

    config = {
        'epochs': 10,
        'batch_size': 32,
        'learning_rate': 0.0002,
        'beta1': 0.5,
        'latent_dim': 100
    }

    trainer = MultiFruitGANTrainer(
        model_name='test',
        data_version='v1',
        config=config
    )

    print("Trainer initialized successfully!")
