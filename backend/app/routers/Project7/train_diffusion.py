#!/usr/bin/env python3
"""
Diffusion Model Training Script
Trains a conditional diffusion model for grayscale-to-RGB colorization.
"""

import os
import sys
import json
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from typing import List, Tuple, Dict
import matplotlib.pyplot as plt

from diffusion_model import ConditionalUNet, DDPMScheduler, count_parameters
from cost_analysis_training import (
    DiffusionTrainingCostModel,
    generate_cost_report_from_training_history,
    print_cost_report
)


class ColorizationDataset(Dataset):
    """Dataset for loading grayscale-RGB pairs from NPZ files."""

    def __init__(self, npz_path: str, normalize: bool = True):
        """
        Args:
            npz_path: Path to NPZ file (train.npz or test.npz)
            normalize: Whether to normalize to [-1, 1] if data is uint8
        """
        data = np.load(npz_path)

        self.grayscale = data['grayscale']
        self.rgb = data['rgb']
        self.labels = data['labels']

        # Normalize if needed
        if normalize and self.grayscale.dtype == np.uint8:
            self.grayscale = (self.grayscale.astype(np.float32) / 127.5) - 1.0
            self.rgb = (self.rgb.astype(np.float32) / 127.5) - 1.0

        # Convert to torch tensors and fix dimensions
        # From (N, H, W, C) to (N, C, H, W)
        self.grayscale = torch.from_numpy(self.grayscale).permute(0, 3, 1, 2).float()
        self.rgb = torch.from_numpy(self.rgb).permute(0, 3, 1, 2).float()
        self.labels = torch.from_numpy(self.labels).long()

    def __len__(self):
        return len(self.grayscale)

    def __getitem__(self, idx):
        return {
            'grayscale': self.grayscale[idx],
            'rgb': self.rgb[idx],
            'label': self.labels[idx]
        }


class DiffusionTrainer:
    """Handles training of the diffusion model."""

    def __init__(
        self,
        model: nn.Module,
        scheduler: DDPMScheduler,
        train_loader: DataLoader,
        val_loader: DataLoader,
        device: str,
        output_dir: Path,
        config: dict
    ):
        self.model = model.to(device)
        self.scheduler = scheduler
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.output_dir = output_dir
        self.config = config

        # Setup optimizer
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=config['learning_rate']
        )

        # Loss function
        self.criterion = nn.MSELoss()

        # Training history
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'epoch_times': []
        }

        # Early stopping
        self.early_stopping_patience = config.get('early_stopping_patience', 0)
        self.early_stopping_counter = 0
        self.early_stopping_triggered = False

        # Cumulative training time
        self.total_training_time_seconds = 0.0

    def train_epoch(self, epoch: int) -> float:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        num_batches = len(self.train_loader)

        for batch_idx, batch in enumerate(self.train_loader):
            grayscale = batch['grayscale'].to(self.device)
            rgb = batch['rgb'].to(self.device)

            # Sample random timesteps
            timesteps = self.scheduler.sample_timesteps(rgb.shape[0])

            # Add noise to RGB images
            noise = torch.randn_like(rgb)
            noisy_rgb = self.scheduler.add_noise(rgb, noise, timesteps)

            # Concatenate noisy RGB + grayscale condition
            model_input = torch.cat([noisy_rgb, grayscale], dim=1)

            # Predict noise
            predicted_noise = self.model(model_input, timesteps)

            # Calculate loss
            loss = self.criterion(predicted_noise, noise)

            # Backpropagation
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()

            # Print progress
            if (batch_idx + 1) % 10 == 0 or (batch_idx + 1) == num_batches:
                print(f'  Batch [{batch_idx + 1}/{num_batches}] Loss: {loss.item():.4f}')

        return total_loss / num_batches

    @torch.no_grad()
    def validate(self) -> float:
        """Validate the model."""
        self.model.eval()
        total_loss = 0.0
        num_batches = len(self.val_loader)

        for batch in self.val_loader:
            grayscale = batch['grayscale'].to(self.device)
            rgb = batch['rgb'].to(self.device)

            # Sample random timesteps
            timesteps = self.scheduler.sample_timesteps(rgb.shape[0])

            # Add noise to RGB images
            noise = torch.randn_like(rgb)
            noisy_rgb = self.scheduler.add_noise(rgb, noise, timesteps)

            # Concatenate noisy RGB + grayscale condition
            model_input = torch.cat([noisy_rgb, grayscale], dim=1)

            # Predict noise
            predicted_noise = self.model(model_input, timesteps)

            # Calculate loss
            loss = self.criterion(predicted_noise, noise)
            total_loss += loss.item()

        return total_loss / num_batches

    def save_checkpoint(self, epoch: int, is_best: bool = False, keep_last_n: int = 3):
        """
        Save model checkpoint with smart management.

        Args:
            epoch: Current epoch number
            is_best: Whether this is the best model so far
            keep_last_n: Number of recent checkpoints to keep (default: 5)
        """
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config,
            'history': self.history,
            'early_stopping_counter': self.early_stopping_counter,
            'total_training_time_seconds': self.total_training_time_seconds
        }

        # Save regular checkpoint
        checkpoint_path = self.output_dir / f'checkpoint_epoch_{epoch}.pth'
        torch.save(checkpoint, checkpoint_path)

        # Save best model
        if is_best:
            best_path = self.output_dir / 'best_model.pth'
            torch.save(checkpoint, best_path)
            print(f'  Saved best model to {best_path}')

        # Clean up old checkpoints (keep only last N)
        all_checkpoints = sorted(
            self.output_dir.glob('checkpoint_epoch_*.pth'),
            key=lambda p: int(p.stem.split('_')[-1])
        )

        # Remove old checkpoints beyond keep_last_n
        if len(all_checkpoints) > keep_last_n:
            for old_checkpoint in all_checkpoints[:-keep_last_n]:
                old_checkpoint.unlink()
                # Silently delete old checkpoints

    def load_checkpoint(self, checkpoint_path: Path):
        """
        Load checkpoint and restore training state.

        Args:
            checkpoint_path: Path to checkpoint file

        Returns:
            Starting epoch number
        """
        print(f"\nLoading checkpoint from {checkpoint_path}")

        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.history = checkpoint['history']
        self.early_stopping_counter = checkpoint.get('early_stopping_counter', 0)
        self.total_training_time_seconds = checkpoint.get('total_training_time_seconds', 0.0)

        start_epoch = checkpoint['epoch']

        print(f"  Resuming from epoch {checkpoint['epoch']}")
        print(f"  Previous best val loss: {min(self.history['val_loss']):.4f}")
        print(f"  Early stopping counter: {self.early_stopping_counter}")
        if self.total_training_time_seconds > 0:
            print(f"  Previous training time: {self.total_training_time_seconds / 3600:.2f} hours")

        return start_epoch

    @torch.no_grad()
    def generate_samples(self, epoch: int, num_samples: int = 4):
        """Generate sample colorizations."""
        self.model.eval()

        # Get a batch from validation set
        batch = next(iter(self.val_loader))
        grayscale = batch['grayscale'][:num_samples].to(self.device)
        rgb_target = batch['rgb'][:num_samples]

        # Generate RGB predictions
        rgb_pred = self.scheduler.generate(self.model, grayscale, self.device)

        # Move to CPU and denormalize
        grayscale_vis = (grayscale.cpu() + 1.0) / 2.0
        rgb_target_vis = (rgb_target + 1.0) / 2.0
        rgb_pred_vis = (rgb_pred.cpu() + 1.0) / 2.0

        # Create visualization
        fig, axes = plt.subplots(num_samples, 3, figsize=(12, 4 * num_samples))

        for i in range(num_samples):
            # Grayscale input
            axes[i, 0].imshow(grayscale_vis[i, 0], cmap='gray')
            axes[i, 0].set_title('Grayscale Input')
            axes[i, 0].axis('off')

            # Target RGB
            axes[i, 1].imshow(rgb_target_vis[i].permute(1, 2, 0))
            axes[i, 1].set_title('Target RGB')
            axes[i, 1].axis('off')

            # Predicted RGB
            axes[i, 2].imshow(rgb_pred_vis[i].permute(1, 2, 0))
            axes[i, 2].set_title('Generated RGB')
            axes[i, 2].axis('off')

        plt.tight_layout()
        sample_path = self.output_dir / f'samples_epoch_{epoch}.png'
        plt.savefig(sample_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f'  Saved samples to {sample_path}')

    def train(self, num_epochs: int, resume_from_epoch: int = 0):
        """
        Main training loop with time estimation.

        Args:
            num_epochs: Total number of epochs to train
            resume_from_epoch: Epoch to resume from (0 = start fresh)
        """
        print("\n" + "=" * 70)
        if resume_from_epoch > 0:
            print(f"Resuming Training from Epoch {resume_from_epoch}")
        else:
            print("Starting Training")
        print("=" * 70)

        # Initialize best validation loss from history if resuming
        if resume_from_epoch > 0 and self.history['val_loss']:
            best_val_loss = min(self.history['val_loss'])
        else:
            best_val_loss = float('inf')

        start_time = time.time()
        epoch_times = self.history.get('epoch_times', []).copy()

        for epoch in range(resume_from_epoch + 1, num_epochs + 1):
            epoch_start = time.time()

            print(f"\nEpoch [{epoch}/{num_epochs}]")
            print("-" * 70)

            # Train
            train_loss = self.train_epoch(epoch)
            print(f"  Train Loss: {train_loss:.4f}")

            # Validate
            val_loss = self.validate()
            print(f"  Val Loss: {val_loss:.4f}")

            # Update history
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)

            # Time tracking
            epoch_time = time.time() - epoch_start
            epoch_times.append(epoch_time)
            self.history['epoch_times'].append(epoch_time)
            self.total_training_time_seconds += epoch_time  # Accumulate total time

            # Estimate remaining time
            avg_epoch_time = np.mean(epoch_times)
            remaining_epochs = num_epochs - epoch
            estimated_remaining = avg_epoch_time * remaining_epochs

            print(f"  Epoch Time: {format_time(epoch_time)}")
            print(f"  Estimated Time Remaining: {format_time(estimated_remaining)}")

            # Save checkpoint
            is_best = val_loss < best_val_loss
            if is_best:
                best_val_loss = val_loss
                self.early_stopping_counter = 0  # Reset counter on improvement
            else:
                self.early_stopping_counter += 1

            self.save_checkpoint(epoch, is_best)

            # Early stopping check
            if self.early_stopping_patience > 0:
                if self.early_stopping_counter >= self.early_stopping_patience:
                    print(f"\n  Early stopping triggered! No improvement for {self.early_stopping_patience} epochs.")
                    self.early_stopping_triggered = True
                    break
                else:
                    print(f"  Early stopping: {self.early_stopping_counter}/{self.early_stopping_patience} (no improvement)")

            # Generate samples every N epochs
            if epoch % self.config.get('sample_every', 5) == 0 or epoch == num_epochs:
                print("  Generating samples...")
                self.generate_samples(epoch)

        # Training complete
        session_time = time.time() - start_time
        print("\n" + "=" * 70)
        if self.early_stopping_triggered:
            print("Training Stopped Early!")
            print(f"Completed {epoch} / {num_epochs} epochs")
        else:
            print("Training Complete!")
        print(f"This Session Time: {format_time(session_time)}")
        print(f"Total Cumulative Time: {format_time(self.total_training_time_seconds)}")
        print(f"Best Validation Loss: {best_val_loss:.4f}")
        print("=" * 70)

        # Save final training history
        self.save_training_history()

        # Return training stats for cost analysis
        return {
            'total_time': self.total_training_time_seconds,
            'session_time': session_time,
            'num_epochs': num_epochs,
            'best_val_loss': best_val_loss
        }

    def save_training_history(self):
        """Save training history and plot loss curves."""
        # Save JSON
        history_path = self.output_dir / 'training_history.json'
        with open(history_path, 'w') as f:
            json.dump(self.history, f, indent=2)

        # Plot loss curves
        plt.figure(figsize=(10, 6))
        plt.plot(self.history['train_loss'], label='Train Loss')
        plt.plot(self.history['val_loss'], label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training History')
        plt.legend()
        plt.grid(True)
        plt.savefig(self.output_dir / 'loss_curve.png', dpi=150, bbox_inches='tight')
        plt.close()


def format_time(seconds: float) -> str:
    """Format seconds into human-readable time string."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)

    if hours > 0:
        return f"{hours}h {minutes}m {secs}s"
    elif minutes > 0:
        return f"{minutes}m {secs}s"
    else:
        return f"{secs}s"


def get_available_datasets(script_dir: Path) -> List[str]:
    """Scan npzData/ directory for available datasets."""
    npz_data_dir = script_dir / "npzData"

    if not npz_data_dir.exists():
        return []

    datasets = []
    for item in npz_data_dir.iterdir():
        if item.is_dir():
            # Check if train.npz exists
            if (item / "train.npz").exists():
                datasets.append(item.name)

    return sorted(datasets)


def display_available_datasets(datasets: List[str], script_dir: Path):
    """Display available datasets with statistics."""
    print("=" * 70)
    print("Available Datasets:")
    print("=" * 70)

    if not datasets:
        print("  No datasets found in npzData/")
        return

    for idx, dataset_name in enumerate(datasets, 1):
        dataset_path = script_dir / "npzData" / dataset_name
        metadata_path = dataset_path / "metadata.json"

        if metadata_path.exists():
            try:
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                train_samples = metadata.get('train_samples', 0)
                test_samples = metadata.get('test_samples', 0)
                img_size = metadata.get('image_shape', {}).get('rgb', [64, 64, 3])
                print(f"  {idx}. {dataset_name}")
                print(f"     Train: {train_samples}, Test: {test_samples}")
                print(f"     Image size: {img_size[0]}x{img_size[1]}")
            except:
                print(f"  {idx}. {dataset_name} (metadata unavailable)")
        else:
            print(f"  {idx}. {dataset_name}")

    print("=" * 70)


def select_dataset(datasets: List[str]) -> str:
    """Let user select a dataset."""
    print("\nEnter dataset number or name:")
    print("(Press Enter to cancel)")

    user_input = input("\nSelection: ").strip()

    if not user_input:
        return ""

    # Check if input is a number
    try:
        idx = int(user_input)
        if 1 <= idx <= len(datasets):
            return datasets[idx - 1]
        else:
            print(f"Error: Number must be between 1 and {len(datasets)}")
            return ""
    except ValueError:
        # Input is a name
        if user_input in datasets:
            return user_input
        else:
            print(f"Error: Dataset '{user_input}' not found")
            return ""


def get_model_name(dataset_name: str, script_dir: Path) -> tuple[str, bool]:
    """
    Let user choose model name.

    Returns:
        (model_name, override_existing) - model name and whether to override if exists
    """
    print("\n" + "=" * 70)
    print("Model Name Configuration")
    print("=" * 70)
    print(f"\nDefault model name: {dataset_name}")
    print("You can create a new model or override an existing one")

    custom_name = input("\nEnter custom model name (or press Enter to use default): ").strip()
    model_name = custom_name if custom_name else dataset_name

    # Check if model directory already exists
    model_dir = script_dir / "models" / model_name
    override = False

    if model_dir.exists():
        print(f"\n! Model '{model_name}' already exists at:")
        print(f"  {model_dir}")

        # Check for existing checkpoints
        existing_checkpoints = list(model_dir.glob('checkpoint_epoch_*.pth'))
        if existing_checkpoints:
            latest_checkpoint = max(existing_checkpoints, key=lambda p: int(p.stem.split('_')[-1]))
            latest_epoch = int(latest_checkpoint.stem.split('_')[-1])
            print(f"\n  Found checkpoint at epoch {latest_epoch}")
            print("\nOptions:")
            print("  1. Resume training from latest checkpoint")
            print("  2. Override existing model (start fresh training)")
            print("  3. Cancel and choose different name")

            choice = input("\nYour choice (1/2/3): ").strip()

            if choice == '1':
                override = False
                return model_name, override, latest_checkpoint
            elif choice == '2':
                confirm = input("\n! This will DELETE all existing checkpoints. Continue? (yes/no): ").strip().lower()
                if confirm == 'yes':
                    override = True
                    return model_name, override, None
                else:
                    print("Cancelled. Please choose a different name.")
                    return get_model_name(dataset_name, script_dir)
            else:
                print("Cancelled. Please choose a different name.")
                return get_model_name(dataset_name, script_dir)
        else:
            print("  No checkpoints found in this directory")
            override = True

    return model_name, override, None


def get_model_config() -> dict:
    """Get model configuration from user."""
    print("\n" + "=" * 70)
    print("Model Configuration")
    print("=" * 70)

    config = {}

    # U-Net features
    print("\nU-Net Architecture:")
    print("  Default features: [64, 128, 256, 512]")
    print("  Recommended for better quality: [128, 256, 512, 1024]")
    features_input = input("  Enter custom features (comma-separated) or press Enter for default: ").strip()

    if features_input:
        try:
            config['features'] = [int(x.strip()) for x in features_input.split(',')]
        except:
            print("  Invalid input, using default")
            config['features'] = [64, 128, 256, 512]
    else:
        config['features'] = [64, 128, 256, 512]

    # Time embedding dimension
    time_emb_input = input("\nTime embedding dimension (default: 256): ").strip()
    config['time_emb_dim'] = int(time_emb_input) if time_emb_input else 256

    # Diffusion timesteps
    timesteps_input = input("\nDiffusion timesteps (default: 1000): ").strip()
    config['timesteps'] = int(timesteps_input) if timesteps_input else 1000

    # Training parameters
    print("\nTraining Parameters:")

    epochs_input = input("  Number of epochs (default: 50): ").strip()
    config['num_epochs'] = int(epochs_input) if epochs_input else 50

    batch_input = input("  Batch size (default: 16): ").strip()
    config['batch_size'] = int(batch_input) if batch_input else 16

    lr_input = input("  Learning rate (default: 1e-4): ").strip()
    config['learning_rate'] = float(lr_input) if lr_input else 1e-4

    sample_input = input("  Generate samples every N epochs (default: 5): ").strip()
    config['sample_every'] = int(sample_input) if sample_input else 5

    # Early stopping
    print("\nEarly Stopping:")
    print("  Stop training if validation loss doesn't improve for N epochs")
    print("  Set to 0 to disable early stopping")
    early_stop_input = input("  Early stopping patience (default: 10): ").strip()
    config['early_stopping_patience'] = int(early_stop_input) if early_stop_input else 10

    return config


def estimate_training_time(
    train_loader: DataLoader,
    device: str,
    num_epochs: int
) -> float:
    """
    Estimate total training time by running a few test batches.
    """
    print("\nEstimating training time...")

    # Create dummy model for timing
    model = ConditionalUNet().to(device)
    scheduler = DDPMScheduler(device=device)
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-4)

    # Time a few batches
    batch_times = []
    for i, batch in enumerate(train_loader):
        if i >= 3:  # Time 3 batches
            break

        batch_start = time.time()

        grayscale = batch['grayscale'].to(device)
        rgb = batch['rgb'].to(device)

        timesteps = scheduler.sample_timesteps(rgb.shape[0])
        noise = torch.randn_like(rgb)
        noisy_rgb = scheduler.add_noise(rgb, noise, timesteps)
        model_input = torch.cat([noisy_rgb, grayscale], dim=1)

        predicted_noise = model(model_input, timesteps)
        loss = criterion(predicted_noise, noise)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_times.append(time.time() - batch_start)

    avg_batch_time = np.mean(batch_times)
    batches_per_epoch = len(train_loader)
    estimated_total = avg_batch_time * batches_per_epoch * num_epochs

    print(f"  Average batch time: {avg_batch_time:.2f}s")
    print(f"  Batches per epoch: {batches_per_epoch}")
    print(f"  Estimated total training time: {format_time(estimated_total)}")

    del model, scheduler, optimizer
    torch.cuda.empty_cache() if device == 'cuda' else None

    return estimated_total


def main():
    """Main entry point."""
    script_dir = Path(__file__).parent.resolve()

    print(f"Script directory: {script_dir}")

    # Select dataset
    available_datasets = get_available_datasets(script_dir)

    if not available_datasets:
        print("\nError: No datasets found in npzData/")
        print("Please run image_to_npz.py first.")
        sys.exit(1)

    display_available_datasets(available_datasets, script_dir)
    dataset_name = select_dataset(available_datasets)

    if not dataset_name:
        print("Training cancelled.")
        sys.exit(0)

    print(f"\nSelected dataset: {dataset_name}")

    # Get model name and check for override
    model_name, should_override, resume_checkpoint = get_model_name(dataset_name, script_dir)

    print(f"\nModel name: {model_name}")

    # Setup device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # Create output directory
    output_dir = script_dir / "models" / model_name
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir}")

    resume_from_epoch = 0
    config = None

    # Handle override
    if should_override:
        print("\n! Overriding existing model...")
        # Delete all existing checkpoints and training files
        for file in output_dir.glob('checkpoint_epoch_*.pth'):
            file.unlink()
        for file in output_dir.glob('best_model.pth'):
            file.unlink()
        for file in output_dir.glob('samples_epoch_*.png'):
            file.unlink()
        print("  Deleted all existing checkpoints and samples")

    # Get configuration
    if resume_checkpoint:
        # Load existing config
        config_path = output_dir / 'config.json'
        if config_path.exists():
            with open(config_path, 'r') as f:
                config = json.load(f)
            print("Loaded existing configuration from checkpoint")
            resume_from_epoch = int(resume_checkpoint.stem.split('_')[-1])

            # Ask if user wants to extend training
            print(f"\nCurrent configuration: {config['num_epochs']} total epochs")
            print(f"Training completed up to epoch {resume_from_epoch}")

            if resume_from_epoch >= config['num_epochs']:
                print("\n! Training already reached target epochs.")
                extend = input("Extend training beyond current limit? (y/n): ").strip().lower()
                if extend == 'y':
                    new_epochs = input(f"New total epochs (current: {config['num_epochs']}): ").strip()
                    if new_epochs:
                        config['num_epochs'] = int(new_epochs)
                        print(f"Extended to {config['num_epochs']} epochs")
                        # Save updated config
                        with open(config_path, 'w') as f:
                            json.dump(config, f, indent=2)
                else:
                    print("Training cancelled.")
                    sys.exit(0)
    else:
        # Get new configuration
        config = get_model_config()

        # Save configuration
        with open(output_dir / 'config.json', 'w') as f:
            json.dump(config, f, indent=2)
        print("\nSaved configuration")

    # Load datasets
    print("\nLoading datasets...")
    dataset_path = script_dir / "npzData" / dataset_name

    train_dataset = ColorizationDataset(dataset_path / "train.npz")
    val_dataset = ColorizationDataset(dataset_path / "test.npz")

    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=0
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=0
    )

    print(f"  Train samples: {len(train_dataset)}")
    print(f"  Val samples: {len(val_dataset)}")
    print(f"  Batches per epoch: {len(train_loader)}")

    # Create model
    print("\nCreating model...")
    model = ConditionalUNet(
        in_channels=4,
        out_channels=3,
        features=config['features'],
        time_emb_dim=config['time_emb_dim']
    )
    scheduler = DDPMScheduler(
        timesteps=config['timesteps'],
        device=device
    )

    num_params = count_parameters(model)
    print(f"  Model parameters: {num_params:,}")

    # Estimate training time
    estimated_time = estimate_training_time(
        train_loader,
        device,
        config['num_epochs']
    )

    # Confirm before starting
    print("\n" + "=" * 70)
    print("Ready to start training")
    print("=" * 70)
    confirm = input("\nProceed with training? (y/n): ").strip().lower()

    if confirm != 'y':
        print("Training cancelled.")
        sys.exit(0)

    # Train
    trainer = DiffusionTrainer(
        model=model,
        scheduler=scheduler,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        output_dir=output_dir,
        config=config
    )

    # Load checkpoint if resuming
    if resume_checkpoint:
        resume_from_epoch = trainer.load_checkpoint(resume_checkpoint)
    else:
        resume_from_epoch = 0

    training_stats = trainer.train(config['num_epochs'], resume_from_epoch=resume_from_epoch)

    # Generate cost analysis report
    print("\n" + "=" * 70)
    print("Generating Cost Analysis Report...")
    print("=" * 70)

    try:
        # Estimate memory usage based on model size and batch size
        estimated_memory_gb = (num_params * 4) / (1024 ** 3)  # Model weights in GB
        estimated_memory_gb += (config['batch_size'] * 64 * 64 * 4 * 4) / (1024 ** 3)  # Batch data
        estimated_memory_gb = max(estimated_memory_gb, 2.0)  # Minimum 2GB

        report = generate_cost_report_from_training_history(
            model_dir=output_dir,
            model_name=model_name,
            cpus_used=2.0,
            peak_memory_gb=estimated_memory_gb,
            use_gpu=(device == 'cuda')
        )

        print_cost_report(report)
        print(f"\nCost report saved to: {output_dir / 'cost_analysis_report.json'}")

    except Exception as e:
        print(f"Warning: Could not generate cost analysis report: {e}")


if __name__ == "__main__":
    main()
