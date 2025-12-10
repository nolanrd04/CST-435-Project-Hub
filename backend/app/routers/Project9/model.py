"""
Multi-Object Image Colorizer - PyTorch Training Script

This script creates, configures, and trains a U-Net model for image colorization.
Prompts user for model name, hyperparameters, and dataset path.
Uses script-relative directory management.
"""

import os
import json
import time
from pathlib import Path
from datetime import datetime
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms


# ==================== MODEL ARCHITECTURE ====================

class DoubleConv(nn.Module):
    """Double Convolution block: (Conv2D -> BatchNorm -> ReLU) x 2"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class UNetColorizer(nn.Module):
    """
    U-Net architecture for image colorization

    Input: Grayscale image (B, 1, H, W)
    Output: RGB image (B, 3, H, W)
    """

    def __init__(self, config):
        super().__init__()
        self.config = config

        # Encoder (downsampling path)
        self.enc1 = DoubleConv(1, 64)
        self.pool1 = nn.MaxPool2d(2)

        self.enc2 = DoubleConv(64, 128)
        self.pool2 = nn.MaxPool2d(2)

        self.enc3 = DoubleConv(128, 256)
        self.pool3 = nn.MaxPool2d(2)

        self.enc4 = DoubleConv(256, 512)
        self.pool4 = nn.MaxPool2d(2)

        # Bottleneck
        self.bottleneck = DoubleConv(512, 1024)

        # Decoder (upsampling path)
        self.upconv4 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.dec4 = DoubleConv(1024, 512)

        self.upconv3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec3 = DoubleConv(512, 256)

        self.upconv2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec2 = DoubleConv(256, 128)

        self.upconv1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec1 = DoubleConv(128, 64)

        # Output layer
        self.out = nn.Conv2d(64, 3, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Encoder with skip connections
        enc1_out = self.enc1(x)
        enc2_out = self.enc2(self.pool1(enc1_out))
        enc3_out = self.enc3(self.pool2(enc2_out))
        enc4_out = self.enc4(self.pool3(enc3_out))

        # Bottleneck
        bottleneck_out = self.bottleneck(self.pool4(enc4_out))

        # Decoder with skip connections
        dec4_in = self.upconv4(bottleneck_out)
        dec4_in = torch.cat([dec4_in, enc4_out], dim=1)
        dec4_out = self.dec4(dec4_in)

        dec3_in = self.upconv3(dec4_out)
        dec3_in = torch.cat([dec3_in, enc3_out], dim=1)
        dec3_out = self.dec3(dec3_in)

        dec2_in = self.upconv2(dec3_out)
        dec2_in = torch.cat([dec2_in, enc2_out], dim=1)
        dec2_out = self.dec2(dec2_in)

        dec1_in = self.upconv1(dec2_out)
        dec1_in = torch.cat([dec1_in, enc1_out], dim=1)
        dec1_out = self.dec1(dec1_in)

        # Output
        out = self.out(dec1_out)
        out = self.sigmoid(out)

        return out

    def count_parameters(self):
        """Count total trainable parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ==================== DATASET ====================

class ColorizerDataset(Dataset):
    """Dataset for loading grayscale and color image pairs"""

    def __init__(self, color_dir, image_size=256, augment=False, augmentation_config=None):
        """
        Args:
            color_dir: Directory containing color images
            image_size: Size to resize images to
            augment: Whether to apply data augmentation
            augmentation_config: Dict with augmentation parameters
        """
        self.color_dir = Path(color_dir)
        self.image_size = image_size
        self.augment = augment
        self.augmentation_config = augmentation_config or {}

        # Find all images
        self.image_files = []
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']:
            self.image_files.extend(list(self.color_dir.rglob(ext)))

        if len(self.image_files) == 0:
            raise ValueError(f"No images found in {color_dir}")

        print(f"[OK] Found {len(self.image_files)} images in {color_dir}")

        # Base transforms
        self.to_tensor = transforms.ToTensor()
        self.resize = transforms.Resize((image_size, image_size))

        # Augmentation transforms
        if augment and augmentation_config:
            aug_transforms = []
            if augmentation_config.get('horizontal_flip', False):
                aug_transforms.append(transforms.RandomHorizontalFlip(p=0.5))
            if augmentation_config.get('rotation_degrees', 0) > 0:
                aug_transforms.append(
                    transforms.RandomRotation(augmentation_config['rotation_degrees'])
                )
            self.augment_transform = transforms.Compose(aug_transforms) if aug_transforms else None
        else:
            self.augment_transform = None

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        # Load color image
        img_path = self.image_files[idx]
        color_img = Image.open(img_path).convert('RGB')

        # Resize
        color_img = self.resize(color_img)

        # Apply augmentation (if enabled)
        if self.augment_transform:
            color_img = self.augment_transform(color_img)

        # Convert to grayscale
        grayscale_img = color_img.convert('L')

        # Convert to tensors
        color_tensor = self.to_tensor(color_img)
        grayscale_tensor = self.to_tensor(grayscale_img)

        # Apply brightness augmentation to color image only
        if self.augment and self.augmentation_config.get('brightness_factor', 0) > 0:
            brightness_factor = self.augmentation_config['brightness_factor']
            brightness_change = 1.0 + np.random.uniform(-brightness_factor, brightness_factor)
            color_tensor = torch.clamp(color_tensor * brightness_change, 0, 1)

        return grayscale_tensor, color_tensor


# ==================== TRAINING ====================

class Trainer:
    """Handles model training and validation"""

    def __init__(self, model, train_loader, val_loader, config, directories, device):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.directories = directories
        self.device = device

        # Setup loss function
        if config['loss_function'] == 'mse':
            self.criterion = nn.MSELoss()
        elif config['loss_function'] == 'l1':
            self.criterion = nn.L1Loss()
        else:
            self.criterion = nn.MSELoss()

        # Setup optimizer
        if config['optimizer'] == 'adam':
            self.optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])
        elif config['optimizer'] == 'adamw':
            self.optimizer = optim.AdamW(model.parameters(), lr=config['learning_rate'])
        elif config['optimizer'] == 'sgd':
            self.optimizer = optim.SGD(model.parameters(), lr=config['learning_rate'], momentum=0.9)
        else:
            self.optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])

        # Setup scheduler
        self.scheduler = None
        if config['scheduler']:
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                patience=config['scheduler']['patience'],
                factor=config['scheduler']['factor']
            )

        # Training history
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'learning_rates': [],
            'epoch_times': []
        }

        # Early stopping
        self.best_val_loss = float('inf')
        self.epochs_without_improvement = 0
        self.early_stopping_patience = config['early_stopping_patience']

        # Cumulative training time
        self.total_training_time_seconds = 0.0

    def train_epoch(self, batch_times, current_epoch, total_epochs):
        """Train for one epoch with batch-level time estimates"""
        self.model.train()
        total_loss = 0
        num_batches = len(self.train_loader)

        for batch_idx, (grayscale, color) in enumerate(self.train_loader):
            batch_start = time.time()

            grayscale = grayscale.to(self.device)
            color = color.to(self.device)

            # Forward pass
            self.optimizer.zero_grad()
            output = self.model(grayscale)
            loss = self.criterion(output, color)

            # Backward pass
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()

            # Track batch time
            batch_time = time.time() - batch_start
            batch_times.append(batch_time)

            # Show progress - detailed every 10 batches, simple indicator every 2 batches
            if (batch_idx + 1) % 10 == 0 or (batch_idx + 1) == num_batches:
                # Detailed progress with time estimates
                avg_batch_time = np.mean(batch_times[-min(50, len(batch_times)):])
                remaining_batches_epoch = num_batches - (batch_idx + 1)
                eta_epoch = avg_batch_time * remaining_batches_epoch

                # Estimate total training time
                batches_per_epoch = num_batches
                epochs_remaining = total_epochs - current_epoch
                total_batches_remaining = (remaining_batches_epoch +
                                          (epochs_remaining * batches_per_epoch))
                eta_total = avg_batch_time * total_batches_remaining

                print(f"  Batch [{batch_idx + 1}/{num_batches}] "
                      f"Loss: {loss.item():.6f} | "
                      f"Batch Time: {batch_time:.2f}s")
                print(f"    Epoch ETA: {format_time(eta_epoch)} | "
                      f"Total Training ETA: {format_time(eta_total)}")
            elif (batch_idx + 1) % 2 == 0:
                # Simple progress indicator every 2 batches
                print(f"  Batch [{batch_idx + 1}/{num_batches}] Loss: {loss.item():.6f}", flush=True)

        avg_loss = total_loss / num_batches
        return avg_loss

    def validate(self):
        """Validate the model"""
        self.model.eval()
        total_loss = 0
        num_batches = len(self.val_loader)

        with torch.no_grad():
            for grayscale, color in self.val_loader:
                grayscale = grayscale.to(self.device)
                color = color.to(self.device)

                output = self.model(grayscale)
                loss = self.criterion(output, color)

                total_loss += loss.item()

        avg_loss = total_loss / num_batches
        return avg_loss

    def save_checkpoint(self, epoch, is_best=False):
        """Save model checkpoint with full state for resuming"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config,
            'history': self.history,
            'best_val_loss': self.best_val_loss,
            'epochs_without_improvement': self.epochs_without_improvement,
            'total_training_time_seconds': self.total_training_time_seconds
        }

        # Include scheduler state if using one
        if self.scheduler is not None:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()

        if is_best:
            save_path = self.directories['saved_models'] / f"{self.config['model_name']}_best.pth"
            torch.save(checkpoint, save_path)
            print(f"  [OK] Saved best model to {save_path.name}")

        # Save latest checkpoint (for resuming)
        save_path = self.directories['saved_models'] / f"{self.config['model_name']}_latest.pth"
        torch.save(checkpoint, save_path)

    def load_checkpoint(self, checkpoint_path):
        """
        Load checkpoint and restore training state.

        Args:
            checkpoint_path: Path to checkpoint file

        Returns:
            Starting epoch number
        """
        print(f"\n[RESUME] Loading checkpoint from {checkpoint_path.name}")

        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.history = checkpoint['history']
        self.best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        self.epochs_without_improvement = checkpoint.get('epochs_without_improvement', 0)
        self.total_training_time_seconds = checkpoint.get('total_training_time_seconds', 0.0)

        # Load scheduler state if it exists
        if self.scheduler is not None and 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        start_epoch = checkpoint['epoch']

        print(f"  Resuming from epoch {start_epoch}/{self.config['epochs']}")
        print(f"  Best validation loss: {self.best_val_loss:.6f}")
        print(f"  Early stopping counter: {self.epochs_without_improvement}/{self.early_stopping_patience}")
        if self.total_training_time_seconds > 0:
            print(f"  Previous training time: {format_time(self.total_training_time_seconds)}")

        return start_epoch

    def train(self, resume_from_epoch=0):
        """
        Main training loop with time estimation and resume support.

        Args:
            resume_from_epoch: Epoch to resume from (0 = start fresh)
        """
        print("\n" + "=" * 60)
        if resume_from_epoch > 0:
            print(f"Resuming Training from Epoch {resume_from_epoch + 1}")
        else:
            print("Starting Training")
        print("=" * 60)

        # Initialize best validation loss from history if resuming
        if resume_from_epoch > 0 and self.history['val_loss']:
            self.best_val_loss = min(self.history['val_loss'])

        start_time = time.time()
        epoch_times = self.history.get('epoch_times', []).copy()
        batch_times = []

        for epoch in range(resume_from_epoch, self.config['epochs']):
            epoch_start = time.time()

            print(f"\nEpoch [{epoch + 1}/{self.config['epochs']}]")
            print("-" * 60)

            # Train
            train_loss = self.train_epoch(batch_times, epoch + 1, self.config['epochs'])

            # Validate
            val_loss = self.validate()

            # Update history
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            current_lr = self.optimizer.param_groups[0]['lr']
            self.history['learning_rates'].append(current_lr)

            # Time tracking
            epoch_time = time.time() - epoch_start
            epoch_times.append(epoch_time)
            self.history['epoch_times'].append(epoch_time)
            self.total_training_time_seconds += epoch_time  # Accumulate total time

            # Calculate time estimates
            avg_epoch_time = np.mean(epoch_times)
            remaining_epochs = self.config['epochs'] - (epoch + 1)
            estimated_remaining = avg_epoch_time * remaining_epochs

            # Print epoch summary
            print(f"\n  Train Loss: {train_loss:.6f}")
            print(f"  Val Loss:   {val_loss:.6f}")
            print(f"  LR:         {current_lr:.6f}")
            print(f"  Epoch Time: {format_time(epoch_time)}")
            print(f"  Estimated Time Remaining: {format_time(estimated_remaining)}")

            # Check for best model
            is_best = val_loss < self.best_val_loss
            if is_best:
                self.best_val_loss = val_loss
                self.epochs_without_improvement = 0
                print(f"  [OK] New best validation loss: {val_loss:.6f}")
            else:
                self.epochs_without_improvement += 1

            # Save checkpoint
            self.save_checkpoint(epoch + 1, is_best=is_best)

            # Learning rate scheduler
            if self.scheduler:
                old_lr = self.optimizer.param_groups[0]['lr']
                self.scheduler.step(val_loss)
                new_lr = self.optimizer.param_groups[0]['lr']
                if new_lr != old_lr:
                    print(f"  [SCHEDULER] Learning rate reduced: {old_lr:.6f} -> {new_lr:.6f}")

            # Early stopping
            if self.early_stopping_patience and self.epochs_without_improvement >= self.early_stopping_patience:
                print(f"\n[EARLY STOPPING] No improvement for {self.early_stopping_patience} epochs")
                break

        # Training complete
        session_time = time.time() - start_time
        print("\n" + "=" * 60)
        print("Training Complete!")
        print("=" * 60)
        print(f"This Session Time: {format_time(session_time)}")
        print(f"Total Cumulative Time: {format_time(self.total_training_time_seconds)}")
        print(f"Average Epoch Time: {format_time(np.mean(epoch_times))}")
        print(f"Best Validation Loss: {self.best_val_loss:.6f}")

        return self.history


def format_time(seconds):
    """Format seconds into human-readable time string"""
    if seconds < 0:
        return "0s"

    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)

    if hours > 0:
        return f"{hours}h {minutes}m {secs}s"
    elif minutes > 0:
        return f"{minutes}m {secs}s"
    else:
        return f"{secs}s"


def plot_training_history(history, save_path):
    """Plot and save training history"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    # Loss plot
    ax1.plot(history['train_loss'], label='Train Loss', marker='o')
    ax1.plot(history['val_loss'], label='Val Loss', marker='s')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.legend()
    ax1.grid(True)

    # Learning rate plot
    ax2.plot(history['learning_rates'], label='Learning Rate', marker='o', color='orange')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Learning Rate')
    ax2.set_title('Learning Rate Schedule')
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"[OK] Saved training history plot: {save_path.name}")


# ==================== USER INPUT ====================

def get_user_input():
    """Prompt user for model configuration"""
    print("=" * 60)
    print("Multi-Object Image Colorizer - Model Training")
    print("=" * 60)
    print()

    # Model name
    while True:
        model_name = input("Enter model name (e.g., 'colorizer_v1'): ").strip()
        if model_name:
            break
        print("Model name cannot be empty. Please try again.")

    print("\n" + "-" * 60)
    print("Model Configuration (press Enter for defaults)")
    print("-" * 60)

    # Image size
    image_size = input("Image size [256]: ").strip()
    image_size = int(image_size) if image_size else 256

    # Batch size
    batch_size = input("Batch size [16]: ").strip()
    batch_size = int(batch_size) if batch_size else 16

    # Learning rate
    learning_rate = input("Learning rate [0.0001]: ").strip()
    learning_rate = float(learning_rate) if learning_rate else 0.0001

    # Epochs
    epochs = input("Number of epochs [50]: ").strip()
    epochs = int(epochs) if epochs else 50

    # Optimizer
    print("\nOptimizer options:")
    print("  1. Adam (recommended)")
    print("  2. SGD")
    print("  3. AdamW")
    optimizer_choice = input("Select optimizer [1]: ").strip()
    optimizer_map = {'1': 'adam', '2': 'sgd', '3': 'adamw', '': 'adam'}
    optimizer = optimizer_map.get(optimizer_choice, 'adam')

    # Loss function
    print("\nLoss function options:")
    print("  1. MSE (Mean Squared Error)")
    print("  2. L1 (Mean Absolute Error)")
    loss_choice = input("Select loss function [1]: ").strip()
    loss_map = {'1': 'mse', '2': 'l1', '': 'mse'}
    loss_function = loss_map.get(loss_choice, 'mse')

    # Learning rate scheduler
    use_scheduler = input("\nUse learning rate scheduler? (y/n) [y]: ").strip().lower()
    use_scheduler = use_scheduler != 'n'

    scheduler_config = None
    if use_scheduler:
        scheduler_patience = input("  Scheduler patience (epochs) [5]: ").strip()
        scheduler_patience = int(scheduler_patience) if scheduler_patience else 5
        scheduler_factor = input("  Scheduler factor [0.5]: ").strip()
        scheduler_factor = float(scheduler_factor) if scheduler_factor else 0.5
        scheduler_config = {'patience': scheduler_patience, 'factor': scheduler_factor}

    # Early stopping
    use_early_stopping = input("\nUse early stopping? (y/n) [y]: ").strip().lower()
    use_early_stopping = use_early_stopping != 'n'

    early_stopping_patience = None
    if use_early_stopping:
        early_stopping_patience = input("  Early stopping patience (epochs) [10]: ").strip()
        early_stopping_patience = int(early_stopping_patience) if early_stopping_patience else 10

    # Data augmentation
    use_augmentation = input("\nUse data augmentation during training? (y/n) [y]: ").strip().lower()
    use_augmentation = use_augmentation != 'n'

    augmentation_config = None
    if use_augmentation:
        print("\n  Augmentation options:")
        horizontal_flip = input("    Horizontal flip? (y/n) [y]: ").strip().lower() != 'n'
        rotation_degrees = input("    Rotation degrees [15]: ").strip()
        rotation_degrees = int(rotation_degrees) if rotation_degrees else 15
        brightness_factor = input("    Brightness adjustment factor [0.2]: ").strip()
        brightness_factor = float(brightness_factor) if brightness_factor else 0.2
        augmentation_config = {
            'horizontal_flip': horizontal_flip,
            'rotation_degrees': rotation_degrees,
            'brightness_factor': brightness_factor
        }

    # Validation split
    print("\n" + "-" * 60)
    print("Dataset Configuration")
    print("-" * 60)
    print("Dataset will be loaded from: dataset/ (in script directory)")

    val_split = input("Validation split ratio [0.15]: ").strip()
    val_split = float(val_split) if val_split else 0.15

    # Build configuration
    config = {
        'model_name': model_name,
        'image_size': image_size,
        'batch_size': batch_size,
        'learning_rate': learning_rate,
        'epochs': epochs,
        'optimizer': optimizer,
        'loss_function': loss_function,
        'scheduler': scheduler_config,
        'early_stopping_patience': early_stopping_patience,
        'augmentation': augmentation_config,
        'val_split': val_split,
        'created_at': datetime.now().isoformat(),
        'input_channels': 1,
        'output_channels': 3,
    }

    return config


def create_directories(script_dir, model_name):
    """Create necessary directories under models/model_name/"""
    model_dir = script_dir / 'models' / model_name
    saved_models_dir = model_dir / 'saved_models'
    visualizations_dir = model_dir / 'visualizations'
    logs_dir = model_dir / 'logs'

    saved_models_dir.mkdir(parents=True, exist_ok=True)
    visualizations_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n[OK] Created model directory: {model_dir}")
    print(f"  - Saved models: saved_models/")
    print(f"  - Visualizations: visualizations/")
    print(f"  - Logs: logs/")

    return {
        'model_dir': model_dir,
        'saved_models': saved_models_dir,
        'visualizations': visualizations_dir,
        'logs': logs_dir
    }


def save_config(config, directories):
    """Save configuration to JSON"""
    config_path = directories['saved_models'] / 'model_config.json'
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2)
    print(f"[OK] Saved configuration: {config_path.name}")


# ==================== MAIN ====================

def main():
    """Main execution function"""
    script_dir = Path(__file__).parent.resolve()
    print(f"\nWorking directory: {script_dir}\n")

    # Get configuration
    config = get_user_input()

    # Create directories
    directories = create_directories(script_dir, config['model_name'])

    # Save configuration
    save_config(config, directories)

    # Check for GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n[OK] Using device: {device}")
    if device.type == 'cuda':
        print(f"  GPU: {torch.cuda.get_device_name(0)}")

    # Check for dataset directory
    dataset_dir = script_dir / 'dataset'
    if not dataset_dir.exists():
        print(f"\n[ERROR] Dataset directory not found: {dataset_dir}")
        print("Please create a 'dataset/' folder in the script directory and add images.")
        return

    # Create datasets
    print("\n" + "-" * 60)
    print("Loading Datasets")
    print("-" * 60)
    print(f"Dataset directory: {dataset_dir}")

    # Load full dataset
    full_dataset = ColorizerDataset(
        dataset_dir,
        image_size=config['image_size'],
        augment=False  # Don't augment yet
    )

    # Split into train/val
    total_size = len(full_dataset)
    val_size = int(total_size * config['val_split'])
    train_size = total_size - val_size

    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)  # For reproducibility
    )

    print(f"[OK] Total images: {total_size}")
    print(f"[OK] Training images: {train_size}")
    print(f"[OK] Validation images: {val_size}")

    # Apply augmentation to training set
    if config['augmentation']:
        # Re-create train dataset with augmentation
        train_indices = train_dataset.indices
        train_image_files = [full_dataset.image_files[i] for i in train_indices]

        # Create a new dataset with just train images and augmentation
        train_dataset_augmented = ColorizerDataset(
            dataset_dir,
            image_size=config['image_size'],
            augment=True,
            augmentation_config=config['augmentation']
        )
        train_dataset_augmented.image_files = train_image_files
        train_dataset = train_dataset_augmented

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=0,
        pin_memory=True if device.type == 'cuda' else False
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=0,
        pin_memory=True if device.type == 'cuda' else False
    )

    print(f"[OK] Training batches: {len(train_loader)}")
    print(f"[OK] Validation batches: {len(val_loader)}")

    # Create model
    print("\n" + "-" * 60)
    print("Creating Model")
    print("-" * 60)

    model = UNetColorizer(config).to(device)

    # Initialize weights
    def init_weights(m):
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)

    model.apply(init_weights)

    print(f"[OK] Model created with {model.count_parameters():,} parameters")

    # Create trainer
    trainer = Trainer(model, train_loader, val_loader, config, directories, device)

    # Check for existing checkpoint to resume from
    resume_from_epoch = 0
    latest_checkpoint = directories['saved_models'] / f"{config['model_name']}_latest.pth"

    if latest_checkpoint.exists():
        print("\n" + "=" * 60)
        print("CHECKPOINT FOUND")
        print("=" * 60)
        print(f"Found existing checkpoint: {latest_checkpoint.name}")

        # Try to load and show checkpoint info
        try:
            checkpoint = torch.load(latest_checkpoint, map_location=device)
            checkpoint_epoch = checkpoint.get('epoch', 0)
            total_epochs = config['epochs']

            print(f"Checkpoint is at epoch {checkpoint_epoch}/{total_epochs}")

            if checkpoint_epoch < total_epochs:
                resume = input("\nDo you want to resume training from this checkpoint? (y/n) [y]: ").strip().lower()

                if resume != 'n':
                    resume_from_epoch = trainer.load_checkpoint(latest_checkpoint)
                else:
                    print("[OK] Starting fresh training (checkpoint will be overwritten)")
            else:
                print("Checkpoint has already completed all epochs.")
                print("[OK] Starting fresh training (checkpoint will be overwritten)")
        except Exception as e:
            print(f"[WARNING] Could not read checkpoint: {e}")
            print("[OK] Starting fresh training")

    # Train
    history = trainer.train(resume_from_epoch=resume_from_epoch)

    # Save training history plot
    plot_path = directories['visualizations'] / 'training_history.png'
    plot_training_history(history, plot_path)

    # Save final history
    history_path = directories['saved_models'] / 'training_history.json'
    with open(history_path, 'w', encoding='utf-8') as f:
        json.dump(history, f, indent=2)
    print(f"[OK] Saved training history: {history_path.name}")

    print("\n" + "=" * 60)
    print("Training Complete!")
    print("=" * 60)
    print(f"\nAll model files saved to: {directories['model_dir']}")
    print(f"  - Saved models: {directories['model_dir']}/saved_models/")
    print(f"  - Visualizations: {directories['model_dir']}/visualizations/")
    print(f"  - Logs: {directories['model_dir']}/logs/")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nTraining cancelled by user.")
    except Exception as e:
        print(f"\n[ERROR] {e}")
        raise
