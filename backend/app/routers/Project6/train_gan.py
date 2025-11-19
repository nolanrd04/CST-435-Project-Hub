"""
Multi-Fruit GAN Training Script
Trains separate Generator/Discriminator pairs for each fruit in one session
"""

import torch
import os
import sys
from pathlib import Path
import argparse
from datetime import datetime
import json

# Script-relative directory management
SCRIPT_DIR = Path(__file__).parent
NPZ_DATA_DIR = SCRIPT_DIR / 'npzData'

# Add parent directory to path for imports
sys.path.insert(0, str(SCRIPT_DIR))

from gan_model import create_gan_models, create_optimizers
from gan_trainer import MultiFruitGANTrainer
from data_loader import (
    get_available_versions, 
    get_fruit_types_for_version,
    load_dataset_for_version,
    create_data_loader
)


def display_menu(options, title=""):
    """
    Display a menu and get user selection
    
    Args:
        options (list): List of option strings
        title (str): Menu title
        
    Returns:
        int: Index of selected option
    """
    print(f"\n{'='*50}")
    if title:
        print(f"{title}")
        print('='*50)
    
    for idx, option in enumerate(options, 1):
        print(f"  {idx}. {option}")
    
    while True:
        try:
            choice = int(input("\nSelect option: "))
            if 1 <= choice <= len(options):
                return choice - 1
            else:
                print(f"Please enter a number between 1 and {len(options)}")
        except ValueError:
            print("Please enter a valid number")


def select_version(npz_dir=None):
    """
    Allow user to select dataset version
    
    Args:
        npz_dir (str or Path): Path to npzData directory. If None, uses script-relative path.
        
    Returns:
        str: Selected version name
    """
    if npz_dir is None:
        npz_dir = NPZ_DATA_DIR
    else:
        npz_dir = Path(npz_dir)
    
    versions = list(get_available_versions(npz_dir))
    
    if not versions:
        print("ERROR: No dataset versions found in npzData directory")
        print(f"Checked directory: {npz_dir}")
        sys.exit(1)
    
    # Check if v1 exists as default
    default_version = 'v1' if 'v1' in versions else versions[0]
    
    print(f"\nFound {len(versions)} dataset version(s): {', '.join(versions)}")
    print(f"Default: {default_version}")
    
    choice_input = input(f"Select version (press Enter for {default_version}): ").strip()
    
    if not choice_input:
        selected = default_version
    else:
        try:
            choice = int(choice_input)
            if 1 <= choice <= len(versions):
                selected = versions[choice - 1]
            else:
                print(f"Invalid choice. Using default: {default_version}")
                selected = default_version
        except ValueError:
            print(f"Invalid input. Using default: {default_version}")
            selected = default_version
    
    print(f"✓ Selected version: {selected}")
    return selected


def select_model_name():
    """
    Allow user to select or create a model name
    Default is 'v1', existing models get overridden
    
    Returns:
        str: Model name
    """
    print(f"\n{'='*50}")
    print("MODEL NAME")
    print('='*50)
    print("Enter a name for this model set (e.g., 'v1', 'apple_focus', 'attempt_2')")
    print("Default: v1")
    print("Note: If model already exists, it will be overridden.")
    
    model_name = input("\nModel name (press Enter for 'v1'): ").strip()
    
    if not model_name:
        model_name = 'v1'
    
    # Check if model exists
    models_dir = SCRIPT_DIR / 'models'
    existing_model = models_dir / f'model_{model_name}'
    
    if existing_model.exists():
        print(f"\n⚠️  Model 'model_{model_name}' already exists!")
        confirm = input("Override? (yes/no, default: no): ").strip().lower()
        if confirm != 'yes':
            print("Using default name instead: v1")
            return 'v1'
    
    print(f"✓ Using model name: {model_name}")
    return model_name


def select_version(npz_dir=None):
    """
    Allow user to select dataset version
    
    Args:
        npz_dir (str or Path): Path to npzData directory. If None, uses script-relative path.
        
    Returns:
        str: Selected version name
    """
    if npz_dir is None:
        npz_dir = NPZ_DATA_DIR
    else:
        npz_dir = Path(npz_dir)
    
    versions = list(get_available_versions(npz_dir))
    
    if not versions:
        print("ERROR: No dataset versions found in npzData directory")
        print(f"Checked directory: {npz_dir}")
        sys.exit(1)
    
    # Check if v1 exists as default
    default_version = 'v1' if 'v1' in versions else versions[0]
    
    print(f"\nFound {len(versions)} dataset version(s): {', '.join(versions)}")
    print(f"Default: {default_version}")
    
    choice_input = input(f"Select version (press Enter for {default_version}): ").strip()
    
    if not choice_input:
        selected = default_version
    else:
        try:
            choice = int(choice_input)
            if 1 <= choice <= len(versions):
                selected = versions[choice - 1]
            else:
                print(f"Invalid choice. Using default: {default_version}")
                selected = default_version
        except ValueError:
            print(f"Invalid input. Using default: {default_version}")
            selected = default_version
    
    print(f"✓ Selected version: {selected}")
    return selected


def select_fruits(npz_dir=None, version='v1'):
    """
    DEPRECATED - No longer used in multi-fruit training
    All fruits are trained automatically
    """
    pass


def get_training_params():
    """
    Get training parameters from user with sensible defaults
    
    Returns:
        dict: Training parameters
    """
    print(f"\n{'='*50}")
    print("TRAINING PARAMETERS")
    print('='*50)
    
    params = {}
    
    # Number of epochs
    while True:
        try:
            epochs_input = input("Number of epochs (default 400, min 50): ").strip()
            if not epochs_input:
                params['epochs'] = 400
                print("Using default: 400 epochs")
                break
            else:
                epochs = int(epochs_input)
                if epochs >= 50:
                    params['epochs'] = epochs
                    break
                else:
                    print("Please enter at least 50 epochs")
        except ValueError:
            print("Please enter a valid number")
    
    # Batch size
    while True:
        try:
            batch_input = input("Batch size (default 32, common: 16, 32, 64): ").strip()
            if not batch_input:
                params['batch_size'] = 32
                print("Using default: 32")
                break
            else:
                batch_size = int(batch_input)
                if batch_size > 0:
                    params['batch_size'] = batch_size
                    break
                else:
                    print("Please enter a positive number")
        except ValueError:
            print("Please enter a valid number")
    
    # Learning rate
    while True:
        try:
            lr_input = input("Learning rate (default 0.0002): ").strip()
            if not lr_input:
                params['learning_rate'] = 0.0002
                print("Using default: 0.0002")
                break
            else:
                lr = float(lr_input)
                if 0 < lr < 1:
                    params['learning_rate'] = lr
                    break
                else:
                    print("Please enter a value between 0 and 1")
        except ValueError:
            print("Please enter a valid number")
    
    # Latent dimension
    while True:
        try:
            latent_input = input("Latent dimension (default 100): ").strip()
            if not latent_input:
                params['latent_dim'] = 100
                print("Using default: 100")
                break
            else:
                latent_dim = int(latent_input)
                if latent_dim > 0:
                    params['latent_dim'] = latent_dim
                    break
                else:
                    print("Please enter a positive number")
        except ValueError:
            print("Please enter a valid number")
    
    print("\n✓ Training parameters set")
    return params


def display_summary(version, fruits, params, total_images):
    """
    Display training configuration summary
    
    Args:
        version (str): Selected version
        fruits (list): Fruits to train
        params (dict): Training parameters
        total_images (dict): Total images per fruit
    """
    print(f"\n{'='*70}")
    print("TRAINING CONFIGURATION SUMMARY")
    print('='*70)
    print(f"Dataset Version: {version}")
    print(f"\nFruits to Train: {len(fruits)}")
    for fruit in sorted(fruits):
        print(f"  - {fruit}: {total_images.get(fruit, 'N/A')} images")
    print(f"\nTraining Parameters:")
    print(f"  - Epochs (per fruit): {params['epochs']}")
    print(f"  - Batch Size: {params['batch_size']}")
    print(f"  - Learning Rate: {params['learning_rate']}")
    print(f"  - Latent Dimension: {params['latent_dim']}")
    print(f"  - Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    print(f"\n✓ Total training time estimate: ~{len(fruits)} × (epoch_time per fruit)")
    print('='*70)


def main():
    """
    Main multi-fruit GAN training script
    """
    parser = argparse.ArgumentParser(description='Train GAN for multiple fruits')
    parser.add_argument('--version', type=str, help='Dataset version (e.g., v1)')
    parser.add_argument('--model-name', type=str, help='Model name (default: v1)')
    parser.add_argument('--epochs', type=int, default=400, help='Number of epochs per fruit')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
    parser.add_argument('--learning-rate', type=float, default=0.0002, help='Learning rate')
    parser.add_argument('--latent-dim', type=int, default=100, help='Latent dimension')
    parser.add_argument('--npz-dir', type=str, help='Path to npzData directory (default: script-relative)')
    parser.add_argument('--save-dir', type=str, help='Directory to save checkpoints')
    
    args = parser.parse_args()
    
    # Resolve npz directory
    if args.npz_dir:
        npz_dir = Path(args.npz_dir).resolve()
    else:
        npz_dir = NPZ_DATA_DIR.resolve()
    
    if not npz_dir.exists():
        print(f"ERROR: npzData directory not found: {npz_dir}")
        sys.exit(1)
    
    print("\n" + "="*70)
    print("MULTI-FRUIT GAN TRAINING")
    print("="*70)
    print("Trains separate Generator/Discriminator pairs for each fruit")
    print("="*70)
    
    # Select version
    if args.version:
        version = args.version
        print(f"\nUsing version: {version}")
    else:
        version = select_version(npz_dir)
    
    # Select model name
    if args.model_name:
        model_name = args.model_name
        print(f"\nUsing model name: {model_name}")
    else:
        model_name = select_model_name()
    
    # Get training parameters
    if args.epochs != 400 or args.batch_size != 32 or args.learning_rate != 0.0002 or args.latent_dim != 100:
        # Use command line parameters
        params = {
            'epochs': args.epochs,
            'batch_size': args.batch_size,
            'learning_rate': args.learning_rate,
            'latent_dim': args.latent_dim
        }
    else:
        # Interactive mode
        params = get_training_params()
    
    # Get all fruit types for this version
    print(f"\nLoading fruit types for version '{version}'...")
    fruits = get_fruit_types_for_version(npz_dir, version)
    print(f"Found {len(fruits)} fruit types: {', '.join(fruits)}")
    
    # Load dataloaders for each fruit
    print(f"\nLoading datasets for each fruit...")
    dataloaders = {}
    total_images = {}
    channels = None
    
    for fruit in fruits:
        print(f"  Loading {fruit}...", end=" ")
        try:
            dataset, loaded_fruits, detected_channels = load_dataset_for_version(
                version, npz_dir, [fruit]
            )
            
            if channels is None:
                channels = detected_channels
            
            dataloader = create_data_loader(dataset, batch_size=params['batch_size'], shuffle=True)
            dataloaders[fruit] = dataloader
            total_images[fruit] = len(dataset)
            print(f"✓ ({len(dataset)} images)")
        except Exception as e:
            print(f"✗ Error: {e}")
    
    if not dataloaders:
        print("ERROR: No datasets loaded")
        sys.exit(1)
    
    print(f"\nDetected image channels: {channels}")
    
    # Display summary
    display_summary(version, fruits, params, total_images)
    
    # Confirm before training
    confirm = input("\nStart training all fruits? (type yes/no to confirm): ").strip().lower()
    if confirm != 'yes':
        print("Training cancelled")
        sys.exit(0)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")
    
    # Create multi-fruit trainer
    multi_trainer = MultiFruitGANTrainer(device=device, latent_dim=params['latent_dim'])
    
    # Create and add trainer for each fruit
    print("\nCreating GAN models for each fruit...")
    for fruit in fruits:
        print(f"  {fruit}...", end=" ")
        
        generator, discriminator = create_gan_models(
            latent_dim=params['latent_dim'],
            channels=channels,
            device=device
        )
        
        gen_optimizer, disc_optimizer = create_optimizers(
            generator,
            discriminator,
            learning_rate=params['learning_rate']
        )
        
        multi_trainer.add_fruit_trainer(fruit, generator, discriminator, gen_optimizer, disc_optimizer)
        print("✓")
    
    # Models directory (script-relative)
    models_dir = SCRIPT_DIR / 'models'
    models_dir.mkdir(parents=True, exist_ok=True)
    
    # Save training configuration
    training_config = {
        'model_name': model_name,
        'version': version,
        'fruits': fruits,
        'epochs': params['epochs'],
        'batch_size': params['batch_size'],
        'learning_rate': params['learning_rate'],
        'latent_dim': params['latent_dim'],
        'image_channels': channels,
        'device': str(device),
        'timestamp': datetime.now().isoformat(),
        'dataset_sizes': total_images
    }
    
    config_path = models_dir / f'model_{model_name}' / 'info' / f'training_config_{model_name}.json'
    config_path.parent.mkdir(parents=True, exist_ok=True)
    with open(config_path, 'w') as f:
        json.dump(training_config, f, indent=2)
    
    print(f"\nTraining configuration saved to: {config_path}")
    
    # Start multi-fruit training
    multi_trainer.train_all_fruits(
        dataloaders=dataloaders,
        num_epochs=params['epochs'],
        save_dir=str(models_dir),
        model_name=model_name,
        save_interval_count=5
    )
    
    print("\n" + "="*70)
    print("ALL TRAINING COMPLETE!")
    print(f"Model location: {models_dir / f'model_{model_name}'}")
    print("="*70)


if __name__ == "__main__":
    main()
