"""
GAN Inference - Generate images with trained models
"""

import torch
import os
import sys
from pathlib import Path
import argparse
import matplotlib.pyplot as plt
import numpy as np
import json

# Script-relative directory management
SCRIPT_DIR = Path(__file__).parent
MODELS_DIR = SCRIPT_DIR / 'models'

# Add parent directory to path for imports
sys.path.insert(0, str(SCRIPT_DIR))

from gan_model import Generator


def load_generator(model_path, latent_dim=100, img_size=28, channels=1, device='cpu'):
    """
    Load trained generator model
    
    Args:
        model_path (str or Path): Path to generator checkpoint
        latent_dim (int): Latent dimension used in training
        img_size (int): Image size used in training
        channels (int): Number of channels used in training
        device (str): Device to load on
        
    Returns:
        nn.Module: Loaded generator model
    """
    model_path = Path(model_path)
    
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    
    generator = Generator(latent_dim=latent_dim, img_size=img_size, channels=channels)
    generator.load_state_dict(torch.load(model_path, map_location=device))
    generator = generator.to(device)
    generator.eval()
    
    print(f"✓ Loaded generator from: {model_path}")
    return generator


def generate_images(generator, num_images=16, latent_dim=100, device='cpu'):
    """
    Generate images using trained generator
    
    Args:
        generator (nn.Module): Generator model
        num_images (int): Number of images to generate
        latent_dim (int): Latent dimension
        device (str): Device
        
    Returns:
        torch.Tensor: Generated images
    """
    with torch.no_grad():
        noise = torch.randn(num_images, latent_dim, device=device)
        images = generator(noise)
    
    return images


def plot_generated_images(images, title="Generated Images", filename=None):
    """
    Plot generated images in a grid
    
    Args:
        images (torch.Tensor): Generated images
        title (str): Plot title
        filename (str or Path): Save plot to file
    """
    images = images.cpu().detach()
    
    # Denormalize from [-1, 1] to [0, 1]
    images = (images + 1) / 2
    
    # Create grid
    num_images = len(images)
    grid_size = int(np.ceil(np.sqrt(num_images)))
    
    fig, axes = plt.subplots(grid_size, grid_size, figsize=(10, 10))
    axes = axes.flatten()
    
    for idx, ax in enumerate(axes):
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
    
    plt.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if filename:
        filename = Path(filename)
        filename.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        print(f"✓ Saved plot to: {filename}")
    else:
        plt.show()
    
    plt.close()


def interpolate_images(generator, num_steps=5, latent_dim=100, device='cpu'):
    """
    Generate images by interpolating between two noise vectors
    Shows smooth transition in generated image space
    
    Args:
        generator (nn.Module): Generator model
        num_steps (int): Number of interpolation steps
        latent_dim (int): Latent dimension
        device (str): Device
        
    Returns:
        torch.Tensor: Interpolated images
    """
    with torch.no_grad():
        # Create two random noise vectors
        noise1 = torch.randn(1, latent_dim, device=device)
        noise2 = torch.randn(1, latent_dim, device=device)
        
        # Interpolate between them
        images = []
        for t in np.linspace(0, 1, num_steps):
            interpolated_noise = (1 - t) * noise1 + t * noise2
            img = generator(interpolated_noise)
            images.append(img)
        
        images = torch.cat(images, dim=0)
    
    return images


def load_model_config(model_path):
    """
    Load training configuration for a model

    Args:
        model_path (Path): Path to model directory

    Returns:
        dict: Model configuration or None if not found
    """
    model_name = model_path.name.replace('model_', '')
    config_path = model_path / 'info' / f'training_config_{model_name}.json'

    if not config_path.exists():
        print(f"Warning: Config file not found at {config_path}")
        return None

    try:
        with open(config_path, 'r') as f:
            config_data = json.load(f)
            return config_data.get('config', {})
    except Exception as e:
        print(f"Warning: Could not load config: {e}")
        return None


def list_available_models(models_dir=None):
    """
    List all available trained models with their fruits and config

    Args:
        models_dir (str or Path): Path to models directory

    Returns:
        dict: Available models organized by model name
    """
    if models_dir is None:
        models_dir = MODELS_DIR
    else:
        models_dir = Path(models_dir)

    if not models_dir.exists():
        print(f"Models directory not found: {models_dir}")
        return {}

    models = {}

    # NEW: Updated to work with model_name folders (e.g., model_v1)
    for item in models_dir.iterdir():
        if item.is_dir() and item.name.startswith('model_'):
            model_name = item.name.replace('model_', '')

            # Load model config
            config = load_model_config(item)

            # Find all fruit-specific generators in this model
            fruits = {}
            for gen_file in item.glob('generator_*.pt'):
                fruit_name = gen_file.stem.replace('generator_', '')
                disc_file = item / f'discriminator_{fruit_name}.pt'

                if disc_file.exists():
                    fruits[fruit_name] = {
                        'generator': gen_file,
                        'discriminator': disc_file
                    }

            if fruits:
                models[model_name] = {
                    'path': item,
                    'fruits': fruits,
                    'config': config or {}  # Include config in model info
                }

    return models


def generate_images_for_api(model_name, fruit, num_images=16, seed=None, models_dir=None):
    """
    Generate images programmatically for API/frontend use

    Args:
        model_name (str): Model name (e.g., 'v1', 'medium_v2')
        fruit (str): Fruit name (e.g., 'apple', 'banana')
        num_images (int): Number of images to generate
        seed (int, optional): Random seed for reproducibility
        models_dir (str or Path, optional): Path to models directory

    Returns:
        tuple: (images_tensor, config_dict) or (None, error_message)
    """
    # Resolve models directory
    if models_dir:
        models_dir = Path(models_dir).resolve()
    else:
        models_dir = MODELS_DIR.resolve()

    # Get available models
    available = list_available_models(models_dir)

    if model_name not in available:
        return None, f"Model '{model_name}' not found. Available: {list(available.keys())}"

    model_info = available[model_name]
    available_fruits = list(model_info['fruits'].keys())

    if fruit not in available_fruits:
        return None, f"Fruit '{fruit}' not found in model '{model_name}'. Available: {available_fruits}"

    fruit_info = model_info['fruits'][fruit]
    model_path = fruit_info['generator']

    # Get config from model
    model_config = model_info.get('config', {})
    img_size = model_config.get('img_size', 64)
    latent_dim = model_config.get('latent_dim', 100)
    channels = 1  # Default to grayscale

    # Set seed if provided
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    try:
        # Load generator
        generator = load_generator(
            model_path,
            latent_dim=latent_dim,
            img_size=img_size,
            channels=channels,
            device=device
        )

        # Generate images
        images = generate_images(
            generator,
            num_images=num_images,
            latent_dim=latent_dim,
            device=device
        )

        # Prepare response config
        response_config = {
            'model_name': model_name,
            'fruit': fruit,
            'num_images': num_images,
            'img_size': img_size,
            'latent_dim': latent_dim,
            'channels': channels,
            'device': str(device)
        }

        return images, response_config

    except Exception as e:
        return None, f"Error generating images: {str(e)}"


def main():
    """
    Main inference script - select model and fruit, then generate images
    """
    parser = argparse.ArgumentParser(description='Generate images with trained GAN')
    parser.add_argument('model_name', nargs='?', help='Model name (e.g., v1). If not provided, lists available models.')
    parser.add_argument('fruit', nargs='?', help='Fruit to generate (e.g., apple). If not provided, lists available fruits for model.')
    parser.add_argument('--num-images', type=int, default=16, help='Number of images to generate')
    parser.add_argument('--latent-dim', type=int, default=100, help='Latent dimension')
    parser.add_argument('--channels', type=int, default=1, help='Number of image channels')
    parser.add_argument('--interpolate', action='store_true', help='Generate interpolation sequence')
    parser.add_argument('--save', type=str, help='Save generated images to file')
    parser.add_argument('--seed', type=int, help='Random seed for reproducibility')
    parser.add_argument('--models-dir', type=str, help='Path to models directory (default: script-relative)')
    
    args = parser.parse_args()
    
    # Resolve models directory
    if args.models_dir:
        models_dir = Path(args.models_dir).resolve()
    else:
        models_dir = MODELS_DIR.resolve()
    
    # List available models if model_name not provided
    if not args.model_name:
        print("\n" + "="*70)
        print("AVAILABLE MODELS")
        print("="*70)
        
        available = list_available_models(models_dir)
        
        if not available:
            print(f"No models found in: {models_dir}")
            print("\nUsage: python generate_images.py model_name fruit [--num-images 16]")
            return
        
        print(f"Found {len(available)} model(s):\n")
        for model_name, model_info in sorted(available.items()):
            fruits_list = ', '.join(sorted(model_info['fruits'].keys()))
            print(f"  model_{model_name}/")
            print(f"    Fruits: {fruits_list}")
        
        print("\n" + "="*70)
        print("Usage: python generate_images.py <model_name> <fruit> [options]")
        print("  Example: python generate_images.py v1 apple --num-images 16")
        print("="*70)
        return
    
    # Check if model exists
    available = list_available_models(models_dir)
    
    if args.model_name not in available:
        print(f"ERROR: Model '{args.model_name}' not found")
        print(f"Available models: {', '.join(available.keys())}")
        sys.exit(1)
    
    model_info = available[args.model_name]
    available_fruits = list(model_info['fruits'].keys())
    
    # List available fruits for this model if fruit not provided
    if not args.fruit:
        print(f"\nModel: {args.model_name}")
        print(f"Available fruits: {', '.join(sorted(available_fruits))}")
        print("\nUsage: python generate_images.py {0} <fruit> [options]".format(args.model_name))
        print("  Example: python generate_images.py {0} apple --num-images 16".format(args.model_name))
        return
    
    # Check if fruit exists in model
    if args.fruit not in available_fruits:
        print(f"ERROR: Fruit '{args.fruit}' not found in model '{args.model_name}'")
        print(f"Available fruits: {', '.join(sorted(available_fruits))}")
        sys.exit(1)
    
    fruit_info = model_info['fruits'][args.fruit]
    model_path = fruit_info['generator']

    # Get config from model (override defaults)
    model_config = model_info.get('config', {})
    img_size = model_config.get('img_size', 64)  # Use config img_size or default to 64
    latent_dim = model_config.get('latent_dim', args.latent_dim)  # Use config latent_dim

    print(f"\nModel Configuration:")
    print(f"  Image size: {img_size}x{img_size}")
    print(f"  Latent dimension: {latent_dim}")
    print(f"  Channels: {args.channels}")

    # Set seed if provided
    if args.seed is not None:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"  Device: {device}")

    # Load generator
    print(f"\nLoading {args.fruit} generator from: {model_path.name}")
    generator = load_generator(
        model_path,
        latent_dim=latent_dim,
        img_size=img_size,
        channels=args.channels,
        device=device
    )
    
    # Generate images
    print(f"\nGenerating {args.num_images} {args.fruit} images...")
    
    if args.interpolate:
        images = interpolate_images(
            generator,
            num_steps=args.num_images,
            latent_dim=latent_dim,
            device=device
        )
        title = f"Interpolated {args.fruit.capitalize()} Images - {args.model_name} (smooth transition)"
    else:
        images = generate_images(
            generator,
            num_images=args.num_images,
            latent_dim=latent_dim,
            device=device
        )
        title = f"{args.fruit.capitalize()} Images - {args.model_name} ({args.num_images} samples)"
    
    # Plot images
    print("\nPlotting images...")
    if args.save:
        save_path = Path(args.save)
    else:
        save_path = model_info['path'] / f'generated_epoch_images_{args.fruit}' / f'generated_{args.num_images}.png'
    
    plot_generated_images(images, title=title, filename=save_path)
    
    print("\n✓ Done!")


if __name__ == "__main__":
    main()
