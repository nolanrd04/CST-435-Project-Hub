#!/usr/bin/env python3
"""
Image to NPZ Converter for Diffusion Model
Converts RGB images to paired grayscale/RGB NPZ arrays for training.
"""

import os
import sys
import json
import numpy as np
from pathlib import Path
from PIL import Image
from typing import Tuple, List, Dict
from sklearn.model_selection import train_test_split


class ImageToNPZConverter:
    """Converts images to NPZ format for diffusion model training."""

    DEFAULT_TEST_SIZE = 0.2
    DEFAULT_RANDOM_STATE = 42
    DEFAULT_IMAGE_SIZE = 64

    def __init__(self, script_dir: Path, model_name: str):
        """Initialize converter with script directory and model name."""
        self.script_dir = script_dir
        self.model_name = model_name
        self.image_data_dir = script_dir / "imageData" / model_name
        self.npz_data_dir = script_dir / "npzData" / model_name
        self.metadata = {}

    def get_user_input(self) -> dict:
        """Prompt user for conversion configuration."""
        print("=" * 60)
        print("Image to NPZ Converter")
        print("=" * 60)
        print()

        # Check if imageData exists
        if not self.image_data_dir.exists():
            print(f"Error: Image data directory not found: {self.image_data_dir}")
            print("Please run import_tiny_imagenet.py first.")
            sys.exit(1)

        # Load import metadata
        metadata_path = self.image_data_dir / "import_metadata.json"
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                self.metadata = json.load(f)
            print(f"Found {self.metadata['total_images']} images in {self.metadata['categories']} categories")
        else:
            print("Warning: import_metadata.json not found. Proceeding anyway.")

        print()

        # Test split size
        test_size_input = input(
            f"Enter test split size (0.0-1.0, default: {self.DEFAULT_TEST_SIZE}): "
        ).strip()
        try:
            test_size = float(test_size_input) if test_size_input else self.DEFAULT_TEST_SIZE
            if not 0.0 < test_size < 1.0:
                print("Error: Test size must be between 0.0 and 1.0")
                sys.exit(1)
        except ValueError:
            print("Error: Invalid test size")
            sys.exit(1)

        # Random state for reproducibility
        random_state_input = input(
            f"Enter random state for reproducibility (default: {self.DEFAULT_RANDOM_STATE}): "
        ).strip()
        random_state = int(random_state_input) if random_state_input else self.DEFAULT_RANDOM_STATE

        # Image size
        image_size_input = input(
            f"Enter target image size (default: {self.DEFAULT_IMAGE_SIZE}x{self.DEFAULT_IMAGE_SIZE}): "
        ).strip()
        image_size = int(image_size_input) if image_size_input else self.DEFAULT_IMAGE_SIZE

        # Normalization option
        print()
        print("Normalization options:")
        print("  1. [0, 1] range (divide by 255)")
        print("  2. [-1, 1] range (standard for diffusion models)")
        print("  3. None (keep as [0, 255])")
        normalize_input = input("Enter normalization option (default: 2): ").strip()
        normalize_option = int(normalize_input) if normalize_input else 2

        print()
        print("-" * 60)
        print("Configuration Summary:")
        print(f"  Model: {self.model_name}")
        print(f"  Image size: {image_size}x{image_size}")
        print(f"  Test split: {test_size * 100:.1f}%")
        print(f"  Train split: {(1 - test_size) * 100:.1f}%")
        print(f"  Random state: {random_state}")
        normalize_desc = {1: "[0, 1]", 2: "[-1, 1]", 3: "None [0, 255]"}
        print(f"  Normalization: {normalize_desc.get(normalize_option, 'Unknown')}")
        print("-" * 60)

        confirm = input("\nProceed with conversion? (y/n): ").strip().lower()
        if confirm != 'y':
            print("Conversion cancelled.")
            sys.exit(0)

        return {
            'test_size': test_size,
            'random_state': random_state,
            'image_size': image_size,
            'normalize_option': normalize_option
        }

    def load_images(self, config: dict) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str]]:
        """Load all images and convert to grayscale/RGB pairs."""
        print("\n[1/3] Loading images...")

        rgb_images = []
        grayscale_images = []
        labels = []
        category_names = []

        # Get all category directories
        category_dirs = sorted([d for d in self.image_data_dir.iterdir() if d.is_dir()])

        if not category_dirs:
            print("Error: No category directories found")
            sys.exit(1)

        print(f"  Found {len(category_dirs)} categories")

        # Create label mapping
        category_to_label = {cat.name: idx for idx, cat in enumerate(category_dirs)}

        total_images = 0
        for category_dir in category_dirs:
            category_name = category_dir.name
            label = category_to_label[category_name]

            # Get all JPEG images
            image_files = list(category_dir.glob("*.JPEG"))

            for img_file in image_files:
                try:
                    # Load RGB image
                    img = Image.open(img_file)

                    # Ensure RGB mode
                    if img.mode != 'RGB':
                        img = img.convert('RGB')

                    # Resize if needed
                    if img.size != (config['image_size'], config['image_size']):
                        img = img.resize((config['image_size'], config['image_size']), Image.LANCZOS)

                    # Convert to numpy array
                    rgb_array = np.array(img)

                    # Convert to grayscale
                    gray_img = img.convert('L')
                    gray_array = np.array(gray_img)

                    # Add channel dimension to grayscale
                    gray_array = np.expand_dims(gray_array, axis=-1)

                    rgb_images.append(rgb_array)
                    grayscale_images.append(gray_array)
                    labels.append(label)
                    category_names.append(category_name)

                    total_images += 1

                    if total_images % 500 == 0:
                        print(f"\r  Loaded: {total_images} images", end='', flush=True)

                except Exception as e:
                    print(f"\n  Warning: Failed to load {img_file}: {e}")
                    continue

        print(f"\n  Loaded {total_images} images successfully")

        # Convert lists to numpy arrays
        rgb_images = np.array(rgb_images, dtype=np.uint8)
        grayscale_images = np.array(grayscale_images, dtype=np.uint8)
        labels = np.array(labels, dtype=np.int32)

        return rgb_images, grayscale_images, labels, category_names

    def normalize_images(self, rgb: np.ndarray, grayscale: np.ndarray, option: int) -> Tuple[np.ndarray, np.ndarray]:
        """Normalize images according to selected option."""
        if option == 1:
            # [0, 1] range
            rgb = rgb.astype(np.float32) / 255.0
            grayscale = grayscale.astype(np.float32) / 255.0
        elif option == 2:
            # [-1, 1] range (standard for diffusion models)
            rgb = (rgb.astype(np.float32) / 127.5) - 1.0
            grayscale = (grayscale.astype(np.float32) / 127.5) - 1.0
        # option 3: keep as uint8 [0, 255]

        return rgb, grayscale

    def split_and_save(self, rgb_images: np.ndarray, grayscale_images: np.ndarray,
                       labels: np.ndarray, category_names: List[str], config: dict):
        """Split data into train/test and save as NPZ files."""
        print("\n[2/3] Splitting and normalizing data...")

        # Split the data
        indices = np.arange(len(rgb_images))
        train_idx, test_idx = train_test_split(
            indices,
            test_size=config['test_size'],
            random_state=config['random_state'],
            stratify=labels  # Ensure balanced splits across categories
        )

        # Split arrays
        rgb_train, rgb_test = rgb_images[train_idx], rgb_images[test_idx]
        gray_train, gray_test = grayscale_images[train_idx], grayscale_images[test_idx]
        labels_train, labels_test = labels[train_idx], labels[test_idx]

        # Category names for reference
        category_names_array = np.array(category_names)
        names_train, names_test = category_names_array[train_idx], category_names_array[test_idx]

        print(f"  Train set: {len(train_idx)} images")
        print(f"  Test set: {len(test_idx)} images")

        # Normalize if needed
        if config['normalize_option'] != 3:
            print(f"  Normalizing images...")
            rgb_train, gray_train = self.normalize_images(rgb_train, gray_train, config['normalize_option'])
            rgb_test, gray_test = self.normalize_images(rgb_test, gray_test, config['normalize_option'])

        print("\n[3/3] Saving NPZ files...")

        # Create output directory
        self.npz_data_dir.mkdir(parents=True, exist_ok=True)

        # Save training data
        train_path = self.npz_data_dir / "train.npz"
        np.savez_compressed(
            train_path,
            grayscale=gray_train,
            rgb=rgb_train,
            labels=labels_train,
            category_names=names_train
        )
        train_size_mb = train_path.stat().st_size / (1024 * 1024)
        print(f"  Saved: {train_path} ({train_size_mb:.2f} MB)")

        # Save test data
        test_path = self.npz_data_dir / "test.npz"
        np.savez_compressed(
            test_path,
            grayscale=gray_test,
            rgb=rgb_test,
            labels=labels_test,
            category_names=names_test
        )
        test_size_mb = test_path.stat().st_size / (1024 * 1024)
        print(f"  Saved: {test_path} ({test_size_mb:.2f} MB)")

        # Save metadata
        conversion_metadata = {
            'model_name': self.model_name,
            'source_dir': str(self.image_data_dir),
            'config': config,
            'train_samples': int(len(train_idx)),
            'test_samples': int(len(test_idx)),
            'image_shape': {
                'grayscale': list(gray_train.shape[1:]),
                'rgb': list(rgb_train.shape[1:])
            },
            'dtype': str(gray_train.dtype),
            'normalization': {
                1: '[0, 1]',
                2: '[-1, 1]',
                3: 'None [0, 255]'
            }.get(config['normalize_option']),
            'categories': len(set(labels)),
            'original_metadata': self.metadata
        }

        metadata_path = self.npz_data_dir / "metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(conversion_metadata, f, indent=2)
        print(f"  Saved: {metadata_path}")

        return conversion_metadata

    def run(self):
        """Execute the conversion process."""
        try:
            # Get user configuration
            config = self.get_user_input()

            # Load images
            rgb_images, grayscale_images, labels, category_names = self.load_images(config)

            # Split and save
            metadata = self.split_and_save(rgb_images, grayscale_images, labels, category_names, config)

            print("\n" + "=" * 60)
            print("Conversion Summary:")
            print(f"  Model: {self.model_name}")
            print(f"  Train samples: {metadata['train_samples']}")
            print(f"  Test samples: {metadata['test_samples']}")
            print(f"  Grayscale shape: {tuple(metadata['image_shape']['grayscale'])}")
            print(f"  RGB shape: {tuple(metadata['image_shape']['rgb'])}")
            print(f"  Data type: {metadata['dtype']}")
            print(f"  Normalization: {metadata['normalization']}")
            print(f"  Output directory: {self.npz_data_dir}")
            print("=" * 60)
            print("\nConversion completed successfully!")

        except KeyboardInterrupt:
            print("\n\nConversion cancelled by user.")
            sys.exit(1)
        except Exception as e:
            print(f"\n\nError during conversion: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)


def get_available_models(script_dir: Path) -> List[str]:
    """
    Scan imageData/ directory and find all available model folders.

    Returns:
        List of model names (folder names in imageData/)
    """
    image_data_dir = script_dir / "imageData"

    if not image_data_dir.exists():
        return []

    models = []
    for item in image_data_dir.iterdir():
        if item.is_dir():
            models.append(item.name)

    return sorted(models)


def display_available_models(models: List[str], script_dir: Path):
    """Display available models with their statistics."""
    print("=" * 60)
    print("Available Models:")
    print("=" * 60)

    if not models:
        print("  No models found in imageData/")
        return

    for idx, model_name in enumerate(models, 1):
        model_path = script_dir / "imageData" / model_name
        metadata_path = model_path / "import_metadata.json"

        # Try to load metadata
        if metadata_path.exists():
            try:
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                total_images = metadata.get('total_images', 0)
                categories = metadata.get('categories', 0)
                size_mb = metadata.get('total_size_mb', 0)
                print(f"  {idx}. {model_name}")
                print(f"     {total_images} images, {categories} categories, {size_mb:.2f} MB")
            except:
                print(f"  {idx}. {model_name} (metadata unavailable)")
        else:
            # Count manually
            image_count = 0
            category_count = 0
            for cat_dir in model_path.iterdir():
                if cat_dir.is_dir():
                    category_count += 1
                    image_count += len(list(cat_dir.glob("*.JPEG")))
            print(f"  {idx}. {model_name}")
            print(f"     {image_count} images, {category_count} categories")

    print("=" * 60)


def select_model(models: List[str]) -> str:
    """
    Let user select a model from available options.

    Args:
        models: List of available model names

    Returns:
        Selected model name, or empty string if cancelled
    """
    print()
    print("Enter model number or name:")
    print("(Press Enter to cancel)")

    user_input = input("\nSelection: ").strip()

    if not user_input:
        return ""

    # Check if input is a number
    try:
        idx = int(user_input)
        if 1 <= idx <= len(models):
            return models[idx - 1]
        else:
            print(f"Error: Number must be between 1 and {len(models)}")
            return ""
    except ValueError:
        # Input is a name
        if user_input in models:
            return user_input
        else:
            print(f"Error: Model '{user_input}' not found")
            return ""


def main():
    """Main entry point."""
    # Get script directory (relative to this file)
    script_dir = Path(__file__).parent.resolve()

    print(f"Script directory: {script_dir}")
    print()

    # Scan for available models
    available_models = get_available_models(script_dir)

    if not available_models:
        print("Error: No models found in imageData/")
        print("Please run import_tiny_imagenet.py first.")
        sys.exit(1)

    # Display available models
    display_available_models(available_models, script_dir)

    # Let user select model
    model_name = select_model(available_models)

    if not model_name:
        print("Conversion cancelled.")
        sys.exit(0)

    image_data_path = script_dir / "imageData" / model_name
    print(f"\nSelected: {model_name}")
    print(f"Data location: {image_data_path}")
    print()

    # Create and run converter
    converter = ImageToNPZConverter(script_dir, model_name)
    converter.run()


if __name__ == "__main__":
    main()
