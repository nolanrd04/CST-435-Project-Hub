"""
ImageNet Data Importer for Project 8: Image Enhancement
Downloads and imports ImageNet dataset with customizable resolution, size limits, and diverse sampling.

Requirements:
    pip install kaggle pillow

Setup Kaggle API:
    1. Go to https://www.kaggle.com/settings
    2. Click "Create New API Token" to download kaggle.json
    3. Place kaggle.json in:
       - Windows: C:\\Users\\<username>\\.kaggle\\kaggle.json
       - Linux/Mac: ~/.kaggle/kaggle.json
"""

import os
import sys
import zipfile
import json
from pathlib import Path
from PIL import Image
import shutil
from typing import Dict, List, Tuple
import random

# Script directory for relative paths
SCRIPT_DIR = Path(__file__).parent
IMAGE_DATA_DIR = SCRIPT_DIR / "imageData"
TEMP_DIR = SCRIPT_DIR / "temp_download"
KAGGLE_DATASET_DIR = SCRIPT_DIR / "kaggle_imagenet"


class ImageNetImporter:
    """Handles ImageNet dataset download and import with flexible options."""

    def __init__(self):
        self.downloaded_path = None
        self.extracted_path = None

    def check_kaggle_setup(self) -> bool:
        """Check if Kaggle API is properly configured."""
        try:
            import kaggle
            return True
        except ImportError:
            print("❌ Kaggle package not installed.")
            print("   Install with: pip install kaggle")
            return False
        except OSError as e:
            if "Could not find kaggle.json" in str(e):
                print("❌ Kaggle API credentials not found.")
                print()
                print("   Setup instructions:")
                print("   1. Go to https://www.kaggle.com/settings")
                print("   2. Scroll to 'API' section")
                print("   3. Click 'Create New API Token'")
                print("   4. Save kaggle.json to:")
                print(f"      Windows: C:\\Users\\<username>\\.kaggle\\kaggle.json")
                print(f"      Linux/Mac: ~/.kaggle/kaggle.json")
                return False
            raise

    def download_kaggle_dataset(self, dataset_name: str, dest_dir: Path) -> Path:
        """Download dataset from Kaggle using Kaggle API."""
        dest_dir.mkdir(parents=True, exist_ok=True)

        # Check if already downloaded
        extracted_path = dest_dir / "imagenet-256"
        if extracted_path.exists():
            print(f"  Dataset already exists at: {extracted_path}")
            return extracted_path

        print(f"  Downloading '{dataset_name}' from Kaggle...")
        print(f"  Destination: {dest_dir}")
        print()
        print("  Note: This is a large download (~37 GB compressed)")
        print("  Download may take a while depending on your connection.")
        print()

        try:
            import kaggle

            # Download dataset
            kaggle.api.dataset_download_files(
                dataset_name,
                path=str(dest_dir),
                unzip=True,
                quiet=False
            )

            # Find extracted directory
            # The dataset might extract to different folder names
            possible_names = ["imagenet-256", "archive", dataset_name.split('/')[-1]]

            for name in possible_names:
                check_path = dest_dir / name
                if check_path.exists() and check_path.is_dir():
                    extracted_path = check_path
                    break

            # If not found in expected locations, look for train/val folders
            if not extracted_path.exists():
                for item in dest_dir.iterdir():
                    if item.is_dir():
                        train_exists = (item / "train").exists()
                        val_exists = (item / "val").exists()
                        if train_exists or val_exists:
                            extracted_path = item
                            break

            if not extracted_path.exists():
                # Just use dest_dir if files are extracted directly there
                if (dest_dir / "train").exists() or (dest_dir / "val").exists():
                    extracted_path = dest_dir

            print(f"  Download and extraction complete!")
            print(f"  Dataset location: {extracted_path}")
            self.extracted_path = extracted_path
            return extracted_path

        except Exception as e:
            print(f"\n  Error downloading from Kaggle: {e}")
            raise

    def extract_dataset(self, zip_path: Path, extract_to: Path) -> Path:
        """Extract downloaded zip file."""
        print(f"\n[2/4] Extracting dataset...")
        extract_to.mkdir(parents=True, exist_ok=True)

        try:
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(extract_to)

            # Find the extracted folder (usually tiny-imagenet-200)
            extracted_dirs = [d for d in extract_to.iterdir() if d.is_dir()]
            if extracted_dirs:
                self.extracted_path = extracted_dirs[0]
            else:
                self.extracted_path = extract_to

            print(f"  Extraction complete!")
            print(f"  Extracted to: {self.extracted_path}")
            return self.extracted_path
        except Exception as e:
            print(f"  Error extracting: {e}")
            raise

    def get_all_images_from_directory(self, directory: Path) -> List[Tuple[Path, str]]:
        """
        Recursively find all image files and their categories.
        Returns list of (image_path, category_name) tuples.

        Supports multiple ImageNet structures:
        - Kaggle ImageNet 256x256: train/category/*.JPEG and val/category/*.JPEG
        - Standard ImageNet: category/*.JPEG
        - Tiny ImageNet: train/category/images/*.JPEG
        """
        images = []
        image_extensions = {'.jpg', '.jpeg', '.png', '.JPEG', '.JPG', '.PNG'}

        # Try different directory structures
        search_dirs = []

        # Check for train/val split (Kaggle ImageNet structure)
        train_dir = directory / "train"
        val_dir = directory / "val"

        if train_dir.exists():
            search_dirs.append(train_dir)
        if val_dir.exists():
            search_dirs.append(val_dir)

        # If no train/val, search in root directory
        if not search_dirs:
            search_dirs.append(directory)

        print(f"  Scanning directories: {[str(d.name) for d in search_dirs]}")

        for search_dir in search_dirs:
            for category_dir in search_dir.iterdir():
                if not category_dir.is_dir():
                    continue

                category_name = category_dir.name

                # Try different structures:
                # 1. Images directly in category folder (Kaggle ImageNet)
                # 2. Images in category/images/ subfolder (Tiny ImageNet)

                found_images = False

                # First try: category/images/ subdirectory
                images_subdir = category_dir / "images"
                if images_subdir.exists() and images_subdir.is_dir():
                    for img_file in images_subdir.iterdir():
                        if img_file.suffix.lower() in image_extensions:
                            images.append((img_file, category_name))
                            found_images = True

                # Second try: directly in category directory
                if not found_images:
                    for img_file in category_dir.iterdir():
                        if img_file.is_file() and img_file.suffix.lower() in image_extensions:
                            images.append((img_file, category_name))

        return images

    def load_category_names(self, directory: Path) -> Dict[str, str]:
        """Load human-readable category names from words.txt."""
        words_file = directory / "words.txt"
        category_names = {}

        if words_file.exists():
            with open(words_file, 'r', encoding='utf-8') as f:
                for line in f:
                    parts = line.strip().split('\t')
                    if len(parts) >= 2:
                        wnid = parts[0]
                        name = parts[1]
                        category_names[wnid] = name
            print(f"  Loaded {len(category_names)} category names")

        return category_names

    def import_images(
        self,
        source_images: List[Tuple[Path, str]],
        output_dir: Path,
        target_resolution: int,
        max_images: int,
        max_size_mb: float,
        category_names: Dict[str, str]
    ) -> Dict:
        """
        Import images with specified constraints and diverse sampling.

        Args:
            source_images: List of (image_path, category) tuples
            output_dir: Destination directory
            target_resolution: Target image size (e.g., 256 for 256x256)
            max_images: Maximum number of images to import
            max_size_mb: Maximum storage size in MB
            category_names: Dictionary mapping category IDs to names

        Returns:
            Dictionary with import statistics
        """
        output_dir.mkdir(parents=True, exist_ok=True)

        # Organize images by category for diverse sampling
        images_by_category = {}
        for img_path, category in source_images:
            if category not in images_by_category:
                images_by_category[category] = []
            images_by_category[category].append(img_path)

        num_categories = len(images_by_category)
        print(f"  Found {num_categories} categories with {len(source_images)} total images")

        # Calculate images per category for even distribution
        images_per_category = max(1, max_images // num_categories)
        print(f"  Sampling ~{images_per_category} images per category for diversity")

        # Sample images from each category
        sampled_images = []
        for category, imgs in images_by_category.items():
            # Randomly sample from this category
            sample_count = min(images_per_category, len(imgs))
            sampled = random.sample(imgs, sample_count)
            sampled_images.extend([(img, category) for img in sampled])

        # Shuffle the final list for better distribution
        random.shuffle(sampled_images)

        # Limit to max_images
        sampled_images = sampled_images[:max_images]

        print(f"  Selected {len(sampled_images)} images for import")
        print(f"\n  Starting import...")

        imported_count = 0
        total_size = 0
        max_size_bytes = max_size_mb * 1024 * 1024
        category_stats = {}
        failed_imports = 0

        for img_path, category in sampled_images:
            # Check size limit
            if total_size >= max_size_bytes:
                print(f"\n  Reached size limit ({max_size_mb} MB)")
                break

            # Check image count limit
            if imported_count >= max_images:
                print(f"\n  Reached image count limit ({max_images})")
                break

            try:
                # Create category folder with readable name
                category_readable = category_names.get(category, category)
                # Sanitize folder name
                category_readable = category_readable.replace('/', '_').replace('\\', '_')
                category_folder_name = f"{category}_{category_readable}"
                category_dir = output_dir / category_folder_name
                category_dir.mkdir(parents=True, exist_ok=True)

                # Open and resize image
                with Image.open(img_path) as img:
                    # Convert to RGB if needed
                    if img.mode != 'RGB':
                        img = img.convert('RGB')

                    # Resize to target resolution
                    img_resized = img.resize(
                        (target_resolution, target_resolution),
                        Image.Resampling.LANCZOS
                    )

                    # Save with original filename
                    output_filename = f"{category}_{imported_count}{img_path.suffix}"
                    output_path = category_dir / output_filename
                    img_resized.save(output_path, quality=95)

                    # Track statistics
                    file_size = output_path.stat().st_size
                    total_size += file_size
                    imported_count += 1

                    # Update category stats
                    if category not in category_stats:
                        category_stats[category] = {
                            'wnid': category,
                            'name': category_readable,
                            'folder': category_folder_name,
                            'count': 0
                        }
                    category_stats[category]['count'] += 1

                    # Progress update every 100 images
                    if imported_count % 100 == 0:
                        print(f"\r  Imported: {imported_count} images ({total_size / (1024*1024):.2f} MB)", end='')

            except Exception as e:
                failed_imports += 1
                if failed_imports <= 5:  # Only show first 5 errors
                    print(f"\n  Warning: Failed to import {img_path.name}: {e}")
                continue

        print(f"\n  Import complete! Total: {imported_count} images ({total_size / (1024*1024):.2f} MB)")

        if failed_imports > 0:
            print(f"  Note: {failed_imports} images failed to import")

        return {
            'total_images': imported_count,
            'total_size_mb': total_size / (1024 * 1024),
            'categories': len(category_stats),
            'category_stats': category_stats,
            'failed_imports': failed_imports
        }


def main():
    print(f"Script directory: {SCRIPT_DIR}")
    print()

    # Ensure imageData directory exists
    IMAGE_DATA_DIR.mkdir(parents=True, exist_ok=True)

    # Step 1: Check Kaggle setup and download dataset
    print("="*60)
    print("ImageNet Data Importer for Project 8")
    print("="*60)
    print()
    print("This script downloads ImageNet 256x256 from Kaggle")
    print("Dataset: https://www.kaggle.com/datasets/dimensi0n/imagenet-256")
    print()

    importer = ImageNetImporter()

    # Check if Kaggle is set up
    if not importer.check_kaggle_setup():
        print()
        print("Setup Kaggle API first, then run this script again.")
        return

    print("✅ Kaggle API is configured")
    print()

    # Ask if user wants to download or use existing
    print("Download Options:")
    print("  1. Download dataset from Kaggle (auto-downloads to Project8/kaggle_imagenet)")
    print("  2. Use existing downloaded dataset (provide path)")
    print()

    choice = input("Enter choice (1 or 2, default: 1): ").strip()

    if choice == "2":
        # Manual path option
        source_path_input = input("Enter path to extracted ImageNet dataset: ").strip()
        if not source_path_input:
            print("Error: Path is required")
            return

        source_path = Path(source_path_input)
        if not source_path.exists():
            print(f"Error: Path does not exist: {source_path}")
            return

        if not source_path.is_dir():
            print(f"Error: Path is not a directory: {source_path}")
            return

        print(f"Using existing dataset: {source_path}")
    else:
        # Auto-download option
        print()
        print("Dataset will be downloaded to:")
        print(f"  {KAGGLE_DATASET_DIR}")
        print()

        proceed = input("Proceed with download? (y/n): ").strip().lower()
        if proceed != 'y':
            print("Download cancelled.")
            return

        try:
            print()
            print("[Step 1/3] Downloading dataset from Kaggle...")
            print()
            source_path = importer.download_kaggle_dataset(
                "dimensi0n/imagenet-256",
                KAGGLE_DATASET_DIR
            )
        except Exception as e:
            print(f"Failed to download dataset: {e}")
            return

    print()
    print(f"Source dataset: {source_path}")
    print()

    # Step 2: Get dataset name
    dataset_name = input("Enter dataset name (saves to imageData/dataset_name): ").strip()
    if not dataset_name:
        print("Error: Dataset name is required")
        return

    output_dir = IMAGE_DATA_DIR / dataset_name
    print(f"Images will be saved to: {output_dir}")
    print()

    # Check if dataset already exists
    if output_dir.exists() and any(output_dir.iterdir()):
        print(f"Warning: Dataset '{dataset_name}' already exists!")
        overwrite = input("Overwrite existing dataset? (y/n): ").strip().lower()
        if overwrite != 'y':
            print("Import cancelled.")
            return
        else:
            print("Removing existing dataset...")
            shutil.rmtree(output_dir)

    # Step 3: Get configuration
    print("="*60)
    print("Configuration")
    print("="*60)
    print()

    # Resolution
    print("Target Resolution:")
    print("  256: Keep original (256x256) - Recommended")
    print("  128: Downsample to 128x128")
    print("  64: Downsample to 64x64")
    print("  512: Upsample to 512x512")
    resolution_input = input("Enter target resolution (default: 256): ").strip()
    target_resolution = int(resolution_input) if resolution_input else 256

    # Max images
    max_images_input = input("Enter maximum number of images (default: 10000): ").strip()
    max_images = int(max_images_input) if max_images_input else 10000

    # Max size
    max_size_input = input("Enter maximum storage size in MB (default: 500): ").strip()
    max_size_mb = float(max_size_input) if max_size_input else 500.0

    print()
    print("-"*60)
    print("Configuration Summary:")
    print(f"  Dataset name: {dataset_name}")
    print(f"  Resolution: {target_resolution}x{target_resolution}")
    print(f"  Max images: {max_images}")
    print(f"  Max size: {max_size_mb} MB")
    print("-"*60)
    print()

    proceed = input("Proceed with import? (y/n): ").strip().lower()
    if proceed != 'y':
        print("Import cancelled.")
        return

    # Step 4: Process dataset
    try:
        print()
        print("[Step 2/3] Scanning source dataset...")

        # Load category names (if words.txt exists)
        category_names = importer.load_category_names(source_path)

        # Get all available images
        all_images = importer.get_all_images_from_directory(source_path)

        if not all_images:
            print("Error: No images found in dataset")
            print("Expected structure:")
            print("  - train/category/*.JPEG")
            print("  - val/category/*.JPEG")
            return

        print(f"  Found {len(all_images)} total images")

        # Step 5: Import with constraints
        print()
        print("[Step 3/3] Importing images...")

        stats = importer.import_images(
            source_images=all_images,
            output_dir=output_dir,
            target_resolution=target_resolution,
            max_images=max_images,
            max_size_mb=max_size_mb,
            category_names=category_names
        )

        # Save metadata
        metadata = {
            'dataset_name': dataset_name,
            'source_path': str(source_path),
            'config': {
                'resolution': target_resolution,
                'max_images': max_images,
                'max_size_mb': max_size_mb
            },
            'stats': stats
        }

        metadata_path = output_dir / 'import_metadata.json'
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2)

        print(f"  Metadata saved to: {metadata_path.name}")

        # Final summary
        print()
        print("="*60)
        print("Import Summary:")
        print(f"  Dataset: {dataset_name}")
        print(f"  Source: {source_path}")
        print(f"  Images imported: {stats['total_images']}")
        print(f"  Total size: {stats['total_size_mb']:.2f} MB")
        print(f"  Categories: {stats['categories']}")
        print(f"  Resolution: {target_resolution}x{target_resolution}")
        print(f"  Output directory: {output_dir}")
        print("="*60)
        print()
        print("Import completed successfully!")

    except Exception as e:
        print(f"\n❌ Error during import: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()