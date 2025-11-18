"""
Converts PNG images from imageData/ to .npz arrays for GAN training.
Example: imageData/apple/version1/ -> npzData/apple_version1.npz
"""

import os
import numpy as np
from PIL import Image
from typing import Tuple, Dict, List
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
IMAGE_DATA_DIR = os.path.join(SCRIPT_DIR, 'imageData')
NPZ_DATA_DIR = os.path.join(SCRIPT_DIR, 'npzData')

CATEGORIES = [
    'apple', 'banana', 'blackberry', 'grapes', 'pear', 'strawberry', 'watermelon'
]

IMAGE_SIZE = 28


def load_images_from_directory(image_dir: str) -> np.ndarray:
    """
    Load all PNG images from a directory into a numpy array.
    
    Args:
        image_dir: Directory containing PNG images
    
    Returns:
        numpy array of shape (N, IMAGE_SIZE, IMAGE_SIZE) with values in [0, 1]
    """
    images = []
    
    if not os.path.exists(image_dir):
        print(f"   ⚠️  Directory not found: {image_dir}")
        return np.array([])
    
    image_files = sorted([f for f in os.listdir(image_dir) if f.endswith('.png')])
    
    if not image_files:
        print(f"   ⚠️  No PNG files found in {image_dir}")
        return np.array([])
    
    print(f"   Loading {len(image_files)} images...")
    
    for idx, filename in enumerate(image_files):
        try:
            img_path = os.path.join(image_dir, filename)
            # Open image and ensure it's grayscale
            img = Image.open(img_path).convert('L')
            
            # Check size
            if img.size != (IMAGE_SIZE, IMAGE_SIZE):
                img = img.resize((IMAGE_SIZE, IMAGE_SIZE), Image.Resampling.LANCZOS)
            
            # Convert to numpy array and normalize to [0, 1]
            img_array = np.array(img, dtype=np.float32) / 255.0
            images.append(img_array)
            
            # Print progress every 100 images
            if (idx + 1) % 100 == 0:
                print(f"      Loaded {idx + 1}/{len(image_files)} images")
        
        except Exception as e:
            print(f"      ⚠️  Error loading {img_path}: {e}")
            continue
    
    # Stack into single array: (N, HEIGHT, WIDTH)
    if images:
        images_array = np.stack(images, axis=0)
        print(f"   ✓ Loaded shape: {images_array.shape}")
        return images_array
    else:
        return np.array([])


def images_to_npz(category: str, version_name: str) -> bool:
    """
    Convert images for a category+version to .npz format.
    
    Args:
        category: Category name (e.g., 'apple')
        version_name: Version name (e.g., 'v1')
    
    Returns:
        True if successful, False otherwise
    """
    # Define paths
    image_dir = os.path.join(IMAGE_DATA_DIR, category, version_name)
    npz_output_file = os.path.join(NPZ_DATA_DIR, f'{category}_{version_name}.npz')
    
    # Check if image directory exists
    if not os.path.exists(image_dir):
        print(f"❌ Image directory not found: {image_dir}")
        return False
    
    print(f"\n📊 Processing: {category}/{version_name}")
    print(f"   Input: {image_dir}")
    print(f"   Output: {npz_output_file}")
    
    # Load images
    images = load_images_from_directory(image_dir)
    
    if images.size == 0:
        print(f"❌ No images loaded for {category}/{version_name}")
        return False
    
    # Create output directory
    os.makedirs(NPZ_DATA_DIR, exist_ok=True)
    
    # Save as .npz with metadata
    try:
        np.savez_compressed(
            npz_output_file,
            images=images,
            category=category,
            version=version_name,
            image_count=len(images),
            image_size=IMAGE_SIZE
        )
        
        # Get file size
        file_size_mb = os.path.getsize(npz_output_file) / (1024 * 1024)
        print(f"   ✅ Saved: {len(images)} images ({file_size_mb:.2f}MB)")
        
        return True
    
    except Exception as e:
        print(f"❌ Error saving {npz_output_file}: {e}")
        return False


def get_available_versions() -> Dict[str, List[str]]:
    """
    Scan imageData/ and find all category/version combinations.
    
    Returns:
        Dictionary mapping category -> list of versions
    """
    versions = {}
    
    if not os.path.exists(IMAGE_DATA_DIR):
        print("❌ imageData/ directory not found!")
        return versions
    
    for category in os.listdir(IMAGE_DATA_DIR):
        category_path = os.path.join(IMAGE_DATA_DIR, category)
        if not os.path.isdir(category_path):
            continue
        
        version_dirs = [d for d in os.listdir(category_path) if os.path.isdir(os.path.join(category_path, d))]
        
        if version_dirs:
            versions[category] = version_dirs
    
    return versions


def display_available_versions(versions: Dict[str, List[str]]):
    """Display available category/version combinations."""
    print(f"\n{'='*60}")
    print("📂 Available Categories & Versions:")
    print(f"{'='*60}")
    
    if not versions:
        print("   ⚠️  No categories found in imageData/")
        return
    
    for category in sorted(versions.keys()):
        print(f"   {category}:")
        for version in sorted(versions[category]):
            version_path = os.path.join(IMAGE_DATA_DIR, category, version)
            img_count = len([f for f in os.listdir(version_path) if f.endswith('.png')])
            print(f"      - {version} ({img_count} images)")


def main():
    """Main entry point for converting images to NPZ."""
    
    # Scan for available versions
    available_versions = get_available_versions()
    display_available_versions(available_versions)
    
    if not available_versions:
        print("\n❌ No image data found. Please run rawDataToImage.py first!")
        return
    
    print(f"\n{'='*60}")
    print("🔄 Starting Image to NPZ Conversion")
    print(f"{'='*60}")
    
    # Process each category/version combination
    successful = 0
    failed = 0
    
    for category in sorted(available_versions.keys()):
        for version in sorted(available_versions[category]):
            if images_to_npz(category, version):
                successful += 1
            else:
                failed += 1
    
    # Final summary
    print(f"\n{'='*60}")
    print(f"✅ Conversion Complete!")
    print(f"   Successful: {successful}")
    print(f"   Failed: {failed}")
    print(f"   Output directory: {NPZ_DATA_DIR}")
    print(f"{'='*60}\n")


if __name__ == '__main__':
    main()
