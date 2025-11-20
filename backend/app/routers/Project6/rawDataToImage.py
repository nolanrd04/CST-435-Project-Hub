"""
Converts ndjson files from rawData/ to 28x28 B&W images in imageData/
Example: rawData/apple.ndjson -> imageData/apple/version1/[images].png
"""

import os
import json
import ndjson
import numpy as np
from PIL import Image, ImageDraw
from typing import List, Tuple
import sys

# Categories to process
CATEGORIES = [
    'apple', 'banana', 'blackberry', 'grapes', 'pear', 'strawberry', 'watermelon'
]

IMAGE_SIZE = 28
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RAW_DATA_DIR = os.path.join(SCRIPT_DIR, 'rawData')
IMAGE_DATA_DIR = os.path.join(SCRIPT_DIR, 'imageData')


def strokes_to_image(drawing: List[List[List[int]]], size: int = 28) -> np.ndarray:
    """
    Convert QuickDraw stroke data to 28x28 B&W image.
    
    Args:
        drawing: QuickDraw format - list of strokes where each stroke is [[x coords], [y coords]]
        size: Output image size (default 28x28)
    
    Returns:
        numpy array of shape (size, size) with values in [0, 1]
    """
    # Create white background image
    img = Image.new('L', (size, size), 255)
    draw = ImageDraw.Draw(img)
    
    # Draw each stroke
    for stroke in drawing:
        if len(stroke) != 2 or len(stroke[0]) < 2:
            continue
        
        # QuickDraw format: stroke = [x_coords, y_coords]
        x_coords = stroke[0]
        y_coords = stroke[1]
        
        if len(x_coords) != len(y_coords):
            continue
        
        # Convert stroke coordinates to image coordinates
        # QuickDraw coordinates are in 0-255 space
        points = []
        for x, y in zip(x_coords, y_coords):
            # Scale from 0-255 to image_size x image_size
            scaled_x = int((x / 255) * (size - 1))
            scaled_y = int((y / 255) * (size - 1))
            # Clamp to valid range
            scaled_x = max(0, min(size - 1, scaled_x))
            scaled_y = max(0, min(size - 1, scaled_y))
            points.append((scaled_x, scaled_y))
        
        # Draw line connecting the points
        if len(points) > 1:
            draw.line(points, fill=0, width=2)
    
    # Convert to numpy array and normalize to [0, 1]
    img_array = np.array(img, dtype=np.float32) / 255.0
    
    return img_array


def process_category(category: str, version_name: str, max_images: int = 2000, image_size: int = 28, quality_filters: dict = None) -> int:
    """
    Process a single category from ndjson to images.

    Args:
        category: Category name (e.g., 'apple')
        version_name: Version name for organizing data (e.g., 'v1')
        max_images: Maximum number of images per category
        image_size: Resolution of output images (default: 28)
        quality_filters: Dictionary with filtering thresholds

    Returns:
        Number of images processed
    """
    # Default quality filters if none provided
    if quality_filters is None:
        quality_filters = {
            'enabled': True,
            'min_strokes': 2,
            'max_strokes': 30,
            'min_points': 15,
            'max_points': 500
        }
    input_file = os.path.join(RAW_DATA_DIR, f'full_simplified_{category}.ndjson')
    output_dir = os.path.join(IMAGE_DATA_DIR, category, version_name)

    # Check if input file exists
    if not os.path.exists(input_file):
        print(f"❌ {input_file} not found!")
        return 0

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    image_count = 0

    print(f"\n📊 Processing '{category}'...")
    print(f"   Input: {input_file}")
    print(f"   Output: {output_dir}")
    print(f"   Resolution: {image_size}×{image_size}")
    print(f"   Max images: {max_images}")
    
    # Read and process ndjson file
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            reader = ndjson.reader(f)
            
            total_drawings = 0
            recognized_count = 0
            quality_filtered_count = 0

            for idx, drawing in enumerate(reader):
                total_drawings += 1
                
                # Check if we've reached image limit BEFORE processing
                if image_count >= max_images:
                    print(f"   ✓ Reached image limit ({max_images})")
                    break
                
                # Only process recognized drawings
                if not drawing.get('recognized', False):
                    continue

                recognized_count += 1

                try:
                    # Extract drawing (QuickDraw format) and convert to image
                    drawing_data = drawing.get('drawing', [])
                    if not drawing_data:
                        continue

                    # Quality filtering based on drawing complexity
                    if quality_filters['enabled']:
                        stroke_count = len(drawing_data)
                        total_points = sum(len(stroke[0]) for stroke in drawing_data if len(stroke) >= 2)

                        # Filter out low-quality drawings
                        # Skip drawings with too few strokes (rushed/simple)
                        if stroke_count < quality_filters['min_strokes']:
                            quality_filtered_count += 1
                            continue

                        # Skip drawings with too few points (not detailed enough)
                        if total_points < quality_filters['min_points']:
                            quality_filtered_count += 1
                            continue

                        # Skip extremely complex outliers (may be noise/scribbles)
                        if stroke_count > quality_filters['max_strokes'] or total_points > quality_filters['max_points']:
                            quality_filtered_count += 1
                            continue

                    image_array = strokes_to_image(drawing_data, size=image_size)
                    
                    # Convert to PIL Image for saving (convert float to uint8)
                    image_uint8 = (image_array * 255).astype(np.uint8)
                    img = Image.fromarray(image_uint8, mode='L')
                    
                    # Save image
                    image_filename = os.path.join(output_dir, f'{image_count:06d}.png')
                    img.save(image_filename)
                    
                    # Update image count
                    image_count += 1
                    
                    # Check if we've reached image limit
                    if image_count >= max_images:
                        print(f"   Processing... {image_count} images - Limit reached")
                        break
                    
                    # Print progress every 100 images
                    if image_count % 100 == 0:
                        print(f"   Processing... {image_count} images")
                
                except Exception as e:
                    if image_count == 0:
                        print(f"   ❌ Error: {e}")
                    continue
    
    except Exception as e:
        print(f"❌ Error reading {input_file}: {e}")
        return 0
    
    # Final summary
    print(f"   ✅ Complete: {image_count} images saved")
    print(f"   📈 Total drawings in file: {total_drawings}")
    print(f"   ✓ Recognized drawings: {recognized_count}")
    print(f"   🔍 Quality filtered: {quality_filtered_count} (low complexity)")
    
    return image_count


def get_user_inputs() -> Tuple[int, str, int, dict]:
    """
    Get user customization inputs for data processing.

    Returns:
        Tuple of (max_images, version_name, image_size, quality_filters)
    """
    print("\n" + "="*60)
    print("🎨 QuickDraw ndjson to Image Converter")
    print("="*60)

    # Get max images
    while True:
        try:
            max_images = int(input("\n📦 Maximum images per category [default: 2000]: ") or "2000")
            if max_images <= 0:
                print("   ❌ Number must be positive!")
                continue
            break
        except ValueError:
            print("   ❌ Please enter a valid number!")

    # Get version name
    version_name = input("\n📂 Version name (e.g., 'v1', 'v2', 'test') [default: 'default']: ").strip() or "default"

    # Remove spaces and special characters from version name
    version_name = version_name.replace(' ', '_')

    # Get image resolution
    print("\n📐 Image Resolution:")
    print("   Common options: 28, 32, 64, 128")
    print("   Larger = more detail but slower training")
    while True:
        try:
            image_size = int(input("   Enter resolution [default: 32]: ") or "32")
            if image_size <= 0:
                print("   ❌ Resolution must be positive!")
                continue
            if image_size > 256:
                print("   ⚠️  Warning: Very large resolution may be slow!")
            break
        except ValueError:
            print("   ❌ Please enter a valid number!")

    # Get quality filtering level
    print("\n🔍 Quality Filtering (based on drawing complexity):")
    print("   1. None       - No filtering (all recognized drawings)")
    print("   2. Lenient    - Light filtering (70-80% kept)")
    print("   3. Balanced   - Medium filtering (40-50% kept) ⭐ RECOMMENDED")
    print("   4. Strict     - Heavy filtering (15-25% kept, best quality)")
    print("   5. Custom     - Set your own thresholds")

    while True:
        choice = input("\n   Select filtering level [default: 1]: ").strip() or "1"

        if choice == "1":
            quality_filters = {
                'enabled': False,
                'min_strokes': 0,
                'max_strokes': 999,
                'min_points': 0,
                'max_points': 9999
            }
            break
        elif choice == "2":
            quality_filters = {
                'enabled': True,
                'min_strokes': 1,
                'max_strokes': 50,
                'min_points': 10,
                'max_points': 800
            }
            break
        elif choice == "3":
            quality_filters = {
                'enabled': True,
                'min_strokes': 2,
                'max_strokes': 30,
                'min_points': 15,
                'max_points': 500
            }
            break
        elif choice == "4":
            quality_filters = {
                'enabled': True,
                'min_strokes': 4,
                'max_strokes': 20,
                'min_points': 40,
                'max_points': 300
            }
            break
        elif choice == "5":
            print("\n   Custom Thresholds:")
            quality_filters = {
                'enabled': True,
                'min_strokes': int(input("     Min strokes [default: 2]: ") or "2"),
                'max_strokes': int(input("     Max strokes [default: 30]: ") or "30"),
                'min_points': int(input("     Min points [default: 15]: ") or "15"),
                'max_points': int(input("     Max points [default: 500]: ") or "500")
            }
            break
        else:
            print("   ❌ Invalid choice. Please enter 1-5.")

    return max_images, version_name, image_size, quality_filters


def main():
    """Main entry point for the converter."""

    # Get user inputs
    max_images, version_name, image_size, quality_filters = get_user_inputs()

    print(f"\n{'='*60}")
    print(f"🔄 Starting conversion with:")
    print(f"   Max images: {max_images} per category")
    print(f"   Version: '{version_name}'")
    print(f"   Resolution: {image_size}×{image_size}")
    if quality_filters['enabled']:
        print(f"   Quality Filter: Enabled")
        print(f"     Strokes: {quality_filters['min_strokes']}-{quality_filters['max_strokes']}")
        print(f"     Points: {quality_filters['min_points']}-{quality_filters['max_points']}")
    else:
        print(f"   Quality Filter: Disabled")
    print(f"   Categories: {len(CATEGORIES)}")
    print(f"{'='*60}")

    # Create imageData directory if it doesn't exist
    os.makedirs(IMAGE_DATA_DIR, exist_ok=True)

    # Process each category
    total_images = 0
    for category in CATEGORIES:
        images_processed = process_category(category, version_name, max_images, image_size, quality_filters)
        total_images += images_processed
    
    # Final summary
    print(f"\n{'='*60}")
    print(f"✅ Conversion Complete!")
    print(f"   Total images: {total_images}")
    print(f"   Categories: {len(CATEGORIES)}")
    print(f"   Resolution: {image_size}×{image_size}")
    print(f"   Output directory: {IMAGE_DATA_DIR}")
    print(f"{'='*60}\n")


if __name__ == '__main__':
    main()