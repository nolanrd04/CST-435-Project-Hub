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


def get_category_thresholds(category: str, strictness: int) -> dict:
    """
    Get category-specific filtering thresholds based on data analysis.
    
    Args:
        category: Category name (e.g., 'apple')
        strictness: 1=Lenient, 2=Balanced, 3=Strict
    
    Returns:
        Dictionary with category-specific thresholds
    """
    # Data-driven thresholds based on actual statistics
    category_data = {
        'apple': {
            'strokes': {'min': 1, 'max': 26, 'avg': 2.7, 'median': 3},
            'points': {'min': 2, 'max': 190, 'avg': 35.7, 'median': 35},
            'per_stroke': {'min': 2, 'max': 126, 'avg': 13.1}
        },
        'banana': {
            'strokes': {'min': 1, 'max': 27, 'avg': 2.2, 'median': 2},
            'points': {'min': 2, 'max': 205, 'avg': 27.8, 'median': 25},
            'per_stroke': {'min': 2, 'max': 179, 'avg': 12.7}
        },
        'blackberry': {
            'strokes': {'min': 1, 'max': 322, 'avg': 7.3, 'median': 6},
            'points': {'min': 2, 'max': 911, 'avg': 126.8, 'median': 117},
            'per_stroke': {'min': 2, 'max': 910, 'avg': 17.3}
        },
        'grapes': {
            'strokes': {'min': 1, 'max': 54, 'avg': 8.2, 'median': 8},
            'points': {'min': 2, 'max': 374, 'avg': 87.9, 'median': 85},
            'per_stroke': {'min': 2, 'max': 266, 'avg': 10.8}
        },
        'pear': {
            'strokes': {'min': 1, 'max': 25, 'avg': 2.0, 'median': 2},
            'points': {'min': 2, 'max': 186, 'avg': 28.0, 'median': 28},
            'per_stroke': {'min': 2, 'max': 161, 'avg': 14.0}
        },
        'strawberry': {
            'strokes': {'min': 1, 'max': 81, 'avg': 8.7, 'median': 8},
            'points': {'min': 2, 'max': 285, 'avg': 51.9, 'median': 49},
            'per_stroke': {'min': 2, 'max': 215, 'avg': 6.0}
        },
        'watermelon': {
            'strokes': {'min': 1, 'max': 170, 'avg': 6.4, 'median': 5},
            'points': {'min': 2, 'max': 359, 'avg': 42.2, 'median': 40},
            'per_stroke': {'min': 2, 'max': 150, 'avg': 6.6}
        }
    }
    
    if category not in category_data:
        # Fallback to balanced defaults if category not found
        return {
            'min_strokes': 1,
            'max_strokes': 111,
            'min_points': 2,
            'max_points': 459,
            'min_points_per_stroke': 2,
            'max_points_per_stroke': 307
        }
    
    data = category_data[category]
    
    if strictness == 1:  # Lenient
        return {
            'min_strokes': max(1, int(data['strokes']['avg'] - 1)),
            'max_strokes': min(int(data['strokes']['max'] * 0.8), int(data['strokes']['avg'] + 15)),
            'min_points': max(2, int(data['points']['avg'] - 20)),
            'max_points': min(int(data['points']['max'] * 0.9), int(data['points']['avg'] + 100)),
            'min_points_per_stroke': 2,
            'max_points_per_stroke': min(int(data['per_stroke']['max'] * 0.7), int(data['per_stroke']['avg'] + 50))
        }
    elif strictness == 2:  # Balanced (median-based)
        return {
            'min_strokes': max(1, int(data['strokes']['median'] - 1)),
            'max_strokes': min(int(data['strokes']['max'] * 0.6), int(data['strokes']['avg'] + 10)),
            'min_points': max(2, int(data['points']['median'] - 10)),
            'max_points': min(int(data['points']['max'] * 0.7), int(data['points']['avg'] + 50)),
            'min_points_per_stroke': 3,
            'max_points_per_stroke': min(int(data['per_stroke']['max'] * 0.5), int(data['per_stroke']['avg'] + 20))
        }
    else:  # Strict
        return {
            'min_strokes': max(1, int(data['strokes']['median'])),
            'max_strokes': min(int(data['strokes']['max'] * 0.4), int(data['strokes']['avg'] + 5)),
            'min_points': max(5, int(data['points']['median'])),
            'max_points': min(int(data['points']['max'] * 0.5), int(data['points']['avg'] + 20)),
            'min_points_per_stroke': 4,
            'max_points_per_stroke': min(int(data['per_stroke']['max'] * 0.3), int(data['per_stroke']['avg'] + 10))
        }


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

    print(f"\nProcessing '{category}'...")
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
                        # Get category-specific thresholds if per-category filtering is enabled
                        if quality_filters.get('per_category', False):
                            cat_thresholds = get_category_thresholds(
                                category, 
                                quality_filters.get('category_strictness', 2)
                            )
                            current_filters = cat_thresholds
                        else:
                            current_filters = quality_filters
                        
                        stroke_count = len(drawing_data)
                        total_points = 0
                        max_stroke_points = 0
                        min_stroke_points = float('inf')
                        stroke_points_list = []
                        
                        # Analyze each stroke individually
                        for stroke in drawing_data:
                            if len(stroke) >= 2:
                                stroke_points = len(stroke[0])
                                total_points += stroke_points
                                max_stroke_points = max(max_stroke_points, stroke_points)
                                min_stroke_points = min(min_stroke_points, stroke_points)
                                stroke_points_list.append(stroke_points)
                        
                        # Handle case where no valid strokes found
                        if not stroke_points_list:
                            quality_filtered_count += 1
                            continue
                        
                        if min_stroke_points == float('inf'):
                            min_stroke_points = 0

                        # Filter out low-quality drawings
                        # Skip drawings with too few strokes (rushed/simple)
                        if stroke_count < current_filters['min_strokes']:
                            quality_filtered_count += 1
                            continue

                        # Skip drawings with too few points overall (not detailed enough)
                        if total_points < current_filters['min_points']:
                            quality_filtered_count += 1
                            continue

                        # Skip extremely complex outliers (may be noise/scribbles)
                        if stroke_count > current_filters['max_strokes'] or total_points > current_filters['max_points']:
                            quality_filtered_count += 1
                            continue
                        
                        # Per-stroke filtering (if enabled)
                        if quality_filters.get('per_stroke_enabled', False):
                            # Use category-specific or global per-stroke thresholds
                            min_per_stroke = current_filters.get('min_points_per_stroke', quality_filters.get('min_points_per_stroke', 3))
                            max_per_stroke = current_filters.get('max_points_per_stroke', quality_filters.get('max_points_per_stroke', 150))
                            
                            # Filter individual strokes that are too complex (likely scribbles/noise)
                            if max_stroke_points > max_per_stroke:
                                quality_filtered_count += 1
                                continue
                            
                            # Filter individual strokes that are too simple
                            if min_stroke_points < min_per_stroke:
                                quality_filtered_count += 1
                                continue
                        
                        # Console monitoring (if enabled)
                        if quality_filters.get('monitor_per_stroke', False) and image_count < 10:
                            avg_points_per_stroke = total_points / len(stroke_points_list)
                            if quality_filters.get('per_category', False):
                                print(f"     🔍 {category} Drawing {image_count}: {stroke_count} strokes, "
                                      f"total: {total_points} pts, per-stroke: {min_stroke_points}-{max_stroke_points} "
                                      f"(avg: {avg_points_per_stroke:.1f}) [Thresholds: {current_filters['min_strokes']}-{current_filters['max_strokes']} strokes, {current_filters['min_points']}-{current_filters['max_points']} pts]")
                            else:
                                print(f"     🔍 Drawing {image_count}: {stroke_count} strokes, "
                                      f"total: {total_points} pts, per-stroke: {min_stroke_points}-{max_stroke_points} "
                                      f"(avg: {avg_points_per_stroke:.1f})")

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
    print(f"   Complete: {image_count} images saved")
    print(f"   📈 Total drawings in file: {total_drawings}")
    print(f"   ✓ Recognized drawings: {recognized_count}")
    print(f"   🔍 Quality filtered: {quality_filtered_count} (complexity + per-stroke analysis)")
    
    return image_count


def get_user_inputs() -> Tuple[int, str, int, dict]:
    """
    Get user customization inputs for data processing.

    Returns:
        Tuple of (max_images, version_name, image_size, quality_filters)
    """
    print("\n" + "="*60)
    print("QuickDraw ndjson to Image Converter")
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
    print("\nQuality Filtering (based on drawing complexity):")
    print("   Filters based on: stroke count, total points, and optionally per-stroke complexity")
    print("   Values are data-driven from actual QuickDraw statistics")
    print("   1. None       - No filtering (all recognized drawings)")
    print("   2. Lenient    - Light filtering (70-80% kept) - Based on data analysis")
    print("   3. Balanced   - Medium filtering (40-50% kept) [RECOMMENDED] - Based on data analysis")
    print("   4. Strict     - Heavy filtering (15-25% kept, best quality) - Based on data analysis")
    print("   5. Custom     - Set your own thresholds")
    print("   6. Per-Category - Use category-specific thresholds")

    while True:
        choice = input("\n   Select filtering level [default: 1]: ").strip() or "1"

        if choice == "1":
            quality_filters = {
                'enabled': False,
                'min_strokes': 0,
                'max_strokes': 999,
                'min_points': 0,
                'max_points': 9999,
                'per_stroke_enabled': False,
                'monitor_per_stroke': False,
                'per_category': False
            }
            break
        elif choice == "2":
            # Lenient - Data-driven values
            quality_filters = {
                'enabled': True,
                'min_strokes': 1,
                'max_strokes': 121,
                'min_points': 5,
                'max_points': 659,
                'per_stroke_enabled': False,
                'monitor_per_stroke': False,
                'per_category': False
            }
            break
        elif choice == "3":
            # Balanced - Data-driven values
            quality_filters = {
                'enabled': True,
                'min_strokes': 1,
                'max_strokes': 111,
                'min_points': 2,
                'max_points': 459,
                'per_stroke_enabled': False,
                'monitor_per_stroke': False,
                'per_category': False
            }
            break
        elif choice == "4":
            # Strict - Data-driven values
            quality_filters = {
                'enabled': True,
                'min_strokes': 2,
                'max_strokes': 96,
                'min_points': 17,
                'max_points': 309,
                'per_stroke_enabled': False,
                'monitor_per_stroke': False,
                'per_category': False
            }
            break
        elif choice == "5":
            print("\n   Custom Thresholds:")
            quality_filters = {
                'enabled': True,
                'min_strokes': int(input("     Min strokes [default: 1]: ") or "1"),
                'max_strokes': int(input("     Max strokes [default: 111]: ") or "111"),
                'min_points': int(input("     Min points [default: 2]: ") or "2"),
                'max_points': int(input("     Max points [default: 459]: ") or "459"),
                'per_stroke_enabled': False,
                'monitor_per_stroke': False,
                'per_category': False
            }
            break
        elif choice == "6":
            # Per-category filtering
            quality_filters = {
                'enabled': True,
                'per_category': True,
                'per_stroke_enabled': False,
                'monitor_per_stroke': False
            }
            break
        else:
            print("   Invalid choice. Please enter 1-6.")
    
    # Ask about per-stroke filtering if quality filtering is enabled
    if quality_filters['enabled'] and not quality_filters.get('per_category', False):
        print("\nPer-Stroke Analysis Options:")
        
        # Option 1: Per-stroke filtering
        enable_per_stroke = input("   Enable per-stroke filtering? (y/n) [default: n]: ").strip().lower().startswith('y')
        if enable_per_stroke:
            quality_filters['per_stroke_enabled'] = True
            quality_filters['min_points_per_stroke'] = int(input("     Min points per stroke [default: 3]: ") or "3")
            quality_filters['max_points_per_stroke'] = int(input("     Max points per stroke [default: 150]: ") or "150")
        
        # Option 2: Console monitoring
        enable_monitoring = input("   Enable console monitoring of per-stroke stats? (y/n) [default: n]: ").strip().lower().startswith('y')
        if enable_monitoring:
            quality_filters['monitor_per_stroke'] = True
            print("     Will show per-stroke statistics for first 10 drawings of each category")
    
    # Configure per-category filtering if selected
    elif quality_filters.get('per_category', False):
        print("\nPer-Category Filtering Configuration:")
        print("   Using data-driven thresholds optimized for each fruit category")
        
        # Ask for strictness level for per-category
        print("\n   Select strictness level:")
        print("   1. Lenient   - Keep more drawings (based on category averages + margin)")
        print("   2. Balanced  - Medium filtering (based on category medians) [RECOMMENDED]")
        print("   3. Strict    - Keep only best drawings (based on category averages - margin)")
        
        while True:
            strictness = input("\n   Select strictness [default: 2]: ").strip() or "2"
            if strictness in ['1', '2', '3']:
                quality_filters['category_strictness'] = int(strictness)
                break
            else:
                print("   ❌ Invalid choice. Please enter 1, 2, or 3.")
        
        # Ask about per-stroke filtering for per-category mode
        enable_per_stroke = input("\n   Enable per-stroke filtering for categories? (y/n) [default: y]: ").strip().lower()
        quality_filters['per_stroke_enabled'] = not enable_per_stroke.startswith('n')
        
        # Ask about monitoring
        enable_monitoring = input("   Enable console monitoring? (y/n) [default: n]: ").strip().lower().startswith('y')
        quality_filters['monitor_per_stroke'] = enable_monitoring
        
        if enable_monitoring:
            print("     🔍 Will show category-specific statistics for first 10 drawings of each category")

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
        if quality_filters.get('per_category', False):
            strictness_names = {1: 'Lenient', 2: 'Balanced', 3: 'Strict'}
            strictness = strictness_names.get(quality_filters.get('category_strictness', 2), 'Balanced')
            print(f"   Quality Filter: Per-Category ({strictness})")
            print(f"   Category-specific thresholds based on data analysis")
            if quality_filters.get('per_stroke_enabled', False):
                print(f"     Per-stroke filtering: Enabled")
            if quality_filters.get('monitor_per_stroke', False):
                print(f"     Console monitoring: Enabled")
        else:
            print(f"   Quality Filter: Global")
            print(f"     Strokes: {quality_filters['min_strokes']}-{quality_filters['max_strokes']}")
            print(f"     Points: {quality_filters['min_points']}-{quality_filters['max_points']}")
            if quality_filters.get('per_stroke_enabled', False):
                print(f"     Per-stroke: {quality_filters['min_points_per_stroke']}-{quality_filters['max_points_per_stroke']} points")
            if quality_filters.get('monitor_per_stroke', False):
                print(f"     Console monitoring: Enabled")
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