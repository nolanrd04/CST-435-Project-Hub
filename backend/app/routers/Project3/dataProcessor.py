import os
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
import sys

# Define constants
IMG_SIZE = (128, 128)  # converts images to 128x128
CLASS_NAMES = ["car", "motorbike", "airplane"]  # must match folder names
BASE_DIR = os.path.dirname(__file__)  # get the base directory
DATA_DIRS = [os.path.join(BASE_DIR, cls) for cls in CLASS_NAMES]  # gets the directories of the folders and their images. These are saved locally.


def preprocess_image(path):
    """Load image with PIL, resize, convert to grayscale, normalize"""
    try:
        # Open image and convert to grayscale (L mode in PIL)
        img = Image.open(path).convert('L')
        
        # Resize to target size
        img = img.resize(IMG_SIZE, Image.Resampling.LANCZOS)
        
        # Convert to numpy array and expand dims to match TensorFlow format (H, W, 1)
        img_array = np.array(img, dtype=np.float32)
        img_array = np.expand_dims(img_array, axis=2)  # Add channel dimension: (128, 128, 1)
        
        # Normalize to [0, 1]
        img_array = img_array / 255.0
        
        return img_array
    except Exception as e:
        print(f"Error processing image {path}: {e}")
        raise


def one_hot_encode(labels, num_classes):
    """
    Convert integer labels to one-hot encoded labels.
    
    Args:
        labels: Array of integer labels
        num_classes: Number of classes
    
    Returns:
        One-hot encoded array of shape (num_samples, num_classes)
    """
    num_samples = len(labels)
    one_hot = np.zeros((num_samples, num_classes), dtype=np.float32)
    one_hot[np.arange(num_samples), labels] = 1
    return one_hot


def get_image_files(directory):
    """Get list of valid image files from a directory"""
    image_files = []
    if not os.path.exists(directory):
        return image_files
    
    for fname in os.listdir(directory):
        if fname.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".gif")):
            image_files.append(fname)
    
    return image_files


def estimate_dataset_size(images_per_class):
    """Estimate the size of a dataset in MB given images per class"""
    # Each image is 128x128x1 float32 = 65,536 bytes
    bytes_per_image = 128 * 128 * 1 * 4  # float32 = 4 bytes
    # Overhead for numpy compression ~10-15%
    bytes_per_image_with_overhead = bytes_per_image * 1.15
    
    total_images = images_per_class * len(CLASS_NAMES)
    total_bytes = total_images * bytes_per_image_with_overhead
    total_mb = total_bytes / (1024 * 1024)
    
    return total_mb


def load_dataset(max_file_size_mb=None):
    """
    Load and preprocess images from class directories with round-robin processing.
    
    Processes images in order: car, motorbike, airplane, car, motorbike, airplane, ...
    continuing until either all images are processed or max file size is reached.
    
    Args:
        max_file_size_mb: Maximum desired output file size in MB. If None, processes all images.
    
    Returns:
        X: Array of preprocessed images (N, 128, 128, 1)
        y: One-hot encoded labels (N, 3)
    """
    images, labels = [], []
    
    # Get image files for each class
    class_image_files = []
    total_images_available = 0
    
    for label, folder in enumerate(DATA_DIRS):
        image_files = get_image_files(folder)
        class_image_files.append((label, folder, image_files))
        total_images_available += len(image_files)
        print(f"📁 Class '{CLASS_NAMES[label]}': {len(image_files)} images available")
    
    print(f"\n📊 Total images available: {total_images_available}")
    
    if total_images_available == 0:
        raise ValueError(f"No images found in directories: {DATA_DIRS}")
    
    # Calculate target number of images based on file size if specified
    target_images = None
    if max_file_size_mb is not None:
        # Work backwards from compressed file size to estimate number of images
        bytes_per_image = 128 * 128 * 1 * 4  # float32 = 4 bytes
        compression_ratio = 0.30  # NumPy compression achieves ~70% compression on grayscale images
        
        # Convert compressed MB target to uncompressed bytes
        max_bytes_uncompressed = (max_file_size_mb * 1024 * 1024) / compression_ratio
        target_images = int(max_bytes_uncompressed / bytes_per_image)
        target_images = max(1, min(target_images, total_images_available))
        print(f"📏 Target output file size (compressed): {max_file_size_mb} MB")
        print(f"📊 Target images: ~{target_images} images")
    else:
        target_images = total_images_available
        print(f"📏 Processing all available images ({total_images_available})")
    
    # Round-robin processing through classes
    print(f"\n🔄 Round-robin processing:")
    processed_count = 0
    class_indices = [0] * len(class_image_files)  # Track position in each class
    
    while processed_count < target_images:
        any_processed = False
        
        # Go through each class in order
        for class_idx, (label, folder, image_files) in enumerate(class_image_files):
            if processed_count >= target_images:
                break
            
            # Get current image index for this class
            file_idx = class_indices[class_idx]
            
            # Skip if we've processed all images in this class
            if file_idx >= len(image_files):
                continue
            
            fname = image_files[file_idx]
            fpath = os.path.join(folder, fname)
            
            try:
                img_array = preprocess_image(fpath)
                images.append(img_array)
                labels.append(label)
                processed_count += 1
                class_indices[class_idx] += 1
                any_processed = True
                
                # Progress update every 50 images
                if processed_count % 50 == 0:
                    print(f"  ✓ Processed {processed_count}/{target_images} images...")
                
            except Exception as e:
                print(f"  ✗ Skipping {fname}: {e}")
                class_indices[class_idx] += 1
                continue
        
        # If no images were processed in this round, we've exhausted all classes
        if not any_processed:
            print(f"  ℹ️  Exhausted all classes after {processed_count} images")
            break
    
    print(f"  ✓ Finished processing: {processed_count} images")
    
    if not images:
        raise ValueError("No images were successfully processed")
    
    X = np.array(images, dtype=np.float32)
    y_labels = np.array(labels, dtype=np.int64)
    y = one_hot_encode(y_labels, num_classes=len(CLASS_NAMES))
    
    print(f"\n✓ Dataset loaded:")
    print(f"  - Total images: {len(X)}")
    print(f"  - Image shape: {X.shape}")
    print(f"  - Label shape: {y.shape}")
    
    # Show class distribution
    unique, counts = np.unique(y_labels, return_counts=True)
    print(f"\n📊 Class distribution:")
    for class_idx, count in zip(unique, counts):
        percentage = count / len(X) * 100
        print(f"  - {CLASS_NAMES[class_idx]}: {count} images ({percentage:.1f}%)")
    
    return X, y


def save_dataset(X, y, out_path="dataset.npz"):
    """
    Save preprocessed dataset to compressed NumPy format.
    
    Args:
        X: Images array
        y: One-hot encoded labels array
        out_path: Output file path
    """
    print(f"\n💾 Saving dataset to: {out_path}")
    np.savez_compressed(out_path, X=X, y=y)
    
    # Print file size
    file_size_mb = os.path.getsize(out_path) / (1024 * 1024)
    print(f"✓ Dataset saved successfully ({file_size_mb:.1f} MB)")


if __name__ == "__main__":
    print("="*70)
    print("VEHICLE DATASET PROCESSOR (PyTorch)")
    print("="*70)
    print("\nLoading and converting dataset...")
    print(f"Base directory: {BASE_DIR}")
    print(f"Image size: {IMG_SIZE}")
    print(f"Classes: {CLASS_NAMES}")
    
    try:
        # Prompt user for file size limit
        print("\n" + "="*70)
        print("OUTPUT FILE SIZE CONFIGURATION")
        print("="*70)
        print("\nThe dataset will be processed in round-robin fashion:")
        print("  car → motorbike → airplane → car → motorbike → airplane → ...")
        print("\nEnter desired output file size, or press Enter to process all images.")
        print("Examples: 50 (50 MB), 100 (100 MB), 200 (200 MB)")
        
        file_size_input = input("\nDesired output file size in MB (or press Enter for all): ").strip()
        
        max_file_size = None
        if file_size_input:
            try:
                max_file_size = float(file_size_input)
                if max_file_size <= 0:
                    print("⚠️  Invalid file size. Processing all images instead.")
                    max_file_size = None
            except ValueError:
                print("⚠️  Invalid input. Processing all images instead.")
                max_file_size = None
        
        print("\n" + "="*70)
        X, y = load_dataset(max_file_size_mb=max_file_size)
        print("\n✓ Dataset loaded successfully")
        
        # Estimate actual file size
        bytes_uncompressed = X.nbytes + y.nbytes
        mb_uncompressed = bytes_uncompressed / (1024 * 1024)
        
        # NumPy's npz compression achieves ~70% compression on image data
        compression_ratio = 0.30  # Typical for grayscale images
        mb_compressed_estimate = mb_uncompressed * compression_ratio
        
        print(f"\n📊 Size estimates:")
        print(f"  - Uncompressed: {mb_uncompressed:.1f} MB")
        print(f"  - Compressed (estimated): {mb_compressed_estimate:.1f} MB")
        
        # Prompt for filename
        print("\nEnter a filename to save the dataset to. Default is 'vehicles_dataset.npz'.")
        filename = input("Filename (or press Enter to use default): ").strip()
        if not filename:
            filename = "vehicles_dataset.npz"
        
        print(f"\nSaving dataset to '{filename}', please wait...")
        save_dataset(X, y, out_path=os.path.join(BASE_DIR, filename))
        print("\n🎉 Dataset processing completed!")
        
    except Exception as e:
        print(f"\n❌ Error processing dataset: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
