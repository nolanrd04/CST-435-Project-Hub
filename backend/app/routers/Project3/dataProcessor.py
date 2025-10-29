import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split

# Define constants
IMG_SIZE = (128, 128)  # converts images to 128x128
CLASS_NAMES = ["car", "motorbike", "airplane"]  # must match folder names
BASE_DIR = os.path.dirname(__file__) # get the base directory
DATA_DIRS = [os.path.join(BASE_DIR, cls) for cls in CLASS_NAMES] # gets the directories of the folders and their images. These are saved locally.

imageCount = 0

def preprocess_image(path):
    """Load image, resize, normalize"""
    img = tf.keras.utils.load_img(path, target_size=IMG_SIZE, color_mode = 'grayscale') # opens image as a grayscale so color is not a factor
    img_array = tf.keras.utils.img_to_array(img)
    img_array = img_array / 255.0  # normalize to [0,1]
    return img_array

def load_dataset():
    images, labels = [], []
    
    for label, folder in enumerate(DATA_DIRS):
        for fname in os.listdir(folder):
            fpath = os.path.join(folder, fname)
            if not fname.lower().endswith((".png", ".jpg", ".jpeg")):
                continue
            try:
                img_array = preprocess_image(fpath)
                images.append(img_array)
                labels.append(label)  # 0=car, 1=motorbike, 2=airplane
                print(f"Converting {fname} with label: {label}...")
            except Exception as e:
                print(f"Skipping {fpath}: {e}")
    
    X = np.array(images, dtype=np.float32)
    y = to_categorical(labels, num_classes=len(CLASS_NAMES))  # one-hot encode
    
    return X, y

def save_dataset(X, y, out_path="dataset.npz"):
    np.savez_compressed(out_path, X=X, y=y)
    print(f"Dataset saved to {out_path}")

if __name__ == "__main__":
    print("Loading and converting dataset:")
    X, y = load_dataset()
    print("Dataset shape:", X.shape, y.shape)
    print("Saving dataset, please wait:")
    save_dataset(X, y, out_path=os.path.join(BASE_DIR, "vehicles_dataset.npz"))
