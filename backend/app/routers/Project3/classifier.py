import os
import tensorflow as tf
import numpy as np
from PIL import Image
from io import BytesIO
import urllib.request

BASE_DIR = os.path.dirname(__file__)
MODEL_PATH = os.path.join(BASE_DIR, "vehicle_classifier_model.keras")

# HuggingFace model URL (uploaded model - used when local model is unavailable)
HUGGINGFACE_MODEL_URL = "https://huggingface.co/nolanrd04/CST-435_Project3_vehicle_classifier_model/resolve/main/vehicle_classifier_model.keras"

# Constants
IMG_SIZE = (128, 128)
CLASS_NAMES = ["car", "motorbike", "airplane"]


class VehicleClassifier:
    def __init__(self):
        self.model = None
        self.load_model()

    def load_model(self):
        """Load the trained model from disk or HuggingFace.

        Priority:
        1. Try to load local model from: vehicle_classifier_model.keras
        2. If local model not found or fails, download from HuggingFace
        3. If both fail, model remains None (will fail on classification)
        """
        # Try local model first
        if os.path.exists(MODEL_PATH):
            try:
                self.model = tf.keras.models.load_model(MODEL_PATH)
                print(f"✅ Model loaded from local path: {MODEL_PATH}")
                return
            except Exception as e:
                print(f"❌ Error loading local model: {e}")
                import traceback
                traceback.print_exc()
        else:
            print(f"⚠️ Local model not found at {MODEL_PATH}")

        # Fallback to HuggingFace model (used when deployed on Render)
        print(f"📥 Attempting to load model from HuggingFace: {HUGGINGFACE_MODEL_URL}")
        try:
            import tempfile
            import socket

            # Increase timeout for large model download
            socket.setdefaulttimeout(120)

            with tempfile.NamedTemporaryFile(suffix=".keras", delete=False) as tmp_file:
                tmp_path = tmp_file.name

            print(f"📥 Downloading from {HUGGINGFACE_MODEL_URL} (this may take a moment)...")
            urllib.request.urlretrieve(HUGGINGFACE_MODEL_URL, tmp_path)
            print(f"📥 Download complete. Loading model from {tmp_path}...")
            self.model = tf.keras.models.load_model(tmp_path)
            os.remove(tmp_path)
            print(f"✅ Model loaded from HuggingFace successfully")
        except Exception as e:
            print(f"❌ Error loading model from HuggingFace: {e}")
            print(f"⚠️ Model failed to load from both local and remote sources")
            import traceback
            traceback.print_exc()
            self.model = None

    def preprocess_image(self, image_bytes):
        """Convert image bytes to preprocessed array."""
        try:
            # Open image from bytes
            img = Image.open(BytesIO(image_bytes))

            # Convert to grayscale
            img = img.convert('L')

            # Resize to expected size
            img = img.resize(IMG_SIZE)

            # Convert to array and normalize
            img_array = np.array(img, dtype=np.float32)
            img_array = img_array / 255.0

            # Add batch and channel dimensions: (1, 128, 128, 1)
            img_array = np.expand_dims(img_array, axis=-1)
            img_array = np.expand_dims(img_array, axis=0)

            return img_array
        except Exception as e:
            raise ValueError(f"Error preprocessing image: {e}")

    def classify(self, image_bytes):
        """Classify the provided image."""
        if self.model is None:
            raise RuntimeError("Model not loaded. Cannot classify image.")

        try:
            # Preprocess image
            img_array = self.preprocess_image(image_bytes)

            # Make prediction
            prediction = self.model.predict(img_array, verbose=0)

            # Get predicted class and confidence
            predicted_class_idx = np.argmax(prediction[0])
            confidence = float(np.max(prediction[0]))
            predicted_class = CLASS_NAMES[predicted_class_idx]

            # Get all class probabilities
            class_probabilities = {
                CLASS_NAMES[i]: float(prediction[0][i])
                for i in range(len(CLASS_NAMES))
            }

            return {
                "predicted_class": predicted_class,
                "confidence": confidence,
                "class_probabilities": class_probabilities
            }
        except Exception as e:
            raise RuntimeError(f"Error during classification: {e}")


# Global classifier instance
_classifier = None


def get_classifier():
    """Get or initialize the global classifier instance."""
    global _classifier
    if _classifier is None:
        _classifier = VehicleClassifier()
    return _classifier
