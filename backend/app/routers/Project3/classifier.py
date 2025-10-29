import os
import tensorflow as tf
import numpy as np
from PIL import Image
from io import BytesIO

BASE_DIR = os.path.dirname(__file__)
MODEL_PATH = os.path.join(BASE_DIR, "vehicle_classifier_model.keras")

# Constants
IMG_SIZE = (128, 128)
CLASS_NAMES = ["car", "motorbike", "airplane"]


class VehicleClassifier:
    def __init__(self):
        self.model = None
        self.load_model()

    def load_model(self):
        """Load the trained model from disk."""
        if os.path.exists(MODEL_PATH):
            try:
                self.model = tf.keras.models.load_model(MODEL_PATH)
                print(f"✅ Model loaded from {MODEL_PATH}")
            except Exception as e:
                print(f"❌ Error loading model: {e}")
                self.model = None
        else:
            print(f"⚠️ Model not found at {MODEL_PATH}")
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
