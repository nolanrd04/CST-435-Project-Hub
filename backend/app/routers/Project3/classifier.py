import os
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from io import BytesIO
import urllib.request
import tempfile

BASE_DIR = os.path.dirname(__file__)
MODEL_DIR = os.path.join(BASE_DIR, "model")
MODEL_PATH = os.path.join(MODEL_DIR, "vehicle_classifier_model.pth")
MODEL_PATH_BEST = os.path.join(MODEL_DIR, "vehicle_classifier_model_best.pth")

# HuggingFace model URL (uploaded model - used when local model is unavailable)
HUGGINGFACE_MODEL_URL = "https://huggingface.co/nolanrd04/CST-435_Project3_vehicle_classifier_model/resolve/main/vehicle_classifier_model.pth"

# Constants
IMG_SIZE = (128, 128)
CLASS_NAMES = ["car", "motorbike", "airplane"]

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class VehicleCNN(nn.Module):
    """
    CNN model architecture for vehicle classification
    """
    
    def __init__(self, input_shape=(128, 128, 1), num_classes=3, 
                 conv_filters=None, dense_units=128, dropout=0.0):
        super(VehicleCNN, self).__init__()
        
        if conv_filters is None:
            conv_filters = [32, 64, 128]
        
        height, width, channels = input_shape
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(channels, conv_filters[0], kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(conv_filters[0], conv_filters[1], kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv3 = nn.Conv2d(conv_filters[1], conv_filters[2], kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Calculate flattened size
        flat_height = height // 4
        flat_width = width // 4
        flatten_size = flat_height * flat_width * conv_filters[2]
        
        # Dense layers
        self.dense1 = nn.Linear(flatten_size, dense_units)
        self.dropout_layer = nn.Dropout(dropout)
        self.output = nn.Linear(dense_units, num_classes)
    
    def forward(self, x):
        # Conv block 1
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = self.pool1(x)
        
        # Conv block 2
        x = torch.relu(self.conv3(x))
        x = self.pool2(x)
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # Dense layers
        x = torch.relu(self.dense1(x))
        x = self.dropout_layer(x)
        x = self.output(x)
        
        return x


class VehicleClassifier:
    def __init__(self):
        self.model = None
        self.model_loaded = False
        self.model_config = None
        self.current_model_source = None  # Track which model is loaded: 'local' or 'huggingface'
        # Don't load model here - load it lazily on first use

    def load_model(self, model_source: str = 'local'):
        """
        Load the trained model from disk or HuggingFace.
        
        Args:
            model_source: Either 'local' or 'huggingface'

        Priority for 'local':
        1. Try to load local best model: vehicle_classifier_model_best.pth
        2. Try to load local model: vehicle_classifier_model.pth
        3. If local models not found, fallback to HuggingFace
        
        Priority for 'huggingface':
        1. Download from HuggingFace directly
        """
        if model_source == 'local':
            self._load_local_model()
        elif model_source == 'huggingface':
            self._load_huggingface_model()
        else:
            raise ValueError(f"Unknown model_source: {model_source}. Use 'local' or 'huggingface'")
    
    def _load_local_model(self):
        """Load model from local filesystem."""
        # Try local best model first
        if os.path.exists(MODEL_PATH_BEST):
            try:
                print(f"📂 Loading best local model from: {MODEL_PATH_BEST}")
                self._load_checkpoint(MODEL_PATH_BEST)
                self.current_model_source = 'local'
                print(f"✅ Best model loaded from local path")
                return
            except Exception as e:
                print(f"❌ Error loading best local model: {e}")

        # Try regular local model
        if os.path.exists(MODEL_PATH):
            try:
                print(f"📂 Loading local model from: {MODEL_PATH}")
                self._load_checkpoint(MODEL_PATH)
                self.current_model_source = 'local'
                print(f"✅ Model loaded from local path")
                return
            except Exception as e:
                print(f"❌ Error loading local model: {e}")
        else:
            print(f"⚠️ Local model not found at {MODEL_PATH}")
        
        # If local loading fails, fallback to HuggingFace
        print("📥 Local model unavailable, falling back to HuggingFace...")
        self._load_huggingface_model()
    
    def _load_huggingface_model(self):
        """Load model from HuggingFace."""
        print(f"📥 Attempting to load model from HuggingFace...")
        print(f"⏱️  Note: First load may take 1-2 minutes due to model size")
        try:
            import socket
            socket.setdefaulttimeout(300)  # 5 minute timeout

            with tempfile.NamedTemporaryFile(suffix=".pth", delete=False) as tmp_file:
                tmp_path = tmp_file.name
                print(f"📥 Temporary file created at: {tmp_path}")

            print(f"📥 Downloading from HuggingFace (this may take 1-2 minutes)...")
            urllib.request.urlretrieve(HUGGINGFACE_MODEL_URL, tmp_path)
            print(f"✅ Download complete ({os.path.getsize(tmp_path) / 1024 / 1024:.1f}MB)")

            print(f"📥 Loading model from temporary file...")
            self._load_checkpoint(tmp_path)
            self.current_model_source = 'huggingface'

            print(f"🧹 Cleaning up temporary file...")
            os.remove(tmp_path)
            print(f"✅ Model loaded from HuggingFace successfully")
        except Exception as e:
            print(f"❌ Error loading model from HuggingFace: {e}")
            print(f"⚠️ Model failed to load from both local and remote sources")
            import traceback
            traceback.print_exc()
            self.model = None
            self.current_model_source = None
    
    def switch_model(self, model_source: str):
        """
        Switch between local and HuggingFace models.
        
        Args:
            model_source: Either 'local' or 'huggingface'
        """
        if self.current_model_source == model_source:
            print(f"✓ Model is already using {model_source} source")
            return
        
        print(f"\n🔄 Switching to {model_source} model...")
        self.model = None
        self.model_loaded = False
        self.model_config = None
        self.load_model(model_source)

    def _load_checkpoint(self, checkpoint_path):
        """
        Load a PyTorch checkpoint file.
        
        Args:
            checkpoint_path: Path to the .pth file
        """
        try:
            checkpoint = torch.load(checkpoint_path, map_location=device)
            
            # Extract model configuration
            if 'input_shape' in checkpoint:
                input_shape = checkpoint['input_shape']
            else:
                input_shape = (128, 128, 1)
            
            if 'num_classes' in checkpoint:
                num_classes = checkpoint['num_classes']
            else:
                num_classes = 3
            
            conv_filters = checkpoint.get('conv_filters', [32, 64, 128])
            dense_units = checkpoint.get('dense_units', 128)
            dropout = checkpoint.get('dropout', 0.0)
            
            # Create model with saved configuration
            self.model = VehicleCNN(
                input_shape=input_shape,
                num_classes=num_classes,
                conv_filters=conv_filters,
                dense_units=dense_units,
                dropout=dropout
            ).to(device)
            
            # Load state dict
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.eval()  # Set to evaluation mode
            
            # Store config for reference
            self.model_config = {
                'input_shape': input_shape,
                'num_classes': num_classes,
                'conv_filters': conv_filters,
                'dense_units': dense_units,
                'dropout': dropout
            }
            
            print(f"  Model config: {self.model_config}")
            
        except Exception as e:
            raise RuntimeError(f"Failed to load checkpoint from {checkpoint_path}: {e}")

    def preprocess_image(self, image_bytes):
        """
        Convert image bytes to preprocessed tensor.
        
        Args:
            image_bytes: Image data as bytes
        
        Returns:
            Preprocessed image tensor of shape (1, 1, 128, 128)
        """
        try:
            # Open image from bytes
            img = Image.open(BytesIO(image_bytes))

            # Convert to grayscale
            img = img.convert('L')

            # Resize to expected size
            img = img.resize(IMG_SIZE, Image.Resampling.LANCZOS)

            # Convert to array and normalize
            img_array = np.array(img, dtype=np.float32)
            img_array = img_array / 255.0

            # Add channel dimension to match training format: (128, 128) -> (128, 128, 1)
            img_array = np.expand_dims(img_array, axis=2)

            # Transpose to PyTorch format (C, H, W): (128, 128, 1) -> (1, 128, 128)
            img_array = np.transpose(img_array, (2, 0, 1))

            # Add batch dimension: (1, 128, 128) -> (1, 1, 128, 128)
            img_array = np.expand_dims(img_array, axis=0)

            # Convert to tensor
            img_tensor = torch.from_numpy(img_array).to(device)

            return img_tensor
        except Exception as e:
            raise ValueError(f"Error preprocessing image: {e}")

    def classify(self, image_bytes, model_source: str = 'local'):
        """
        Classify the provided image.
        
        Args:
            image_bytes: Image data as bytes
            model_source: Either 'local' or 'huggingface'
        
        Returns:
            Dictionary with predicted class, confidence, and all class probabilities
        """
        # Load model on first use or switch if requested (lazy loading)
        if not self.model_loaded or self.current_model_source != model_source:
            print(f"📂 Loading {model_source} model...")
            self.load_model(model_source)
            self.model_loaded = True

        if self.model is None:
            raise RuntimeError("Model not loaded. Cannot classify image.")

        try:
            # Preprocess image
            img_tensor = self.preprocess_image(image_bytes)

            # Make prediction
            with torch.no_grad():
                output = self.model(img_tensor)
                # Apply softmax to get probabilities
                probabilities = torch.softmax(output, dim=1)

            # Get predicted class and confidence
            predicted_class_idx = torch.argmax(probabilities[0]).item()
            confidence = float(torch.max(probabilities[0]).item())
            predicted_class = CLASS_NAMES[predicted_class_idx]

            # Get all class probabilities
            class_probabilities = {
                CLASS_NAMES[i]: float(probabilities[0][i].item())
                for i in range(len(CLASS_NAMES))
            }

            return {
                "predicted_class": predicted_class,
                "confidence": confidence,
                "class_probabilities": class_probabilities,
                "model_source": self.current_model_source
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
