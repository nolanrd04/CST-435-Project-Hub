# Multi-Object Image Colorizer - Technical Specification

## Project Overview

**Objective:** Develop a deep learning system that colorizes grayscale images of five specific fruit objects using their learned shape and texture patterns.

**Supported Objects:**
1. **Strawberry** - Red fruit + green leaves (2-color object)
2. **Orange** - Orange peel (single-color object)
3. **Blackberry** - Dark purple/black compound berry structure (1-2 colors)
4. **Banana** - Yellow elongated fruit (single-color object)
5. **Pineapple** - Yellow/golden body + green/brown crown (multi-color, complex texture)

**Why These Objects:**
- Diverse shape characteristics: round, elongated, compound, textured
- Color complexity spectrum: single-color to multi-color objects
- Distinct visual features enabling shape-to-color learning
- Readily available training data
- Demonstrates model's ability to generalize across object classes

---

## System Architecture

### Core Design Decisions

#### Single Unified Model Architecture

The system employs **one U-Net model** trained on all five object types simultaneously.

**Rationale:**
- Leverages deep learning's strength in learning shared visual features across classes
- Single deployment artifact (~100MB model file)
- Demonstrates true generalization capability
- Simplified inference pipeline and API design
- Scalable for future object additions

**How It Works:**
The model learns implicit shape-to-color mappings through training on mixed object batches:
- Round + textured surface → orange tones
- Elongated + smooth curvature → yellow
- Conical shape + leafy top → red body + green leaves
- Compound bumpy texture → deep purple/black
- Diamond scale pattern + crown structure → golden yellow + green top

**Alternative Rejected:** Five separate specialist models
- Would require object classification preprocessing
- 5x storage and memory footprint
- Doesn't leverage DL's generalization strength
- Complex deployment and maintenance

#### Dataset Composition Strategy

Training exclusively on **isolated target objects with simple backgrounds**.

**Rationale:**
- Well-defined problem scope suitable for academic project
- Clear evaluation metrics and success criteria
- Simplified dataset curation and quality control
- Predictable model behavior and outputs

**Training Data Characteristics:**
- Objects isolated on white/gray/neutral backgrounds
- All five classes mixed in training batches
- No extraneous objects, complex scenes, or backgrounds
- Focus on learning specific shape-to-color relationships

**Handling Out-of-Distribution Objects:**
The model will attempt to colorize any input, potentially producing incorrect colors for unknown objects. This is acceptable within project scope and documented as a known limitation. Future enhancement: integrate object detection for filtering non-target items.

---

## Model Architecture

### U-Net Design

**Architecture:** Encoder-Decoder with Skip Connections

**Why U-Net:**
- State-of-the-art for image-to-image translation tasks
- Preserves spatial information through skip connections (critical for precise colorization)
- Proven success in segmentation and colorization domains
- Balances model capacity with training efficiency

### Network Structure

```
Input: Grayscale image (256×256×1)
Output: RGB image (256×256×3)

┌─────────────────────────────────────────────┐
│              ENCODER (Downsampling)          │
├─────────────────────────────────────────────┤
│ Conv2D(64) + BatchNorm + ReLU               │
│ Conv2D(64) + BatchNorm + ReLU → MaxPool ───┐│
│                                           Skip1
│ Conv2D(128) + BatchNorm + ReLU              │
│ Conv2D(128) + BatchNorm + ReLU → MaxPool ──┤│
│                                           Skip2
│ Conv2D(256) + BatchNorm + ReLU              │
│ Conv2D(256) + BatchNorm + ReLU → MaxPool ──┤│
│                                           Skip3
│ Conv2D(512) + BatchNorm + ReLU              │
│ Conv2D(512) + BatchNorm + ReLU → MaxPool ──┤│
│                                           Skip4
├─────────────────────────────────────────────┤
│               BOTTLENECK                     │
├─────────────────────────────────────────────┤
│ Conv2D(1024) + BatchNorm + ReLU             │
│ Conv2D(1024) + BatchNorm + ReLU             │
├─────────────────────────────────────────────┤
│             DECODER (Upsampling)             │
├─────────────────────────────────────────────┤
│ UpConv(512) + Concat[Skip4]                 │
│ Conv2D(512) + BatchNorm + ReLU              │
│ Conv2D(512) + BatchNorm + ReLU              │
│                                              │
│ UpConv(256) + Concat[Skip3]                 │
│ Conv2D(256) + BatchNorm + ReLU              │
│ Conv2D(256) + BatchNorm + ReLU              │
│                                              │
│ UpConv(128) + Concat[Skip2]                 │
│ Conv2D(128) + BatchNorm + ReLU              │
│ Conv2D(128) + BatchNorm + ReLU              │
│                                              │
│ UpConv(64) + Concat[Skip1]                  │
│ Conv2D(64) + BatchNorm + ReLU               │
│ Conv2D(64) + BatchNorm + ReLU               │
│                                              │
│ Conv2D(3, activation='sigmoid')              │
└─────────────────────────────────────────────┘

Total Parameters: ~30-35 million
```

### Training Configuration

**Hyperparameters:**
```python
IMAGE_SIZE = 256                # Input/output resolution
CHANNELS_GRAYSCALE = 1          # Single-channel input
CHANNELS_RGB = 3                # Three-channel output
BATCH_SIZE = 16                 # Training batch size
LEARNING_RATE = 0.0001          # Initial learning rate
EPOCHS = 50                     # Maximum training epochs
```

**Loss Function:** Combined MSE + Perceptual Loss
- MSE: Pixel-level reconstruction accuracy
- Perceptual: High-level feature similarity using pretrained VGG

**Optimizer:** Adam with adaptive learning rate scheduling
- ReduceLROnPlateau: Reduce LR when validation loss plateaus
- Factor: 0.5
- Patience: 5 epochs

**Training Callbacks:**
- ModelCheckpoint: Save best model based on validation loss
- EarlyStopping: Halt training if no improvement (patience=10)
- TensorBoard: Real-time training monitoring
- Custom visualization: Sample predictions each epoch

**Evaluation Metrics:**
- **MSE/MAE:** Training loss metrics
- **PSNR (Peak Signal-to-Noise Ratio):** Image quality measure (target: >25 dB)
- **SSIM (Structural Similarity Index):** Perceptual similarity (target: >0.85)
- **Per-object performance:** Breakdown by fruit type

---

## Dataset Specification

### Collection Requirements

**Target Size:** 200-500 images per object (1000-2500 total)

**Sources:**
- Kaggle fruit datasets (e.g., Fruits-360)
- ImageNet fruit subsets
- Google Images (filtered for usage rights)
- Manual photography in controlled settings

**Image Specifications:**
- Format: PNG or JPG
- Minimum resolution: 256×256 pixels
- Isolated objects on simple backgrounds (white/gray preferred)
- Variety in:
  - Scale and rotation
  - Lighting conditions
  - Camera distance/perspective

### Data Organization

```
backend/app/routers/Project7/
├── data/
│   ├── raw/                       # Original color images
│   │   ├── strawberry/
│   │   ├── orange/
│   │   ├── blackberry/
│   │   ├── banana/
│   │   └── pineapple/
│   │
│   ├── processed/                 # Preprocessed training data
│   │   ├── train/
│   │   │   ├── color/            # RGB ground truth
│   │   │   └── grayscale/        # Model input
│   │   ├── validation/
│   │   │   ├── color/
│   │   │   └── grayscale/
│   │   └── test/
│   │       ├── color/
│   │       └── grayscale/
│   │
│   └── metadata.json              # Dataset statistics
```

### Preprocessing Pipeline

**Operations:**
1. **Validation:** Check image integrity, filter corrupted files
2. **Resizing:** Standardize to 256×256 resolution
3. **Normalization:** Scale pixel values to [0, 1] range
4. **Grayscale Conversion:** Create input images from color originals
5. **Augmentation:** (applied during training only)
   - Horizontal flip (50% probability)
   - Rotation (±15 degrees)
   - Brightness adjustment (±20%)
   - Zoom variation (90-110%)
   - **No color augmentation** (would corrupt ground truth)

**Data Split:**
- Training: 70%
- Validation: 15%
- Testing: 15%
- Stratified sampling ensures equal class representation

---

## Implementation Components

### Python Modules

#### `colorizer_model.py`
Defines the U-Net architecture and model management.

```python
class ColorizerModel:
    """U-Net model for image colorization"""

    def __init__(self, input_shape=(256, 256, 1)):
        """Initialize model with input dimensions"""

    def build_unet(self):
        """Construct encoder-decoder architecture with skip connections"""

    def compile_model(self, loss='mse', optimizer='adam'):
        """Configure loss function and optimizer"""

    def get_summary(self):
        """Return model architecture summary"""
```

#### `data_generator.py`
Efficient data loading with on-the-fly augmentation.

```python
class ColorizerDataGenerator(tf.keras.utils.Sequence):
    """Memory-efficient batch generator for training"""

    def __init__(self, image_paths, batch_size=16, augment=True):
        """Initialize with dataset paths and configuration"""

    def __len__(self):
        """Return number of batches per epoch"""

    def __getitem__(self, index):
        """Generate batch: (grayscale_images, color_images)"""

    def on_epoch_end(self):
        """Shuffle dataset after each epoch"""
```

#### `train_colorizer.py`
Training orchestration and model persistence.

```python
def train_model():
    """Main training loop"""
    # 1. Initialize data generators
    # 2. Build and compile model
    # 3. Setup callbacks (checkpoint, early stopping, etc.)
    # 4. Train with validation monitoring
    # 5. Save final model and training history
```

#### `colorizer_inference.py`
Production inference pipeline.

```python
class ImageColorizer:
    """Inference wrapper for trained model"""

    def __init__(self, model_path):
        """Load trained model from disk"""

    def colorize_image(self, grayscale_image):
        """
        Colorize single grayscale image

        Input: numpy array (H×W×1) or (H×W)
        Output: numpy array (H×W×3) RGB
        """

    def colorize_from_path(self, image_path):
        """Load and colorize image from file"""

    def batch_colorize(self, image_paths):
        """Efficiently colorize multiple images"""
```

#### `evaluate_model.py`
Model evaluation and analysis.

**Quantitative Metrics:**
- MSE on test set
- PSNR (Peak Signal-to-Noise Ratio)
- SSIM (Structural Similarity Index)
- Per-object breakdown

**Qualitative Analysis:**
- Visual comparison grids (grayscale → prediction → ground truth)
- Success/failure case studies
- Error heatmaps

**Test Scenarios:**
- Held-out test set
- Internet images (generalization test)
- Edge cases: unusual lighting, occlusion, multiple objects
- Originally B&W images (not color-to-gray-to-color)

---

## Backend API

### FastAPI Integration

#### Endpoints

**File:** `backend/app/routers/Project7/colorizer_router.py`

```python
@router.post("/colorize-image")
async def colorize_image(file: UploadFile) -> ColorizeResponse:
    """
    Upload grayscale image, receive colorized version

    Input: Multipart file upload (PNG/JPG)
    Output: Base64-encoded RGB image + metadata
    """

@router.post("/colorize-batch")
async def colorize_batch(files: List[UploadFile]) -> List[ColorizeResponse]:
    """Batch colorization for multiple images"""

@router.get("/model-info")
async def get_model_info() -> ModelInfo:
    """
    Retrieve model metadata:
    - Supported objects
    - Architecture summary
    - Training performance metrics
    """

@router.get("/visualizations")
async def get_visualizations() -> dict:
    """
    Return training artifacts:
    - Model architecture diagram
    - Training/validation curves
    - Sample predictions
    """
```

#### Data Models

**File:** `backend/app/routers/Project7/models.py`

```python
from pydantic import BaseModel
from typing import List, Optional

class ColorizeRequest(BaseModel):
    image_data: str              # Base64-encoded grayscale image
    return_metrics: bool = False # Include PSNR/SSIM in response

class ColorizeResponse(BaseModel):
    colorized_image: str         # Base64-encoded RGB result
    processing_time_ms: float    # Inference latency
    metrics: Optional[dict]      # PSNR, SSIM if requested

class ModelInfo(BaseModel):
    supported_objects: List[str] # ["strawberry", "orange", ...]
    image_size: int              # 256
    total_parameters: int        # ~30-35 million
    training_metrics: dict       # Final PSNR, SSIM, loss
```

#### Main App Integration

```python
# backend/main.py
from app.routers.Project7.colorizer_router import router as colorizer_router

app.include_router(
    colorizer_router,
    prefix="/api/colorizer",
    tags=["Image Colorization"]
)
```

---

## Frontend Application

### React Component Architecture

**Directory:** `frontend/src/projects/projectColorizer/`

```
projectColorizer/
├── ImageColorizer.tsx         # Main component (orchestrator)
├── ImageUploader.tsx          # Drag-and-drop file upload
├── ColorizedResults.tsx       # Side-by-side comparison view
├── ModelVisualizer.tsx        # Training info and metrics
├── api.ts                     # Backend API client
└── types.ts                   # TypeScript interfaces
```

### Main Component Features

- Drag-and-drop image upload
- Real-time colorization preview
- Side-by-side comparison (original vs colorized)
- Download colorized results
- Batch processing support
- Loading states and error handling
- Processing time display

### UI Layout

```
┌─────────────────────────────────────────────────────┐
│         Multi-Object Image Colorizer                 │
├─────────────────────────────────────────────────────┤
│  [Upload Image]  [Upload Batch]  [Model Info]       │
├────────────────────────┬────────────────────────────┤
│                        │                            │
│   Grayscale Input      │     Colorized Output       │
│   (Original/Upload)    │     (Model Prediction)     │
│                        │                            │
│   [Download]           │     [Download]             │
│                        │                            │
├────────────────────────┴────────────────────────────┤
│  Performance Metrics:                               │
│  • Processing Time: 234ms                           │
│  • PSNR: 28.4 dB                                    │
│  • SSIM: 0.89                                       │
└─────────────────────────────────────────────────────┘
```

### API Client

**File:** `frontend/src/projects/projectColorizer/api.ts`

```typescript
export const colorizerAPI = {
  colorizeImage: async (file: File): Promise<ColorizeResponse> => {
    const formData = new FormData();
    formData.append('file', file);

    const response = await axios.post(
      `${API_URL}/api/colorizer/colorize-image`,
      formData
    );
    return response.data;
  },

  getModelInfo: async (): Promise<ModelInfo> => {
    const response = await axios.get(
      `${API_URL}/api/colorizer/model-info`
    );
    return response.data;
  }
};
```

### App Router Integration

```typescript
// frontend/src/App.tsx
import ImageColorizer from './projects/projectColorizer/ImageColorizer';

<Route
  path="/project/colorizer"
  element={<ImageColorizer />}
/>
```

---

## Model Artifacts

### Generated Files

```
saved_models/
├── colorizer_best.h5          # Best validation loss checkpoint
├── colorizer_final.h5         # Final epoch model
└── model_config.json          # Architecture + hyperparameters

visualizations/
├── model_architecture.png     # U-Net diagram
├── training_history.png       # Loss/metric curves
├── predictions_samples.png    # Example colorizations
└── per_object_metrics.png     # Performance by fruit type

logs/
└── tensorboard/               # TensorBoard training logs
```

---

## Technology Stack

### Backend

```
tensorflow>=2.13.0             # Deep learning framework
keras>=2.13.0                  # High-level neural network API
numpy>=1.24.0                  # Numerical computing
opencv-python>=4.8.0           # Image processing
pillow>=10.0.0                 # Image I/O
matplotlib>=3.7.0              # Visualization
scikit-image>=0.21.0           # Image metrics (PSNR, SSIM)
fastapi>=0.100.0               # Web framework
uvicorn>=0.23.0                # ASGI server
python-multipart>=0.0.6        # File upload support
```

### Frontend

```
react>=18.0.0                  # UI framework
typescript>=5.0.0              # Type-safe JavaScript
axios>=1.4.0                   # HTTP client
```

---

## Success Criteria

### Minimum Viable Product (MVP)

- ✅ Model colorizes all 5 objects with recognizable colors
- ✅ PSNR > 20 dB on test set (good reconstruction quality)
- ✅ SSIM > 0.80 (strong structural similarity)
- ✅ Inference time < 1 second per image
- ✅ Functional web interface (upload → colorize → display)
- ✅ Model generalizes to unseen images of target objects

### Stretch Goals

- Handle multiple objects in a single image
- Integrate object detection preprocessing (filter non-target objects)
- Preserve original background colors
- Real-time colorization (webcam feed)
- Transfer learning: add new objects with minimal training data
- Model optimization (quantization, pruning) for faster inference

---

## Known Limitations

1. **Out-of-Distribution Objects:** Model will attempt to colorize any input, potentially producing incorrect colors for objects outside the training set (not strawberries, oranges, blackberries, bananas, or pineapples).

2. **Background Handling:** Trained only on simple backgrounds; complex scenes may produce artifacts.

3. **Occlusion:** Partially visible or overlapping objects may not colorize correctly.

4. **Lighting Extremes:** Very dark or overexposed images may challenge the model.

5. **Object Variants:** Uncommon varieties (e.g., green strawberries, red bananas) may not colorize as expected.

---

## Testing Strategy

### Unit Tests
- Data preprocessing functions
- Model input/output shape validation
- API endpoint responses
- Image encoding/decoding

### Integration Tests
- End-to-end: upload → inference → response
- Batch processing pipeline
- Error handling (invalid files, timeouts, corrupted images)

### Visual Validation
- Manual inspection of colorized outputs
- Comparison against ground truth
- Edge case evaluation:
  - Unusual angles/rotations
  - Lighting variations (bright, dark, shadowed)
  - Partial visibility (cropped objects)
  - Multiple objects in frame
  - Unknown objects (graceful failure)

### Performance Benchmarks

**Targets:**
- Inference: <500ms on CPU, <100ms on GPU
- PSNR: >25 dB (good quality)
- SSIM: >0.85 (high similarity)
- Memory: <2GB during inference

---

## Risk Mitigation

| Risk | Mitigation Strategy |
|------|---------------------|
| Insufficient training data | Aggressive augmentation; synthetic data generation; use public datasets |
| Model convergence issues | Simpler architecture fallback; learning rate tuning; verify data pipeline |
| Poor generalization | Increase dataset diversity; stronger regularization (dropout, L2) |
| Slow inference | Model optimization (pruning, quantization); batch processing; GPU acceleration |
| API integration issues | Independent endpoint testing; mock data for frontend development |
| Overfitting | Early stopping; validation monitoring; dropout layers; data augmentation |

---

## Development Setup

### Prerequisites

**Hardware:**
- GPU strongly recommended (CUDA-compatible NVIDIA GPU)
- Minimum 8GB RAM (16GB preferred)
- 5GB+ free disk space (datasets + models)

**Software:**
- Python 3.8+
- Node.js 16+ (for React frontend)
- Git

### Installation

**Backend:**
```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Verify TensorFlow GPU support (if available)
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```

**Frontend:**
```bash
cd frontend
npm install
```

### Running the Application

**Backend:**
```bash
# From project root
uvicorn backend.main:app --reload --host localhost --port 8000

# API docs available at: http://localhost:8000/docs
```

**Frontend:**
```bash
cd frontend
npm start

# Application runs at: http://localhost:3000
```

---

## Implementation Workflow

### Stage 1: Data Preparation
1. Collect 200-500 images per fruit type
2. Organize into directory structure
3. Implement preprocessing pipeline
4. Generate train/val/test splits
5. Verify data quality and balance

### Stage 2: Model Development
1. Implement U-Net architecture
2. Create data generator
3. Configure training pipeline
4. Train model with validation monitoring
5. Evaluate on test set
6. Generate visualizations

### Stage 3: Backend API
1. Implement inference pipeline
2. Create FastAPI endpoints
3. Define Pydantic models
4. Test endpoints with Postman/curl
5. Document API

### Stage 4: Frontend Integration
1. Design React component structure
2. Implement file upload UI
3. Create API client
4. Build visualization components
5. Add error handling and loading states
6. Test end-to-end workflow

### Stage 5: Testing & Refinement
1. Unit test critical functions
2. Integration test full pipeline
3. Visual validation of outputs
4. Performance profiling
5. Bug fixes and optimization
6. Documentation updates

### Stage 6: Deployment & Documentation
1. Finalize code documentation
2. Create demo notebook
3. Prepare presentation materials
4. Document known issues
5. Provide usage examples
6. Record demo video (optional)

---

## References & Resources

**Research Papers:**
- U-Net: Convolutional Networks for Biomedical Image Segmentation (Ronneberger et al., 2015)
- Colorful Image Colorization (Zhang et al., 2016)
- Image-to-Image Translation with Conditional Adversarial Networks (Isola et al., 2017)

**Documentation:**
- TensorFlow/Keras Image-to-Image Translation Guide
- FastAPI File Upload Documentation
- React File Upload Best Practices

**Datasets:**
- Kaggle Fruits-360 Dataset
- ImageNet Fruit Categories
- Custom photography guidelines

**Tools:**
- TensorBoard: Training visualization
- Weights & Biases: Experiment tracking (optional)
- Netron: Model architecture visualization

---

*This document serves as the technical blueprint for the Multi-Object Image Colorizer project. Update as implementation progresses and new insights emerge.*