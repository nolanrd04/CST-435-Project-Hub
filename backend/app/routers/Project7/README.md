# Project 7: Diffusion Model for Grayscale-to-RGB Colorization

A complete implementation of a conditional diffusion model built from scratch using PyTorch. This project trains a neural network to convert grayscale images to RGB using the DDPM (Denoising Diffusion Probabilistic Model) algorithm.

## Model Architecture Overview

### System Components

**1. Conditional U-Net (Neural Network)**
- Architecture: Encoder-decoder with skip connections
- Input channels: 4 (3 RGB noisy + 1 grayscale condition)
- Output channels: 3 (predicted noise for RGB)
- Default features: [64, 128, 256, 512] at each level (configurable)
- Timestep embedding: 256-dimensional sinusoidal position encodings (configurable)
- **NEW: Dropout regularization** - Configurable dropout (default 0.1) to prevent overfitting

**Network Structure:**
```
Input (noisy RGB + grayscale) → Encoder (downsampling with dropout)
                                    ↓
                            Bottleneck (with timestep injection + dropout)
                                    ↓
                                Decoder (upsampling with skip connections + dropout)
                                    ↓
                            Output (predicted noise)
```

**2. DDPM Scheduler (Diffusion Algorithm)**
- Timesteps: 1000 (configurable, 500 recommended for faster training)
- **Beta schedule options:**
  - **Linear**: Original DDPM schedule (0.0001 → 0.02)
  - **Cosine**: Better for image generation - recommended (default)
- Forward process: Gradually adds Gaussian noise to RGB images
- Reverse process: Iteratively removes noise conditioned on grayscale input

### Training Process

**Forward Diffusion (during training):**
1. Take clean RGB image from dataset
2. Convert to grayscale (condition)
3. Sample random timestep t ∈ [0, 1000)
4. Add noise to RGB: `noisy_RGB = sqrt(α_t) * RGB + sqrt(1 - α_t) * noise`
5. Concatenate: `input = [noisy_RGB, grayscale]` (4 channels)
6. Neural network predicts the noise
7. Loss: MSE between predicted noise and actual noise

**Reverse Diffusion (during generation):**
1. Start with pure random noise (RGB shape)
2. For timestep t = 999 → 0:
   - Concatenate current noisy RGB + grayscale condition
   - Neural network predicts noise at timestep t
   - Remove predicted noise from image
   - Add small controlled noise (except final step)
3. Result: Clean RGB image

### Mathematical Foundation

**Forward Process:**
```
q(x_t | x_0) = N(x_t; √(ᾱ_t) x_0, (1 - ᾱ_t)I)
```

**Reverse Process:**
```
p_θ(x_{t-1} | x_t, condition) = N(x_{t-1}; μ_θ(x_t, t, condition), σ_t²I)
```

**Training Objective:**
```
L = E[||ε - ε_θ(x_t, t, condition)||²]
```
Where ε is true noise and ε_θ is predicted noise.

### Key Features

- **Conditional generation**: Grayscale image guides the colorization
- **Iterative refinement**: 1000 denoising steps for high-quality output
- **Learned noise prediction**: Network learns structure, not direct RGB values
- **Stochastic sampling**: Can generate diverse colorizations for same grayscale input
- **Regularization**: Dropout throughout network prevents overfitting
- **Flexible beta schedules**: Cosine schedule provides better training stability

### Model Parameters & Configuration

**Default configuration (recommended for preventing overfitting):**
- U-Net parameters: ~30-50 million (depends on feature sizes)
- Features: [64, 128, 256, 512] (can increase to [128, 256, 512, 1024] for better quality)
- **Dropout: 0.15** (aggressive regularization to prevent overfitting)
- Beta schedule: Cosine (better than linear for image generation)
- Timesteps: 1000 (can reduce to 500 for faster training)
- Learning rate: 2e-4 (updated from 1e-4)
- Weight decay: 0.01 (regularization)
- Batch size: 32 (increased from 16)
- **Epochs: 50** (conservative default, early stopping will catch earlier convergence)
- **Early stopping patience: 10 epochs** (stops when validation loss plateaus)
- Input resolution: 64×64 (Tiny ImageNet)
- Memory: ~4-8 GB GPU RAM during training (batch size 32)
- Training time: ~2-3 hours (50 epochs, consumer GPU; often stops earlier with early stopping)

**All parameters are fully customizable during training setup.**

### Improvements from Original Implementation

**Architecture Enhancements:**
1. **Dropout layers**: Added configurable dropout (0.0-0.5) throughout U-Net
   - Prevents overfitting on limited datasets
   - Improves generalization to unseen images

2. **Cosine beta schedule**: Alternative to linear schedule
   - Better noise distribution for image generation
   - Improved training stability
   - Based on "Improved Denoising Diffusion Probabilistic Models" (Nichol & Dhariwal, 2021)

**Training Improvements:**
1. **Higher learning rate**: 2e-4 (from 1e-4)
   - Faster convergence for diffusion models
2. **Weight decay**: 0.01 regularization
   - Prevents overfitting alongside dropout
3. **Larger batch sizes**: 32 (from 16)
   - More stable gradients for diffusion training
4. **Aggressive dropout**: 0.15 (from 0.0)
   - Prevents overfitting observed in original training runs
5. **Conservative epochs**: 50 with early stopping patience of 10
   - Prevents wasted compute on overfitted training

**Workflow Enhancements:**
1. **Custom model naming**: Name models independently of dataset
2. **Model override protection**: Prevents accidental deletion of trained models
3. **Resume training**: Continue from any checkpoint
4. **Full parameter customization**: All hyperparameters configurable at runtime

---

## Training the Diffusion Model

### Quick Start

```bash
python train_diffusion.py
```

Or from project root:
```bash
python backend/app/routers/Project7/train_diffusion.py
```

### Interactive Training Workflow

The training script provides a fully interactive workflow with comprehensive customization:

#### 1. Dataset Selection

The script automatically discovers all available datasets in `npzData/` and displays them:

```
============================================================
Available Datasets:
============================================================
  1. sample_model
     Train: 48000, Test: 12000
     Image size: 64x64
============================================================

Enter dataset number or name:
Selection: 1
```

#### 2. Model Naming

After selecting a dataset, you can customize the model name:

```
============================================================
Model Name Configuration
============================================================

Default model name: sample_model
You can create a new model or override an existing one

Enter custom model name (or press Enter to use default): my_diffusion_v2

Model name: my_diffusion_v2
```

**If the model already exists:**
- **Option 1**: Resume training from latest checkpoint
- **Option 2**: Override existing model (deletes all checkpoints - requires confirmation)
- **Option 3**: Cancel and choose a different name

#### 3. Model Configuration

Fully customize all model and training parameters:

**U-Net Architecture:**
```
U-Net Architecture:
  Default features: [64, 128, 256, 512]
  Recommended for better quality: [128, 256, 512, 1024]
  Enter custom features (comma-separated) or press Enter for default:

Time embedding dimension (default: 256):

Dropout Rate (regularization):
  0.0 = no dropout, 0.15-0.2 recommended for preventing overfitting
  Dropout rate (default: 0.15):
```

**Diffusion Parameters:**
```
Diffusion Parameters:
  Timesteps: Number of diffusion steps
    Higher = better quality but slower (1000 is standard, 500 is faster)
  Diffusion timesteps (default: 1000): 500

  Beta Schedule:
    linear: Original DDPM schedule
    cosine: Better for image generation (recommended)
  Beta schedule (linear/cosine, default: cosine): cosine
```

**Training Parameters:**
```
Training Parameters:
  Number of epochs (default: 50): 50
  Batch size (larger is better for diffusion models)
  Batch size (default: 32): 32
  Learning rate (default: 2e-4): 2e-4
  Weight decay (default: 0.01): 0.01
  Generate samples every N epochs (default: 10): 10

Early Stopping:
  Stop training if validation loss doesn't improve for N epochs
  Set to 0 to disable early stopping
  Early stopping patience (default: 10): 10
```

#### 4. Training Execution

The script will:
- Estimate training time based on test batches
- Display configuration summary
- Request confirmation before starting
- Show real-time training progress with:
  - Loss metrics (train and validation)
  - Time per epoch
  - Estimated time remaining
  - Early stopping counter

**Sample Output:**
```
Epoch [35/50]
----------------------------------------------------------------------
  Batch [100/100] Loss: 0.0061
  Train Loss: 0.0062
  Val Loss: 0.0058
  Epoch Time: 5m 18s
  Estimated Time Remaining: 1h 19m
  Early stopping: 3/10 (no improvement)
  Generating samples...
  Saved samples to models/my_diffusion_v2/samples_epoch_35.png
```

### Output Files

Training creates the following structure:

```
models/
└── my_diffusion_v2/
    ├── config.json                    # Full training configuration
    ├── training_history.json          # Loss curves and metrics
    ├── loss_curve.png                 # Training/validation loss plot
    ├── checkpoint_epoch_95.pth        # Recent checkpoints (last 5 kept)
    ├── checkpoint_epoch_96.pth
    ├── checkpoint_epoch_97.pth
    ├── checkpoint_epoch_98.pth
    ├── checkpoint_epoch_100.pth
    ├── best_model.pth                 # Best validation loss checkpoint
    ├── samples_epoch_10.png           # Generated sample colorizations
    ├── samples_epoch_20.png
    └── cost_analysis_report.json      # Training cost estimation
```

### Best Practices

**For Best Results:**
1. **Use cosine beta schedule** - Better stability and quality
2. **Keep dropout at 0.15-0.2** - Essential for preventing overfitting
3. **Increase batch size** - If GPU memory allows (32-64 ideal)
4. **Monitor validation loss** - Should decrease and stabilize (watch for plateau around epoch 30-40)
5. **Check generated samples** - Visual quality matters more than loss numbers
6. **Trust early stopping** - The model often converges before 50 epochs
7. **Don't train too long** - Overfitting typically starts around epoch 30-40 without regularization

**Troubleshooting:**
- **Loss not decreasing**: Increase learning rate to 3e-4 or 5e-4
- **Overfitting (val > train after epoch 30)**: This is normal! Increase dropout to 0.2, reduce model size
- **Val loss plateaus early**: Model has converged - stop training, don't increase epochs
- **Training too slow**: Reduce timesteps to 500, increase batch size
- **Out of memory**: Reduce batch size or model features
- **Poor color quality despite low loss**: Loss (MSE on noise) doesn't measure color realism - this is a known limitation of basic diffusion training. Consider adding perceptual loss in future improvements.

---

# Tiny ImageNet Importer

This script downloads and imports the Tiny ImageNet dataset with customizable options, including human-readable category names and organized folder structure.

## Usage

Run the script from the command line:

```bash
python import_tiny_imagenet.py
```

Or from the project root:

```bash
python backend/app/routers/project7/import_tiny_imagenet.py
```

## Features

### Interactive Prompts

The script will prompt you for:

1. **Model Name** (required)
   - Used to organize imports: images are saved to `imageData/model_name/`
   - Allows multiple dataset imports for different models/experiments
   - Example: "vgg16", "resnet50", "my_cnn_v1"

2. **Target Resolution** (default: 64x64)
   - The Tiny ImageNet images are 64x64 by default
   - This option is included for future resizing capabilities

3. **Maximum Number of Images** (default: 10,000)
   - Limits the total number of images imported
   - Import stops when this limit is reached

4. **Maximum Storage Size** (default: 50 MB)
   - Limits the total storage used by imported images
   - Import stops when this size limit is reached

5. **Categories** (default: all)
   - Type `all` to import all 200 categories
   - Or specify specific category IDs separated by commas
   - Example: `n01443537,n01629819,n01641577`

### Output Structure

Images are saved to `imageData/model_name/` with human-readable folder names:

```
backend/app/routers/project7/
├── import_tiny_imagenet.py
├── imageData/
│   └── my_model/                          # Your model name
│       ├── n01443537_goldfish/            # Category folders with readable names
│       │   ├── n01443537_0.JPEG
│       │   ├── n01443537_1.JPEG
│       │   └── ...
│       ├── n01629819_European_fire_salamander/
│       │   └── ...
│       └── import_metadata.json           # Import statistics
└── temp_download/                         # Automatically deleted after import
```

### Folder Naming Convention

Category folders use the format: `{wnid}_{human_readable_name}`
- Example: `n01443537_goldfish`
- This preserves the unique ID while providing readable context
- Category names are loaded from the dataset's `words.txt` file

### Metadata File

After import, `imageData/model_name/import_metadata.json` contains detailed statistics:

```json
{
  "model_name": "my_model",
  "total_images": 5000,
  "total_size_mb": 45.32,
  "categories": 10,
  "category_stats": {
    "n01443537": {
      "wnid": "n01443537",
      "name": "goldfish",
      "folder": "n01443537_goldfish",
      "count": 500
    },
    "n01629819": {
      "wnid": "n01629819",
      "name": "European fire salamander",
      "folder": "n01629819_European_fire_salamander",
      "count": 500
    }
  },
  "config": {
    "resolution": 64,
    "max_images": 10000,
    "max_size_mb": 50,
    "categories": "all"
  }
}
```

## Dataset Information

**Tiny ImageNet** is a subset of ImageNet with:
- 200 object categories
- 500 training images per category (100,000 total)
- 64x64 color images (JPEG format)
- ~240 MB total size (full dataset)

Dataset source: http://cs231n.stanford.edu/tiny-imagenet-200.zip

## Example Run

```
Script directory: C:\...\project7

Enter model name (this will create imageData/model_name/): my_cnn_v1
Images will be saved to: C:\...\project7\imageData\my_cnn_v1

============================================================
Tiny ImageNet Importer
============================================================

Enter target resolution (default: 64x64): [press Enter]
Enter maximum number of images (default: 10000): 5000
Enter maximum storage size in MB (default: 50): 100

Category options:
  - Type 'all' for all categories
  - Type category names separated by commas (e.g., 'n01443537,n01629819')
  - Press Enter for default (all)
Enter categories: all

------------------------------------------------------------
Configuration Summary:
  Resolution: 64x64
  Max Images: 5000
  Max Size: 100 MB
  Categories: all
------------------------------------------------------------

Proceed with import? (y/n): y

[1/4] Downloading Tiny ImageNet dataset...
  Progress: [████████████████████████████████████████] 100.0%
  Download complete!

[2/4] Extracting dataset...
  Extraction complete!

[3/4] Importing images...
  Loaded 200 category names
  Importing from 200 categories...
  Imported: 5000 images (45.32 MB)
  Import complete! Total: 5000 images (45.32 MB)
  Metadata saved to: imageData/my_cnn_v1/import_metadata.json

[4/4] Cleaning up...
  Temporary files deleted.

============================================================
Import Summary:
  Images imported: 5000
  Total size: 45.32 MB
  Categories: 200
  Output directory: C:\...\project7\imageData\my_cnn_v1
============================================================

Import completed successfully!
```

## Notes

- The script uses **script-relative paths**, so `imageData/` will be created in the same directory as the script
- Each model import creates a separate subdirectory under `imageData/model_name/`
- Category folders include human-readable names from the dataset's `words.txt` file
- Temporary files in `temp_download/` are **automatically deleted** after import
- The actual image files are saved, so you can view them directly
- Import can be interrupted with Ctrl+C
- Limits (max images or max size) are checked per-image to avoid overshooting significantly
- Metadata includes both WordNet IDs and human-readable category names for easy reference

---

# Image to NPZ Data Conversion

Converts imported RGB images to paired grayscale/RGB NPZ arrays for diffusion model training. This script prepares data for training a model that converts grayscale images to RGB.

## Usage

Run the script from the command line:

```bash
python image_to_npz.py
```

Or from the project root:

```bash
python backend/app/routers/project7/image_to_npz.py
```

## Features

### Automatic Model Discovery

The script automatically detects all available models in `imageData/` and displays them with statistics:

```
============================================================
Available Models:
============================================================
  1. my_cnn_v1
     5000 images, 200 categories, 45.32 MB
  2. resnet_experiment
     10000 images, 200 categories, 90.15 MB
============================================================
```

You can select by:
- **Number**: Type `1` to select the first model
- **Name**: Type `my_cnn_v1` to select by name
- **Cancel**: Press Enter to exit

### Interactive Prompts

After selecting a model, the script prompts for:

1. **Test Split Size** (default: 0.2)
   - Fraction of data to use for testing (0.0-1.0)
   - Example: 0.2 = 20% test, 80% train
   - Uses stratified split to maintain category balance

2. **Random State** (default: 42)
   - Seed for reproducible train/test splits
   - Same seed = same split across runs
   - Important for comparing different models

3. **Target Image Size** (default: 64)
   - Resizes all images to this resolution
   - Matches Tiny ImageNet native size: 64x64
   - Smaller sizes (32, 28) reduce file size and training time

4. **Normalization Option**:
   - **Option 1: [0, 1] range** - Divide by 255
   - **Option 2: [-1, 1] range** - Standard for diffusion models (recommended)
   - **Option 3: None [0, 255]** - Keep as uint8 (saves 75% storage)

### Data Processing Pipeline

1. **Load RGB images** from `imageData/model_name/`
2. **Convert to grayscale** (input for diffusion model)
3. **Resize** if needed to target resolution
4. **Stratified split** into train/test sets
5. **Normalize** according to selected option
6. **Save** as compressed NPZ files

### Output Structure

```
backend/app/routers/project7/
├── image_to_npz.py
├── imageData/
│   └── my_model/                    # Source images
│       └── ...
└── npzData/
    └── my_model/                    # Converted data
        ├── train.npz                # Training data
        ├── test.npz                 # Testing data
        └── metadata.json            # Conversion statistics
```

### NPZ File Contents

Each NPZ file (train.npz, test.npz) contains:

```python
{
    'grayscale': np.ndarray,  # Shape: (N, H, W, 1) - Model input
    'rgb': np.ndarray,        # Shape: (N, H, W, 3) - Model target
    'labels': np.ndarray,     # Shape: (N,) - Category indices
    'category_names': np.ndarray  # Shape: (N,) - Category folder names
}
```

**Data types:**
- **float32** (options 1-2): Ready for training, larger files (~4x size)
- **uint8** (option 3): Compact storage, normalize during training

### Metadata File

After conversion, `npzData/model_name/metadata.json` contains:

```json
{
  "model_name": "my_model",
  "source_dir": "C:/.../imageData/my_model",
  "config": {
    "test_size": 0.2,
    "random_state": 42,
    "image_size": 64,
    "normalize_option": 2
  },
  "train_samples": 4000,
  "test_samples": 1000,
  "image_shape": {
    "grayscale": [64, 64, 1],
    "rgb": [64, 64, 3]
  },
  "dtype": "float32",
  "normalization": "[-1, 1]",
  "categories": 200
}
```

## Example Run

```
Script directory: C:\...\project7

============================================================
Available Models:
============================================================
  1. my_cnn_v1
     5000 images, 200 categories, 45.32 MB
============================================================

Enter model number or name:
(Press Enter to cancel)

Selection: 1

Selected: my_cnn_v1
Data location: C:\...\project7\imageData\my_cnn_v1

============================================================
Image to NPZ Converter
============================================================

Found 5000 images in 200 categories

Enter test split size (0.0-1.0, default: 0.2): 0.2
Enter random state for reproducibility (default: 42): 42
Enter target image size (default: 64x64): 64

Normalization options:
  1. [0, 1] range (divide by 255)
  2. [-1, 1] range (standard for diffusion models)
  3. None (keep as [0, 255])
Enter normalization option (default: 2): 3

------------------------------------------------------------
Configuration Summary:
  Model: my_cnn_v1
  Image size: 64x64
  Test split: 20.0%
  Train split: 80.0%
  Random state: 42
  Normalization: None [0, 255]
------------------------------------------------------------

Proceed with conversion? (y/n): y

[1/3] Loading images...
  Found 200 categories
  Loaded: 5000 images
  Loaded 5000 images successfully

[2/3] Splitting and normalizing data...
  Train set: 4000 images
  Test set: 1000 images

[3/3] Saving NPZ files...
  Saved: npzData/my_cnn_v1/train.npz (123.45 MB)
  Saved: npzData/my_cnn_v1/test.npz (30.86 MB)
  Saved: npzData/my_cnn_v1/metadata.json

============================================================
Conversion Summary:
  Model: my_cnn_v1
  Train samples: 4000
  Test samples: 1000
  Grayscale shape: (64, 64, 1)
  RGB shape: (64, 64, 3)
  Data type: uint8
  Normalization: None [0, 255]
  Output directory: C:\...\project7\npzData\my_cnn_v1
============================================================

Conversion completed successfully!
```

## Storage Considerations

NPZ file sizes depend on normalization option:

**Example: 10,000 images at 64x64**

| Normalization | Data Type | Size (Train + Test) |
|---------------|-----------|---------------------|
| Option 1 or 2 | float32   | ~1.05 GB           |
| Option 3      | uint8     | ~250 MB            |

**Recommendation:**
- Use **uint8 (option 3)** to save 75% storage
- Normalize to [-1, 1] during training instead:
  ```python
  data = np.load('train.npz')
  grayscale = data['grayscale'].astype(np.float32) / 127.5 - 1.0
  rgb = data['rgb'].astype(np.float32) / 127.5 - 1.0
  ```

## Loading Data for Training

```python
import numpy as np

# Load training data
train_data = np.load('npzData/my_model/train.npz')
train_grayscale = train_data['grayscale']  # Input
train_rgb = train_data['rgb']              # Target
train_labels = train_data['labels']        # Categories (optional)

# Load test data
test_data = np.load('npzData/my_model/test.npz')
test_grayscale = test_data['grayscale']
test_rgb = test_data['rgb']

# If using uint8, normalize during training:
if train_grayscale.dtype == np.uint8:
    train_grayscale = train_grayscale.astype(np.float32) / 127.5 - 1.0
    train_rgb = train_rgb.astype(np.float32) / 127.5 - 1.0
```

## Notes

- **Stratified split**: Maintains category balance in train/test sets
- **Reproducibility**: Same random_state produces identical splits
- **Grayscale conversion**: Uses PIL's standard luminance formula
- **Compression**: NPZ files are automatically compressed (zip format)
- **Memory efficiency**: Loads all images into RAM before splitting (fine for Tiny ImageNet subsets)
- **Category labels**: Included for conditional diffusion models (can ignore if not needed)
