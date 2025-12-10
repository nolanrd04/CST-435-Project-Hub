# PyTorch Colorizer Model Creation

## Overview

This script creates a PyTorch U-Net model for the Multi-Object Image Colorizer project. It provides an interactive interface to customize model architecture and training parameters.

## Prerequisites

```bash
pip install torch torchvision
```

## Usage

Run the script from the Project9 directory:

```bash
python create_model.py
```

## Interactive Prompts

The script will guide you through configuration:

### 1. Model Name
```
Enter model name (e.g., 'colorizer_v1'): fruit_colorizer_v1
```

### 2. Model Configuration

- **Image Size** (default: 256) - Input/output resolution
- **Batch Size** (default: 16) - Training batch size
- **Learning Rate** (default: 0.0001) - Initial learning rate
- **Epochs** (default: 50) - Maximum training epochs

### 3. Optimizer Selection

1. Adam (recommended) - Adaptive learning rate
2. SGD - Stochastic Gradient Descent
3. AdamW - Adam with weight decay

### 4. Loss Function

1. MSE - Mean Squared Error (pixel-level)
2. L1 - Mean Absolute Error
3. Perceptual Loss - Feature-based (requires VGG)

### 5. Training Features

- **Learning Rate Scheduler** - Reduces LR on plateau
  - Patience: epochs to wait before reducing
  - Factor: multiplication factor (e.g., 0.5 = halve LR)

- **Early Stopping** - Halt training if no improvement
  - Patience: epochs without improvement

- **Data Augmentation** - Random transformations
  - Horizontal flip
  - Rotation (degrees)
  - Brightness adjustment

## Output Structure

The script creates the following directory structure (relative to script location):

```
Project9/
├── saved_models/
│   └── <model_name>/
│       ├── <model_name>.pth          # PyTorch model file
│       ├── model_config.json         # Configuration
│       └── model_summary.txt         # Architecture summary
│
├── visualizations/
│   └── <model_name>/                 # Training plots (created during training)
│
└── logs/
    └── <model_name>/                 # TensorBoard logs (created during training)
```

## Model Architecture

**U-Net Structure:**
```
Input: (B, 1, 256, 256) - Grayscale
       ↓
Encoder: 64 → 128 → 256 → 512
       ↓
Bottleneck: 1024
       ↓
Decoder: 512 → 256 → 128 → 64 (with skip connections)
       ↓
Output: (B, 3, 256, 256) - RGB (Sigmoid activation)
```

**Total Parameters:** ~31 million

## Generated Files

### 1. `model_config.json`
Complete configuration including:
- Model name and creation timestamp
- Hyperparameters (batch size, learning rate, epochs)
- Optimizer and loss function settings
- Scheduler and early stopping configuration
- Data augmentation parameters

### 2. `<model_name>.pth`
PyTorch model file containing:
- Model state dictionary (weights and biases)
- Configuration dictionary
- Architecture name

### 3. `model_summary.txt`
Human-readable summary:
- Architecture details
- Parameter count
- Training configuration
- Layer-by-layer breakdown

## Example Session

```
$ python create_model.py

==============================================================
Multi-Object Image Colorizer - Model Creation
==============================================================

Enter model name (e.g., 'colorizer_v1'): fruit_colorizer_v1

------------------------------------------------------------
Model Configuration (press Enter for defaults)
------------------------------------------------------------
Image size [256]:
Batch size [16]: 32
Learning rate [0.0001]: 0.001
Number of epochs [50]: 100

Optimizer options:
  1. Adam (recommended)
  2. SGD
  3. AdamW
Select optimizer [1]: 1

Loss function options:
  1. MSE (Mean Squared Error)
  2. L1 (Mean Absolute Error)
  3. Perceptual Loss (requires pretrained VGG)
Select loss function [1]: 1

Use learning rate scheduler? (y/n) [y]: y
  Scheduler patience (epochs) [5]: 5
  Scheduler factor [0.5]: 0.5

Use early stopping? (y/n) [y]: y
  Early stopping patience (epochs) [10]: 15

Use data augmentation during training? (y/n) [y]: y
  Augmentation options:
    Horizontal flip? (y/n) [y]: y
    Rotation degrees [15]: 15
    Brightness adjustment factor [0.2]: 0.2

✓ Created directories:
  - Models: C:\...\Project9\saved_models\fruit_colorizer_v1
  - Visualizations: C:\...\Project9\visualizations\fruit_colorizer_v1
  - Logs: C:\...\Project9\logs\fruit_colorizer_v1

✓ Saved configuration: ...\saved_models\fruit_colorizer_v1\model_config.json

------------------------------------------------------------
Creating PyTorch model...
------------------------------------------------------------
✓ Model initialized with 31,043,139 parameters
✓ Saved model: ...\saved_models\fruit_colorizer_v1\fruit_colorizer_v1.pth
✓ Saved summary: ...\saved_models\fruit_colorizer_v1\model_summary.txt

==============================================================
Model Created Successfully!
==============================================================
```

## Next Steps

After creating the model:

1. **Prepare Dataset**
   - Collect fruit images (strawberry, orange, blackberry, banana, pineapple)
   - Place in `data/raw/` directory
   - Run preprocessing script

2. **Train Model**
   ```bash
   python train_colorizer.py --model fruit_colorizer_v1
   ```

3. **Evaluate Model**
   ```bash
   python evaluate_model.py --model fruit_colorizer_v1
   ```

4. **Run Inference**
   ```bash
   python colorizer_inference.py --model fruit_colorizer_v1 --input image.jpg
   ```

## Loading a Saved Model

```python
import torch
from create_model import UNetColorizer

# Load checkpoint
checkpoint = torch.load('saved_models/fruit_colorizer_v1/fruit_colorizer_v1.pth')

# Create model with saved config
model = UNetColorizer(checkpoint['config'])
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Use for inference
with torch.no_grad():
    grayscale_input = torch.randn(1, 1, 256, 256)  # Example input
    colorized_output = model(grayscale_input)
```

## Customization Tips

### For Better Quality
- Increase image size: 512 or 1024
- Lower learning rate: 0.00001
- More epochs: 100+
- Use perceptual loss

### For Faster Training
- Smaller image size: 128
- Larger batch size: 32 or 64
- Higher learning rate: 0.001
- Fewer epochs: 25-30

### For Limited Memory
- Smaller batch size: 4 or 8
- Smaller image size: 128
- Disable augmentation during initial testing

## Troubleshooting

**Out of Memory Error:**
- Reduce batch size
- Reduce image size
- Use mixed precision training (FP16)

**Model Not Converging:**
- Lower learning rate
- Check data preprocessing
- Verify dataset quality
- Use different optimizer (try AdamW)

**Poor Generalization:**
- Enable data augmentation
- Increase dataset size
- Add dropout layers (modify architecture)
- Reduce model complexity

## References

- U-Net Paper: [Ronneberger et al., 2015](https://arxiv.org/abs/1505.04597)
- PyTorch Documentation: [pytorch.org](https://pytorch.org/docs/)
- Project Plan: `COLORIZER_PLAN.md`
