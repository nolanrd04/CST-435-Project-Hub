# Enhanced Diffusion Training (V2) - Perceptual Loss & GAN

This directory contains an improved training script (`train_diffusion_v2.py`) that adds **perceptual loss** and **optional GAN discriminator** to improve colorization accuracy.

## What's New in V2?

### 1. **Perceptual Loss (VGG-based)**
Instead of only measuring pixel-level noise prediction (MSE), the model now learns to match colors in **VGG feature space**:
- Compares predicted and target images using VGG16 features (relu1_2, relu2_2, relu3_3, relu4_3)
- Better captures perceptual similarity (what humans perceive as "similar colors")
- **Fixes the black pixel problem**: Model learns semantic color patterns instead of guessing random colors

### 2. **Optional GAN Discriminator (PatchGAN)**
Adds adversarial training to force outputs to look realistic:
- PatchGAN classifies local patches as real/fake instead of whole image
- Helps with local color accuracy and texture details
- Optional - can train with or without it

### 3. **Combined Loss Function**
Three objectives working together:
- **MSE loss** (1.0): Original diffusion objective - predicts noise accurately
- **Perceptual loss** (0.1): Matches colors in VGG feature space
- **Adversarial loss** (0.01, optional): Fools discriminator to produce realistic colors

## Quick Start

### Installation Requirements

First, install torchvision (required for VGG):

```bash
pip install torchvision
```

### Running the Training

```bash
cd backend/app/routers/Project7
python train_diffusion_v2.py
```

### Configuration Workflow

The script will guide you through configuration:

#### 1. **Resume or Create New Model**
```
Do you want to resume from an existing model?

Available models:
  1. model_epoch50
  2. model_v2_perceptual
  3. Create new model

Select option (1-3):
```

**Options:**
- Select existing model to continue training with perceptual loss
- Select "Create new model" to train from scratch

**Recommendation**: Resume from your epoch 71 model to add perceptual loss without starting over!

#### 2. **Model Configuration** (if creating new)
```
Enter model name (e.g., 'model_v2_perceptual'): model_perceptual_50epochs

Model Architecture:
  U-Net features [64,128,256,512]: 64,128,256,512
  Time embedding dim [256]: 256
  Diffusion timesteps [1000]: 1000

Training Hyperparameters:
  Batch size [16]: 16
  Learning rate [1e-4]: 1e-4
  Number of epochs [50]: 50
```

**Recommendation**: Keep same architecture as your original model (64,128,256,512)

#### 3. **Perceptual Loss Configuration**
```
Loss weights determine how much each objective contributes to training.
Recommended starting values:
  - MSE weight: 1.0 (noise prediction)
  - Perceptual weight: 0.1 (color accuracy via VGG features)
  - Adversarial weight: 0.01 (realism via GAN)

MSE weight [1.0]: 1.0
Perceptual weight [0.1]: 0.1

Use GAN discriminator? (y/n) [n]: n
```

**Recommendations:**
- **First run**: Use `MSE=1.0, Perceptual=0.1, GAN=no`
- **If colors still wrong**: Increase perceptual weight to `0.2` or `0.5`
- **If you want max realism**: Use GAN with `Adversarial=0.01`

## Understanding Loss Weights

### MSE Weight (1.0)
- Original diffusion loss - predicts noise in images
- **Keep at 1.0** to maintain diffusion model stability

### Perceptual Weight (0.1 - 0.5)
- How much to prioritize "correct colors" vs "correct noise"
- **Start at 0.1**: Gentle guidance toward correct colors
- **0.2-0.3**: Stronger color correction (recommended if 0.1 doesn't help)
- **0.5+**: Very strong color emphasis (may hurt image quality)

### Adversarial Weight (0.01 - 0.05)
- Only used if GAN is enabled
- How much to prioritize "realistic looking" outputs
- **0.01**: Subtle realism boost
- **0.05**: Stronger realism (may cause instability)

## Resuming from Existing Models

### Resume from Original Model (epoch 71)

You can resume your epoch 71 model and add perceptual loss:

1. Run `python train_diffusion_v2.py`
2. Select your existing model (e.g., `model_epoch50`)
3. Configure perceptual loss weights
4. Training continues from epoch 71 with **additional objectives**

**What happens:**
- Original MSE loss: Still trained (weight 1.0)
- Perceptual loss: Added on top (weight 0.1)
- Model learns to balance both objectives

### Training New Model from Scratch

If you want a clean start with perceptual loss:

1. Create new model name (e.g., `model_v2_perceptual`)
2. Configure architecture (same as before: 64,128,256,512)
3. Set perceptual weight to 0.1-0.2
4. Train for 50 epochs

## Expected Results

### With Perceptual Loss (no GAN)
- ✓ Better color accuracy on animals and objects
- ✓ Black pixels colored more semantically (e.g., black fur stays dark, not random red/blue)
- ✓ Colors match training data patterns better
- ⚠️ May sacrifice some fine texture details for color correctness

### With Perceptual Loss + GAN
- ✓ All benefits of perceptual loss
- ✓ More realistic local textures
- ✓ Sharper color transitions
- ⚠️ Longer training time (discriminator adds overhead)
- ⚠️ Slightly less stable (adversarial training can diverge)

## Troubleshooting

### "ModuleNotFoundError: No module named 'torchvision'"
```bash
pip install torchvision
```

### Colors still wrong after training
- **Increase perceptual weight**: Try 0.2 or 0.3 instead of 0.1
- **Train longer**: Perceptual loss may need more epochs to converge
- **Enable GAN**: Add adversarial loss for stronger realism

### Training is very slow
- **Perceptual loss adds ~30% overhead** (VGG forward passes)
- **GAN adds ~50% overhead** (discriminator training)
- **Solution**: Train on GPU if possible, or use smaller batch size

### Model diverges / loss explodes
- **Lower perceptual weight**: Try 0.05 instead of 0.1
- **Lower adversarial weight**: Try 0.005 instead of 0.01
- **Disable GAN**: Train with perceptual loss only first

### Out of memory (CUDA/RAM)
- **Reduce batch size**: Try 8 instead of 16
- **Disable GAN**: Saves ~1-2GB memory
- **Use CPU**: Slower but works with less RAM

## File Structure

```
Project7/
├── train_diffusion.py          # Original training script (MSE loss only)
├── train_diffusion_v2.py       # New training script (Perceptual + GAN)
├── perceptual_loss.py          # VGG perceptual loss & GAN discriminator
├── diffusion_model.py          # U-Net and DDPM scheduler (unchanged)
├── dataset.py                  # Tiny ImageNet loader (unchanged)
├── models/
│   ├── model_epoch50/          # Your original model
│   │   ├── best_model.pth
│   │   ├── config.json
│   │   └── training_history.json
│   └── model_v2_perceptual/    # New model with perceptual loss
│       ├── best_model.pth
│       ├── config.json
│       └── training_history.json
└── README_V2_TRAINING.md       # This file
```

## Checkpoint Format

V2 checkpoints include additional state:

```python
{
    'epoch': 10,
    'model_state_dict': {...},           # U-Net weights
    'optimizer_state_dict': {...},       # Adam optimizer
    'loss_state_dict': {                 # NEW: Loss module state
        'mse_weight': 1.0,
        'perceptual_weight': 0.1,
        'adversarial_weight': 0.01,
        'use_gan': False,
        'discriminator_state_dict': {...},      # If using GAN
        'discriminator_optimizer_state_dict': {...}
    },
    'train_loss': 0.0234,
    'val_loss': 0.0189,
    'best_val_loss': 0.0189,
    'config': {...}
}
```

## Recommended Training Strategy

### Strategy 1: Resume from Epoch 71 (Fastest)
1. Resume your existing model
2. Add perceptual loss (weight 0.1)
3. Train 10-20 more epochs
4. Evaluate - if colors improved, stop. Otherwise increase weight and continue.

**Pros:** Fastest, builds on existing training
**Cons:** May not fully optimize perceptual objective

### Strategy 2: Train New Model (Most Control)
1. Create new model with same architecture
2. Perceptual loss from epoch 0
3. Train full 50 epochs
4. Compare with original model

**Pros:** Clean optimization of both objectives
**Cons:** Takes longer, may need tuning

### Strategy 3: Two-Stage Training (Recommended)
1. **Stage 1**: Train with MSE only (original script) for 30-40 epochs
2. **Stage 2**: Resume with perceptual loss (v2 script) for 10-20 epochs

**Pros:** Best of both worlds - stable diffusion learning, then color refinement
**Cons:** Requires monitoring and switching scripts

## Next Steps

After training with perceptual loss:

1. **Test on frontend**: Upload images and check if black pixels are colored better
2. **Compare models**: Try both original (epoch 71) and new perceptual model
3. **Adjust weights**: If still not good enough, increase perceptual weight to 0.2-0.3
4. **Enable GAN**: If you want maximum realism, train with discriminator

Good luck! The perceptual loss should significantly improve color accuracy, especially on black/dark regions.
