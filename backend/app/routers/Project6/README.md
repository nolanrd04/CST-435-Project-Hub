# Description:
Fake images have become so ubiquitous on social media that they affect the public discourse on these platforms. It is essential for students to understand how these images are created.

Design a GAN-based application using PyTorch that can generate fake images that look like real ones. The main idea is to train a GAN with a mix of real and fake images so it can learn to distinguish (discriminate) between them.

Complete the following steps. Then, write a comprehensive technical report as a Python Jupyter notebook to include all code, code comments, all outputs, plots, and analysis. Make sure the project documentation contains 
- a) Problem statement
- b) Algorithm of the solution
- c) Analysis of the findings
- d) References

# Requirements

## 1. Build a simple Generator network.
1. Initialize the neural network.
2. Add an input layer.
3. Activate the layer with LeakyReLU.
4. Apply batch normalization.
5. Add a second layer.
6. Add a third layer.
7. Add an output layer.
8. Compile the Generator network.

## 2. Build a simple Discriminator network.
1. Initialize the neural network.
2. Add an input layer.
3. Activate the layer with LeakyReLU.
4. Apply batch normalization.
5. Add a second layer.
6. Add a third layer.
7. Add an output layer.
8. Compile the Discriminator network.

## 3. Build a GAN by stacking the generator and discriminator.
1. Plot the images created by the Generator from a normally distributed noise input.
2. Generate a normally distributed noise of shape 100x100.
3. Generate images for the input noise.
4. Reshape images if needed.
5. Plot.

## 4. Training
1. Train the GAN using the training set, default epoch, and default batch size.
2. Use the noised input of the Generator and trick it as real data.
3. Train the GAN for at least 400 epochs.
4. Print the images generated at several epoch milestones (e.g., epoch 1, epoch 30, epoch 100, epoch 400).
5. Summarize the model, quantify its performance, and explain to what extent it is capable of generating fake images that look like the real ones.

# **Project Pipeline**

## 1. Locally download the data.
The initial data files are large; over 100mb, so those you need to download yourself. You can find them at
```
https://console.cloud.google.com/storage/browser/quickdraw_dataset/full/simplified?pageState=(%22StorageObjectListTable%22:(%22f%22:%22%5B%5D%22))
```
Find the fruit files apple, banana, blackberry, grapes, pear, strawberry, and watermelon.
Create a new folder in
```
backend/app/Project6/
```
and call it rawData, so the new filepath should be
```
backend/app/Project6/rawData.
```
Move the downloaded ndjson files into this folder.

## 1. rawDataToImage.py
Converts all ndjson files in rawData to custome resolution images and saves those images as their subfolder types in imageData. For example, apple.ndjson becomes imageData/apple/versionX/[images]. Prompts the user to pick a quality size, where quality is primarily determined by the number of strokes.
1. Output file size: will stop running when the converter has generated a specific number of files for each fruit that the user picks.
2. Versions: the user might want to create multiple models, in which case we need to maintain data versions. This can be solved by asking the user for a version name and creating a new subfolder version[Name] where [Name] is given by the user.

## 2. imageToNPZ.py
Converts images in imageData/types/versionX to their prospective .npz array files in npzData/. For example. imageData/apple/version1 becomes npzData/apple_version1.npz. Used for the GAN.

## 3. train_gan.py
This will train the model. This will:
1. Prompt you to select a dataset version (v1, v2, etc.)
2. Prompt you to select a model name (user-defined, defaults to v1)
3. Ask for training parameters (epochs, batch size, learning rate)
4. Automatically trains separate Generator/Discriminator pairs for each fruit
5. Trains all fruits in one session with individual time estimates per fruit
6. Saves fruit-specific checkpoints and models
It uses helper python files:
1. ```data_loader.py```
2. ```gan_model.py```
3. ```gan_trainer.py``` (includes `MultiFruitGANTrainer` class)
4. ```cost_analysis_training.py``` (tracks training costs and performance)

## 4. saved models from train_gan.py
The model structure now supports multi-fruit training. Each model folder contains separate generators/discriminators for each fruit:
Project6/
```
├── train_gan.py                 (main script)
├── gan_model.py                 (models)
├── gan_trainer.py               (training logic + MultiFruitGANTrainer)
├── data_loader.py               (data loading)
├── generate_images.py           (inference)
├── npzData/                     (existing dataset)
│   ├── apple_v1.npz
│   ├── banana_v1.npz
│   └── ...
└── models/                      (auto-created)
    └── model_v1/                (user-named model, e.g., v1, attempt_2, etc.)
        ├── generator_apple.pt           (fruit-specific generators)
        ├── discriminator_apple.pt       (fruit-specific discriminators)
        ├── generator_banana.pt
        ├── discriminator_banana.pt
        ├── generator_orange.pt
        ├── discriminator_orange.pt
        ├── ... (remaining 4 fruits)
        ├── generated_epoch_images_apple/    (5 epoch images during training)
        │   ├── epoch_0001.png
        │   ├── epoch_0010.png
        │   ├── epoch_0020.png
        │   ├── epoch_0030.png
        │   └── epoch_0040.png
        ├── generated_epoch_images_banana/   (separate for each fruit)
        │   ├── epoch_0001.png
        │   └── ...
        ├── ... (more fruit folders)
        └── info/                        (training metadata)
            ├── training_config_v1.json      (overall config, all fruits)
            ├── training_summary_v1.json     (summary of trained fruits)
            ├── training_history_apple.json  (per-fruit training history with cost data)
            ├── training_history_banana.json
            ├── cost_analysis_report.json    (detailed cost breakdown)
            ├── description.txt              (user description + auto-generated stats)
            └── ... (more fruit histories)
```

**Key Features:**
- One training session creates 7 separate models (one per fruit)
- Each fruit trains independently but in one script execution
- Time estimates shown for each fruit individually
- All fruits' epochs printed to console
- Models can generate fruit-specific images
- **Cost Analysis:** Tracks training time, memory usage, and estimated costs
- **Training Statistics:** Auto-generated reports with performance metrics

## 5. generate_images.py
Uses the trained models to generate fruit-specific images.

**Usage:**
```bash
# List available models and fruits
python generate_images.py

# Generate apple images from v1 model
python generate_images.py v1 apple --num-images 16

# Generate banana images with interpolation
python generate_images.py v1 banana --num-images 10 --interpolate

# Save to specific file
python generate_images.py v1 orange --save output.png
```

**Parameters:**
- `model_name`: Name of the model folder (e.g., v1)
- `fruit`: Which fruit to generate (e.g., apple, banana, orange)
- `--num-images`: Number of images to generate (default: 16)
- `--interpolate`: Generate smooth transition between images
- `--save`: Save output to file instead of displaying
- `--seed`: Random seed for reproducibility

## 6. Cost Analysis & Training Statistics

Project6 now includes comprehensive cost tracking and performance metrics, similar to Project5. During training, the system automatically:

### Tracked Metrics
- **Training Time:** Per-fruit and total training duration
- **Memory Usage:** Peak memory consumption during training
- **Training Cost:** Estimated compute costs based on Render pricing
- **Cost Per Epoch:** Average cost breakdown per training epoch
- **Performance Metrics:** Generator and discriminator losses per epoch

### Output Files

#### `description.txt`
Auto-generated file containing:
- User-provided model description
- Dataset information (resolution, fruit types, image counts)
- Training configuration (epochs, batch size, learning rate, etc.)
- **Training Statistics:**
  - Total training time (hours and minutes)
  - Average time per fruit
  - Peak memory usage (GB)
  - Total training cost
  - Average cost per fruit
  - Average cost per epoch

#### `cost_analysis_report.json`
Detailed JSON report with:
- Complete training cost breakdown (compute, memory, storage)
- Cost per epoch calculations
- Peak memory usage
- Training hours
- Model configuration snapshot

#### `training_history_[fruit].json`
Per-fruit training history including:
- Generator and discriminator losses per epoch
- Training time (seconds and hours)
- Peak memory usage
- Cost summary (total, per epoch, per hour)

### Pricing Configuration

Cost calculations are based on `render_pricing_config.json`:
```json
{
  "fixed_monthly_cost": 19.0,
  "cost_per_cpu_per_month": 0.296875,
  "cost_per_gb_ram_per_month": 0.037109375,
  "additional_storage_cost_per_gb": 0.10
}
```

You can customize these values to match your deployment platform's pricing.

### Example Cost Summary Output
```
==========================================================
TRAINING COST SUMMARY FOR APPLE
==========================================================
Training time: 2.34 hours (140.5 minutes)
Peak memory: 3.84 GB
Total cost: $0.003256
Cost per epoch: $0.000008
Cost per hour: $0.001391
==========================================================
```

# GAN Architecture

## Overview: What is a GAN?

A **Generative Adversarial Network (GAN)** consists of two neural networks competing against each other:
- **Generator**: Creates fake images from random noise
- **Discriminator**: Tries to distinguish real images from fake ones

Think of it like a counterfeiter (Generator) vs. a detective (Discriminator). The counterfeiter gets better at making fake images, while the detective gets better at spotting fakes. Eventually, the counterfeiter becomes so good that even the detective can't tell the difference!

## Network Types

### Generator: Hybrid Architecture (NOT a Pure CNN)
The Generator is a **hybrid network** consisting of:
1. **Fully Connected Layer**: Expands 100-dimensional noise into a spatial representation
2. **Deconvolutional/Transpose Convolutional Layers**: Progressive upsampling to build images

**Direction**: Expands from compact representation → Full image
- Input: 100D vector (random noise)
- Output: 64×64 image
- Type: **Deconvolutional Network** (reverse of typical CNNs)

### Discriminator: Pure Convolutional Neural Network (CNN)
The Discriminator is a **traditional CNN**, similar to image classifiers (VGG, ResNet):
1. **Convolutional Layers**: Progressively downsample and extract features
2. **Fully Connected Layer**: Final binary classification (real vs. fake)

**Direction**: Compresses from full image → Single probability
- Input: 64×64 image
- Output: Single value (0=fake, 1=real)
- Type: **Standard CNN** for classification

### Why This Design?

| Network | Challenge | Solution |
|---------|-----------|----------|
| **Generator** | Must CREATE spatial structure from nothing | Uses transposed convolutions (upsampling) to build up detail |
| **Discriminator** | Must EXTRACT features to classify | Uses standard convolutions to compress and analyze |

This architecture is called a **DCGAN (Deep Convolutional GAN)**:
- "Deep" = Multiple layers
- "Convolutional" = Uses convolution operations (in both directions)
- "GAN" = Generative Adversarial Network

---

## 1. GENERATOR ARCHITECTURE

  Purpose

  Transform random noise (latent vector) into realistic fruit images.

  Input

  - Latent vector (z): 100-dimensional random noise sampled from normal distribution
  - Shape: (batch_size, 100)

  Architecture Breakdown

  Layer 0: Fully Connected Expansion

  nn.Linear(100, 128 * init_size * init_size)
  - Input: 100-dimensional noise vector
  - Output: Flattened tensor ready to reshape
  - Purpose: Expand the compact noise into a spatial representation
  - Example: For 64×64 images, init_size = 16, so output = 128 × 16 × 16 = 32,768 values

  Reshape

  - Reshape from flat vector to 3D tensor: (batch_size, 128, init_size, init_size)
  - For 64×64 images: (batch, 128, 16, 16)
  - For 32×32 images: (batch, 128, 8, 8)

  ---
  Layer 1: First Upsampling Block

  nn.BatchNorm2d(128)
  nn.Upsample(scale_factor=2)
  nn.Conv2d(128, 128, 3, stride=1, padding=1)
  nn.BatchNorm2d(128)
  nn.LeakyReLU(0.2)

  Components:
  1. BatchNorm2d(128): Normalizes the 128 channels to stabilize training
  2. Upsample(scale_factor=2): Doubles spatial dimensions (16×16 → 32×32)
  3. Conv2d(128→128, kernel=3×3): Refines upsampled features
  4. BatchNorm2d(128): Normalizes again after convolution
  5. LeakyReLU(0.2): Activation that allows small negative gradients (slope=0.2)

  Output shape: (batch, 128, 32, 32) for 64×64 final images

  ---
  Layer 2: Middle Convolution Block

  nn.Conv2d(128, 64, 3, stride=1, padding=1)
  nn.BatchNorm2d(64)
  nn.LeakyReLU(0.2)

  Purpose: Reduce channel depth while maintaining spatial size
  - Channels: 128 → 64
  - Spatial: Stays 32×32
  - Learns mid-level features

  Output shape: (batch, 64, 32, 32)

  ---
  Layer 3: Second Upsampling Block

  nn.Upsample(scale_factor=2)
  nn.Conv2d(64, 64, 3, stride=1, padding=1)
  nn.BatchNorm2d(64)
  nn.LeakyReLU(0.2)

  Purpose: Final upsampling to target resolution
  - Upsample: 32×32 → 64×64
  - Maintains 64 channels
  - Refines details

  Output shape: (batch, 64, 64, 64)

  ---
  Layer 4: Output Layer

  nn.Conv2d(64, channels, 3, stride=1, padding=1)
  nn.Tanh()

  Purpose: Generate final image
  - Channels: 64 → 1 (grayscale) or 3 (RGB)
  - Tanh activation: Outputs in range [-1, 1]
    - This matches the normalized training data range
    - -1 = black, 0 = gray, +1 = white

  Final Output: (batch, 1, 64, 64) - grayscale fruit images

  ---
  Generator Summary

  Input:  (batch, 100) random noise
      ↓
  FC:     (batch, 32768) → reshape → (batch, 128, 16, 16)
      ↓
  Block1: (batch, 128, 16, 16) → (batch, 128, 32, 32)  [Upsample 2×]
      ↓
  Block2: (batch, 128, 32, 32) → (batch, 64, 32, 32)   [Channel reduction]
      ↓
  Block3: (batch, 64, 32, 32) → (batch, 64, 64, 64)    [Upsample 2×]
      ↓
  Output: (batch, 64, 64, 64) → (batch, 1, 64, 64)     [Final image]

  Total upsampling: 16×16 → 32×32 → 64×64 (4× increase)

  ---
  ## 2. DISCRIMINATOR ARCHITECTURE

  Purpose

  Classify images as real (from dataset) or fake (from generator).

  Input

  - Image tensor: Shape (batch_size, channels, img_size, img_size)
  - Example: (32, 1, 64, 64) for batch of 32 grayscale 64×64 images

  Architecture Breakdown

  Layer 1: Input Convolution

  nn.Conv2d(channels, 32, 3, stride=2, padding=1)
  nn.LeakyReLU(0.2)
  nn.Dropout2d(0.25)

  Components:
  - Conv2d: Channels 1→32, stride=2 (downsamples by 2×)
  - LeakyReLU(0.2): Allows small negative gradients
  - Dropout2d(0.25): Randomly zeros 25% of feature maps (prevents overfitting)

  Effect: 64×64 → 32×32, extracts basic features

  Output shape: (batch, 32, 32, 32)

  ---
  Layer 2: Middle Convolution

  nn.Conv2d(32, 64, 3, stride=2, padding=1)
  nn.BatchNorm2d(64)
  nn.LeakyReLU(0.2)
  nn.Dropout2d(0.25)

  Purpose: Learn mid-level features, downsample further
  - Channels: 32 → 64
  - Spatial: 32×32 → 16×16
  - BatchNorm: Stabilizes learning

  Output shape: (batch, 64, 16, 16)

  ---
  Layer 3: Deep Convolution

  nn.Conv2d(64, 128, 3, stride=2, padding=1)
  nn.BatchNorm2d(128)
  nn.LeakyReLU(0.2)
  nn.Dropout2d(0.25)

  Purpose: Learn high-level semantic features
  - Channels: 64 → 128
  - Spatial: 16×16 → 8×8
  - Deepest feature representation

  Output shape: (batch, 128, 8, 8)

  ---
  Layer 4: Flatten & Classification
```
  # Flatten: (batch, 128, 8, 8) → (batch, 8192)
  nn.Linear(8192, 1)
  nn.Sigmoid()
```


  Purpose: Final binary classification
  - Flatten: Convert 3D features to 1D vector
  - Linear layer: Reduce to single value
  - Sigmoid: Output probability in [0, 1]
    - 0 = Fake image
    - 1 = Real image

  Final Output: (batch, 1) - probability scores

  ---
  Discriminator Summary

  Input:  (batch, 1, 64, 64) image
      ↓
  Layer1: (batch, 1, 64, 64) → (batch, 32, 32, 32)   [Downsample 2×]
      ↓
  Layer2: (batch, 32, 32, 32) → (batch, 64, 16, 16)  [Downsample 2×]
      ↓
  Layer3: (batch, 64, 16, 16) → (batch, 128, 8, 8)   [Downsample 2×]
      ↓
  Flatten: (batch, 128, 8, 8) → (batch, 8192)
      ↓
  Output: (batch, 8192) → (batch, 1) probability

  Total downsampling: 64×64 → 32×32 → 16×16 → 8×8 (8× reduction)

  ---