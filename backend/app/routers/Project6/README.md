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

## 1. rawDataToImage.py
Converts all ndjson files in rawData to 28x28 images and saves those images as their subfolder types in imageData. For example, apple.ndjson becomes imageData/apple/versionX/[images].
Certain features like strokes and countrycode are not needed. We just need the images. It should only extract images if they were succesfully guessed. This file will prompt the user for customization:
1. Output file size: will stop running when the converter has generated this storage amount for each category. For example, when 20mb of apple images have been generated, it stops and moves on to the next category.
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
            ├── training_history_apple.json  (per-fruit training history)
            ├── training_history_banana.json
            └── ... (more fruit histories)
```

**Key Features:**
- One training session creates 7 separate models (one per fruit)
- Each fruit trains independently but in one script execution
- Time estimates shown for each fruit individually
- All fruits' epochs printed to console
- Models can generate fruit-specific images

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