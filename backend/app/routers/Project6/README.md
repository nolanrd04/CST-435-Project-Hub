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