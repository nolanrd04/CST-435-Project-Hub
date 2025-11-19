"""
GAN Model Architecture
Defines Generator and Discriminator networks for image generation
"""

import torch
import torch.nn as nn


class Generator(nn.Module):
    """
    Generator network for GAN
    Takes random noise as input and generates images
    """

    def __init__(self, latent_dim=100, channels=1, img_size=28):
        """
        Initialize the Generator

        Args:
            latent_dim (int): Dimension of the random noise input
            channels (int): Number of image channels (1 for grayscale, 3 for RGB)
            img_size (int): Size of the output image (28, 32, 64, etc.)
        """
        super(Generator, self).__init__()

        self.latent_dim = latent_dim
        self.channels = channels
        self.img_size = img_size
        self.img_shape = (channels, img_size, img_size)

        # Calculate the output dimension for the first layer
        # We'll reshape to (128, init_size, init_size) before upsampling
        self.init_size = img_size // 4  # e.g., 28 // 4 = 7, 64 // 4 = 16

        # Linear layer to expand latent vector
        self.fc = nn.Sequential(
            nn.Linear(latent_dim, 128 * self.init_size * self.init_size)
        )

        # Main generator network
        self.model = nn.Sequential(
            # Input: (128, init_size, init_size)

            # First layer - Upsample by 2x
            nn.BatchNorm2d(128),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),

            # Second layer - Keep same size
            nn.Conv2d(128, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),

            # Third layer - Upsample by 2x
            nn.Upsample(scale_factor=2),
            nn.Conv2d(64, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),

            # Output layer - Generate final image (channels, img_size, img_size)
            nn.Conv2d(64, channels, 3, stride=1, padding=1),
            nn.Tanh()  # Output range [-1, 1]
        )

    def forward(self, z):
        """
        Forward pass of the generator

        Args:
            z (torch.Tensor): Random noise tensor of shape (batch_size, latent_dim)

        Returns:
            torch.Tensor: Generated images of shape (batch_size, channels, img_size, img_size)
        """
        # Expand latent vector
        out = self.fc(z)

        # Reshape to (batch_size, 128, init_size, init_size)
        out = out.view(out.shape[0], 128, self.init_size, self.init_size)

        # Generate image
        img = self.model(out)

        return img


class Discriminator(nn.Module):
    """
    Discriminator network for GAN
    Takes an image and predicts if it's real or fake
    """

    def __init__(self, channels=1, img_size=28):
        """
        Initialize the Discriminator

        Args:
            channels (int): Number of image channels (1 for grayscale, 3 for RGB)
            img_size (int): Size of the input image (28, 32, 64, etc.)
        """
        super(Discriminator, self).__init__()

        self.channels = channels
        self.img_size = img_size
        self.img_shape = (channels, img_size, img_size)

        # Main discriminator network
        self.model = nn.Sequential(
            # Input layer - (channels, img_size, img_size) -> downsample by 2
            nn.Conv2d(channels, 32, 3, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(0.25),

            # Second layer - downsample by 2
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(0.25),

            # Third layer - downsample by 2
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(0.25),
        )

        # Calculate the flattened dimension dynamically
        # Create a dummy input to get the actual output size
        with torch.no_grad():
            dummy_input = torch.zeros(1, channels, img_size, img_size)
            dummy_output = self.model(dummy_input)
            self.adv_layer_dim = dummy_output.view(1, -1).size(1)

        # Output layer - Binary classification (real/fake)
        self.adv_layer = nn.Sequential(
            nn.Linear(self.adv_layer_dim, 1),
            nn.Sigmoid()  # Output range [0, 1]
        )

    def forward(self, img):
        """
        Forward pass of the discriminator

        Args:
            img (torch.Tensor): Input images of shape (batch_size, channels, img_size, img_size)

        Returns:
            torch.Tensor: Validity predictions of shape (batch_size, 1)
        """
        # Extract features
        out = self.model(img)

        # Flatten
        out = out.view(out.shape[0], -1)

        # Predict validity (real/fake)
        validity = self.adv_layer(out)

        return validity


def initialize_weights(model):
    """
    Initialize model weights using normal distribution

    Args:
        model (nn.Module): The model to initialize
    """
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
            nn.init.normal_(m.weight.data, 0.0, 0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias.data, 0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.normal_(m.weight.data, 1.0, 0.02)
            nn.init.constant_(m.bias.data, 0)


def test_models():
    """
    Test the Generator and Discriminator models
    """
    print("Testing GAN Models...")

    # Test parameters
    batch_size = 4
    latent_dim = 100
    channels = 1
    img_size = 64  # Test with 64x64

    # Create models
    generator = Generator(latent_dim=latent_dim, channels=channels, img_size=img_size)
    discriminator = Discriminator(channels=channels, img_size=img_size)

    # Initialize weights
    initialize_weights(generator)
    initialize_weights(discriminator)

    print(f"\nGenerator:")
    print(f"  Parameters: {sum(p.numel() for p in generator.parameters()):,}")

    print(f"\nDiscriminator:")
    print(f"  Parameters: {sum(p.numel() for p in discriminator.parameters()):,}")

    # Test generator
    z = torch.randn(batch_size, latent_dim)
    fake_imgs = generator(z)
    print(f"\nGenerator output shape: {fake_imgs.shape}")
    print(f"Generator output range: [{fake_imgs.min():.3f}, {fake_imgs.max():.3f}]")

    # Test discriminator
    validity = discriminator(fake_imgs)
    print(f"\nDiscriminator output shape: {validity.shape}")
    print(f"Discriminator output range: [{validity.min():.3f}, {validity.max():.3f}]")

    print("\nTests passed!")


if __name__ == "__main__":
    test_models()
