"""
GAN Model Architecture
Generator and Discriminator networks for fruit image generation
"""

import torch
import torch.nn as nn
import torch.optim as optim


class Generator(nn.Module):
    """
    Generator Network
    Transforms noise into images that resemble real fruit images.
    
    Architecture:
    - Input: Latent vector (noise) from normal distribution
    - Layer 1: Fully connected + LeakyReLU + BatchNorm
    - Layer 2: Fully connected + LeakyReLU + BatchNorm
    - Layer 3: Fully connected + LeakyReLU + BatchNorm
    - Output: Image tensor reshaped to 28x28x3
    """
    
    def __init__(self, latent_dim=100, img_size=28, channels=3):
        """
        Initialize Generator
        
        Args:
            latent_dim (int): Dimension of noise input (default 100)
            img_size (int): Size of output image (default 28)
            channels (int): Number of image channels (default 3 for RGB)
        """
        super(Generator, self).__init__()
        
        self.latent_dim = latent_dim
        self.img_size = img_size
        self.channels = channels
        self.output_size = img_size * img_size * channels
        
        # Hidden layer dimensions
        hidden1 = 256
        hidden2 = 512
        hidden3 = 1024
        
        # Build the network
        self.fc1 = nn.Linear(latent_dim, hidden1)
        self.bn1 = nn.BatchNorm1d(hidden1)
        self.leaky_relu1 = nn.LeakyReLU(0.2, inplace=True)
        
        self.fc2 = nn.Linear(hidden1, hidden2)
        self.bn2 = nn.BatchNorm1d(hidden2)
        self.leaky_relu2 = nn.LeakyReLU(0.2, inplace=True)
        
        self.fc3 = nn.Linear(hidden2, hidden3)
        self.bn3 = nn.BatchNorm1d(hidden3)
        self.leaky_relu3 = nn.LeakyReLU(0.2, inplace=True)
        
        # Output layer - use tanh to scale output to [-1, 1]
        self.fc_out = nn.Linear(hidden3, self.output_size)
        self.tanh = nn.Tanh()
    
    def forward(self, noise):
        """
        Forward pass through generator
        
        Args:
            noise (torch.Tensor): Batch of noise vectors shape (batch_size, latent_dim)
            
        Returns:
            torch.Tensor: Generated images shape (batch_size, channels, img_size, img_size)
        """
        # Layer 1
        x = self.fc1(noise)
        x = self.bn1(x)
        x = self.leaky_relu1(x)
        
        # Layer 2
        x = self.fc2(x)
        x = self.bn2(x)
        x = self.leaky_relu2(x)
        
        # Layer 3
        x = self.fc3(x)
        x = self.bn3(x)
        x = self.leaky_relu3(x)
        
        # Output layer
        x = self.fc_out(x)
        x = self.tanh(x)
        
        # Reshape to image format
        x = x.view(x.size(0), self.channels, self.img_size, self.img_size)
        
        return x


class Discriminator(nn.Module):
    """
    Discriminator Network
    Classifies images as real or fake.
    
    Architecture:
    - Input: Image tensor (28x28x3)
    - Layer 1: Fully connected + LeakyReLU + BatchNorm
    - Layer 2: Fully connected + LeakyReLU + BatchNorm
    - Layer 3: Fully connected + LeakyReLU + BatchNorm
    - Output: Single value (real=1, fake=0)
    """
    
    def __init__(self, img_size=28, channels=3):
        """
        Initialize Discriminator
        
        Args:
            img_size (int): Size of input image (default 28)
            channels (int): Number of image channels (default 3 for RGB)
        """
        super(Discriminator, self).__init__()
        
        self.img_size = img_size
        self.channels = channels
        self.input_size = img_size * img_size * channels
        
        # Hidden layer dimensions
        hidden1 = 1024
        hidden2 = 512
        hidden3 = 256
        
        # Build the network
        self.fc1 = nn.Linear(self.input_size, hidden1)
        self.leaky_relu1 = nn.LeakyReLU(0.2, inplace=True)
        self.bn1 = nn.BatchNorm1d(hidden1)
        
        self.fc2 = nn.Linear(hidden1, hidden2)
        self.leaky_relu2 = nn.LeakyReLU(0.2, inplace=True)
        self.bn2 = nn.BatchNorm1d(hidden2)
        
        self.fc3 = nn.Linear(hidden2, hidden3)
        self.leaky_relu3 = nn.LeakyReLU(0.2, inplace=True)
        self.bn3 = nn.BatchNorm1d(hidden3)
        
        # Output layer - single output for binary classification
        self.fc_out = nn.Linear(hidden3, 1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, img):
        """
        Forward pass through discriminator
        
        Args:
            img (torch.Tensor): Batch of images shape (batch_size, channels, img_size, img_size)
            
        Returns:
            torch.Tensor: Classification scores shape (batch_size, 1)
        """
        # Flatten image
        x = img.view(img.size(0), -1)
        
        # Layer 1
        x = self.fc1(x)
        x = self.leaky_relu1(x)
        x = self.bn1(x)
        
        # Layer 2
        x = self.fc2(x)
        x = self.leaky_relu2(x)
        x = self.bn2(x)
        
        # Layer 3
        x = self.fc3(x)
        x = self.leaky_relu3(x)
        x = self.bn3(x)
        
        # Output layer
        x = self.fc_out(x)
        x = self.sigmoid(x)
        
        return x


def create_gan_models(latent_dim=100, img_size=28, channels=1, device='cpu'):
    """
    Create Generator and Discriminator models
    
    Args:
        latent_dim (int): Dimension of noise input
        img_size (int): Size of output image
        channels (int): Number of image channels (auto-detected from data)
        device (str): Device to place models on ('cpu' or 'cuda')
        
    Returns:
        tuple: (generator, discriminator)
    """
    generator = Generator(latent_dim=latent_dim, img_size=img_size, channels=channels)
    discriminator = Discriminator(img_size=img_size, channels=channels)
    
    generator = generator.to(device)
    discriminator = discriminator.to(device)
    
    return generator, discriminator


def create_optimizers(generator, discriminator, learning_rate=0.0002, beta1=0.5, beta2=0.999):
    """
    Create optimizers for both networks
    
    Args:
        generator (nn.Module): Generator network
        discriminator (nn.Module): Discriminator network
        learning_rate (float): Learning rate for Adam optimizer
        beta1 (float): Beta1 parameter for Adam
        beta2 (float): Beta2 parameter for Adam
        
    Returns:
        tuple: (generator_optimizer, discriminator_optimizer)
    """
    gen_optimizer = optim.Adam(generator.parameters(), lr=learning_rate, betas=(beta1, beta2))
    disc_optimizer = optim.Adam(discriminator.parameters(), lr=learning_rate, betas=(beta1, beta2))
    
    return gen_optimizer, disc_optimizer


if __name__ == "__main__":
    # Test the models
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create models
    generator, discriminator = create_gan_models(device=device)
    
    print("Generator Model:")
    print(generator)
    print(f"\nTotal Generator Parameters: {sum(p.numel() for p in generator.parameters()):,}")
    
    print("\n" + "="*50)
    print("Discriminator Model:")
    print(discriminator)
    print(f"\nTotal Discriminator Parameters: {sum(p.numel() for p in discriminator.parameters()):,}")
    
    # Test forward pass
    print("\n" + "="*50)
    print("Testing forward passes...")
    
    batch_size = 8
    latent_dim = 100
    
    noise = torch.randn(batch_size, latent_dim, device=device)
    generated_images = generator(noise)
    print(f"Generated images shape: {generated_images.shape}")
    
    discriminator_output = discriminator(generated_images)
    print(f"Discriminator output shape: {discriminator_output.shape}")
    print("Forward pass test successful!")
