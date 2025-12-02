"""
Perceptual Loss and GAN Discriminator for improved colorization quality.

This module provides:
1. VGG-based perceptual loss to measure color similarity in feature space
2. Optional PatchGAN discriminator for adversarial training
"""

import torch
import torch.nn as nn
import torchvision.models as models
from typing import Tuple, Optional


class VGGPerceptualLoss(nn.Module):
    """
    Perceptual loss using VGG16 features.

    This loss compares images in VGG feature space instead of pixel space,
    which better captures perceptual similarity (e.g., color accuracy).

    Uses layers: relu1_2, relu2_2, relu3_3, relu4_3
    """

    def __init__(self, device: str = 'cpu'):
        super().__init__()

        # Load pretrained VGG16
        try:
            # Try new API first (torchvision >= 0.13)
            from torchvision.models import VGG16_Weights
            vgg = models.vgg16(weights=VGG16_Weights.IMAGENET1K_V1).features.eval()
        except ImportError:
            # Fallback to old API
            vgg = models.vgg16(pretrained=True).features.eval()

        # Extract feature layers
        # relu1_2 (layer 4), relu2_2 (layer 9), relu3_3 (layer 16), relu4_3 (layer 23)
        self.slice1 = nn.Sequential(*list(vgg.children())[:4])   # relu1_2
        self.slice2 = nn.Sequential(*list(vgg.children())[:9])   # relu2_2
        self.slice3 = nn.Sequential(*list(vgg.children())[:16])  # relu3_3
        self.slice4 = nn.Sequential(*list(vgg.children())[:23])  # relu4_3

        # Freeze VGG parameters
        for param in self.parameters():
            param.requires_grad = False

        self.to(device)

        # VGG normalization (ImageNet stats)
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def normalize_vgg_input(self, x: torch.Tensor) -> torch.Tensor:
        """
        Normalize input from [-1, 1] to VGG's expected [0, 1] with ImageNet stats.
        """
        # Convert from [-1, 1] to [0, 1]
        x = (x + 1.0) / 2.0

        # Apply ImageNet normalization
        x = (x - self.mean) / self.std

        return x

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Calculate perceptual loss between predicted and target images.

        Args:
            pred: Predicted RGB image (B, 3, H, W) in range [-1, 1]
            target: Target RGB image (B, 3, H, W) in range [-1, 1]

        Returns:
            Perceptual loss (scalar)
        """
        # Normalize inputs
        pred = self.normalize_vgg_input(pred)
        target = self.normalize_vgg_input(target)

        # Extract features
        pred_relu1 = self.slice1(pred)
        pred_relu2 = self.slice2(pred)
        pred_relu3 = self.slice3(pred)
        pred_relu4 = self.slice4(pred)

        target_relu1 = self.slice1(target)
        target_relu2 = self.slice2(target)
        target_relu3 = self.slice3(target)
        target_relu4 = self.slice4(target)

        # Calculate MSE in feature space
        loss = 0.0
        loss += nn.functional.mse_loss(pred_relu1, target_relu1)
        loss += nn.functional.mse_loss(pred_relu2, target_relu2)
        loss += nn.functional.mse_loss(pred_relu3, target_relu3)
        loss += nn.functional.mse_loss(pred_relu4, target_relu4)

        return loss


class PatchGANDiscriminator(nn.Module):
    """
    PatchGAN discriminator for adversarial training.

    Instead of classifying the entire image as real/fake, PatchGAN classifies
    NxN patches, which helps with local color accuracy.

    Architecture: C64-C128-C256-C512 (similar to pix2pix)
    """

    def __init__(self, input_channels: int = 4):
        """
        Args:
            input_channels: Number of input channels (grayscale + RGB = 4)
        """
        super().__init__()

        def discriminator_block(in_channels, out_channels, normalization=True):
            """Basic discriminator block: Conv -> BatchNorm -> LeakyReLU"""
            layers = [nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1)]
            if normalization:
                layers.append(nn.BatchNorm2d(out_channels))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        # Build discriminator
        self.model = nn.Sequential(
            # Input: (B, 4, 64, 64) - grayscale + RGB
            *discriminator_block(input_channels, 64, normalization=False),  # -> (B, 64, 32, 32)
            *discriminator_block(64, 128),                                   # -> (B, 128, 16, 16)
            *discriminator_block(128, 256),                                  # -> (B, 256, 8, 8)
            *discriminator_block(256, 512),                                  # -> (B, 512, 4, 4)

            # Final layer: classify each patch
            nn.Conv2d(512, 1, kernel_size=4, stride=1, padding=1)           # -> (B, 1, 3, 3)
            # Output: (B, 1, 3, 3) - each value is real/fake for a patch
        )

    def forward(self, grayscale: torch.Tensor, rgb: torch.Tensor) -> torch.Tensor:
        """
        Classify if the RGB colorization is real or fake given grayscale input.

        Args:
            grayscale: Grayscale input (B, 1, 64, 64)
            rgb: RGB colorization (B, 3, 64, 64)

        Returns:
            Patch predictions (B, 1, 3, 3) - real/fake for each patch
        """
        x = torch.cat([grayscale, rgb], dim=1)  # (B, 4, 64, 64)
        return self.model(x)


class CombinedLoss:
    """
    Combined loss function for improved colorization.

    Combines:
    1. MSE noise loss (original diffusion loss)
    2. Perceptual loss (VGG features)
    3. Optional adversarial loss (GAN discriminator)
    """

    def __init__(
        self,
        device: str = 'cpu',
        mse_weight: float = 1.0,
        perceptual_weight: float = 0.1,
        adversarial_weight: float = 0.01,
        use_gan: bool = False
    ):
        """
        Args:
            device: Device to run on
            mse_weight: Weight for MSE noise loss
            perceptual_weight: Weight for perceptual loss
            adversarial_weight: Weight for adversarial loss
            use_gan: Whether to use GAN discriminator
        """
        self.device = device
        self.mse_weight = mse_weight
        self.perceptual_weight = perceptual_weight
        self.adversarial_weight = adversarial_weight
        self.use_gan = use_gan

        # Initialize perceptual loss only if needed (lazy loading)
        self.perceptual_loss = None
        if self.perceptual_weight > 0:
            self.perceptual_loss = VGGPerceptualLoss(device=device)

        # Initialize discriminator if using GAN
        self.discriminator = None
        self.discriminator_optimizer = None
        if use_gan:
            self.discriminator = PatchGANDiscriminator(input_channels=4).to(device)
            self.discriminator_optimizer = torch.optim.Adam(
                self.discriminator.parameters(),
                lr=0.0002,
                betas=(0.5, 0.999)
            )
            self.bce_loss = nn.BCEWithLogitsLoss()

    def compute_generator_loss(
        self,
        predicted_noise: torch.Tensor,
        actual_noise: torch.Tensor,
        predicted_rgb: torch.Tensor,
        target_rgb: torch.Tensor,
        grayscale: torch.Tensor
    ) -> Tuple[torch.Tensor, dict]:
        """
        Compute combined generator loss.

        Args:
            predicted_noise: Model's predicted noise (B, 3, 64, 64)
            actual_noise: Actual noise added (B, 3, 64, 64)
            predicted_rgb: Denoised RGB prediction (B, 3, 64, 64)
            target_rgb: Ground truth RGB (B, 3, 64, 64)
            grayscale: Grayscale condition (B, 1, 64, 64)

        Returns:
            total_loss: Combined loss
            loss_dict: Dictionary with individual loss components
        """
        losses = {}

        # 1. MSE noise loss (original diffusion objective)
        mse_loss = nn.functional.mse_loss(predicted_noise, actual_noise)
        losses['mse_loss'] = mse_loss.item()
        total_loss = self.mse_weight * mse_loss

        # 2. Perceptual loss (VGG features)
        if self.perceptual_weight > 0 and self.perceptual_loss is not None:
            perc_loss = self.perceptual_loss(predicted_rgb, target_rgb)
            losses['perceptual_loss'] = perc_loss.item()
            total_loss += self.perceptual_weight * perc_loss

        # 3. Adversarial loss (fool discriminator)
        if self.use_gan and self.discriminator is not None:
            fake_pred = self.discriminator(grayscale, predicted_rgb)
            # Try to fool discriminator (fake should be classified as real)
            real_labels = torch.ones_like(fake_pred)
            adv_loss = self.bce_loss(fake_pred, real_labels)
            losses['adversarial_loss'] = adv_loss.item()
            total_loss += self.adversarial_weight * adv_loss

        losses['total_loss'] = total_loss.item()
        return total_loss, losses

    def train_discriminator(
        self,
        grayscale: torch.Tensor,
        real_rgb: torch.Tensor,
        fake_rgb: torch.Tensor
    ) -> Optional[float]:
        """
        Train discriminator to distinguish real from fake colorizations.

        Args:
            grayscale: Grayscale input (B, 1, 64, 64)
            real_rgb: Real RGB images (B, 3, 64, 64)
            fake_rgb: Fake RGB colorizations (B, 3, 64, 64)

        Returns:
            Discriminator loss value (None if not using GAN)
        """
        if not self.use_gan or self.discriminator is None:
            return None

        self.discriminator_optimizer.zero_grad()

        # Real images
        real_pred = self.discriminator(grayscale, real_rgb)
        real_labels = torch.ones_like(real_pred)
        real_loss = self.bce_loss(real_pred, real_labels)

        # Fake images (detach to avoid backprop through generator)
        fake_pred = self.discriminator(grayscale, fake_rgb.detach())
        fake_labels = torch.zeros_like(fake_pred)
        fake_loss = self.bce_loss(fake_pred, fake_labels)

        # Total discriminator loss
        d_loss = (real_loss + fake_loss) * 0.5

        d_loss.backward()
        self.discriminator_optimizer.step()

        return d_loss.item()

    def state_dict(self) -> dict:
        """Get state dict for saving."""
        state = {
            'mse_weight': self.mse_weight,
            'perceptual_weight': self.perceptual_weight,
            'adversarial_weight': self.adversarial_weight,
            'use_gan': self.use_gan
        }
        if self.use_gan and self.discriminator is not None:
            state['discriminator_state_dict'] = self.discriminator.state_dict()
            state['discriminator_optimizer_state_dict'] = self.discriminator_optimizer.state_dict()
        return state

    def load_state_dict(self, state: dict):
        """Load state dict."""
        self.mse_weight = state.get('mse_weight', 1.0)
        self.perceptual_weight = state.get('perceptual_weight', 0.1)
        self.adversarial_weight = state.get('adversarial_weight', 0.01)
        self.use_gan = state.get('use_gan', False)

        if self.use_gan and 'discriminator_state_dict' in state:
            if self.discriminator is None:
                self.discriminator = PatchGANDiscriminator(input_channels=4).to(self.device)
                self.discriminator_optimizer = torch.optim.Adam(
                    self.discriminator.parameters(),
                    lr=0.0002,
                    betas=(0.5, 0.999)
                )
            self.discriminator.load_state_dict(state['discriminator_state_dict'])
            self.discriminator_optimizer.load_state_dict(state['discriminator_optimizer_state_dict'])
