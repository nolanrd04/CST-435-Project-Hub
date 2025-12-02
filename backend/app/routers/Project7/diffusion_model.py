"""
Diffusion Model Components (U-Net + DDPM Scheduler)
Built from scratch using PyTorch for grayscale-to-RGB image colorization.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Tuple


class SinusoidalPositionEmbeddings(nn.Module):
    """Encodes timestep information for the diffusion process."""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, time: torch.Tensor) -> torch.Tensor:
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


class DoubleConv(nn.Module):
    """Two consecutive Conv2d -> BatchNorm -> ReLU blocks."""

    def __init__(self, in_channels: int, out_channels: int, mid_channels: int = None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels

        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # Handle size mismatch
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class ConditionalUNet(nn.Module):
    """
    U-Net for conditional diffusion model.

    Takes noisy RGB + grayscale condition + timestep -> predicts noise
    """

    def __init__(
        self,
        in_channels: int = 4,      # 3 (RGB) + 1 (grayscale condition)
        out_channels: int = 3,     # RGB noise prediction
        features: list = None,     # Feature channels at each level
        time_emb_dim: int = 256    # Timestep embedding dimension
    ):
        super().__init__()

        if features is None:
            features = [64, 128, 256, 512]

        self.in_channels = in_channels
        self.out_channels = out_channels

        # Timestep embedding
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.ReLU()
        )

        # Initial convolution
        self.inc = DoubleConv(in_channels, features[0])

        # Downsampling path
        self.down1 = Down(features[0], features[1])
        self.down2 = Down(features[1], features[2])
        self.down3 = Down(features[2], features[3])

        # Bottleneck with time embedding
        self.bottleneck = DoubleConv(features[3], features[3] * 2)
        self.time_emb_bottleneck = nn.Linear(time_emb_dim, features[3] * 2)

        # Upsampling path
        self.up1 = Up(features[3] * 2, features[3])
        self.up2 = Up(features[3], features[2])
        self.up3 = Up(features[2], features[1])
        self.up4 = Up(features[1], features[0])

        # Output convolution
        self.outc = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor (batch_size, 4, H, W) - noisy RGB + grayscale
            timestep: Timestep tensor (batch_size,)

        Returns:
            Predicted noise (batch_size, 3, H, W)
        """
        # Get time embedding
        t = self.time_mlp(timestep)

        # Encoder
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)

        # Bottleneck with time conditioning
        x5 = self.bottleneck(x4)
        # Add time embedding
        t_emb = self.time_emb_bottleneck(t)
        t_emb = t_emb[(..., ) + (None, ) * 2]  # Reshape for broadcasting
        x5 = x5 + t_emb

        # Decoder
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)

        # Output
        return self.outc(x)


class DDPMScheduler:
    """
    Denoising Diffusion Probabilistic Model (DDPM) noise scheduler.
    Implements the forward and reverse diffusion processes.
    """

    def __init__(
        self,
        timesteps: int = 1000,
        beta_start: float = 0.0001,
        beta_end: float = 0.02,
        device: str = 'cpu'
    ):
        self.timesteps = timesteps
        self.device = device

        # Linear beta schedule
        self.betas = torch.linspace(beta_start, beta_end, timesteps, device=device)

        # Pre-compute alphas and related terms
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)

        # Calculations for diffusion q(x_t | x_{t-1})
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)

        # Calculations for posterior q(x_{t-1} | x_t, x_0)
        self.sqrt_recip_alphas = torch.sqrt(1.0 / self.alphas)
        self.posterior_variance = self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)

    def add_noise(
        self,
        x_start: torch.Tensor,
        noise: torch.Tensor,
        timesteps: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward diffusion process: add noise to clean images.

        q(x_t | x_0) = sqrt(alpha_cumprod_t) * x_0 + sqrt(1 - alpha_cumprod_t) * noise
        """
        sqrt_alpha_prod = self.sqrt_alphas_cumprod[timesteps]
        sqrt_one_minus_alpha_prod = self.sqrt_one_minus_alphas_cumprod[timesteps]

        # Reshape for broadcasting (batch_size, 1, 1, 1)
        sqrt_alpha_prod = sqrt_alpha_prod.flatten()
        while len(sqrt_alpha_prod.shape) < len(x_start.shape):
            sqrt_alpha_prod = sqrt_alpha_prod.unsqueeze(-1)

        sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.flatten()
        while len(sqrt_one_minus_alpha_prod.shape) < len(x_start.shape):
            sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.unsqueeze(-1)

        return sqrt_alpha_prod * x_start + sqrt_one_minus_alpha_prod * noise

    def sample_timesteps(self, batch_size: int) -> torch.Tensor:
        """Sample random timesteps for training."""
        return torch.randint(0, self.timesteps, (batch_size,), device=self.device, dtype=torch.long)

    def denoise_step(
        self,
        x_t: torch.Tensor,
        predicted_noise: torch.Tensor,
        timestep: int
    ) -> torch.Tensor:
        """
        Single reverse diffusion step: remove predicted noise.

        p(x_{t-1} | x_t) = N(mean, variance)
        """
        t = timestep

        # Get coefficients
        beta_t = self.betas[t]
        sqrt_one_minus_alpha_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t]
        sqrt_recip_alpha_t = self.sqrt_recip_alphas[t]

        # Predict x_0
        model_mean = sqrt_recip_alpha_t * (
            x_t - beta_t / sqrt_one_minus_alpha_cumprod_t * predicted_noise
        )

        if t == 0:
            return model_mean
        else:
            posterior_variance_t = self.posterior_variance[t]
            noise = torch.randn_like(x_t)
            return model_mean + torch.sqrt(posterior_variance_t) * noise

    @torch.no_grad()
    def generate(
        self,
        model: nn.Module,
        grayscale: torch.Tensor,
        device: str = 'cpu'
    ) -> torch.Tensor:
        """
        Generate RGB image from grayscale using reverse diffusion.

        Args:
            model: Trained U-Net model
            grayscale: Grayscale condition (batch_size, 1, H, W)
            device: Device to run on

        Returns:
            Generated RGB image (batch_size, 3, H, W)
        """
        model.eval()
        batch_size = grayscale.shape[0]

        # Start from pure noise
        img = torch.randn(batch_size, 3, grayscale.shape[2], grayscale.shape[3], device=device)

        # Reverse diffusion process
        for t in reversed(range(self.timesteps)):
            # Prepare input: concatenate noisy RGB + grayscale
            t_batch = torch.full((batch_size,), t, device=device, dtype=torch.long)
            model_input = torch.cat([img, grayscale], dim=1)

            # Predict noise
            predicted_noise = model(model_input, t_batch)

            # Remove noise
            img = self.denoise_step(img, predicted_noise, t)

        return img


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters in model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
