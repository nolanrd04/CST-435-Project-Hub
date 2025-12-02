"""
FastAPI router for Project7 Diffusion Colorization endpoints
"""
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import os
import json
import base64
from pathlib import Path
import torch
import numpy as np
from PIL import Image
import io

# Import diffusion modules
from .diffusion_model import ConditionalUNet, DDPMScheduler

router = APIRouter(prefix="/project7", tags=["project7"])

# Base paths
PROJECT_DIR = Path(__file__).parent
MODELS_DIR = PROJECT_DIR / "models"

# Pydantic Models
class ColorizeRequest(BaseModel):
    model_name: str
    image: str  # Base64 encoded image
    num_inference_steps: Optional[int] = 1000


class ColorizeResponse(BaseModel):
    success: bool
    grayscale_image: str  # Base64
    colorized_image: str  # Base64
    model_name: str
    inference_time_seconds: float


class ModelInfo(BaseModel):
    model_name: str
    config: Dict[str, Any]
    training_progress: Optional[Dict[str, Any]] = None
    model_parameters: int
    image_size: int


class TrainingHistory(BaseModel):
    epochs: List[int]
    train_loss: List[float]
    val_loss: List[float]
    epoch_times: List[float]


class CostAnalysis(BaseModel):
    model_name: str
    training_cost_breakdown: Dict[str, float]
    cost_per_epoch: Dict[str, Any]
    peak_memory_gb: float
    training_hours: float
    cpus_used: float
    gpu_used: bool


# Helper Functions
def get_available_models() -> List[str]:
    """Get list of available trained models"""
    if not MODELS_DIR.exists():
        return []

    models = []
    for item in MODELS_DIR.iterdir():
        if item.is_dir():
            # Check if model has checkpoints or best_model.pth
            if (item / "best_model.pth").exists() or list(item.glob("checkpoint_epoch_*.pth")):
                models.append(item.name)

    return sorted(models)


def load_model(model_name: str, device: str = "cpu"):
    """Load a trained diffusion model"""
    model_dir = MODELS_DIR / model_name

    if not model_dir.exists():
        raise HTTPException(status_code=404, detail=f"Model {model_name} not found")

    # Load config
    config_file = model_dir / "config.json"
    if not config_file.exists():
        raise HTTPException(status_code=404, detail=f"Config not found for {model_name}")

    with open(config_file, 'r') as f:
        config = json.load(f)

    # Create model
    model = ConditionalUNet(
        in_channels=4,
        out_channels=3,
        features=config['features'],
        time_emb_dim=config['time_emb_dim']
    )

    # Load weights
    checkpoint_path = model_dir / "best_model.pth"
    if not checkpoint_path.exists():
        # Try to find latest checkpoint
        checkpoints = sorted(model_dir.glob("checkpoint_epoch_*.pth"))
        if not checkpoints:
            raise HTTPException(status_code=404, detail=f"No checkpoints found for {model_name}")
        checkpoint_path = checkpoints[-1]

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    # Create scheduler
    scheduler = DDPMScheduler(
        timesteps=config['timesteps'],
        device=device
    )

    return model, scheduler, config


def preprocess_image(image_b64: str) -> np.ndarray:
    """
    Preprocess base64 image to grayscale tensor
    Returns: numpy array (64, 64, 1) in range [-1, 1]
    """
    # Decode base64
    image_data = base64.b64decode(image_b64)
    image = Image.open(io.BytesIO(image_data)).convert('L')  # Convert to grayscale

    # Resize to 64x64
    image = image.resize((64, 64), Image.Resampling.LANCZOS)

    # Convert to numpy array and normalize to [-1, 1]
    image_np = np.array(image).astype(np.float32)
    image_np = (image_np / 127.5) - 1.0  # [0, 255] -> [-1, 1]

    # Add channel dimension: (64, 64) -> (64, 64, 1)
    image_np = image_np[:, :, np.newaxis]

    return image_np


def tensor_to_base64(tensor: torch.Tensor) -> str:
    """
    Convert tensor to base64 image
    Input: tensor (3, 64, 64) or (1, 64, 64) in range [-1, 1]
    Output: base64 PNG string
    """
    # Convert to numpy
    if tensor.dim() == 4:
        tensor = tensor.squeeze(0)  # Remove batch dimension

    # Move to CPU and denormalize
    image_np = tensor.cpu().numpy()
    image_np = (image_np + 1.0) / 2.0  # [-1, 1] -> [0, 1]
    image_np = (image_np * 255).astype(np.uint8)

    # Convert to HWC format if needed
    if image_np.shape[0] in [1, 3]:  # CHW format
        image_np = np.transpose(image_np, (1, 2, 0))

    # Handle grayscale
    if image_np.shape[2] == 1:
        image_np = image_np.squeeze(2)

    # Convert to PIL Image
    if len(image_np.shape) == 2:
        image = Image.fromarray(image_np, mode='L')
    else:
        image = Image.fromarray(image_np, mode='RGB')

    # Encode to base64
    buffer = io.BytesIO()
    image.save(buffer, format='PNG')
    image_b64 = base64.b64encode(buffer.getvalue()).decode('utf-8')

    return image_b64


# API Endpoints
@router.get("/models", response_model=List[str])
async def list_models():
    """Get list of available trained models"""
    return get_available_models()


@router.get("/models/{model_name}/info", response_model=ModelInfo)
async def get_model_info(model_name: str):
    """Get detailed information about a specific model"""
    model_dir = MODELS_DIR / model_name

    if not model_dir.exists():
        raise HTTPException(status_code=404, detail=f"Model {model_name} not found")

    # Load config
    config_file = model_dir / "config.json"
    if not config_file.exists():
        raise HTTPException(status_code=404, detail=f"Config not found for {model_name}")

    with open(config_file, 'r') as f:
        config = json.load(f)

    # Count model parameters
    temp_model = ConditionalUNet(
        in_channels=4,
        out_channels=3,
        features=config['features'],
        time_emb_dim=config['time_emb_dim']
    )
    model_parameters = sum(p.numel() for p in temp_model.parameters())

    # Load training history if available
    history_file = model_dir / "training_history.json"
    training_progress = None

    if history_file.exists():
        with open(history_file, 'r') as f:
            history = json.load(f)

            if history['train_loss'] and history['val_loss']:
                current_epoch = len(history['train_loss'])
                training_progress = {
                    "current_epoch": current_epoch,
                    "total_epochs": config['num_epochs'],
                    "train_loss": history['train_loss'][-1],
                    "val_loss": history['val_loss'][-1],
                    "best_val_loss": min(history['val_loss'])
                }

    return ModelInfo(
        model_name=model_name,
        config=config,
        training_progress=training_progress,
        model_parameters=model_parameters,
        image_size=64  # Fixed for Tiny ImageNet
    )


@router.get("/models/{model_name}/training-history", response_model=TrainingHistory)
async def get_training_history(model_name: str):
    """Get training loss history for visualization"""
    model_dir = MODELS_DIR / model_name
    history_file = model_dir / "training_history.json"

    if not history_file.exists():
        raise HTTPException(status_code=404, detail=f"Training history not found for {model_name}")

    with open(history_file, 'r') as f:
        history = json.load(f)

    # Generate epoch list
    epochs = list(range(1, len(history['train_loss']) + 1))

    return TrainingHistory(
        epochs=epochs,
        train_loss=history['train_loss'],
        val_loss=history['val_loss'],
        epoch_times=history.get('epoch_times', [])
    )


@router.get("/models/{model_name}/cost-analysis", response_model=CostAnalysis)
async def get_cost_analysis(model_name: str):
    """Get training cost analysis"""
    model_dir = MODELS_DIR / model_name
    cost_file = model_dir / "cost_analysis_report.json"

    if not cost_file.exists():
        raise HTTPException(status_code=404, detail=f"Cost analysis not found for {model_name}")

    with open(cost_file, 'r') as f:
        cost_data = json.load(f)

    training_cost = cost_data.get("training_cost", {})
    cost_per_epoch = cost_data.get("cost_per_epoch", {})
    system_info = cost_data.get("system_info", {})

    return CostAnalysis(
        model_name=model_name,
        training_cost_breakdown={
            "compute_cost": training_cost.get("compute", 0.0),
            "memory_cost": training_cost.get("memory", 0.0),
            "storage_cost": training_cost.get("storage", 0.0),
            "total_cost": training_cost.get("total", 0.0)
        },
        cost_per_epoch={
            "avg_cost_per_epoch": cost_per_epoch.get("cost_per_epoch", 0.0),
            "total_epochs": cost_per_epoch.get("total_epochs", 0)
        },
        peak_memory_gb=training_cost.get("peak_memory_gb", 0.0),
        training_hours=training_cost.get("training_hours", 0.0),
        cpus_used=system_info.get("cpus", 0.0),
        gpu_used=system_info.get("gpu", False)
    )


@router.post("/colorize", response_model=ColorizeResponse)
async def colorize_image(request: ColorizeRequest):
    """Colorize a grayscale image using trained diffusion model"""
    import time

    # Check if CUDA is available
    device = "cuda" if torch.cuda.is_available() else "cpu"

    start_time = time.time()

    # Load model
    model, scheduler, config = load_model(request.model_name, device=device)

    # Preprocess image
    grayscale_np = preprocess_image(request.image)

    # Convert to torch tensor
    grayscale_tensor = torch.from_numpy(grayscale_np).permute(2, 0, 1).unsqueeze(0).to(device)  # (1, 1, 64, 64)

    # Generate colorized image
    with torch.no_grad():
        # Start from pure noise
        rgb_noisy = torch.randn(1, 3, 64, 64, device=device)

        # Use fewer steps if requested
        num_steps = min(request.num_inference_steps, scheduler.timesteps)
        step_size = scheduler.timesteps // num_steps

        # Reverse diffusion process
        for i in range(num_steps - 1, -1, -1):
            t = i * step_size
            t_batch = torch.full((1,), t, device=device, dtype=torch.long)

            # Concatenate noisy RGB + grayscale condition
            model_input = torch.cat([rgb_noisy, grayscale_tensor], dim=1)  # (1, 4, 64, 64)

            # Predict noise
            predicted_noise = model(model_input, t_batch)

            # Denoise step
            rgb_noisy = scheduler.denoise_step(rgb_noisy, predicted_noise, t)

        colorized_tensor = rgb_noisy

    # Convert tensors to base64
    grayscale_b64 = tensor_to_base64(grayscale_tensor)
    colorized_b64 = tensor_to_base64(colorized_tensor)

    inference_time = time.time() - start_time

    return ColorizeResponse(
        success=True,
        grayscale_image=grayscale_b64,
        colorized_image=colorized_b64,
        model_name=request.model_name,
        inference_time_seconds=inference_time
    )
