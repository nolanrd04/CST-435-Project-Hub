"""
FastAPI router for Project9 Image Colorizer endpoints
"""
from fastapi import APIRouter, HTTPException, File, UploadFile
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
import torchvision.transforms as transforms

# Import the U-Net colorizer model
from .model import UNetColorizer

router = APIRouter(prefix="/project9", tags=["project9"])

# Base paths
PROJECT_DIR = Path(__file__).parent
MODELS_DIR = PROJECT_DIR / "models"


# Pydantic Models
class ColorizeImageRequest(BaseModel):
    image: str  # Base64 encoded image
    model_name: str


class ColorizeImageResponse(BaseModel):
    success: bool
    colorized_image: str  # Base64 encoded
    grayscale_image: str  # Base64 encoded (echoed back)
    model_name: str


class ModelInfo(BaseModel):
    model_name: str
    description: Optional[str] = None
    training_config: Optional[Dict[str, Any]] = None
    image_size: int
    total_parameters: Optional[int] = None
    training_stats: Optional[Dict[str, Any]] = None


class TrainingCostAnalysis(BaseModel):
    training_cost_breakdown: Dict[str, float]
    cost_per_epoch: Dict[str, Any]
    peak_memory_gb: float
    training_hours: float


# Helper Functions
def get_available_models() -> List[str]:
    """Get list of available trained models"""
    if not MODELS_DIR.exists():
        return []

    models = []
    for item in MODELS_DIR.iterdir():
        if item.is_dir():
            # Check if it has a saved model file
            saved_models_dir = item / "saved_models"
            if saved_models_dir.exists():
                # Look for best model checkpoint
                best_model = saved_models_dir / f"{item.name}_best.pth"
                if best_model.exists():
                    models.append(item.name)

    return sorted(models)


def get_model_info_data(model_name: str) -> Dict[str, Any]:
    """Load model information from model directory"""
    model_dir = MODELS_DIR / model_name
    saved_models_dir = model_dir / "saved_models"

    if not model_dir.exists():
        raise HTTPException(status_code=404, detail=f"Model not found: {model_name}")

    # Load config
    config_file = saved_models_dir / "model_config.json"
    training_config = None
    image_size = 256
    total_parameters = None

    if config_file.exists():
        with open(config_file, 'r') as f:
            training_config = json.load(f)
            image_size = training_config.get('image_size', 256)

    # Load training history if available
    history_file = saved_models_dir / "training_history.json"
    training_stats = None

    if history_file.exists():
        with open(history_file, 'r') as f:
            history = json.load(f)
            training_stats = {
                "total_epochs": len(history.get('train_loss', [])),
                "best_val_loss": min(history.get('val_loss', [float('inf')])) if history.get('val_loss') else None,
                "final_train_loss": history.get('train_loss', [])[-1] if history.get('train_loss') else None,
                "final_val_loss": history.get('val_loss', [])[-1] if history.get('val_loss') else None,
            }

    # Try to get parameter count from config or calculate it
    if training_config:
        # Create a dummy model to count parameters
        try:
            device = torch.device('cpu')
            model = UNetColorizer(training_config)
            total_parameters = model.count_parameters()
        except Exception as e:
            print(f"Warning: Could not count parameters: {e}")
            total_parameters = None

    # Description
    description = training_config.get('description', 'U-Net based fruit image colorizer') if training_config else 'U-Net based fruit image colorizer'

    return {
        "model_name": model_name,
        "description": description,
        "training_config": training_config,
        "image_size": image_size,
        "total_parameters": total_parameters,
        "training_stats": training_stats
    }


def load_model(model_name: str, device: torch.device):
    """Load a trained model from disk"""
    model_dir = MODELS_DIR / model_name
    saved_models_dir = model_dir / "saved_models"

    # Load config
    config_file = saved_models_dir / "model_config.json"
    if not config_file.exists():
        raise HTTPException(status_code=404, detail=f"Model config not found for {model_name}")

    with open(config_file, 'r') as f:
        config = json.load(f)

    # Load best model checkpoint
    model_path = saved_models_dir / f"{model_name}_best.pth"
    if not model_path.exists():
        raise HTTPException(status_code=404, detail=f"Model checkpoint not found for {model_name}")

    # Create model
    model = UNetColorizer(config)

    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    return model, config


def preprocess_image(image_data: str, target_size: int = 256):
    """
    Preprocess uploaded image for colorization

    Args:
        image_data: Base64 encoded image
        target_size: Size to resize image to

    Returns:
        tuple: (grayscale_tensor, grayscale_pil)
    """
    # Decode base64
    image_bytes = base64.b64decode(image_data)
    image = Image.open(io.BytesIO(image_bytes)).convert('RGB')

    # Resize
    resize_transform = transforms.Resize((target_size, target_size))
    image = resize_transform(image)

    # Convert to grayscale
    grayscale_image = image.convert('L')

    # Convert to tensor
    to_tensor = transforms.ToTensor()
    grayscale_tensor = to_tensor(grayscale_image).unsqueeze(0)  # Add batch dimension

    return grayscale_tensor, grayscale_image


def postprocess_image(output_tensor: torch.Tensor) -> Image.Image:
    """
    Convert model output tensor to PIL Image

    Args:
        output_tensor: Model output (B, 3, H, W) in range [0, 1]

    Returns:
        PIL Image
    """
    # Remove batch dimension and convert to numpy
    output_np = output_tensor.squeeze(0).cpu().numpy()  # (3, H, W)

    # Transpose to (H, W, 3)
    output_np = np.transpose(output_np, (1, 2, 0))

    # Convert to [0, 255]
    output_np = (output_np * 255).clip(0, 255).astype(np.uint8)

    # Create PIL Image
    image = Image.fromarray(output_np, mode='RGB')

    return image


def encode_image_to_base64(image: Image.Image) -> str:
    """Encode PIL Image to base64 string"""
    buffer = io.BytesIO()
    image.save(buffer, format='PNG')
    img_b64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
    return img_b64


# API Endpoints
@router.get("/models", response_model=List[str])
async def list_models():
    """Get list of available trained models"""
    return get_available_models()


@router.get("/models/{model_name}/info", response_model=ModelInfo)
async def get_model_info(model_name: str):
    """Get detailed information about a specific model"""
    try:
        info = get_model_info_data(model_name)
        return info
    except Exception as e:
        raise HTTPException(status_code=404, detail=str(e))


@router.post("/colorize", response_model=ColorizeImageResponse)
async def colorize_image(request: ColorizeImageRequest):
    """Colorize a grayscale or color image"""
    try:
        # Load model
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model, config = load_model(request.model_name, device)

        image_size = config.get('image_size', 256)

        # Preprocess image
        grayscale_tensor, grayscale_pil = preprocess_image(request.image, image_size)
        grayscale_tensor = grayscale_tensor.to(device)

        # Run inference
        with torch.no_grad():
            colorized_tensor = model(grayscale_tensor)

        # Postprocess
        colorized_image = postprocess_image(colorized_tensor)

        # Encode images to base64
        colorized_b64 = encode_image_to_base64(colorized_image)
        grayscale_b64 = encode_image_to_base64(grayscale_pil)

        return ColorizeImageResponse(
            success=True,
            colorized_image=colorized_b64,
            grayscale_image=grayscale_b64,
            model_name=request.model_name
        )

    except HTTPException:
        raise
    except Exception as e:
        import traceback
        print(f"❌ Error colorizing image: {str(e)}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error colorizing image: {str(e)}")


@router.get("/cost-analysis", response_model=TrainingCostAnalysis)
@router.get("/cost-analysis/{model_name}", response_model=TrainingCostAnalysis)
async def get_training_cost_analysis(model_name: Optional[str] = None):
    """Get training cost analysis for a model"""
    try:
        if model_name:
            model_dir = MODELS_DIR / model_name
            visualizations_dir = model_dir / "visualizations"
            cost_file = visualizations_dir / "cost_analysis.json"

            if cost_file.exists():
                with open(cost_file, 'r') as f:
                    cost_data = json.load(f)

                return TrainingCostAnalysis(
                    training_cost_breakdown=cost_data.get("training_cost_breakdown", {
                        "compute_cost": 0.0,
                        "memory_cost": 0.0,
                        "storage_cost": 0.0,
                        "total_cost": 0.0
                    }),
                    cost_per_epoch=cost_data.get("cost_per_epoch", {
                        "avg_cost_per_epoch": 0.0,
                        "total_epochs": 0
                    }),
                    peak_memory_gb=cost_data.get("peak_memory_gb", 0.0),
                    training_hours=cost_data.get("training_hours", 0.0)
                )

        # Return default cost analysis
        return TrainingCostAnalysis(
            training_cost_breakdown={
                "compute_cost": 0.0,
                "memory_cost": 0.0,
                "storage_cost": 0.0,
                "total_cost": 0.0
            },
            cost_per_epoch={
                "avg_cost_per_epoch": 0.0,
                "total_epochs": 0
            },
            peak_memory_gb=0.0,
            training_hours=0.0
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/training/status")
async def get_training_status():
    """Get current training status"""
    return {
        "is_training": False,
        "message": "No training in progress. Use model.py to start training."
    }
