"""
FastAPI router for Project6 GAN endpoints
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

# Import your existing GAN modules
from .gan_model import Generator

router = APIRouter(prefix="/project6", tags=["project6"])

# Base paths
PROJECT_DIR = Path(__file__).parent
MODELS_DIR = PROJECT_DIR / "models"
NPZ_DATA_DIR = PROJECT_DIR / "npzData"


# Pydantic Models
class GenerateImageRequest(BaseModel):
    model_name: str
    fruit: str
    num_images: int = 16
    seed: Optional[int] = None


class GenerateImageResponse(BaseModel):
    success: bool
    images: List[str]  # Base64 encoded
    model_name: str
    fruit: str
    num_images: int


class CreateModelRequest(BaseModel):
    model_name: str
    data_version: str
    description: str
    create_new_data: bool
    override_existing: Optional[bool] = False
    # Data creation params (optional)
    image_count: Optional[int] = 100
    image_resolution: Optional[int] = 64
    stroke_importance: Optional[int] = 5
    # Training params
    epochs: Optional[int] = 50
    batch_size: Optional[int] = 32
    learning_rate: Optional[float] = 0.0002


class CreateModelResponse(BaseModel):
    success: bool
    message: str
    model_name: str
    estimated_training_time: Optional[int] = None
    warning: Optional[str] = None


class ModelInfo(BaseModel):
    model_name: str
    data_version: str
    description: str
    training_config: Optional[Dict[str, Any]] = None
    training_stats: Optional[Dict[str, Any]] = None
    fruits: List[str]
    model_architecture: Optional[Dict[str, Any]] = None
    total_parameters: Optional[int] = None
    parameters_per_fruit: Optional[Dict[str, int]] = None


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
        if item.is_dir() and item.name.startswith("model_"):
            models.append(item.name)

    return sorted(models)


def get_model_info_data(model_name: str) -> Dict[str, Any]:
    """Load model information from info folder"""
    model_dir = MODELS_DIR / model_name
    info_dir = model_dir / "info"

    if not info_dir.exists():
        raise HTTPException(status_code=404, detail=f"Model info not found for {model_name}")

    # Load description
    desc_file = info_dir / "description.txt"
    description = ""
    if desc_file.exists():
        with open(desc_file, 'r') as f:
            description = f.read()

    # Load training config
    config_file = info_dir / f"training_config_{model_name}.json"
    training_config = None
    if config_file.exists():
        with open(config_file, 'r') as f:
            training_config = json.load(f)

    # Load training summary
    summary_file = info_dir / f"training_summary_{model_name}.json"
    training_stats = None
    if summary_file.exists():
        with open(summary_file, 'r') as f:
            summary_data = json.load(f)
            training_stats = {
                "total_training_time_hours": summary_data.get("total_training_time_hours", 0),
                "peak_memory_gb": summary_data.get("peak_memory_gb", 0),
                "total_training_cost": summary_data.get("total_training_cost", 0),
                "avg_cost_per_fruit": summary_data.get("avg_cost_per_fruit", 0),
                "avg_cost_per_epoch": summary_data.get("avg_cost_per_epoch", 0),
            }
    
    # If no training stats from summary, try loading from cost analysis report
    if not training_stats or all(v == 0 for v in training_stats.values()):
        cost_file = info_dir / "cost_analysis_report.json"
        if cost_file.exists():
            try:
                with open(cost_file, 'r') as f:
                    cost_data = json.load(f)
                    training_cost = cost_data.get("training_cost", {})
                    cost_per_epoch_data = cost_data.get("cost_per_epoch", {})
                    training_stats = {
                        "total_training_time_hours": training_cost.get("training_hours", 0),
                        "peak_memory_gb": training_cost.get("peak_memory_gb", 0),
                        "total_training_cost": training_cost.get("total", 0),
                        "avg_cost_per_fruit": training_cost.get("total", 0) / max(len(fruits), 1) if fruits else 0,
                        "avg_cost_per_epoch": cost_per_epoch_data.get("cost_per_epoch", 0),
                    }
            except Exception as e:
                print(f"Warning: Could not load cost analysis: {e}")

    # Get list of trained fruits (check for generator files)
    fruits = []
    fruit_file_map = {
        "apple": "apple",
        "banana": "banana",
        "blackberry": "blackberry",
        "grape": "grapes",  # Singular for frontend, plural in files
        "pear": "pear",
        "strawberry": "strawberry",
        "watermelon": "watermelon"
    }
    for fruit_display, fruit_file in fruit_file_map.items():
        gen_file = model_dir / f"generator_{fruit_file}.pt"
        if gen_file.exists():
            fruits.append(fruit_display)  # Return singular name to frontend

    # Calculate model parameters
    img_size = training_config.get("image_size", 64) if training_config else 64
    total_parameters = 0
    parameters_per_fruit = {}

    try:
        device = torch.device('cpu')  # Use CPU for counting
        # Load one generator to get architecture info
        if fruits:
            first_fruit = fruits[0]
            fruit_file = fruit_file_map.get(first_fruit, first_fruit)
            gen_path = model_dir / f"generator_{fruit_file}.pt"
            disc_path = model_dir / f"discriminator_{fruit_file}.pt"

            # Count generator parameters
            generator = Generator(img_size=img_size, latent_dim=100, channels=1)
            if gen_path.exists():
                generator.load_state_dict(torch.load(gen_path, map_location=device))
            gen_params = sum(p.numel() for p in generator.parameters())

            # Count discriminator parameters
            from .gan_model import Discriminator
            discriminator = Discriminator(img_size=img_size, channels=1)
            if disc_path.exists():
                discriminator.load_state_dict(torch.load(disc_path, map_location=device))
            disc_params = sum(p.numel() for p in discriminator.parameters())

            # Each fruit has both a generator and discriminator
            params_per_fruit = gen_params + disc_params

            # Calculate for all fruits
            for fruit in fruits:
                parameters_per_fruit[fruit] = params_per_fruit

            total_parameters = params_per_fruit * len(fruits)
    except Exception as e:
        print(f"Warning: Could not calculate parameter counts: {e}")
        total_parameters = None
        parameters_per_fruit = None

    # Get model architecture details
    model_architecture = None
    if training_config:
        model_architecture = {
            "image_size": img_size,
            "latent_dim": 100,
            "channels": 1,
            "generator_layers": "FC + 3 Conv blocks with upsampling",
            "discriminator_layers": "3 Conv blocks with downsampling + FC"
        }

    return {
        "model_name": model_name,
        "data_version": training_config.get("data_version", model_name) if training_config else model_name,
        "description": description,
        "training_config": training_config,
        "training_stats": training_stats,
        "fruits": fruits,
        "model_architecture": model_architecture,
        "total_parameters": total_parameters,
        "parameters_per_fruit": parameters_per_fruit
    }


def get_available_data_versions() -> List[str]:
    """Get list of available data versions from NPZ files"""
    if not NPZ_DATA_DIR.exists():
        return []

    versions = set()
    for npz_file in NPZ_DATA_DIR.glob("*.npz"):
        # Extract version from filename like "apple_v1.npz"
        parts = npz_file.stem.split('_')
        if len(parts) >= 2:
            version = '_'.join(parts[1:])  # Everything after fruit name
            versions.add(version)

    return sorted(list(versions))


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


@router.post("/generate", response_model=GenerateImageResponse)
async def generate_images(request: GenerateImageRequest):
    """Generate images using trained GAN model"""
    try:
        # Handle singular/plural fruit name mapping
        fruit_name_map = {
            "grape": "grapes",  # Frontend sends singular, but file uses plural
        }
        fruit_file_name = fruit_name_map.get(request.fruit, request.fruit)

        model_dir = MODELS_DIR / request.model_name
        generator_path = model_dir / f"generator_{fruit_file_name}.pt"

        if not generator_path.exists():
            raise HTTPException(
                status_code=404,
                detail=f"Generator not found for {request.fruit} in model {request.model_name}"
            )

        # Load generator
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Determine image size from config or default to 64
        info_dir = model_dir / "info"
        config_file = info_dir / f"training_config_{request.model_name}.json"
        img_size = 64
        if config_file.exists():
            with open(config_file, 'r') as f:
                config = json.load(f)
                img_size = config.get('image_size', 64)

        # Initialize generator
        generator = Generator(img_size=img_size, latent_dim=100, channels=1)
        generator.load_state_dict(torch.load(generator_path, map_location=device))
        generator.to(device)
        generator.eval()

        # Generate images
        images_b64 = []
        with torch.no_grad():
            if request.seed is not None:
                torch.manual_seed(request.seed)
                np.random.seed(request.seed)

            # Generate random noise
            z = torch.randn(request.num_images, 100, device=device)

            # Generate images
            generated = generator(z)

            # Convert to numpy and encode as base64
            generated = generated.cpu().numpy()

            for i in range(request.num_images):
                # Denormalize from [-1, 1] to [0, 255]
                img_array = ((generated[i, 0] + 1) * 127.5).astype(np.uint8)

                # Convert to PIL Image
                img = Image.fromarray(img_array, mode='L')

                # Convert to base64
                buffer = io.BytesIO()
                img.save(buffer, format='PNG')
                img_b64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
                images_b64.append(img_b64)

        return GenerateImageResponse(
            success=True,
            images=images_b64,
            model_name=request.model_name,
            fruit=request.fruit,
            num_images=request.num_images
        )

    except HTTPException:
        raise
    except Exception as e:
        import traceback
        print(f"❌ Error generating {request.fruit} images: {str(e)}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error generating images: {str(e)}")


@router.get("/cost-analysis", response_model=TrainingCostAnalysis)
@router.get("/cost-analysis/{model_name}", response_model=TrainingCostAnalysis)
async def get_training_cost_analysis(model_name: Optional[str] = None):
    """Get training cost analysis for a model or general pricing"""
    try:
        if model_name:
            model_dir = MODELS_DIR / model_name
            info_dir = model_dir / "info"
            cost_file = info_dir / "cost_analysis_report.json"

            if not cost_file.exists():
                raise HTTPException(
                    status_code=404,
                    detail=f"Cost analysis not found for model {model_name}"
                )

            with open(cost_file, 'r') as f:
                cost_data = json.load(f)

            # Map the actual JSON structure to the expected frontend format
            training_cost = cost_data.get("training_cost", {})
            cost_per_epoch_data = cost_data.get("cost_per_epoch", {})
            
            return TrainingCostAnalysis(
                training_cost_breakdown={
                    "compute_cost": training_cost.get("compute", 0),
                    "memory_cost": training_cost.get("memory", 0),
                    "storage_cost": training_cost.get("storage", 0),
                    "total_cost": training_cost.get("total", 0)
                },
                cost_per_epoch={
                    "avg_cost_per_epoch": cost_per_epoch_data.get("cost_per_epoch", 0),
                    "total_epochs": cost_per_epoch_data.get("total_epochs", 0)
                },
                peak_memory_gb=training_cost.get("peak_memory_gb", 0),
                training_hours=training_cost.get("training_hours", 0)
            )
        else:
            # Return default/example cost analysis
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

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/data-versions", response_model=List[str])
async def list_data_versions():
    """Get list of available data versions"""
    return get_available_data_versions()


@router.post("/models/create", response_model=CreateModelResponse)
async def create_model(request: CreateModelRequest):
    """Create and train a new GAN model in background"""
    import threading
    import shutil

    model_dir = MODELS_DIR / request.model_name

    # Check if model exists
    if model_dir.exists():
        if not request.override_existing:
            # Return error asking for confirmation
            return CreateModelResponse(
                success=False,
                message=f"Model {request.model_name} already exists. Please use a different name or delete the existing model.",
                model_name=request.model_name,
                warning=f"Model directory already exists at {model_dir}"
            )
        else:
            # User confirmed override - delete existing model
            try:
                shutil.rmtree(model_dir)
                print(f"🗑️ Deleted existing model directory: {model_dir}")
            except Exception as e:
                return CreateModelResponse(
                    success=False,
                    message=f"Failed to delete existing model: {str(e)}",
                    model_name=request.model_name,
                    warning="Could not remove existing model directory"
                )

    # Estimate training time (rough calculation: ~1 min per epoch per fruit)
    estimated_minutes = request.epochs * 7  # 7 fruits

    def train_in_background():
        """Run training in a background thread"""
        try:
            from .train_gan_api import train_model_programmatic

            fruits = ['apple', 'banana', 'blackberry', 'grapes', 'pear', 'strawberry', 'watermelon']

            train_model_programmatic(
                model_name=request.model_name,
                data_version=request.data_version,
                epochs=request.epochs,
                batch_size=request.batch_size,
                learning_rate=request.learning_rate,
                description=request.description,
                fruits=fruits
            )

            print(f"\n✅ Training completed successfully for model '{request.model_name}'")

        except Exception as e:
            print(f"\n❌ Training failed for model '{request.model_name}': {str(e)}")
            import traceback
            traceback.print_exc()

    try:
        # Start training in background thread
        training_thread = threading.Thread(target=train_in_background, daemon=True)
        training_thread.start()

        print(f"\n🚀 Started background training for model '{request.model_name}'")
        print(f"📊 Training {len(['apple', 'banana', 'blackberry', 'grapes', 'pear', 'strawberry', 'watermelon'])} fruits for {request.epochs} epochs each")
        print(f"⏱️  Estimated time: ~{estimated_minutes} minutes ({estimated_minutes//60}h {estimated_minutes%60}m)")
        print(f"📁 Model will be saved to: {model_dir}")
        print(f"👀 Watch this console for training progress...\n")

        return CreateModelResponse(
            success=True,
            message=f"Training started in background for model '{request.model_name}'. Check server console for progress. Training will take approximately {estimated_minutes//60} hours {estimated_minutes%60} minutes.",
            model_name=request.model_name,
            estimated_training_time=estimated_minutes,
            warning="Training is running in the background. Do not stop the server until training completes. Monitor progress in the server console."
        )

    except Exception as e:
        print(f"❌ Error starting training: {str(e)}")
        import traceback
        traceback.print_exc()
        return CreateModelResponse(
            success=False,
            message=f"Error starting training: {str(e)}",
            model_name=request.model_name,
            warning="Failed to start training"
        )


@router.get("/models/{model_name}/training-history/{fruit}")
async def get_training_history(model_name: str, fruit: str):
    """Get training history for a specific fruit"""
    try:
        # Handle singular/plural fruit name mapping
        fruit_name_map = {
            "grape": "grapes",
        }
        fruit_file_name = fruit_name_map.get(fruit, fruit)

        model_dir = MODELS_DIR / model_name
        info_dir = model_dir / "info"
        history_file = info_dir / f"training_history_{fruit_file_name}.json"

        if not history_file.exists():
            raise HTTPException(
                status_code=404,
                detail=f"Training history not found for {fruit} in model {model_name}"
            )

        with open(history_file, 'r') as f:
            history = json.load(f)

        return history

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/training/status")
async def get_training_status():
    """Get current training status (placeholder for real-time monitoring)"""
    # This would connect to a running training process
    # For now, return that no training is in progress
    return {
        "is_training": False,
        "message": "No training in progress. Use train_gan.py to start training."
    }
