from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import os
import time
import psutil
import json
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

app = FastAPI()

# Download NLTK data on startup (required for Project4 sentiment analysis)
# Do this after app creation to avoid blocking startup
try:
    import nltk
    nltk.download('stopwords', quiet=True)
    nltk.download('punkt', quiet=True)
    nltk.download('punkt_tab', quiet=True)
    print("✅ NLTK data downloaded successfully")
except Exception as e:
    print(f"⚠️ Warning: Could not download NLTK data: {e}")
    # Don't crash if NLTK download fails - it may already be cached

# Add CORS middleware - MUST be first middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for Render deployment
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
)

# Mount visualizations directory if it exists
if os.path.exists("visualizations"):
    app.mount("/visualizations", StaticFiles(directory="visualizations"), name="visualizations")

@app.get("/")
@app.head("/")
def read_root():
    return {"message": "Welcome to the Hub API!"}

# Define a Pydantic model for the request body
class GenerateTextRequest(BaseModel):
    seed_text: str
    num_words: int
    temperature: float

class GenerateTextResponse(BaseModel):
    generated_text: str
    seed_text: str
    num_words: int
    temperature: float
    query_cost: float  # Add query cost to response model

class ModelInfo(BaseModel):
    vocab_size: int
    sequence_length: int
    embedding_dim: int
    lstm_units: int
    num_layers: int

class ClassificationResponse(BaseModel):
    predicted_class: str
    confidence: float
    class_probabilities: dict

class SentimentRequest(BaseModel):
    review_text: str

class SentimentResponse(BaseModel):
    original_text: str
    processed_text: str
    classification: str
    confidence: float
    positive_probability: float
    negative_probability: float
    sentiment_score: float
    sentiment_label: str
    success: bool

class CostReport(BaseModel):
    pricing_config: dict

    class Config:
        extra = "allow"  # Allow extra fields

class PricingUpdate(BaseModel):
    compute_hourly_rate: float = None
    storage_per_gb_month: float = None
    data_transfer_per_gb: float = None
    instance_type: str = None

# Global variable to store the text generator
text_generator = None

# Global variable to store cost model
cost_model = None

def get_text_generator():
    """Initialize or get the text generator instance."""
    global text_generator
    if text_generator is None:
        from backend.app.routers.projectRNN.text_generator import TextGenerator
        text_generator = TextGenerator()

        # Try to load existing PyTorch model (model_best.pt)
        model_path = "backend/app/routers/projectRNN/saved_models/model_best.pt"
        tokenizer_path = "backend/app/routers/projectRNN/saved_models/tokenizer.pkl"

        # Only require model_path to exist; tokenizer can be reconstructed from config
        if os.path.exists(model_path):
            try:
                text_generator.load_model(model_path, tokenizer_path)
            except Exception as e:
                print(f"❌ Error loading model: {e}")
                return None
        else:
            # Return None if no model exists
            return None
    return text_generator

@app.post("/generate-text", response_model=GenerateTextResponse)
def generate_text_endpoint(request: GenerateTextRequest):
    """Generate text using the trained model and calculate query cost."""
    generator = get_text_generator()

    if generator is None:
        raise HTTPException(
            status_code=503,
            detail="Model not trained yet. Please train the model first using /train-model endpoint."
        )

    try:
        # Start time and memory usage
        start_time = time.time()
        process = psutil.Process()
        start_memory = process.memory_info().rss / (1024 * 1024 * 1024)  # Convert to GB

        # Generate text
        generated = generator.generate_text(
            seed_text=request.seed_text,
            num_words=request.num_words,
            temperature=request.temperature
        )

        # End time and memory usage
        end_time = time.time()
        end_memory = process.memory_info().rss / (1024 * 1024 * 1024)  # Convert to GB

        # Calculate time and memory usage
        elapsed_time = end_time - start_time
        memory_used = max(0, end_memory - start_memory)

        # Load pricing configuration
        try:
            pricing_config_path = os.path.join(
                os.path.dirname(__file__),
                "app/routers/projectRNN/render_pricing_config.json"
            )
            with open(pricing_config_path, "r", encoding="utf-8") as f:
                pricing_config = json.load(f)
        except FileNotFoundError:
            print(f"⚠️ Warning: Pricing config not found at {pricing_config_path}, using defaults")
            pricing_config = {
                "cost_per_cpu_per_month": 0.0,
                "cost_per_gb_ram_per_month": 0.0
            }

        # Compute cost per hour
        compute_cost_per_hour = (
            pricing_config.get("cost_per_cpu_per_month", 0) * (1 / 720)
        ) + (
            pricing_config.get("cost_per_gb_ram_per_month", 0) * (1 / 720) * memory_used
        )

        # Calculate query cost
        query_cost = compute_cost_per_hour * (elapsed_time / 3600)

        return GenerateTextResponse(
            generated_text=generated,
            seed_text=request.seed_text,
            num_words=request.num_words,
            temperature=request.temperature,
            query_cost=query_cost  # Include cost in the response
        )
    except Exception as e:
        print(f"❌ Error in generate_text_endpoint: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error generating text: {str(e)}")

@app.get("/model/info", response_model=ModelInfo)
def get_model_info():
    """Get information about the current model."""
    generator = get_text_generator()

    if generator is None:
        raise HTTPException(
            status_code=503,
            detail="Model not trained yet."
        )

    return ModelInfo(
        vocab_size=generator.vocab_size,
        sequence_length=generator.sequence_length,
        embedding_dim=generator.embedding_dim,
        lstm_units=generator.lstm_units,
        num_layers=generator.num_layers
    )

@app.post("/train-model")
def train_model_endpoint():
    """Train a new model with sample data."""
    from backend.app.routers.projectRNN.train import train_model

    try:
        train_model()
        # Reset the global text_generator so it reloads the new model
        global text_generator
        text_generator = None
        return {"message": "Model training completed successfully."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error training model: {str(e)}")

@app.get("/health")
def health_check():
    """Check if the API and model are ready."""
    generator = get_text_generator()
    return {
        "status": "healthy",
        "model_loaded": generator is not None
    }

@app.post("/classify-image", response_model=ClassificationResponse)
async def classify_image(file: UploadFile = File(...)):
    """Classify an uploaded image using the trained CNN model."""
    from backend.app.routers.Project3.classifier import get_classifier

    try:
        # Read the uploaded file
        image_bytes = await file.read()

        # Get classifier and classify
        classifier = get_classifier()
        result = classifier.classify(image_bytes)

        return ClassificationResponse(
            predicted_class=result["predicted_class"],
            confidence=result["confidence"],
            class_probabilities=result["class_probabilities"]
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error classifying image: {str(e)}")

@app.post("/analyze-sentiment", response_model=SentimentResponse)
async def analyze_sentiment(request: SentimentRequest):
    """Analyze sentiment of hotel review text using NLP model."""
    import sys
    from pathlib import Path

    try:
        # Extract review text from request
        review_text = request.review_text.strip()

        if not review_text:
            raise HTTPException(
                status_code=400,
                detail="Review text is required"
            )

        # Ensure Project4 directory is in path for imports
        project4_path = str(Path(__file__).parent / "app" / "routers" / "Project4")
        if project4_path not in sys.path:
            sys.path.insert(0, project4_path)

        print(f"🔍 Loading sentiment classifier from: {project4_path}")

        from backend.app.routers.Project4.classifier import get_classifier

        # Get classifier and analyze
        classifier = get_classifier()

        if classifier.model is None:
            print("❌ Model is None - failed to load")
            raise HTTPException(
                status_code=503,
                detail="Sentiment model not loaded. Please ensure trained_nlp_model.pkl exists in backend/app/routers/Project4/"
            )

        print(f"✅ Model loaded successfully")
        print(f"📝 Analyzing review: {review_text[:50]}...")

        result = classifier.classify(review_text)

        print(f"✅ Analysis complete. Classification: {result['classification']}")

        return SentimentResponse(
            original_text=result["original_text"],
            processed_text=result["processed_text"],
            classification=result["classification"],
            confidence=result["confidence"],
            positive_probability=result["positive_probability"],
            negative_probability=result["negative_probability"],
            sentiment_score=result["sentiment_score"],
            sentiment_label=result["sentiment_label"],
            success=result["success"]
        )
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Error during sentiment analysis: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error analyzing sentiment: {str(e)}")

@app.get("/cost-analysis/report")
def get_cost_analysis_report():
    """Get comprehensive cost analysis report for RNN deployment."""
    from backend.app.routers.projectRNN.cost_analysis import RenderPricingConfig, RNNCostModel

    global cost_model
    if cost_model is None:
        pricing = RenderPricingConfig()
        cost_model = RNNCostModel(pricing)

    try:
        report = cost_model.generate_cost_report()
        print(f"✅ Cost report generated successfully")
        return report
    except Exception as e:
        print(f"❌ Error generating cost analysis: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error generating cost analysis: {str(e)}")

@app.post("/cost-analysis/update-pricing")
def update_pricing(pricing_update: PricingUpdate):
    """Update Render pricing configuration."""
    from backend.app.routers.projectRNN.cost_analysis import RenderPricingConfig, RNNCostModel

    global cost_model
    if cost_model is None:
        pricing = RenderPricingConfig()
        cost_model = RNNCostModel(pricing)

    try:
        update_dict = pricing_update.dict(exclude_none=True)
        cost_model.update_pricing(update_dict)
        report = cost_model.generate_cost_report()
        return {"success": True, "message": "Pricing updated successfully", "report": report}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error updating pricing: {str(e)}")

@app.get("/cost-analysis/images/{image_name}")
def get_cost_analysis_image(image_name: str):
    """Get cost analysis visualization images."""
    import os
    from fastapi.responses import FileResponse

    valid_images = [
        "cost_breakdown.png",
        "cost_scaling.png",
        "training_vs_inference.png",
        "cost_per_inference.png"
    ]

    if image_name not in valid_images:
        raise HTTPException(status_code=404, detail="Image not found")

    image_path = f"backend/app/routers/projectRNN/visualizations/{image_name}"

    if not os.path.exists(image_path):
        raise HTTPException(status_code=404, detail="Image file not generated yet")

    return FileResponse(image_path)