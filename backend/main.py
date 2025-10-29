from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import os

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
    allow_origins=["https://cst-435-project-hub-frontend.onrender.com",
                   "http://localhost:3000",  # Local React frontend
                   "http://127.0.0.1:3000"],
    allow_credentials=False,  # Cannot use True with allow_origins=["*"]
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

# Global variable to store the text generator
text_generator = None

def get_text_generator():
    """Initialize or get the text generator instance."""
    global text_generator
    if text_generator is None:
        from backend.app.routers.projectRNN.text_generator import TextGenerator
        text_generator = TextGenerator()

        # Try to load existing model
        model_path = "backend/app/routers/projectRNN/saved_models/model.h5"
        tokenizer_path = "backend/app/routers/projectRNN/saved_models/tokenizer.pkl"

        if os.path.exists(model_path) and os.path.exists(tokenizer_path):
            text_generator.load_model(model_path, tokenizer_path)
        else:
            # Return None if no model exists
            return None
    return text_generator

@app.post("/generate-text", response_model=GenerateTextResponse)
def generate_text_endpoint(request: GenerateTextRequest):
    """Generate text using the trained model."""
    generator = get_text_generator()

    if generator is None:
        raise HTTPException(
            status_code=503,
            detail="Model not trained yet. Please train the model first using /train-model endpoint."
        )

    try:
        generated = generator.generate_text(
            seed_text=request.seed_text,
            num_words=request.num_words,
            temperature=request.temperature
        )

        return GenerateTextResponse(
            generated_text=generated,
            seed_text=request.seed_text,
            num_words=request.num_words,
            temperature=request.temperature
        )
    except Exception as e:
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