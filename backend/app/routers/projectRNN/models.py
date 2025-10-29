from pydantic import BaseModel, Field
from typing import Optional

class GenerateRequest(BaseModel):
    """Request model for text generation."""
    seed_text: str = Field(..., min_length=1, description="Starting text")
    num_words: int = Field(50, ge=10, le=200, description="Number of words to generate")
    temperature: float = Field(1.0, ge=0.1, le=2.0, description="Sampling temperature")

class GenerateResponse(BaseModel):
    """Response model for text generation."""
    generated_text: str
    seed_text: str
    num_words: int
    temperature: float

class ModelInfo(BaseModel):
    """Model information."""
    vocab_size: int
    sequence_length: int
    embedding_dim: int
    lstm_units: int
    num_layers: int

class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    model_loaded: bool