"""
Song Lyric Generator Module
Handles loading the trained LSTM model and generating song lyrics from seed text.
"""

import torch
import torch.nn.functional as F
import os
import json
import sys
from typing import Dict, Optional, Tuple

# Add current directory to path for imports
DIR_PATH = os.path.dirname(os.path.realpath(__file__))
if DIR_PATH not in sys.path:
    sys.path.insert(0, DIR_PATH)

from Model import LyricsLSTM, generate_lyrics, load_model
from DataPreprocessor import LyricsTokenizer


class SongLyricGenerator:
    """
    Generates song lyrics using the trained LSTM model.
    Handles model loading, seed text validation, and generation with various parameters.
    """
    
    # Available models
    AVAILABLE_MODELS = {
        'lyrics_model': 'lyrics_model.pth',
        'lyrics_model_2': 'lyrics_model_2.pth'
    }
    
    def __init__(self, model_name: str = 'lyrics_model', tokenizer_path: str = None):
        """
        Initialize the lyric generator.
        
        Args:
            model_name: Name of the model to use ('lyrics_model' or 'lyrics_model_2')
            tokenizer_path: Path to the saved tokenizer file (e.g., 'tokenizer.pkl')
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.tokenizer = None
        self.model_loaded = False
        self.current_model_name = model_name
        self.tokenizer_path = tokenizer_path
        self.generation_config = {}
        
        # Set model path based on model name
        DIR_PATH = os.path.dirname(os.path.realpath(__file__))
        model_filename = self.AVAILABLE_MODELS.get(model_name, 'lyrics_model.pth')
        self.model_path = os.path.join(DIR_PATH, "model", model_filename)
        
        if self.tokenizer_path is None:
            self.tokenizer_path = os.path.join(DIR_PATH, "model", "tokenizer.pkl")
    
    def load_model(self) -> bool:
        """
        Load the trained model and tokenizer.
        
        Returns:
            True if model loaded successfully, False otherwise
        """
        try:
            if not os.path.exists(self.model_path):
                print(f"❌ Model file not found: {self.model_path}")
                return False
            
            if not os.path.exists(self.tokenizer_path):
                print(f"❌ Tokenizer file not found: {self.tokenizer_path}")
                return False
            
            print(f"📖 Loading model '{self.current_model_name}' from {self.model_path}...")
            self.model, _, _ = load_model(self.model_path)
            
            print(f"📚 Loading tokenizer from {self.tokenizer_path}...")
            self.tokenizer = LyricsTokenizer.load(self.tokenizer_path)
            
            self.model_loaded = True
            print(f"✅ Model '{self.current_model_name}' and tokenizer loaded successfully")
            return True
            
        except Exception as e:
            print(f"❌ Error loading model: {str(e)}")
            import traceback
            traceback.print_exc()
            return False
    
    def switch_model(self, model_name: str) -> bool:
        """
        Switch to a different model.
        
        Args:
            model_name: Name of the model to switch to ('lyrics_model' or 'lyrics_model_2')
        
        Returns:
            True if model switched successfully, False otherwise
        """
        if model_name not in self.AVAILABLE_MODELS:
            print(f"❌ Unknown model: {model_name}. Available models: {list(self.AVAILABLE_MODELS.keys())}")
            return False
        
        self.current_model_name = model_name
        DIR_PATH = os.path.dirname(os.path.realpath(__file__))
        model_filename = self.AVAILABLE_MODELS[model_name]
        self.model_path = os.path.join(DIR_PATH, "model", model_filename)
        self.model_loaded = False
        
        return self.load_model()
    
    def validate_seed_text(self, seed_text: str) -> Tuple[bool, str]:
        """
        Validate the seed text for generation.
        
        Args:
            seed_text: User-provided seed text
        
        Returns:
            Tuple of (is_valid, message)
        """
        if not seed_text or not isinstance(seed_text, str):
            return False, "Seed text must be a non-empty string"
        
        seed_text = seed_text.strip()
        
        if len(seed_text) == 0:
            return False, "Seed text cannot be empty or only whitespace"
        
        if len(seed_text) > 500:
            return False, "Seed text is too long (max 500 characters)"
        
        # Check if seed text contains mostly valid words
        words = seed_text.lower().split()
        if len(words) == 0:
            return False, "Seed text must contain at least one word"
        
        if len(words) > 50:
            return False, "Seed text contains too many words (max 50 words)"
        
        return True, "Seed text is valid"
    
    def generate(self, 
                 seed_text: str, 
                 max_length: int = 50,
                 temperature: float = 1.0,
                 top_k: int = 50,
                 num_variations: int = 1,
                 model: str = None) -> Dict:
        """
        Generate song lyrics from seed text.
        
        Args:
            seed_text: Starting text for generation
            max_length: Maximum number of words to generate (default 50)
            temperature: Sampling temperature - higher = more random (default 1.0)
                        Values: 0.1-2.0 (lower = more deterministic)
            top_k: Only sample from top k most likely words (default 50)
                   0 = disabled, higher = more conservative
            num_variations: Number of different variations to generate (default 1)
            model: Name of model to use ('lyrics_model' or 'lyrics_model_2')
        
        Returns:
            Dictionary with generated lyrics and metadata
        """
        # Switch model if requested
        if model and model != self.current_model_name:
            print(f"🔄 Switching to model: {model}")
            if not self.switch_model(model):
                return {
                    "success": False,
                    "error": f"Failed to load model: {model}",
                    "generated_text": None,
                    "variations": []
                }
        
        # Check if model is loaded
        if not self.model_loaded:
            if not self.load_model():
                return {
                    "success": False,
                    "error": "Model not loaded. Please ensure model files exist.",
                    "generated_text": None,
                    "variations": []
                }
        
        # Validate seed text
        is_valid, message = self.validate_seed_text(seed_text)
        if not is_valid:
            return {
                "success": False,
                "error": message,
                "generated_text": None,
                "variations": []
            }
        
        # Validate parameters
        max_length = max(1, min(max_length, 200))  # Clamp between 1 and 200
        temperature = max(0.1, min(temperature, 3.0))  # Clamp between 0.1 and 3.0
        top_k = max(0, min(top_k, 1000))  # Clamp between 0 and 1000
        num_variations = max(1, min(num_variations, 5))  # Allow 1-5 variations
        
        try:
            variations = []
            primary_text = None
            
            print(f"🎵 Generating {num_variations} variation(s) from seed: '{seed_text}'")
            print(f"   Model: {self.current_model_name}")
            print(f"   Parameters: max_length={max_length}, temperature={temperature}, top_k={top_k}")
            
            for i in range(num_variations):
                print(f"   Generating variation {i+1}/{num_variations}...")
                
                generated_text = generate_lyrics(
                    model=self.model,
                    tokenizer=self.tokenizer,
                    seed_text=seed_text,
                    max_length=max_length,
                    temperature=temperature,
                    top_k=top_k
                )
                
                # Clean up debug output
                generated_text = generated_text.split('\n')[-1] if '\n' in generated_text else generated_text
                
                variations.append({
                    "variation": i + 1,
                    "generated_text": generated_text,
                    "word_count": len(generated_text.split())
                })
                
                if i == 0:
                    primary_text = generated_text
            
            print(f"✅ Successfully generated {num_variations} variation(s) using {self.current_model_name}")
            
            return {
                "success": True,
                "seed_text": seed_text,
                "generated_text": primary_text,
                "variations": variations,
                "parameters": {
                    "max_length": max_length,
                    "temperature": temperature,
                    "top_k": top_k,
                    "num_variations": num_variations
                },
                "model_used": self.current_model_name,
                "model_info": {
                    "vocab_size": self.model.vocab_size if self.model else None,
                    "embedding_dim": self.model.embedding_dim if self.model else None,
                    "hidden_size": self.model.hidden_size if self.model else None,
                    "num_layers": self.model.num_layers if self.model else None
                }
            }
            
        except Exception as e:
            print(f"❌ Error during generation: {str(e)}")
            import traceback
            traceback.print_exc()
            
            return {
                "success": False,
                "error": f"Generation failed: {str(e)}",
                "generated_text": None,
                "variations": []
            }
    
    def get_model_info(self) -> Dict:
        """
        Get information about the loaded model.
        
        Returns:
            Dictionary with model architecture and config info
        """
        if not self.model_loaded:
            if not self.load_model():
                return {"success": False, "error": "Model not loaded"}
        
        try:
            return {
                "success": True,
                "model_loaded": self.model_loaded,
                "current_model": self.current_model_name,
                "available_models": list(self.AVAILABLE_MODELS.keys()),
                "model_architecture": {
                    "vocab_size": self.model.vocab_size,
                    "embedding_dim": self.model.embedding_dim,
                    "hidden_size": self.model.hidden_size,
                    "num_layers": self.model.num_layers,
                    "dropout": self.model.dropout
                },
                "vocabulary_size": len(self.tokenizer.word_to_index) if self.tokenizer else 0,
                "parameter_count": self.model.get_parameter_count(),
                "parameter_breakdown": self.model.get_parameter_breakdown(),
                "special_tokens": {
                    "pad_token": self.tokenizer.pad_token if self.tokenizer else None,
                    "oov_token": self.tokenizer.oov_token if self.tokenizer else None,
                    "start_token": self.tokenizer.start_token if self.tokenizer else None,
                    "end_token": self.tokenizer.end_token if self.tokenizer else None
                },
                "model_file": self.model_path,
                "tokenizer_file": self.tokenizer_path,
                "device": str(self.device)
            }
        except Exception as e:
            return {
                "success": False,
                "error": f"Error retrieving model info: {str(e)}"
            }
    
    @staticmethod
    def get_available_models() -> Dict:
        """
        Get list of available models.
        
        Returns:
            Dictionary with available models info
        """
        return {
            "available_models": SongLyricGenerator.AVAILABLE_MODELS,
            "models": [
                {"id": "lyrics_model", "name": "Large Model", "description": "Full model with better quality"},
                {"id": "lyrics_model_2", "name": "Small Model", "description": "Compact model with faster inference"}
            ]
        }


# Global instance for the API
_lyric_generator = None


def get_lyric_generator() -> SongLyricGenerator:
    """Get or create the global lyric generator instance."""
    global _lyric_generator
    if _lyric_generator is None:
        _lyric_generator = SongLyricGenerator()
    return _lyric_generator
