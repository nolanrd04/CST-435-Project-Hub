import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
import re
import json
from typing import List, Tuple, Dict
import matplotlib.pyplot as plt
import seaborn as sns

class TextGenerator:
    """
    Advanced RNN-based text generator with LSTM architecture.

    This class handles:
    - Text preprocessing and tokenization
    - Sequence generation for training
    - LSTM model construction
    - Training with visualization
    - Text generation with temperature sampling
    """

    def __init__(
        self,
        sequence_length: int = 50,
        embedding_dim: int = 100,
        lstm_units: int = 150,
        num_layers: int = 2,
        dropout_rate: float = 0.2
    ):
        """
        Initialize the text generator.

        Args:
            sequence_length: Number of words to consider for context
            embedding_dim: Dimension of word embeddings
            lstm_units: Number of units in each LSTM layer
            num_layers: Number of LSTM layers
            dropout_rate: Dropout rate for regularization
        """
        self.sequence_length = sequence_length
        self.embedding_dim = embedding_dim
        self.lstm_units = lstm_units
        self.num_layers = num_layers
        self.dropout_rate = dropout_rate

        self.tokenizer = Tokenizer()
        self.model = None
        self.vocab_size = 0
        self.max_sequence_len = 0
        self.history = None

    def preprocess_text(self, text: str) -> str:
        """
        Clean and normalize input text.

        Steps:
        1. Convert to lowercase
        2. Remove special characters (keep punctuation)
        3. Normalize whitespace
        4. Remove extra newlines

        Args:
            text: Raw input text

        Returns:
            Cleaned text string
        """
        # Convert to lowercase
        text = text.lower()

        # Remove special characters but keep basic punctuation
        text = re.sub(r'[^a-z0-9\s.,!?\'\-]', '', text)

        # Normalize whitespace
        text = ' '.join(text.split())

        return text

    def prepare_sequences(self, text: str) -> Tuple[np.ndarray, np.ndarray, int]:
        """
        Convert text to training sequences.

        Process:
        1. Tokenize text into words
        2. Build vocabulary
        3. Create sliding window sequences
        4. Encode as integer sequences
        5. Separate into X (input) and y (target)

        Args:
            text: Preprocessed text

        Returns:
            Tuple of (X, y, max_sequence_len)
            - X: Input sequences (context words)
            - y: Target words (one-hot encoded)
            - max_sequence_len: Length of longest sequence
        """
        # Preprocess
        text = self.preprocess_text(text)

        # Tokenize text
        self.tokenizer.fit_on_texts([text])
        self.vocab_size = len(self.tokenizer.word_index) + 1

        print(f"Vocabulary size: {self.vocab_size}")

        # Create input sequences using sliding window
        input_sequences = []
        words = text.split()

        for i in range(self.sequence_length, len(words)):
            # Take sequence_length + 1 words
            # First sequence_length words = input
            # Last word = target
            seq = words[i - self.sequence_length : i + 1]
            input_sequences.append(seq)

        print(f"Total sequences: {len(input_sequences)}")

        # Convert words to integer sequences
        token_sequences = self.tokenizer.texts_to_sequences(input_sequences)

        # Pad sequences to same length
        self.max_sequence_len = max([len(seq) for seq in token_sequences])
        padded_sequences = pad_sequences(
            token_sequences,
            maxlen=self.max_sequence_len,
            padding='pre'
        )

        # Split into inputs and labels
        X = padded_sequences[:, :-1]  # All but last word
        y = padded_sequences[:, -1]   # Last word

        # Convert y to one-hot encoding
        y = keras.utils.to_categorical(y, num_classes=self.vocab_size)

        print(f"X shape: {X.shape}")
        print(f"y shape: {y.shape}")

        return X, y, self.max_sequence_len

    def build_model(self) -> keras.Model:
        """
        Build LSTM model architecture.

        Architecture:
        1. Embedding layer (word → dense vector)
        2. Multiple LSTM layers with dropout
        3. Dense output layer with softmax

        Returns:
            Compiled Keras model
        """
        model = keras.Sequential(name="Text_Generator_LSTM")

        # Embedding layer
        model.add(layers.Embedding(
            input_dim=self.vocab_size,
            output_dim=self.embedding_dim,
            input_length=self.max_sequence_len - 1,
            name="Embedding"
        ))

        # LSTM layers
        for i in range(self.num_layers):
            return_sequences = (i < self.num_layers - 1)
            model.add(layers.LSTM(
                units=self.lstm_units,
                return_sequences=return_sequences,
                name=f"LSTM_{i+1}"
            ))
            model.add(layers.Dropout(
                rate=self.dropout_rate,
                name=f"Dropout_{i+1}"
            ))

        # Output layer
        model.add(layers.Dense(
            units=self.vocab_size,
            activation='softmax',
            name="Output"
        ))

        # Compile model
        model.compile(
            loss='categorical_crossentropy',
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            metrics=['accuracy']
        )

        self.model = model
        return model

    def train(
        self,
        X: np.ndarray,
        y: np.ndarray,
        epochs: int = 100,
        batch_size: int = 128,
        validation_split: float = 0.1,
        save_path: str = "saved_models"
    ) -> Dict:
        """
        Train the LSTM model with visualization.

        Args:
            X: Input sequences
            y: Target words (one-hot)
            epochs: Number of training epochs
            batch_size: Batch size for training
            validation_split: Fraction of data for validation
            save_path: Directory to save checkpoints

        Returns:
            Dictionary with training history
        """
        # Callbacks
        callbacks = [
            keras.callbacks.ModelCheckpoint(
                filepath=f"{save_path}/model_best.h5",
                monitor='val_loss',
                save_best_only=True,
                verbose=1
            ),
            keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True,
                verbose=1
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-6,
                verbose=1
            ),
            keras.callbacks.TensorBoard(
                log_dir=f"{save_path}/logs",
                histogram_freq=1
            )
        ]

        # Train model
        self.history = self.model.fit(
            X, y,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            callbacks=callbacks,
            verbose=1
        )

        return self.history.history

    def generate_text(
        self,
        seed_text: str,
        num_words: int = 50,
        temperature: float = 1.0
    ) -> str:
        """
        Generate text using trained model.

        Temperature controls randomness:
        - Low (0.5): More predictable, coherent
        - Medium (1.0): Balanced
        - High (1.5-2.0): More creative, random

        Args:
            seed_text: Starting text
            num_words: Number of words to generate
            temperature: Sampling temperature

        Returns:
            Generated text string
        """
        if self.model is None:
            raise ValueError("Model not trained yet!")

        generated_text = seed_text.lower()

        for _ in range(num_words):
            # Tokenize current text
            token_list = self.tokenizer.texts_to_sequences([generated_text])[0]

            # Take last sequence_length tokens
            token_list = token_list[-(self.sequence_length):]

            # Pad to model input size
            token_list = pad_sequences(
                [token_list],
                maxlen=self.max_sequence_len - 1,
                padding='pre'
            )

            # Predict next word probabilities
            predicted_probs = self.model.predict(token_list, verbose=0)[0]

            # Apply temperature
            predicted_probs = np.log(predicted_probs + 1e-10) / temperature
            predicted_probs = np.exp(predicted_probs)
            predicted_probs = predicted_probs / np.sum(predicted_probs)

            # Sample from distribution
            predicted_index = np.random.choice(
                len(predicted_probs),
                p=predicted_probs
            )

            # Convert index to word
            for word, index in self.tokenizer.word_index.items():
                if index == predicted_index:
                    generated_text += " " + word
                    break

        return generated_text

    def visualize_architecture(self, save_path: str = "visualizations"):
        """Generate model architecture visualization."""
        keras.utils.plot_model(
            self.model,
            to_file=f"{save_path}/model_architecture.png",
            show_shapes=True,
            show_layer_names=True,
            rankdir='TB',
            dpi=150
        )

    def plot_training_history(self, save_path: str = "visualizations"):
        """
        Plot training and validation metrics.

        Creates two subplots:
        1. Loss over epochs
        2. Accuracy over epochs
        """
        if self.history is None:
            raise ValueError("No training history available!")

        history_dict = self.history.history

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        # Loss plot
        ax1.plot(history_dict['loss'], label='Training Loss', linewidth=2)
        ax1.plot(history_dict['val_loss'], label='Validation Loss', linewidth=2)
        ax1.set_xlabel('Epoch', fontsize=12)
        ax1.set_ylabel('Loss', fontsize=12)
        ax1.set_title('Model Loss Over Time', fontsize=14, fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Accuracy plot
        ax2.plot(history_dict['accuracy'], label='Training Accuracy', linewidth=2)
        ax2.plot(history_dict['val_accuracy'], label='Validation Accuracy', linewidth=2)
        ax2.set_xlabel('Epoch', fontsize=12)
        ax2.set_ylabel('Accuracy', fontsize=12)
        ax2.set_title('Model Accuracy Over Time', fontsize=14, fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(f"{save_path}/training_history.png", dpi=300, bbox_inches='tight')
        plt.close()

    def save_model(self, model_path: str, tokenizer_path: str):
        """Save model and tokenizer."""
        self.model.save(model_path)
        with open(tokenizer_path, 'wb') as f:
            pickle.dump(self.tokenizer, f)

        # Save configuration
        config = {
            'sequence_length': self.sequence_length,
            'embedding_dim': self.embedding_dim,
            'lstm_units': self.lstm_units,
            'num_layers': self.num_layers,
            'dropout_rate': self.dropout_rate,
            'vocab_size': self.vocab_size,
            'max_sequence_len': self.max_sequence_len
        }

        with open(model_path.replace('.h5', '_config.json'), 'w') as f:
            json.dump(config, f, indent=2)

    def load_model(self, model_path: str, tokenizer_path: str):
        """Load saved model and tokenizer."""
        self.model = keras.models.load_model(model_path)

        with open(tokenizer_path, 'rb') as f:
            self.tokenizer = pickle.load(f)

        # Load configuration
        config_path = model_path.replace('.h5', '_config.json')
        with open(config_path, 'r') as f:
            config = json.load(f)

        self.sequence_length = config['sequence_length']
        self.embedding_dim = config['embedding_dim']
        self.lstm_units = config['lstm_units']
        self.num_layers = config['num_layers']
        self.dropout_rate = config['dropout_rate']
        self.vocab_size = config['vocab_size']
        self.max_sequence_len = config['max_sequence_len']

