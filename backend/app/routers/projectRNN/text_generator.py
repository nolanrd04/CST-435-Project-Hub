import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pickle
import re
import json
from typing import List, Tuple, Dict
import matplotlib.pyplot as plt
import seaborn as sns
import os


class SimpleTokenizer:
    """Simple PyTorch-compatible tokenizer."""
    def __init__(self):
        self.word_index = {}
        self.index_word = {}

    def fit_on_texts(self, texts):
        """Build vocabulary from texts."""
        words = set()
        for text in texts:
            words.update(text.split())
        self.word_index = {word: idx + 1 for idx, word in enumerate(sorted(words))}
        self.index_word = {idx: word for word, idx in self.word_index.items()}

    def texts_to_sequences(self, texts):
        """Convert texts to token sequences."""
        sequences = []
        for text in texts:
            seq = [self.word_index.get(word, 0) for word in text.split()]
            sequences.append(seq)
        return sequences


def pad_sequences_fn(sequences, maxlen, padding='pre'):
    """Pad sequences to fixed length."""
    padded = []
    for seq in sequences:
        if len(seq) >= maxlen:
            padded.append(seq[-maxlen:])
        else:
            pad_len = maxlen - len(seq)
            if padding == 'pre':
                padded.append([0] * pad_len + seq)
            else:
                padded.append(seq + [0] * pad_len)
    return np.array(padded, dtype=np.int32)


class LSTMModel(nn.Module):
    """
    PyTorch LSTM model for text generation.

    Architecture:
    1. Embedding layer (word → dense vector)
    2. Multiple LSTM layers with dropout
    3. Dense output layer with softmax
    """

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        lstm_units: int,
        num_layers: int,
        dropout_rate: float,
        input_length: int,
        device: str = 'cpu'
    ):
        super(LSTMModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=lstm_units,
            num_layers=num_layers,
            dropout=dropout_rate if num_layers > 1 else 0.0,
            batch_first=True
        )
        self.dropout = nn.Dropout(dropout_rate)
        self.dense = nn.Linear(lstm_units, vocab_size)
        self.device = device

    def forward(self, x):
        """Forward pass through the network."""
        x = self.embedding(x)
        lstm_out, _ = self.lstm(x)
        # Take the last output from LSTM
        lstm_out = lstm_out[:, -1, :]
        lstm_out = self.dropout(lstm_out)
        output = self.dense(lstm_out)
        return output


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

        self.tokenizer = SimpleTokenizer()
        self.model = None
        self.vocab_size = 0
        self.max_sequence_len = 0
        self.history = {'loss': [], 'val_loss': [], 'accuracy': [], 'val_accuracy': []}
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

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
        # Join each sequence of words into a single string
        token_sequences = self.tokenizer.texts_to_sequences([' '.join(seq) for seq in input_sequences])

        # Pad sequences to same length
        self.max_sequence_len = max([len(seq) for seq in token_sequences])
        padded_sequences = pad_sequences_fn(
            token_sequences,
            maxlen=self.max_sequence_len,
            padding='pre'
        )

        # Split into inputs and labels
        X = padded_sequences[:, :-1]  # All but last word
        y = padded_sequences[:, -1]   # Last word

        print(f"X shape: {X.shape}")
        print(f"y shape: {y.shape}")

        return X, y, self.max_sequence_len

    def build_model(self) -> LSTMModel:
        """
        Build LSTM model architecture.

        Architecture:
        1. Embedding layer (word → dense vector)
        2. Multiple LSTM layers with dropout
        3. Dense output layer with softmax

        Returns:
            PyTorch LSTM model
        """
        model = LSTMModel(
            vocab_size=self.vocab_size,
            embedding_dim=self.embedding_dim,
            lstm_units=self.lstm_units,
            num_layers=self.num_layers,
            dropout_rate=self.dropout_rate,
            input_length=self.max_sequence_len - 1,
            device=self.device
        )
        model.to(self.device)
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
        os.makedirs(save_path, exist_ok=True)

        # Convert numpy arrays to torch tensors
        X_tensor = torch.from_numpy(X).long().to(self.device)
        y_tensor = torch.from_numpy(y).long().to(self.device)

        # Split into train and validation
        split_idx = int(len(X_tensor) * (1 - validation_split))
        X_train = X_tensor[:split_idx]
        y_train = y_tensor[:split_idx]
        X_val = X_tensor[split_idx:]
        y_val = y_tensor[split_idx:]

        # Create data loaders
        train_dataset = TensorDataset(X_train, y_train)
        val_dataset = TensorDataset(X_val, y_val)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)

        # Loss function and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.5,
            patience=5,
            min_lr=1e-6
        )

        # Training loop
        best_val_loss = float('inf')
        patience_counter = 0
        patience = 10

        self.history = {'loss': [], 'val_loss': [], 'accuracy': [], 'val_accuracy': []}

        for epoch in range(epochs):
            # Training phase
            self.model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0

            for batch_X, batch_y in train_loader:
                optimizer.zero_grad()

                # Forward pass
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)

                # Backward pass
                loss.backward()
                optimizer.step()

                train_loss += loss.item() * batch_X.size(0)
                _, predicted = torch.max(outputs.data, 1)
                train_total += batch_y.size(0)
                train_correct += (predicted == batch_y).sum().item()

            avg_train_loss = train_loss / train_total
            train_accuracy = train_correct / train_total

            # Validation phase
            self.model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0

            with torch.no_grad():
                for batch_X, batch_y in val_loader:
                    outputs = self.model(batch_X)
                    loss = criterion(outputs, batch_y)

                    val_loss += loss.item() * batch_X.size(0)
                    _, predicted = torch.max(outputs.data, 1)
                    val_total += batch_y.size(0)
                    val_correct += (predicted == batch_y).sum().item()

            avg_val_loss = val_loss / val_total
            val_accuracy = val_correct / val_total

            # Store history
            self.history['loss'].append(avg_train_loss)
            self.history['val_loss'].append(avg_val_loss)
            self.history['accuracy'].append(train_accuracy)
            self.history['val_accuracy'].append(val_accuracy)

            print(f"Epoch {epoch + 1}/{epochs} - "
                  f"loss: {avg_train_loss:.4f} - accuracy: {train_accuracy:.4f} - "
                  f"val_loss: {avg_val_loss:.4f} - val_accuracy: {val_accuracy:.4f}")

            # Learning rate scheduling
            scheduler.step(avg_val_loss)
            for param_group in optimizer.param_groups:
                print(f"Learning rate: {param_group['lr']}")

            # Early stopping
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                patience_counter = 0
                # Save best model
                torch.save(self.model.state_dict(), f"{save_path}/model_best.pt")
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping triggered after {epoch + 1} epochs")
                    break

        return self.history

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
        self.model.eval()

        with torch.no_grad():
            for _ in range(num_words):
                # Tokenize current text
                token_list = self.tokenizer.texts_to_sequences([generated_text])[0]

                # Take last sequence_length tokens
                token_list = token_list[-(self.sequence_length):]

                # Pad to model input size (returns torch tensor)
                token_tensor = pad_sequences(
                    [token_list],
                    maxlen=self.max_sequence_len - 1,
                    padding='pre'
                )

                # Move tensor to device
                token_tensor = token_tensor.to(self.device)

                # Predict next word probabilities
                outputs = self.model(token_tensor)
                predicted_probs = torch.softmax(outputs[0], dim=0).cpu().numpy()

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
        os.makedirs(save_path, exist_ok=True)

        # Create a simple text-based architecture summary
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.axis('off')

        model_summary = f"""
        PyTorch LSTM Text Generator Architecture
        {'=' * 50}

        Input Layer:
          Shape: (batch_size, {self.max_sequence_len - 1})

        Embedding Layer:
          Vocab Size: {self.vocab_size}
          Embedding Dim: {self.embedding_dim}
          Output Shape: (batch_size, {self.max_sequence_len - 1}, {self.embedding_dim})

        LSTM Layers: {self.num_layers}
          Units per layer: {self.lstm_units}
          Dropout Rate: {self.dropout_rate}
          Output Shape: (batch_size, {self.lstm_units})

        Dense Output Layer:
          Units: {self.vocab_size}
          Activation: Softmax
          Output Shape: (batch_size, {self.vocab_size})

        Total Parameters: {sum(p.numel() for p in self.model.parameters())}
        """

        ax.text(0.05, 0.95, model_summary, transform=ax.transAxes,
                fontsize=10, verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        plt.tight_layout()
        plt.savefig(f"{save_path}/model_architecture.png", dpi=150, bbox_inches='tight')
        plt.close()

    def plot_training_history(self, save_path: str = "visualizations"):
        """
        Plot training and validation metrics.

        Creates two subplots:
        1. Loss over epochs
        2. Accuracy over epochs
        """
        if self.history is None or not self.history['loss']:
            raise ValueError("No training history available!")

        os.makedirs(save_path, exist_ok=True)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        # Loss plot
        ax1.plot(self.history['loss'], label='Training Loss', linewidth=2)
        ax1.plot(self.history['val_loss'], label='Validation Loss', linewidth=2)
        ax1.set_xlabel('Epoch', fontsize=12)
        ax1.set_ylabel('Loss', fontsize=12)
        ax1.set_title('Model Loss Over Time', fontsize=14, fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Accuracy plot
        ax2.plot(self.history['accuracy'], label='Training Accuracy', linewidth=2)
        ax2.plot(self.history['val_accuracy'], label='Validation Accuracy', linewidth=2)
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
        os.makedirs(os.path.dirname(model_path) or '.', exist_ok=True)

        # Save PyTorch model state dict
        torch.save(self.model.state_dict(), model_path)

        # Save tokenizer data in a pickle file (for backward compatibility)
        with open(tokenizer_path, 'wb') as f:
            pickle.dump(self.tokenizer, f)

        # Save configuration with tokenizer data for robustness
        config = {
            'sequence_length': self.sequence_length,
            'embedding_dim': self.embedding_dim,
            'lstm_units': self.lstm_units,
            'num_layers': self.num_layers,
            'dropout_rate': self.dropout_rate,
            'vocab_size': self.vocab_size,
            'max_sequence_len': self.max_sequence_len,
            'word_index': self.tokenizer.word_index,
            'index_word': {str(k): v for k, v in self.tokenizer.index_word.items()}
        }

        config_path = model_path.replace('.pt', '_config.json')
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)

    def load_model(self, model_path: str, tokenizer_path: str):
        """Load saved model and tokenizer."""
        # Load configuration first
        config_path = model_path.replace('.pt', '_config.json')
        with open(config_path, 'r') as f:
            config = json.load(f)

        self.sequence_length = config['sequence_length']
        self.embedding_dim = config['embedding_dim']
        self.lstm_units = config['lstm_units']
        self.num_layers = config['num_layers']
        self.dropout_rate = config['dropout_rate']
        self.vocab_size = config['vocab_size']
        self.max_sequence_len = config['max_sequence_len']

        # Build model architecture
        self.model = LSTMModel(
            vocab_size=self.vocab_size,
            embedding_dim=self.embedding_dim,
            lstm_units=self.lstm_units,
            num_layers=self.num_layers,
            dropout_rate=self.dropout_rate,
            input_length=self.max_sequence_len - 1,
            device=self.device
        )
        self.model.to(self.device)

        # Load model weights
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))

        # Load tokenizer with proper error handling
        try:
            if os.path.exists(tokenizer_path):
                with open(tokenizer_path, 'rb') as f:
                    self.tokenizer = pickle.load(f)
            else:
                raise FileNotFoundError(f"Tokenizer file not found: {tokenizer_path}")
        except Exception as e:
            print(f"⚠️ Warning: Could not load pickled tokenizer: {e}")
            print("📝 Recreating tokenizer from scratch...")
            # If pickle fails, create a new tokenizer - it will be used for generation
            self.tokenizer = SimpleTokenizer()
            # Try to load word_index and index_word from config if available
            if 'word_index' in config and 'index_word' in config:
                word_index_data = config.get('word_index', {})
                index_word_data = config.get('index_word', {})

                # Verify word_index format: should map words (str) -> indices (int)
                # If the keys are numeric strings, it might be corrupted - use index_word instead
                sample_key = next(iter(word_index_data.keys())) if word_index_data else None
                if sample_key and sample_key.isdigit():
                    # word_index is corrupted (has numeric keys), reconstruct from index_word
                    print("⚠️ Warning: word_index appears corrupted, reconstructing from index_word...")
                    self.tokenizer.word_index = {v: int(k) for k, v in index_word_data.items()}
                    self.tokenizer.index_word = {int(k): v for k, v in index_word_data.items()}
                else:
                    # word_index is valid
                    self.tokenizer.word_index = word_index_data
                    self.tokenizer.index_word = {int(k): v for k, v in index_word_data.items()}

def pad_sequences(sequences, maxlen, padding='pre', value=0):
    """
    Pads sequences to the same length.

    Args:
        sequences (list of list of int): List of sequences to pad.
        maxlen (int): Maximum length of the sequences after padding.
        padding (str): 'pre' or 'post', where to add padding.
        value (int): Padding value.

    Returns:
        torch.Tensor: Padded sequences as a tensor.
    """
    padded_sequences = []
    for seq in sequences:
        if len(seq) < maxlen:
            if padding == 'pre':
                padded_seq = [value] * (maxlen - len(seq)) + seq
            elif padding == 'post':
                padded_seq = seq + [value] * (maxlen - len(seq))
        else:
            padded_seq = seq[:maxlen]
        padded_sequences.append(padded_seq)
    return torch.tensor(padded_sequences, dtype=torch.long)