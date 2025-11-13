import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import json
import os
import time
import threading
import psutil
from typing import List, Tuple, Dict, Optional
from DataPreprocessor import LyricsTokenizer, prepare_dataset

# Set up paths
DIR_PATH = os.path.dirname(os.path.realpath(__file__))
MODEL_PATH = os.path.join(DIR_PATH, "model")
CONFIG_PATH = os.path.join(DIR_PATH, "render_pricing_config.json")

# Create model directory if it doesn't exist
os.makedirs(MODEL_PATH, exist_ok=True)

# Set device
print("\n" + "="*70)
print("DEVICE DETECTION")
print("="*70)
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA device count: {torch.cuda.device_count()}")
    print(f"Current CUDA device: {torch.cuda.current_device()}")
    print(f"CUDA device name: {torch.cuda.get_device_name(0)}")
    print(f"CUDA capability: {torch.cuda.get_device_capability(0)}")
    device = torch.device('cuda')
else:
    print("⚠️  CUDA not available - using CPU (training will be slower)")
    device = torch.device('cpu')

print(f"Using device: {device}")
print("="*70 + "\n")

# Initialize variables for cost calculation
peak_memory_usage = 0
training_active = False

# Function to monitor memory usage in background thread
def monitor_memory():
    """Background thread function to track peak memory usage during training"""
    global peak_memory_usage, training_active
    process = psutil.Process()
    while training_active:
        try:
            current_memory = process.memory_info().rss / (1024 * 1024 * 1024)  # Convert to GB
            peak_memory_usage = max(peak_memory_usage, current_memory)
        except Exception as e:
            print(f"Warning: Could not read memory: {e}")
        time.sleep(0.5)  # Check every 500ms


class TrainingTimeEstimator:
    """
    Hybrid approach for estimating remaining training time.
    Uses EMA for batch timing initially, then switches to actual epoch times after warmup.
    """
    
    def __init__(self, total_epochs: int, warmup_epochs: int = 3):
        self.total_epochs = total_epochs
        self.warmup_epochs = warmup_epochs
        self.epoch_times = []
        self.batch_times = []
        self.ema_batch_time = None
        self.ema_alpha = 0.15  # EMA smoothing factor
        self.validation_times = []
        
    def record_batch_time(self, batch_time: float):
        """Record a batch execution time and update EMA"""
        self.batch_times.append(batch_time)
        
        if self.ema_batch_time is None:
            self.ema_batch_time = batch_time
        else:
            # Exponential Moving Average: weight recent values more heavily
            self.ema_batch_time = (self.ema_alpha * batch_time) + ((1 - self.ema_alpha) * self.ema_batch_time)
    
    def record_epoch_time(self, epoch_time: float, validation_time: float):
        """Record full epoch time and validation time"""
        self.epoch_times.append(epoch_time)
        self.validation_times.append(validation_time)
    
    def estimate_remaining_time(self, current_epoch: int, batches_processed: int, total_batches: int) -> Dict[str, str]:
        """
        Estimate remaining training time using hybrid approach.
        
        Args:
            current_epoch: Current epoch number (0-indexed)
            batches_processed: Batches completed in current epoch
            total_batches: Total batches per epoch
            
        Returns:
            Dictionary with estimates
        """
        remaining_epochs = self.total_epochs - current_epoch - 1
        remaining_batches_this_epoch = total_batches - batches_processed
        
        if remaining_epochs < 0:
            remaining_epochs = 0
        
        # Strategy based on training progress
        if current_epoch < self.warmup_epochs or len(self.epoch_times) == 0:
            # Warmup phase: use batch-level EMA
            if self.ema_batch_time is None:
                return {'optimistic': 'N/A', 'realistic': 'N/A', 'pessimistic': 'N/A'}
            
            # Estimate this epoch
            this_epoch_estimate = (self.ema_batch_time * remaining_batches_this_epoch) + (self.validation_times[-1] if self.validation_times else self.ema_batch_time * 5)
            
            # Average across remaining epochs
            remaining_time = this_epoch_estimate + (self.ema_batch_time * total_batches + (self.validation_times[-1] if self.validation_times else self.ema_batch_time * 5)) * remaining_epochs
            
        else:
            # Post-warmup: use actual epoch times
            avg_epoch_time = sum(self.epoch_times) / len(self.epoch_times)
            
            # Detect if we're slowing down (learning rate decrease effects)
            if len(self.epoch_times) >= 2:
                recent_avg = sum(self.epoch_times[-2:]) / 2
                trend_factor = recent_avg / avg_epoch_time if avg_epoch_time > 0 else 1.0
            else:
                trend_factor = 1.0
            
            # Estimate this epoch
            this_epoch_estimate = (self.ema_batch_time * remaining_batches_this_epoch) + (self.validation_times[-1] if self.validation_times else avg_epoch_time * 0.3)
            
            # Remaining epochs with trend adjustment
            remaining_time = this_epoch_estimate + (avg_epoch_time * trend_factor * remaining_epochs)
        
        # Generate estimates with confidence intervals
        optimistic_time = remaining_time * 0.8   # 20% faster
        realistic_time = remaining_time
        pessimistic_time = remaining_time * 1.2  # 20% slower
        
        return {
            'optimistic': self._format_time(optimistic_time),
            'realistic': self._format_time(realistic_time),
            'pessimistic': self._format_time(pessimistic_time),
            'remaining_epochs': remaining_epochs,
            'in_warmup': current_epoch < self.warmup_epochs
        }
    
    @staticmethod
    def _format_time(seconds: float) -> str:
        """Format seconds into human-readable format"""
        if seconds < 60:
            return f"{seconds:.0f}s"
        elif seconds < 3600:
            minutes = seconds / 60
            return f"{minutes:.1f}m"
        else:
            hours = seconds / 3600
            return f"{hours:.2f}h"


class LyricsDataset(Dataset):
    """
    PyTorch Dataset for lyrics training data.
    """
    
    def __init__(self, features: List[List[int]], labels: List[int]):
        self.features = torch.tensor(features, dtype=torch.long)
        self.labels = torch.tensor(labels, dtype=torch.long)
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]


class LyricsLSTM(nn.Module):
    """
    LSTM model for lyric generation following the specified architecture:
    1. Embedding layer (100-dimensional vectors)
    2. Masking capability (handled by PyTorch automatically with padding)
    3. LSTM layer with dropout
    4. Dense layer with ReLU
    5. Dropout layer
    6. Output Dense layer with softmax
    """
    
    def __init__(self, vocab_size: int, embedding_dim: int = 200, hidden_size: int = 512, 
                 num_layers: int = 3, dropout: float = 0.2, pretrained_weights: Optional[torch.Tensor] = None):
        super(LyricsLSTM, self).__init__()
        
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        
        # 1. Embedding layer - maps each input word to 100-dimensional vector
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        
        # Load pretrained weights if provided
        if pretrained_weights is not None:
            self.embedding.weight.data.copy_(pretrained_weights)
            # Set trainable to False if using pretrained embeddings (can be changed later)
            self.embedding.weight.requires_grad = True
        
        # 2. Masking is handled automatically by PyTorch when using padding_idx in embedding
        
        # 3. LSTM layer with dropout - heart of the network
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,  # Only apply dropout if multiple layers
            batch_first=True
        )
        
        # 4. Fully connected Dense layer with ReLU
        self.dense = nn.Linear(hidden_size, hidden_size)
        
        # 5. Dropout layer to prevent overfitting
        self.dropout_layer = nn.Dropout(dropout)
        
        # 6. Output Dense layer - produces probability for every word in vocab
        self.output = nn.Linear(hidden_size, vocab_size)
        
    def forward(self, x, hidden=None):
        batch_size = x.size(0)
        
        # 1. Embedding lookup
        embedded = self.embedding(x)  # (batch_size, seq_len, embedding_dim)
        
        # 2. LSTM forward pass
        lstm_out, hidden = self.lstm(embedded, hidden)  # (batch_size, seq_len, hidden_size)
        
        # Take only the last output for prediction (since we're not returning sequences)
        lstm_out = lstm_out[:, -1, :]  # (batch_size, hidden_size)
        
        # 3. Dense layer with ReLU
        dense_out = F.relu(self.dense(lstm_out))  # (batch_size, hidden_size)
        
        # 4. Dropout
        dropped = self.dropout_layer(dense_out)  # (batch_size, hidden_size)
        
        # 5. Output layer with softmax (applied in loss function for numerical stability)
        output = self.output(dropped)  # (batch_size, vocab_size)
        
        return output, hidden
    
    def init_hidden(self, batch_size):
        """Initialize hidden state for LSTM"""
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(device)
        return (h0, c0)
    
    def freeze_embeddings(self):
        """Freeze embedding weights (set trainable=False)"""
        self.embedding.weight.requires_grad = False
        print("Embedding weights frozen")
    
    def unfreeze_embeddings(self):
        """Unfreeze embedding weights (set trainable=True)"""
        self.embedding.weight.requires_grad = True
        print("Embedding weights unfrozen")
    
    def get_parameter_count(self) -> int:
        """
        Calculate the total number of trainable parameters in the model.
        
        Returns:
            Total number of parameters as an integer
        """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def get_parameter_breakdown(self) -> Dict[str, int]:
        """
        Get a breakdown of parameters by layer.
        
        Returns:
            Dictionary with layer names and their parameter counts
        """
        breakdown = {}
        for name, param in self.named_parameters():
            if param.requires_grad:
                breakdown[name] = param.numel()
        return breakdown


def create_model(vocab_size: int, embedding_dim: int = 200, hidden_size: int = 512,
                num_layers: int = 3, dropout: float = 0.2, learning_rate: float = 0.005,
                pretrained_weights: Optional[torch.Tensor] = None) -> Tuple[LyricsLSTM, optim.Adam]:
    """
    Create LSTM model with Adam optimizer.
    
    Args:
        vocab_size: Size of vocabulary
        embedding_dim: Dimension of embedding vectors (default 100)
        hidden_size: Size of LSTM hidden state
        num_layers: Number of LSTM layers
        dropout: Dropout probability
        learning_rate: Learning rate for Adam optimizer
        pretrained_weights: Optional pretrained embedding weights
    
    Returns:
        Tuple of (model, optimizer)
    """
    # Create model
    model = LyricsLSTM(
        vocab_size=vocab_size,
        embedding_dim=embedding_dim,
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=dropout,
        pretrained_weights=pretrained_weights
    )
    
    # Move model to device
    model = model.to(device)
    
    # Create Adam optimizer
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Get parameter count
    total_params = model.get_parameter_count()
    
    print(f"✓ Model created with {total_params:,} parameters")
    print(f"  - Vocabulary size: {vocab_size}")
    print(f"  - Embedding dimension: {embedding_dim}")
    print(f"  - Hidden size: {hidden_size}")
    print(f"  - Number of layers: {num_layers}")
    print(f"  - Dropout: {dropout}")
    print(f"  - Learning rate: {learning_rate}")
    
    return model, optimizer


def train_model(model: LyricsLSTM, optimizer: optim.Adam, 
               train_features: List[List[int]], train_labels: List[int],
               test_features: List[List[int]], test_labels: List[int],
               batch_size: int = 128, epochs: int = 25, 
               save_path: str = None, clip_grad: float = 0.5) -> Dict:
    """
    Train the LSTM model and track training costs.
    
    Args:
        model: The LSTM model
        optimizer: Adam optimizer
        train_features: Training input sequences
        train_labels: Training labels
        test_features: Testing input sequences 
        test_labels: Testing labels
        batch_size: Batch size for training
        epochs: Number of epochs to train
        save_path: Path to save the trained model
    
    Returns:
        Dictionary with training history
    """
    global peak_memory_usage, training_active
    
    # Reset cost tracking variables
    peak_memory_usage = 0
    training_active = True
    
    # Start memory monitoring in background thread
    monitor_thread = threading.Thread(target=monitor_memory, daemon=True)
    monitor_thread.start()
    
    # Record start time
    train_start_time = time.time()
    
    # Initialize time estimator
    time_estimator = TrainingTimeEstimator(total_epochs=epochs, warmup_epochs=3)
    
    # Create datasets and dataloaders
    train_dataset = LyricsDataset(train_features, train_labels)
    test_dataset = LyricsDataset(test_features, test_labels)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # Loss function (CrossEntropyLoss includes softmax)
    criterion = nn.CrossEntropyLoss(ignore_index=0)  # Ignore padding tokens
    
    # Learning rate scheduler with more aggressive reduction for faster convergence
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.3, 
                                                     patience=3, min_lr=1e-5)
    
    # Early stopping parameters
    best_test_loss = float('inf')
    patience_counter = 0
    early_stop_patience = 10  # Stop if no improvement for 10 epochs
    
    # Training history
    history = {
        'train_loss': [],
        'train_acc': [],
        'test_loss': [],
        'test_acc': [],
        'actual_training_seconds': 0,
        'actual_training_hours': 0,
        'peak_memory_gb': 0
    }
    
    print(f"\nStarting training for {epochs} epochs...")
    print(f"Training samples: {len(train_features):,}")
    print(f"Testing samples: {len(test_features):,}")
    print(f"Batch size: {batch_size}")
    
    for epoch in range(epochs):
        epoch_start_time = time.time()

        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            batch_start_time = time.time()
            data, target = data.to(device), target.to(device)
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Forward pass
            output, _ = model(data)
            loss = criterion(output, target)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping to prevent exploding gradients
            if clip_grad > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
            
            optimizer.step()
            
            # Statistics
            train_loss += loss.item()
            _, predicted = torch.max(output.data, 1)
            train_total += target.size(0)
            train_correct += (predicted == target).sum().item()
            
            # Record batch time
            batch_time = time.time() - batch_start_time
            time_estimator.record_batch_time(batch_time)
            
            if batch_idx % 100 == 0:
                print(f'  Epoch {epoch+1}/{epochs}, Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item():.4f}')
                
                # Get time estimates
                estimates = time_estimator.estimate_remaining_time(epoch, batch_idx, len(train_loader))
                
                if estimates['optimistic'] != 'N/A':
                    print(f"  Batch time: {batch_time:.2f}s")
                    print(f"     Estimated remaining time:")
                    print(f"     Optimistic: {estimates['optimistic']}")
                    print(f"     Realistic:  {estimates['realistic']}")
                    print(f"     Pessimistic: {estimates['pessimistic']}")
                    if estimates.get('in_warmup'):
                        print(f"     (Warmup phase - estimates improve after epoch 3)")
                    print()

        # Calculate average training metrics
        avg_train_loss = train_loss / len(train_loader)
        train_accuracy = 100 * train_correct / train_total
        
        # Testing phase
        model.eval()
        test_loss = 0.0
        test_correct = 0
        test_total = 0
        
        validation_start_time = time.time()
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output, _ = model(data)
                loss = criterion(output, target)
                
                test_loss += loss.item()
                _, predicted = torch.max(output.data, 1)
                test_total += target.size(0)
                test_correct += (predicted == target).sum().item()
        
        validation_time = time.time() - validation_start_time
        
        # Calculate average testing metrics
        avg_test_loss = test_loss / len(test_loader)
        test_accuracy = 100 * test_correct / test_total

        # Store history
        history['train_loss'].append(avg_train_loss)
        history['train_acc'].append(train_accuracy)
        history['test_loss'].append(avg_test_loss)
        history['test_acc'].append(test_accuracy)
        
        # Record epoch time for future estimates
        epoch_time = time.time() - epoch_start_time
        time_estimator.record_epoch_time(epoch_time, validation_time)
        
        # Step the scheduler with validation loss
        scheduler.step(avg_test_loss)
        
        # Early stopping logic
        if avg_test_loss < best_test_loss:
            best_test_loss = avg_test_loss
            patience_counter = 0
            # Save best model
            if save_path:
                best_model_path = save_path.replace('.pth', '_best.pth')
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'history': history,
                    'vocab_size': model.vocab_size,
                    'embedding_dim': model.embedding_dim,
                    'hidden_size': model.hidden_size,
                    'num_layers': model.num_layers,
                    'dropout': model.dropout,
                    'epoch': epoch + 1,
                    'best_test_loss': best_test_loss
                }, best_model_path)
        else:
            patience_counter += 1
        
        print(f'Epoch {epoch+1}/{epochs}:')
        print(f'  Train Loss: {avg_train_loss:.4f}, Train Acc: {train_accuracy:.2f}%')
        print(f'  Test Loss: {avg_test_loss:.4f}, Test Acc: {test_accuracy:.2f}%')
        print(f'  Learning Rate: {optimizer.param_groups[0]["lr"]:.6f}')
        print(f'  Best Test Loss: {best_test_loss:.4f} (Patience: {patience_counter}/{early_stop_patience})')
        
        # Early stopping check
        if patience_counter >= early_stop_patience:
            print(f'\n⚠ Early stopping triggered after {epoch+1} epochs (no improvement for {early_stop_patience} epochs)')
            break
            
        print()
    
    # Save model if path provided
    if save_path:
        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'history': history,
            'vocab_size': model.vocab_size,
            'embedding_dim': model.embedding_dim,
            'hidden_size': model.hidden_size,
            'num_layers': model.num_layers,
            'dropout': model.dropout
        }, save_path)
        print(f"✓ Model saved to: {save_path}")
    
    # Stop memory monitoring
    training_active = False
    monitor_thread.join(timeout=1)
    
    # Calculate actual training time and cost
    train_end_time = time.time()
    actual_training_seconds = train_end_time - train_start_time
    actual_training_hours = actual_training_seconds / 3600
    
    # Store actual metrics in history
    history['actual_training_seconds'] = actual_training_seconds
    history['actual_training_hours'] = actual_training_hours
    history['peak_memory_gb'] = peak_memory_usage
    
    # Calculate actual training cost
    try:
        with open(CONFIG_PATH, 'r', encoding='utf-8') as f:
            pricing_config = json.load(f)
        
        # Calculate compute cost per hour based on actual resource usage
        compute_cost_per_hour = (
            pricing_config.get("cost_per_cpu_per_month", 0) * (1 / 720)  # Convert monthly to hourly
        ) + (
            pricing_config.get("cost_per_gb_ram_per_month", 0) * (1 / 720) * peak_memory_usage
        )
        
        # Calculate storage cost
        storage_cost_per_gb = pricing_config.get("additional_storage_cost_per_gb", 0.10)
        model_size_gb = 0.050  # Model checkpoint ~50MB
        storage_cost = storage_cost_per_gb * model_size_gb
        
        # Total actual training cost
        actual_training_cost = (compute_cost_per_hour * actual_training_hours) + storage_cost
        
        # Store cost in config for frontend access
        pricing_config["actual_training_cost"] = actual_training_cost
        pricing_config["actual_training_hours"] = actual_training_hours
        pricing_config["peak_memory_gb"] = peak_memory_usage
        
        # Save updated config
        with open(CONFIG_PATH, 'w', encoding='utf-8') as f:
            json.dump(pricing_config, f, indent=2)
        
        print(f"\n📊 Training Cost Summary:")
        print(f"   Actual training time: {actual_training_hours:.2f} hours ({actual_training_seconds:.0f} seconds)")
        print(f"   Peak memory usage: {peak_memory_usage:.2f} GB")
        print(f"   Actual training cost: ${actual_training_cost:.6f}")
        print(f"   ✓ Cost saved to {CONFIG_PATH}")
        
    except Exception as e:
        print(f"⚠ Warning: Could not calculate actual training cost: {e}")
    
    return history


def generate_lyrics(model: LyricsLSTM, tokenizer: LyricsTokenizer, seed_text: str,
                   max_length: int = 50, temperature: float = 1.0, top_k: int = 50) -> str:
    """
    Generate lyrics using the trained model with improved sampling.
    
    Args:
        model: Trained LSTM model
        tokenizer: Fitted tokenizer
        seed_text: Starting text for generation
        max_length: Maximum number of words to generate
        temperature: Sampling temperature (higher = more random)
        top_k: Only sample from top k most likely words
    
    Returns:
        Generated lyrics as string
    """
    # Define word blacklist - words to skip during generation
    # These words will be skipped and not count toward the word limit
    WORD_BLACKLIST = {
        # Inappropriate language (add more as needed)
        'damn', 'hell', 'crap', 'shit', 'fuck', 'ass', 'bitch', 'bastard', 
        'nigga', 'nigger', 'niggas', 'niggers', 'fag', 'fags', 'faggot', 'faggots',
        # Structural elements that shouldn't be in lyrics
        'verse', 'chorus', 'intro', 'outro', 'bridge', 'hook', 'pre-chorus',
        'verse1', 'verse2', 'verse3', 'chorus1', 'chorus2',
        '[verse', '[chorus', '[bridge', '[intro', '[outro',  # Bracketed versions
        # Common filler that degrades quality
        'oh', 'ah', 'hmm', 'uh', 'um', 'hey', 'yeah', 'yea', 'nah', 'na',
        # Numbers that shouldn't appear alone (often part of "verse 2" type patterns)
        '1', '2', '3', '4', '5', '6', '7', '8', '9', '0',
        # Unknown token representation
        '<unk>', 'unk', '<UNK>',
    }
    
    model.eval()
    
    # Preprocess seed text to match training data format
    seed_text = seed_text.lower().strip()
    print(f"Debug: Original seed: '{seed_text}'")
    
    # Tokenize seed text
    seed_sequence = tokenizer.texts_to_sequences([seed_text], add_start_end=False)[0]
    print(f"Debug: Seed sequence: {seed_sequence}")
    print(f"Debug: Seed words: {[tokenizer.index_to_word.get(idx, f'UNK({idx})') for idx in seed_sequence]}")
    
    # Debug: Print tokenizer info
    print(f"Debug: Tokenizer vocab_size: {tokenizer.vocab_size}")
    print(f"Debug: Tokenizer end_token: '{tokenizer.end_token}'")
    print(f"Debug: Tokenizer end_token index: {tokenizer.word_to_index.get(tokenizer.end_token, 'NOT FOUND')}")
    print(f"Debug: Tokenizer oov_token: '{tokenizer.oov_token}'")
    print(f"Debug: Tokenizer oov_token index: {tokenizer.word_to_index.get(tokenizer.oov_token, 'NOT FOUND')}")
    
    # Convert blacklist words to indices
    blacklist_indices = set()
    for word in WORD_BLACKLIST:
        idx = tokenizer.word_to_index.get(word, -1)
        if idx != -1:
            blacklist_indices.add(idx)
    
    print(f"Debug: Blacklist has {len(blacklist_indices)} words in vocabulary")
    
    # Start with seed sequence
    generated_sequence = seed_sequence.copy()
    
    with torch.no_grad():
        words_added = 0
        skipped_unknown = 0
        hit_end_token = False
        
        for step in range(max_length):
            # Take last n_words as input (assuming n_words=4 from training)
            input_seq = generated_sequence[-4:] if len(generated_sequence) >= 4 else generated_sequence
            
            # Pad if necessary
            while len(input_seq) < 4:
                input_seq = [tokenizer.word_to_index[tokenizer.pad_token]] + input_seq
            
            # Convert to tensor
            input_tensor = torch.tensor([input_seq], dtype=torch.long).to(device)
            
            # Get prediction
            output, _ = model(input_tensor)
            
            # Debug on first iteration
            if step == 0:
                print(f"Debug: Model output shape: {output.shape}")
                print(f"Debug: Model vocab_size attr: {model.vocab_size}")
            
            # Get first batch element and clamp to valid vocabulary size
            output = output[0, :tokenizer.vocab_size]
            
            # Apply temperature sampling with top-k filtering
            output = output / temperature
            
            # Top-k filtering - only keep top k predictions
            if top_k > 0:
                top_k_values, top_k_indices = torch.topk(output, min(top_k, output.size(-1)))
                # Set all non-top-k values to very negative (will become ~0 after softmax)
                output[output < top_k_values[-1]] = -float('inf')
            
            # Remove special tokens from prediction (avoid generating padding, start, end during middle)
            special_indices = [
                tokenizer.word_to_index.get(tokenizer.pad_token, -1),
                tokenizer.word_to_index.get(tokenizer.start_token, -1),
                tokenizer.word_to_index.get(tokenizer.end_token, -1),  # Also exclude end token from generation
            ]
            for idx in special_indices:
                if idx != -1 and idx < len(output):
                    output[idx] = -float('inf')
            
            # Remove blacklisted words from prediction
            for idx in blacklist_indices:
                if idx < len(output):
                    output[idx] = -float('inf')
            
            # Apply softmax to get probabilities
            probabilities = F.softmax(output, dim=0)
            
            # Handle NaN in probabilities (can happen with all -inf values)
            if torch.isnan(probabilities).any():
                # If all values are filtered out, use uniform distribution over remaining indices
                probabilities = torch.ones_like(output)
                probabilities[output == -float('inf')] = 0
                probabilities = probabilities / probabilities.sum()
            
            # Sample next word
            try:
                next_word_idx = torch.multinomial(probabilities, 1).item()
            except RuntimeError:
                # If sampling fails (e.g., no valid probabilities), pick the highest probability
                next_word_idx = torch.argmax(probabilities).item()
            
            # Skip unknown tokens (oov_token) - continue generating instead of stopping
            if next_word_idx == tokenizer.word_to_index.get(tokenizer.oov_token, -1):
                skipped_unknown += 1
                if step < 15:
                    print(f"Debug: Step {step+1}, SKIPPED UNKNOWN TOKEN - continuing")
                continue
            
            # Skip blacklisted words - continue generating without counting toward limit
            if next_word_idx in blacklist_indices:
                word = tokenizer.index_to_word.get(next_word_idx, f'UNK({next_word_idx})')
                if step < 15:
                    print(f"Debug: Step {step+1}, SKIPPED BLACKLISTED WORD '{word}' - continuing")
                continue
            
            generated_sequence.append(next_word_idx)
            words_added += 1
            
            # Debug: show generation progress
            if step < 10:  # Only show first 10 steps
                word = tokenizer.index_to_word.get(next_word_idx, f'UNK({next_word_idx})')
                print(f"Debug: Step {step+1}, predicted word: '{word}' (idx: {next_word_idx})")
        
        # Print generation statistics
        print(f"Debug: Loop completed after {step+1} iterations")
        print(f"Debug: Words added: {words_added}, Skipped unknown: {skipped_unknown}, Hit end token: {hit_end_token}")
        print(f"Debug: Generated {len(generated_sequence) - len(seed_sequence)} new words (total: {len(generated_sequence)})")
    
    # Convert back to text
    generated_text = tokenizer.sequences_to_texts([generated_sequence], skip_special=True)[0]
    
    # Post-process to remove <UNK> tokens from the final output
    generated_text = generated_text.replace('<UNK>', '').strip()
    # Clean up multiple spaces
    generated_text = ' '.join(generated_text.split())
    
    return generated_text


def load_model(model_path: str) -> Tuple[LyricsLSTM, optim.Adam, Dict]:
    """
    Load a saved model from file.
    
    Args:
        model_path: Path to the saved model file
    
    Returns:
        Tuple of (model, optimizer, metadata)
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    # Load checkpoint
    try:
        # Newer PyTorch versions default to weights_only=True which can fail for full checkpoints.
        checkpoint = torch.load(model_path, map_location=device)
    except Exception as e:
        # If loading fails with the new weights_only behavior, retry with weights_only=False
        err_str = str(e)
        print(f"⚠️ torch.load failed: {err_str}")
        try:
            print("⚙️ Retrying torch.load with weights_only=False (required for older checkpoints)")
            checkpoint = torch.load(model_path, map_location=device, weights_only=False)
            print("✓ torch.load succeeded with weights_only=False")
        except Exception as e2:
            print(f"❌ Retry with weights_only=False also failed: {e2}")
            # Re-raise the original exception to preserve stack trace for upstream handling
            raise
    
    # Recreate model
    model = LyricsLSTM(
        vocab_size=checkpoint['vocab_size'],
        embedding_dim=checkpoint['embedding_dim'],
        hidden_size=checkpoint['hidden_size'],
        num_layers=checkpoint['num_layers'],
        dropout=checkpoint['dropout']
    )
    
    # Load model state
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    
    # Recreate optimizer
    optimizer = optim.Adam(model.parameters())
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    # Get metadata
    metadata = {
        'vocab_size': checkpoint['vocab_size'],
        'embedding_dim': checkpoint['embedding_dim'],
        'hidden_size': checkpoint['hidden_size'],
        'num_layers': checkpoint['num_layers'],
        'dropout': checkpoint['dropout'],
        'history': checkpoint.get('history', {}),
        'model_path': model_path
    }
    
    print(f"✓ Model loaded from: {model_path}")
    print(f"  - Vocabulary size: {metadata['vocab_size']}")
    print(f"  - Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    return model, optimizer, metadata


def save_model_config(model_config: Dict, tokenizer_path: str, model_path: str) -> None:
    """
    Save model configuration and metadata for easy loading.
    
    Args:
        model_config: Dictionary containing model configuration
        tokenizer_path: Path to the tokenizer file
        model_path: Path where the model will be saved
    """
    config = {
        'model_path': model_path,
        'tokenizer_path': tokenizer_path,
        'model_config': model_config,
        'created_at': json.dumps({"timestamp": "generated"}),  # Will be filled during actual training
        'training_completed': False
    }
    
    config_path = os.path.join(MODEL_PATH, "model_config.json")
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"✓ Model configuration saved to: {config_path}")


def get_model_parameters_from_input() -> Dict[str, int]:
    """
    Interactive function to get model parameters from user input.
    Shows defaults and provides suggestions for 75% size reduction.
    
    Returns:
        Dictionary with model parameters (vocab_size, embedding_dim, hidden_size, num_layers)
    """
    # Default values
    DEFAULT_VOCAB_SIZE = 25000
    DEFAULT_EMBEDDING_DIM = 200
    DEFAULT_HIDDEN_SIZE = 512
    DEFAULT_NUM_LAYERS = 3
    
    # Suggested values for 75% size reduction
    SUGGESTED_VOCAB_SIZE = 25000  # Keep same - vocab size doesn't affect memory as much
    SUGGESTED_EMBEDDING_DIM = 100  # 50% reduction
    SUGGESTED_HIDDEN_SIZE = 256    # 50% reduction
    SUGGESTED_NUM_LAYERS = 2       # 33% reduction (3 to 2)
    
    print("\n" + "="*70)
    print("LSTM MODEL PARAMETER CONFIGURATION")
    print("="*70)
    
    print("\n📊 CURRENT DEFAULTS:")
    print(f"  • Vocab Size: {DEFAULT_VOCAB_SIZE:,}")
    print(f"  • Embedding Dimension: {DEFAULT_EMBEDDING_DIM}")
    print(f"  • Hidden Size: {DEFAULT_HIDDEN_SIZE}")
    print(f"  • Number of Layers: {DEFAULT_NUM_LAYERS}")
    print(f"  ➜ Estimated Parameters: ~24 million")
    
    print("\n💡 SUGGESTED VALUES FOR 75% MODEL SIZE REDUCTION:")
    print(f"  • Vocab Size: {SUGGESTED_VOCAB_SIZE:,} (keep same)")
    print(f"  • Embedding Dimension: {SUGGESTED_EMBEDDING_DIM} (50% reduction)")
    print(f"  • Hidden Size: {SUGGESTED_HIDDEN_SIZE} (50% reduction)")
    print(f"  • Number of Layers: {SUGGESTED_NUM_LAYERS} (33% reduction)")
    print(f"  ➜ Estimated Parameters: ~6 million (75% reduction)")
    
    print("\n⚙️  ENTER YOUR PARAMETERS (press Enter to use default):")
    print("-"*70)
    
    # Vocab Size Input
    while True:
        try:
            vocab_input = input(f"Vocab Size [{DEFAULT_VOCAB_SIZE:,}]: ").strip()
            vocab_size = int(vocab_input) if vocab_input else DEFAULT_VOCAB_SIZE
            if vocab_size <= 0:
                print("  ❌ Vocab size must be positive. Try again.")
                continue
            break
        except ValueError:
            print("  ❌ Invalid input. Please enter a number.")
    
    # Embedding Dimension Input
    while True:
        try:
            emb_input = input(f"Embedding Dimension [{DEFAULT_EMBEDDING_DIM}]: ").strip()
            embedding_dim = int(emb_input) if emb_input else DEFAULT_EMBEDDING_DIM
            if embedding_dim <= 0:
                print("  ❌ Embedding dimension must be positive. Try again.")
                continue
            break
        except ValueError:
            print("  ❌ Invalid input. Please enter a number.")
    
    # Hidden Size Input
    while True:
        try:
            hidden_input = input(f"Hidden Size [{DEFAULT_HIDDEN_SIZE}]: ").strip()
            hidden_size = int(hidden_input) if hidden_input else DEFAULT_HIDDEN_SIZE
            if hidden_size <= 0:
                print("  ❌ Hidden size must be positive. Try again.")
                continue
            break
        except ValueError:
            print("  ❌ Invalid input. Please enter a number.")
    
    # Number of Layers Input
    while True:
        try:
            layers_input = input(f"Number of Layers [{DEFAULT_NUM_LAYERS}]: ").strip()
            num_layers = int(layers_input) if layers_input else DEFAULT_NUM_LAYERS
            if num_layers <= 0:
                print("  ❌ Number of layers must be positive. Try again.")
                continue
            break
        except ValueError:
            print("  ❌ Invalid input. Please enter a number.")
    
    # Calculate and display final configuration
    print("\n" + "="*70)
    print("✅ FINAL CONFIGURATION:")
    print("="*70)
    print(f"  • Vocab Size: {vocab_size:,}")
    print(f"  • Embedding Dimension: {embedding_dim}")
    print(f"  • Hidden Size: {hidden_size}")
    print(f"  • Number of Layers: {num_layers}")
    
    # Rough parameter count estimation
    # LSTM params ≈ 4 * hidden_size * (embedding_dim + hidden_size + 1) * num_layers
    # Embedding params = vocab_size * embedding_dim
    # Dense layers ≈ hidden_size * vocab_size + hidden_size^2
    lstm_params = 4 * hidden_size * (embedding_dim + hidden_size + 1) * num_layers
    embedding_params = vocab_size * embedding_dim
    dense_params = hidden_size * hidden_size + hidden_size * vocab_size
    total_params = lstm_params + embedding_params + dense_params
    
    print(f"\n📈 ESTIMATED PARAMETER BREAKDOWN:")
    print(f"  • LSTM Parameters: {lstm_params:,}")
    print(f"  • Embedding Parameters: {embedding_params:,}")
    print(f"  • Dense Layer Parameters: {dense_params:,}")
    print(f"  • Total Parameters: {total_params:,}")
    print(f"  • Approx Memory (float32): {(total_params * 4) / (1024*1024):.1f} MB")
    
    print("="*70 + "\n")
    
    return {
        'vocab_size': vocab_size,
        'embedding_dim': embedding_dim,
        'hidden_size': hidden_size,
        'num_layers': num_layers
    }


def get_model_choice() -> str:
    """
    Prompt user to choose between retraining current model or training a new/specific model.
    
    Returns:
        Path to the model file to save/train
    """
    print("\n" + "="*70)
    print("MODEL SELECTION")
    print("="*70)
    print("\n1. Retrain current model (overwrite best existing)")
    print("2. Train/retrain specific model (enter name)")
    
    while True:
        choice = input("\nEnter choice (1 or 2): ").strip()
        
        if choice == "1":
            # Find current best model
            best_model = os.path.join(MODEL_PATH, "lyrics_model_best.pth")
            fallback_model = os.path.join(MODEL_PATH, "lyrics_model.pth")
            
            if os.path.exists(best_model):
                print(f"\n✓ Will retrain: lyrics_model_best.pth")
                return best_model
            else:
                print(f"\n✓ Will retrain: lyrics_model.pth")
                return fallback_model
        
        elif choice == "2":
            model_name = input("\nEnter model name (e.g., 'lyrics_model_1'): ").strip()
            
            if not model_name:
                print("❌ Model name cannot be empty. Try again.")
                continue
            
            # Ensure .pth extension
            if not model_name.endswith('.pth'):
                model_name += '.pth'
            
            model_path = os.path.join(MODEL_PATH, model_name)
            
            if os.path.exists(model_path):
                print(f"\n✓ Will overwrite existing: {model_name}")
            else:
                print(f"\n✓ Will create new: {model_name}")
            
            return model_path
        
        else:
            print("❌ Invalid choice. Enter 1 or 2.")


if __name__ == "__main__":
    # Improved training with better parameters and techniques
    try:
        print("="*60)
        print("TRAINING IMPROVED LYRICS GENERATION MODEL")
        print("="*60)
        
        # Get model choice
        print("\n0. Selecting model...")
        save_path = get_model_choice()
        
        # Get model parameters from user input
        print("\n1. Configuring model parameters...")
        model_params = get_model_parameters_from_input()
        
        # Load dataset with improved parameters
        print("\n2. Preparing dataset with improved parameters...")
        tokenizer, (train_features, train_labels, test_features, test_labels), metadata = prepare_dataset(
            vocab_size=model_params['vocab_size'],
            max_sequence_length=150, # Longer sequences for better context  
            n_words=4,              # 4-word context window for richer patterns
            train_split=0.85        # More training data
        )
        
        print(f"\nDataset Statistics:")
        print(f"  - Vocabulary size: {metadata['vocab_size']:,}")
        print(f"  - Training samples: {len(train_features):,}")
        print(f"  - Testing samples: {len(test_features):,}")
        print(f"  - Context window: 4 words")
        print(f"  - Max sequence length: 150")
        
        # Check vocabulary coverage
        print(f"\nVocabulary Analysis:")
        print(f"  - Most common words: {list(tokenizer.word_counts.most_common(10))}")
        print(f"  - <UNK> token index: {tokenizer.word_to_index.get(tokenizer.oov_token)}")
        print(f"  - <PAD> token index: {tokenizer.word_to_index.get(tokenizer.pad_token)}")
        
        # Create improved model
        print("\n3. Creating improved model architecture...")
        model_config = {
            'vocab_size': metadata['vocab_size'],
            'embedding_dim': model_params['embedding_dim'],
            'hidden_size': model_params['hidden_size'],
            'num_layers': model_params['num_layers'],
            'dropout': 0.2,
            'learning_rate': 0.005
        }
        
        model, optimizer = create_model(**model_config)
        
        print(f"Model Architecture:")
        print(f"  - Embedding dim: {model_config['embedding_dim']}")
        print(f"  - Hidden size: {model_config['hidden_size']}")  
        print(f"  - Layers: {model_config['num_layers']}")
        print(f"  - Dropout: {model_config['dropout']}")
        print(f"  - Learning rate: {model_config['learning_rate']}")
        
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"  - Total parameters: {total_params:,}")
        print(f"  - Trainable parameters: {trainable_params:,}")
        
        # Save model configuration
        save_model_config(model_config, metadata['tokenizer_path'], save_path)
        
        # Use larger subset for better learning with low accuracy
        subset_size = min(500000, len(train_features))  # 500k samples for much better learning
        test_subset_size = min(50000, len(test_features))  # 50k for more robust testing
        
        print(f"\n4. Training with improved techniques...")
        print(f"Using {subset_size:,} training samples and {test_subset_size:,} test samples")
        
        history = train_model(
            model, optimizer,
            train_features[:subset_size],
            train_labels[:subset_size],
            test_features[:test_subset_size],
            test_labels[:test_subset_size],
            batch_size=128,    # Even larger batch size for more stable learning
            epochs=25,         # More epochs needed for complex model to converge  
            save_path=save_path,
            clip_grad=0.5      # Tighter gradient clipping for stability with larger model
        )
        
        # Save tokenizer alongside model
        # Extract model name without .pth extension for tokenizer naming
        model_name = os.path.basename(save_path).replace('.pth', '')
        tokenizer_path = os.path.join(MODEL_PATH, f"{model_name}_tokenizer.pkl")
        tokenizer.save(tokenizer_path)

        
        
        # Also save a shared tokenizer for convenience (same for all models)
        shared_tokenizer_path = os.path.join(MODEL_PATH, "tokenizer.pkl")
        tokenizer.save(shared_tokenizer_path)
        
        # Test generation with improved parameters
        print("\n5. Testing generation quality...")
        test_seeds = [
            "love is",
            "when the sun", 
            "dancing in the",
            "i feel like",
            "never gonna"
        ]
        
        print("\nGeneration Results (Temperature = 0.8, Top-k = 30):")
        print("-" * 50)
        
        for seed in test_seeds:
            try:
                generated = generate_lyrics(
                    model=model, 
                    tokenizer=tokenizer, 
                    seed_text=seed,
                    max_length=15,
                    temperature=0.8,  # Balanced creativity vs coherence
                    top_k=30         # Focus on top predictions
                )
                print(f"Seed: '{seed}'")
                print(f"Generated: '{generated}'")
                print()
            except Exception as e:
                print(f"Error generating for '{seed}': {e}")
                print()
        
        # Print training summary
        print("\n6. Training Summary:")
        print("-" * 50)
        final_train_loss = history['train_loss'][-1]
        final_train_acc = history['train_acc'][-1]
        final_test_loss = history['test_loss'][-1] 
        final_test_acc = history['test_acc'][-1]
        
        print(f"Final Training Loss: {final_train_loss:.4f}")
        print(f"Final Training Accuracy: {final_train_acc:.2f}%")
        print(f"Final Test Loss: {final_test_loss:.4f}")
        print(f"Final Test Accuracy: {final_test_acc:.2f}%")
        
        # Check for improvement indicators
        if final_train_acc > 15:  # Expecting better than 15% now
            print("✓ Training accuracy improved significantly!")
        else:
            print("⚠ Training accuracy still low - may need more data or training time")
        
        if final_test_loss < final_train_loss * 2:  # Reasonable overfitting
            print("✓ Model is not severely overfitting")
        else:
            print("⚠ Model may be overfitting - consider more regularization")
        
        print(f"\n✓ Model saved to: {save_path}")
        print(f"✓ Tokenizer saved to: {tokenizer_path}")
        print("\n🎉 Training completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Training failed: {str(e)}")
        import traceback
        traceback.print_exc()


def get_model_parameters_from_input() -> Dict[str, int]:
    """
    Interactive function to get model parameters from user input.
    Shows defaults and provides suggestions for 75% size reduction.
    
    Returns:
        Dictionary with model parameters (vocab_size, embedding_dim, hidden_size, num_layers)
    """
    # Default values
    DEFAULT_VOCAB_SIZE = 25000
    DEFAULT_EMBEDDING_DIM = 200
    DEFAULT_HIDDEN_SIZE = 512
    DEFAULT_NUM_LAYERS = 3
    
    # Suggested values for 75% size reduction
    SUGGESTED_VOCAB_SIZE = 25000  # Keep same - vocab size doesn't affect memory as much
    SUGGESTED_EMBEDDING_DIM = 100  # 50% reduction
    SUGGESTED_HIDDEN_SIZE = 256    # 50% reduction
    SUGGESTED_NUM_LAYERS = 2       # 33% reduction (3 to 2)
    
    print("\n" + "="*70)
    print("LSTM MODEL PARAMETER CONFIGURATION")
    print("="*70)
    
    print("\n📊 CURRENT DEFAULTS:")
    print(f"  • Vocab Size: {DEFAULT_VOCAB_SIZE:,}")
    print(f"  • Embedding Dimension: {DEFAULT_EMBEDDING_DIM}")
    print(f"  • Hidden Size: {DEFAULT_HIDDEN_SIZE}")
    print(f"  • Number of Layers: {DEFAULT_NUM_LAYERS}")
    print(f"  ➜ Estimated Parameters: ~24 million")
    
    print("\n💡 SUGGESTED VALUES FOR 75% MODEL SIZE REDUCTION:")
    print(f"  • Vocab Size: {SUGGESTED_VOCAB_SIZE:,} (keep same)")
    print(f"  • Embedding Dimension: {SUGGESTED_EMBEDDING_DIM} (50% reduction)")
    print(f"  • Hidden Size: {SUGGESTED_HIDDEN_SIZE} (50% reduction)")
    print(f"  • Number of Layers: {SUGGESTED_NUM_LAYERS} (33% reduction)")
    print(f"  ➜ Estimated Parameters: ~6 million (75% reduction)")
    
    print("\n⚙️  ENTER YOUR PARAMETERS (press Enter to use default):")
    print("-"*70)
    
    # Vocab Size Input
    while True:
        try:
            vocab_input = input(f"Vocab Size [{DEFAULT_VOCAB_SIZE:,}]: ").strip()
            vocab_size = int(vocab_input) if vocab_input else DEFAULT_VOCAB_SIZE
            if vocab_size <= 0:
                print("  ❌ Vocab size must be positive. Try again.")
                continue
            break
        except ValueError:
            print("  ❌ Invalid input. Please enter a number.")
    
    # Embedding Dimension Input
    while True:
        try:
            emb_input = input(f"Embedding Dimension [{DEFAULT_EMBEDDING_DIM}]: ").strip()
            embedding_dim = int(emb_input) if emb_input else DEFAULT_EMBEDDING_DIM
            if embedding_dim <= 0:
                print("  ❌ Embedding dimension must be positive. Try again.")
                continue
            break
        except ValueError:
            print("  ❌ Invalid input. Please enter a number.")
    
    # Hidden Size Input
    while True:
        try:
            hidden_input = input(f"Hidden Size [{DEFAULT_HIDDEN_SIZE}]: ").strip()
            hidden_size = int(hidden_input) if hidden_input else DEFAULT_HIDDEN_SIZE
            if hidden_size <= 0:
                print("  ❌ Hidden size must be positive. Try again.")
                continue
            break
        except ValueError:
            print("  ❌ Invalid input. Please enter a number.")
    
    # Number of Layers Input
    while True:
        try:
            layers_input = input(f"Number of Layers [{DEFAULT_NUM_LAYERS}]: ").strip()
            num_layers = int(layers_input) if layers_input else DEFAULT_NUM_LAYERS
            if num_layers <= 0:
                print("  ❌ Number of layers must be positive. Try again.")
                continue
            break
        except ValueError:
            print("  ❌ Invalid input. Please enter a number.")
    
    # Calculate and display final configuration
    print("\n" + "="*70)
    print("✅ FINAL CONFIGURATION:")
    print("="*70)
    print(f"  • Vocab Size: {vocab_size:,}")
    print(f"  • Embedding Dimension: {embedding_dim}")
    print(f"  • Hidden Size: {hidden_size}")
    print(f"  • Number of Layers: {num_layers}")
    
    # Rough parameter count estimation
    # LSTM params ≈ 4 * hidden_size * (embedding_dim + hidden_size + 1) * num_layers
    # Embedding params = vocab_size * embedding_dim
    # Dense layers ≈ hidden_size * vocab_size + hidden_size^2
    lstm_params = 4 * hidden_size * (embedding_dim + hidden_size + 1) * num_layers
    embedding_params = vocab_size * embedding_dim
    dense_params = hidden_size * hidden_size + hidden_size * vocab_size
    total_params = lstm_params + embedding_params + dense_params
    
    print(f"\n📈 ESTIMATED PARAMETER BREAKDOWN:")
    print(f"  • LSTM Parameters: {lstm_params:,}")
    print(f"  • Embedding Parameters: {embedding_params:,}")
    print(f"  • Dense Layer Parameters: {dense_params:,}")
    print(f"  • Total Parameters: {total_params:,}")
    print(f"  • Approx Memory (float32): {(total_params * 4) / (1024*1024):.1f} MB")
    
    print("="*70 + "\n")
    
    return {
        'vocab_size': vocab_size,
        'embedding_dim': embedding_dim,
        'hidden_size': hidden_size,
        'num_layers': num_layers
    }
