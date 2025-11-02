# THIS FILE SHOULD NOT RUN ON THE FRONT END. DO NOT HAVE ANY API CALLS TO IT.
# THIS FILE IS SOLELY FOR TOKENIZING AND PREPROCESSING DATASET FOR MODEL TRAINING.
# RUNNING THIS FILE OR ITS FUNCTIONS ON THE FRONT END COULD CAUSE COMPUTATIONAL ERRORS WITH THE LIMITED HARDWARE RESOURCES WE HAVE ON RENDER.

import os
import pickle
import json
from collections import Counter
from typing import List, Tuple, Dict, Optional
from DataImporter import import_spotify_dataset

DIR_PATH = os.path.dirname(os.path.realpath(__file__))
DATA_PATH = os.path.join(DIR_PATH, "data")


class LyricsTokenizer:
    """
    A simple tokenizer for lyrics that converts text to sequences of integers.
    Compatible with PyTorch and other web-app friendly frameworks.
    """
    
    def __init__(self, vocab_size: int = 10000, oov_token: str = "<UNK>", 
                 pad_token: str = "<PAD>", start_token: str = "<START>", 
                 end_token: str = "<END>"):
        self.vocab_size = vocab_size
        self.oov_token = oov_token
        self.pad_token = pad_token
        self.start_token = start_token
        self.end_token = end_token
        
        # Vocabulary mappings
        self.word_to_index = {}
        self.index_to_word = {}
        self.word_counts = Counter()
        self.is_fitted = False
        
    def fit_on_texts(self, texts: List[str]) -> None:
        """
        Build vocabulary from a list of texts.
        
        Args:
            texts: List of text strings to build vocabulary from
        """
        print("Building vocabulary from texts...")
        
        # Count all words
        for text in texts:
            words = text.split()
            self.word_counts.update(words)
        
        # Reserve indices for special tokens
        special_tokens = [self.pad_token, self.oov_token, self.start_token, self.end_token]
        
        # Get most common words (excluding special tokens)
        most_common = self.word_counts.most_common(self.vocab_size - len(special_tokens))
        
        # Build vocabulary
        self.word_to_index = {}
        self.index_to_word = {}
        
        # Add special tokens first
        for i, token in enumerate(special_tokens):
            self.word_to_index[token] = i
            self.index_to_word[i] = token
        
        # Add most common words
        for i, (word, count) in enumerate(most_common):
            index = i + len(special_tokens)
            self.word_to_index[word] = index
            self.index_to_word[index] = word
        
        self.is_fitted = True
        
        print(f"✓ Vocabulary built with {len(self.word_to_index)} words")
        print(f"  - Total unique words in corpus: {len(self.word_counts)}")
        print(f"  - Vocabulary size (including special tokens): {len(self.word_to_index)}")
        
    def texts_to_sequences(self, texts: List[str], 
                          add_start_end: bool = True) -> List[List[int]]:
        """
        Convert texts to sequences of integers.
        
        Args:
            texts: List of text strings to convert
            add_start_end: Whether to add start and end tokens
            
        Returns:
            List of sequences (lists of integers)
        """
        if not self.is_fitted:
            raise ValueError("Tokenizer must be fitted before converting texts to sequences")
        
        sequences = []
        oov_index = self.word_to_index[self.oov_token]
        start_index = self.word_to_index[self.start_token]
        end_index = self.word_to_index[self.end_token]
        
        for text in texts:
            words = text.split()
            sequence = []
            
            if add_start_end:
                sequence.append(start_index)
            
            for word in words:
                index = self.word_to_index.get(word, oov_index)
                sequence.append(index)
            
            if add_start_end:
                sequence.append(end_index)
            
            sequences.append(sequence)
        
        return sequences
    
    def sequences_to_texts(self, sequences: List[List[int]], 
                          skip_special: bool = True) -> List[str]:
        """
        Convert sequences of integers back to texts.
        
        Args:
            sequences: List of sequences (lists of integers)
            skip_special: Whether to skip special tokens in output
            
        Returns:
            List of text strings
        """
        if not self.is_fitted:
            raise ValueError("Tokenizer must be fitted before converting sequences to texts")
        
        texts = []
        special_indices = {self.word_to_index[token] for token in 
                         [self.pad_token, self.start_token, self.end_token]}
        
        for sequence in sequences:
            words = []
            for index in sequence:
                if skip_special and index in special_indices:
                    continue
                word = self.index_to_word.get(index, self.oov_token)
                words.append(word)
            texts.append(' '.join(words))
        
        return texts
    
    def save(self, filepath: str) -> None:
        """Save tokenizer to file."""
        tokenizer_data = {
            'vocab_size': self.vocab_size,
            'oov_token': self.oov_token,
            'pad_token': self.pad_token,
            'start_token': self.start_token,
            'end_token': self.end_token,
            'word_to_index': self.word_to_index,
            'index_to_word': self.index_to_word,
            'word_counts': dict(self.word_counts),
            'is_fitted': self.is_fitted
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(tokenizer_data, f)
        
        print(f"✓ Tokenizer saved to: {filepath}")
    
    @classmethod
    def load(cls, filepath: str) -> 'LyricsTokenizer':
        """Load tokenizer from file."""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        tokenizer = cls(
            vocab_size=data['vocab_size'],
            oov_token=data['oov_token'],
            pad_token=data['pad_token'],
            start_token=data['start_token'],
            end_token=data['end_token']
        )
        
        tokenizer.word_to_index = data['word_to_index']
        tokenizer.index_to_word = data['index_to_word']
        tokenizer.word_counts = Counter(data['word_counts'])
        tokenizer.is_fitted = data['is_fitted']
        
        print(f"✓ Tokenizer loaded from: {filepath}")
        return tokenizer


def pad_sequences(sequences: List[List[int]], maxlen: Optional[int] = None, 
                 padding: str = 'pre', truncating: str = 'pre', 
                 value: int = 0) -> List[List[int]]:
    """
    Pad sequences to the same length.
    
    Args:
        sequences: List of sequences (lists of integers)
        maxlen: Maximum length. If None, uses the length of the longest sequence
        padding: 'pre' or 'post' - where to add padding
        truncating: 'pre' or 'post' - where to truncate if sequence is longer than maxlen
        value: Value to use for padding
        
    Returns:
        List of padded sequences
    """
    if maxlen is None:
        maxlen = max(len(seq) for seq in sequences) if sequences else 0
    
    padded_sequences = []
    
    for seq in sequences:
        # Truncate if necessary
        if len(seq) > maxlen:
            if truncating == 'pre':
                seq = seq[-maxlen:]
            else:  # 'post'
                seq = seq[:maxlen]
        
        # Pad if necessary
        if len(seq) < maxlen:
            pad_length = maxlen - len(seq)
            if padding == 'pre':
                seq = [value] * pad_length + seq
            else:  # 'post'
                seq = seq + [value] * pad_length
        
        padded_sequences.append(seq)
    
    return padded_sequences


def create_features_and_labels(sequences: List[List[int]], n_words: int = 3) -> Tuple[List[List[int]], List[int]]:
    """
    Create features and labels for next-word prediction.
    
    Args:
        sequences: List of tokenized sequences
        n_words: Number of words to use as features (context window)
        
    Returns:
        Tuple of (features, labels) where:
        - features: List of sequences with n_words each
        - labels: List of integers representing the next word
    """
    features = []
    labels = []
    
    for sequence in sequences:
        # Skip sequences that are too short
        if len(sequence) <= n_words:
            continue
            
        # Create sliding windows of n_words + 1
        for i in range(len(sequence) - n_words):
            # Features: n_words
            feature_seq = sequence[i:i + n_words]
            # Label: the next word
            label = sequence[i + n_words]
            
            features.append(feature_seq)
            labels.append(label)
    
    return features, labels


def split_dataset(features: List[List[int]], labels: List[int], 
                 train_split: float = 0.8, random_seed: int = 42) -> Tuple[List[List[int]], List[int], List[List[int]], List[int]]:
    """
    Split dataset into training and testing sets.
    
    Args:
        features: List of feature sequences
        labels: List of labels
        train_split: Fraction of data to use for training (0.0 to 1.0)
        random_seed: Random seed for reproducible splits
        
    Returns:
        Tuple of (train_features, train_labels, test_features, test_labels)
    """
    import random
    
    # Set random seed for reproducibility
    random.seed(random_seed)
    
    # Create indices and shuffle them
    indices = list(range(len(features)))
    random.shuffle(indices)
    
    # Calculate split point
    split_point = int(len(indices) * train_split)
    
    # Split indices
    train_indices = indices[:split_point]
    test_indices = indices[split_point:]
    
    # Create splits
    train_features = [features[i] for i in train_indices]
    train_labels = [labels[i] for i in train_indices]
    test_features = [features[i] for i in test_indices]
    test_labels = [labels[i] for i in test_indices]
    
    return train_features, train_labels, test_features, test_labels


def prepare_dataset(vocab_size: int = 25000, max_sequence_length: int = 150, 
                   n_words: int = 4, train_split: float = 0.85) -> Tuple[LyricsTokenizer, Tuple[List[List[int]], List[int], List[List[int]], List[int]], Dict]:
    """
    Prepare the complete dataset for training with features and labels.
    
    Args:
        vocab_size: Size of vocabulary to build
        max_sequence_length: Maximum length for sequences (will pad/truncate)
        n_words: Number of words to use as features for next-word prediction
        train_split: Fraction of data to use for training (0.0 to 1.0)
        
    Returns:
        Tuple of (tokenizer, (train_features, train_labels, test_features, test_labels), metadata)
    """
    print("\n" + "="*50)
    print("PREPARING DATASET FOR TRAINING")
    print("="*50)
    
    # Import data if it doesn't exist
    lyrics_file = import_spotify_dataset()
    
    # Load preprocessed lyrics
    print(f"\nLoading lyrics from: {lyrics_file}")
    with open(lyrics_file, 'r', encoding='utf-8') as f:
        text_data = f.read()
    
    # Split into individual songs (assuming newline-separated)
    songs = [song.strip() for song in text_data.split('\n') if song.strip()]
    print(f"Loaded {len(songs)} songs")
    
    # Create and fit tokenizer
    tokenizer = LyricsTokenizer(vocab_size=vocab_size)
    tokenizer.fit_on_texts(songs)
    
    # Convert to sequences
    print("Converting texts to sequences...")
    sequences = tokenizer.texts_to_sequences(songs, add_start_end=True)
    
    # Pad sequences
    print(f"Padding sequences to max length: {max_sequence_length}")
    pad_value = tokenizer.word_to_index[tokenizer.pad_token]
    padded_sequences = pad_sequences(sequences, maxlen=max_sequence_length, 
                                   padding='post', truncating='post', 
                                   value=pad_value)
    
    # Create features and labels for next-word prediction
    print("Creating features and labels for next-word prediction...")
    features, labels = create_features_and_labels(padded_sequences, n_words=n_words)
    
    # Split into training and testing sets
    print(f"Splitting dataset (train: {train_split:.1%}, test: {1-train_split:.1%})...")
    train_features, train_labels, test_features, test_labels = split_dataset(
        features, labels, train_split=train_split, random_seed=42
    )
    
    # Save tokenizer
    tokenizer_path = os.path.join(DATA_PATH, "lyrics_tokenizer.pkl")
    tokenizer.save(tokenizer_path)
    
    # Create metadata
    metadata = {
        'num_songs': len(songs),
        'vocab_size': len(tokenizer.word_to_index),
        'max_sequence_length': max_sequence_length,
        'n_words': n_words,
        'train_split': train_split,
        'total_samples': len(features),
        'train_samples': len(train_features),
        'test_samples': len(test_features),
        'pad_token_id': pad_value,
        'oov_token_id': tokenizer.word_to_index[tokenizer.oov_token],
        'start_token_id': tokenizer.word_to_index[tokenizer.start_token],
        'end_token_id': tokenizer.word_to_index[tokenizer.end_token],
        'tokenizer_path': tokenizer_path
    }
    
    # Save metadata
    metadata_path = os.path.join(DATA_PATH, "dataset_metadata.json")
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\n✓ Dataset preparation complete!")
    print(f"  - Number of songs: {metadata['num_songs']}")
    print(f"  - Vocabulary size: {metadata['vocab_size']}")
    print(f"  - Sequence length: {metadata['max_sequence_length']}")
    print(f"  - Context window (n_words): {metadata['n_words']}")
    print(f"  - Total samples: {metadata['total_samples']:,}")
    print(f"  - Training samples: {metadata['train_samples']:,}")
    print(f"  - Testing samples: {metadata['test_samples']:,}")
    print(f"  - Tokenizer saved to: {tokenizer_path}")
    print(f"  - Metadata saved to: {metadata_path}")
    
    return tokenizer, (train_features, train_labels, test_features, test_labels), metadata


if __name__ == "__main__":
    # Example usage: prepare dataset for training
    try:
        tokenizer, (train_features, train_labels, test_features, test_labels), metadata = prepare_dataset(
            vocab_size=15000,  # Larger vocabulary for better word coverage
            max_sequence_length=75,  # Slightly longer sequences for better context
            n_words=4,  # 4-word context window for richer patterns
            train_split=0.85  # More training data (85% vs 80%)
        )
        print(f"\n✓ Dataset preparation successful!")
        print(f"  - Ready for model training")
        print(f"  - Training samples: {len(train_features):,}")
        print(f"  - Testing samples: {len(test_features):,}")
        print(f"  - Vocabulary size: {metadata['vocab_size']}")
        print(f"  - Context window: {metadata['n_words']} words")
        
        # Show example of features and labels
        if train_features and train_labels:
            print(f"\n--- EXAMPLE TRAINING SAMPLE ---")
            example_feature = train_features[0]
            example_label = train_labels[0]
            
            # Convert back to words for display
            feature_words = [tokenizer.index_to_word[idx] for idx in example_feature if idx in tokenizer.index_to_word]
            label_word = tokenizer.index_to_word.get(example_label, tokenizer.oov_token)
            
            print(f"  Features (input): {feature_words}")
            print(f"  Label (target): '{label_word}'")
            print(f"  Raw indices: {example_feature} -> {example_label}")
            
    except Exception as e:
        print(f"\n✗ Dataset preparation failed: {str(e)}")
