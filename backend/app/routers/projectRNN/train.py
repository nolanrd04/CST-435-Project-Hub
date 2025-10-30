import sys
import os
sys.path.append(os.path.abspath(".."))
from text_generator import TextGenerator

def main():
    """Main training pipeline."""

    # Configuration
    FILE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_PATH = os.path.join(FILE_DIR, "data/training_text.txt")
    MODEL_DIR = os.path.join(FILE_DIR, "saved_models")
    VIZ_DIR = os.path.join(FILE_DIR, "visualizations")

    # Create directories
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(VIZ_DIR, exist_ok=True)

    # Load training data
    print("Loading training data...")
    with open(DATA_PATH, 'r', encoding='utf-8') as f:
        text = f.read()

    print(f"Loaded {len(text)} characters")
    print(f"Unique words: {len(set(text.split()))}")

    # Initialize generator
    print("\nInitializing text generator...")
    generator = TextGenerator(
        sequence_length=50,
        embedding_dim=100,
        lstm_units=150,
        num_layers=2,
        dropout_rate=0.2
    )

    # Prepare sequences
    print("\nPreparing training sequences...")
    X, y, max_seq_len = generator.prepare_sequences(text)

    # Build model
    print("\nBuilding model...")
    model = generator.build_model()
    print(model)

    # Visualize architecture
    print("\nGenerating architecture visualization...")
    generator.visualize_architecture(save_path=VIZ_DIR)

    # Train model
    print("\nTraining model...")
    history = generator.train(
        X, y,
        epochs=100,
        batch_size=128,
        validation_split=0.1,
        save_path=MODEL_DIR
    )

    # Plot training history
    print("\nGenerating training plots...")
    generator.plot_training_history(save_path=VIZ_DIR)

    # Save final model
    print("\nSaving model as model_best.pt and tokenizer.pkl...")
    generator.save_model(
        f"{MODEL_DIR}/model_best.pt",
        f"{MODEL_DIR}/tokenizer.pkl"
    )

    # Test generation
    print("\n" + "="*50)
    print("Testing text generation...")
    print("="*50)

    seed_text = " ".join(text.split()[:10])
    print(f"\nSeed text: '{seed_text}'")

    for temp in [0.5, 1.0, 1.5]:
        print(f"\n--- Temperature: {temp} ---")
        generated = generator.generate_text(seed_text, num_words=50, temperature=temp)
        print(generated)

    print("\n✓ Training complete!")

def train_model():
    # Placeholder logic for model training
    print("Training the model...")

if __name__ == "__main__":
    main()