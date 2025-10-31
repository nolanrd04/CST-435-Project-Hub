import sys
import os
import json
import time
sys.path.append(os.path.abspath(".."))
from text_generator import TextGenerator
import psutil

# Local Training Cost = (Compute Cost/hour × Training Hours) + (Storage Cost/GB × Dataset Size)
FILE_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(FILE_DIR, "render_pricing_config.json")
MODEL_PATH = os.path.join(FILE_DIR, "saved_models", "model_best.pt")

with open(CONFIG_PATH, 'r', encoding='utf-8') as f:
    pricing_config = json.load(f)

# model file size in mb
storage_cost = pricing_config["database_cost_per_gb"]
file_size_bytes = os.path.getsize(MODEL_PATH)
file_size_mb = file_size_bytes / (1024 * 1024)

# cost for training resources
process = psutil.Process()
memory_usage_mb = process.memory_info().rss / (1024 * 1024)
compute_cost_per_hour = (pricing_config["cost_per_cpu_per_month"] * (1 / 720)) + (pricing_config["cost_per_gb_ram_per_month"] * (1 / 720) * memory_usage_mb) # Assuming 720 hours in a month

training_hours = 0

start_time = time.time()

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


end_time = time.time()
elapsed_time = end_time - start_time
print(f"Elapsed time: {elapsed_time:.2f} seconds")

training_hours = elapsed_time / 3600
local_training_cost = (compute_cost_per_hour * training_hours) + (storage_cost * (file_size_mb / 1024))

# Save local_training_cost to render_pricing_config.json
pricing_config["local_training_cost"] = local_training_cost

with open(CONFIG_PATH, 'w', encoding='utf-8') as f:
    json.dump(pricing_config, f, indent=2)

print(f"Local training cost saved to {CONFIG_PATH}")

