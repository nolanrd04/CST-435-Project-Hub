"""
Non-interactive GAN training script for API calls
"""
from pathlib import Path
import torch
from gan_trainer import MultiFruitGANTrainer


def train_model_programmatic(
    model_name: str,
    data_version: str,
    epochs: int = 50,
    batch_size: int = 32,
    learning_rate: float = 0.0002,
    description: str = "",
    fruits: list = None
):
    """
    Train GAN model programmatically without user interaction

    Args:
        model_name: Name for the model
        data_version: Data version to use
        epochs: Number of training epochs
        batch_size: Batch size for training
        learning_rate: Learning rate
        description: Model description
        fruits: List of fruits to train (None = all)
    """
    if fruits is None:
        fruits = ['apple', 'banana', 'blackberry', 'grapes', 'pear', 'strawberry', 'watermelon']

    print(f"\n{'='*60}")
    print(f"STARTING GAN TRAINING")
    print(f"{'='*60}")
    print(f"Model Name: {model_name}")
    print(f"Data Version: {data_version}")
    print(f"Fruits: {', '.join(fruits)}")
    print(f"Epochs: {epochs}")
    print(f"Batch Size: {batch_size}")
    print(f"Learning Rate: {learning_rate}")
    print(f"{'='*60}\n")

    # Initialize trainer
    trainer = MultiFruitGANTrainer(
        model_name=model_name,
        data_version=data_version,
        fruits=fruits,
        description=description
    )

    # Train all fruits
    trainer.train_all_fruits(
        epochs=epochs,
        batch_size=batch_size,
        lr=learning_rate
    )

    print(f"\n{'='*60}")
    print(f"TRAINING COMPLETE!")
    print(f"Model saved to: models/model_{model_name}/")
    print(f"{'='*60}\n")

    return True


if __name__ == "__main__":
    # Example usage for testing
    train_model_programmatic(
        model_name="test_api",
        data_version="v1",
        epochs=5,
        batch_size=32,
        learning_rate=0.0002,
        description="Test model created via API"
    )
