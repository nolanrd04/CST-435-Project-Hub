import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split
import os
import matplotlib.pyplot as plt

BASE_DIR = os.path.dirname(__file__)
DATASET_PATH = os.path.join(BASE_DIR, "vehicles_dataset.npz")
MODEL_PATH = os.path.join(BASE_DIR, "vehicle_classifier_model.keras")


def load_data(path=DATASET_PATH):
    """Load dataset from .npz file."""
    data = np.load(path, allow_pickle=True)
    X, y = data["X"], data["y"]
    return train_test_split(X, y, test_size=0.2, random_state=42)


def build_model(input_shape=(128, 128, 1), num_classes=3):
    """Build a CNN model with the specified architecture."""
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(
            filters=32,
            kernel_size=(3, 3),
            padding="same",
            activation="relu",
            input_shape=input_shape
        ),
        tf.keras.layers.Conv2D(
            filters=64,
            kernel_size=(3, 3),
            padding="same",
            activation="relu"
        ),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        tf.keras.layers.Conv2D(
            filters=128,
            kernel_size=(3, 3),
            padding="same",
            activation="relu"
        ),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])

    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model


def train_and_evaluate(model, X_train, y_train, X_test, y_test,
                       epochs=50, batch_size=32, validation_split=0.2):
    """Train model and evaluate on test data. Returns history and test metrics."""
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"🚀 Using GPU: {len(gpus)} device(s) found")
        except RuntimeError as e:
            print(f"GPU setup failed: {e}")
    else:
        print("💻 Using CPU for training")
    history = model.fit(
        X_train, y_train,
        batch_size=batch_size,
        epochs=epochs,
        validation_split=validation_split,
        verbose=1
    )

    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"✅ Test accuracy: {test_acc*100:.2f}% | Test loss: {test_loss:.4f}")

    return history, (test_loss, test_acc)


def plot_metrics(history, save_dir=BASE_DIR):
    """Plot training vs validation loss and accuracy."""
    # Accuracy
    plt.figure(figsize=(8, 6))
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title("Model Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, "accuracy_plot.png"))
    plt.close()

    # Loss
    plt.figure(figsize=(8, 6))
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title("Model Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, "loss_plot.png"))
    plt.close()

    print(f"📊 Plots saved to {save_dir}")


def main():
    # Load data
    X_train, X_test, y_train, y_test = load_data()

    # Build and train model
    model = build_model()
    history, (test_loss, test_acc) = train_and_evaluate(
        model, X_train, y_train, X_test, y_test,
        epochs=50
    )

    # Save model + plots
    model.save(MODEL_PATH)
    print(f"💾 Model saved to {MODEL_PATH}")
    plot_metrics(history)


if __name__ == "__main__":
    main()
