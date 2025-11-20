"""
Main Training Script for Multi-Fruit GAN
Handles user interaction and orchestrates training
"""

import sys
from pathlib import Path
import numpy as np

from data_loader import get_available_versions, get_fruit_types_for_version, NPZ_DATA_DIR
from gan_trainer import MultiFruitGANTrainer


def get_user_input(prompt, default=None, input_type=str):
    """
    Get user input with optional default value

    Args:
        prompt (str): Prompt to display
        default: Default value if user presses enter
        input_type (type): Type to convert input to

    Returns:
        User input or default value
    """
    if default is not None:
        prompt = f"{prompt} (default: {default}): "
    else:
        prompt = f"{prompt}: "

    user_input = input(prompt).strip()

    if not user_input and default is not None:
        return default

    try:
        return input_type(user_input)
    except ValueError:
        print(f"Invalid input. Please enter a valid {input_type.__name__}.")
        return get_user_input(prompt, default, input_type)


def detect_image_resolution(data_version):
    """
    Auto-detect image resolution from NPZ data files

    Args:
        data_version (str): Dataset version

    Returns:
        int: Image resolution (width/height, assumes square images)
    """
    # Get first available fruit for this version
    fruits = get_fruit_types_for_version(version=data_version)

    if not fruits:
        print(f"Warning: No fruits found for version '{data_version}'")
        return 28  # Default fallback

    # Load first NPZ file to check dimensions
    npz_file = Path(NPZ_DATA_DIR) / f'{fruits[0]}_{data_version}.npz'

    try:
        npz_data = np.load(npz_file)
        array_name = list(npz_data.keys())[0]
        images = npz_data[array_name]

        # Get image shape
        if len(images.shape) == 3:  # (N, H, W)
            img_size = images.shape[1]  # Height
        elif len(images.shape) == 4:  # (N, C, H, W) or (N, H, W, C)
            # Assume it's (N, H, W, C) if last dim is small (1 or 3)
            if images.shape[-1] in [1, 3]:
                img_size = images.shape[1]  # Height
            else:
                img_size = images.shape[2]  # Height in (N, C, H, W)
        else:
            img_size = 28  # Fallback

        print(f"Auto-detected image resolution: {img_size}x{img_size}")
        return img_size

    except Exception as e:
        print(f"Warning: Could not auto-detect resolution ({e}). Using default: 28x28")
        return 28


def select_data_version():
    """
    Prompt user to select a dataset version

    Returns:
        str: Selected version name
    """
    print("\n" + "="*60)
    print("STEP 1: Select Dataset Version")
    print("="*60)

    available_versions = get_available_versions()

    if not available_versions:
        print("Error: No dataset versions found in npzData directory!")
        print("Please run imageToNPZ.py first to create dataset files.")
        sys.exit(1)

    print("\nAvailable dataset versions:")
    for i, version in enumerate(available_versions, 1):
        print(f"  {i}. {version}")

    # Check if v1 exists for default
    default_version = 'v1' if 'v1' in available_versions else available_versions[0]

    # Get selection
    while True:
        try:
            choice = input(f"\nSelect version (1-{len(available_versions)}) or enter version name [default: {default_version}]: ").strip()

            # Default to v1 if empty
            if not choice:
                return default_version

            # Check if it's a number
            if choice.isdigit():
                idx = int(choice) - 1
                if 0 <= idx < len(available_versions):
                    return available_versions[idx]
                else:
                    print(f"Invalid choice. Please enter 1-{len(available_versions)}.")
            # Check if it's a valid version name
            elif choice in available_versions:
                return choice
            else:
                print(f"Invalid version. Available: {', '.join(available_versions)}")
        except (ValueError, IndexError):
            print("Invalid input. Please try again.")


def select_model_name():
    """
    Prompt user to enter a model name

    Returns:
        str: Model name
    """
    print("\n" + "="*60)
    print("STEP 2: Select Model Name")
    print("="*60)

    print("\nThis will be used to create a model folder: models/model_{name}/")
    print("Examples: v1, v2, attempt_2, baseline, etc.")

    model_name = get_user_input("Enter model name", default="v1")

    # Check if model already exists
    model_dir = Path(__file__).parent / 'models' / f'model_{model_name}'
    if model_dir.exists():
        print(f"\nWarning: Model '{model_name}' already exists!")
        overwrite = input("Do you want to overwrite it? (yes/no): ").strip().lower()
        if overwrite not in ['yes', 'y']:
            print("Please choose a different model name.")
            return select_model_name()

    return model_name


def select_fruits(data_version):
    """
    Prompt user to select which fruits to train

    Args:
        data_version (str): Dataset version

    Returns:
        list: List of selected fruit names
    """
    print("\n" + "="*60)
    print("STEP 3: Select Fruits to Train")
    print("="*60)

    available_fruits = get_fruit_types_for_version(version=data_version)

    if not available_fruits:
        print(f"Error: No fruits found for version '{data_version}'!")
        sys.exit(1)

    print(f"\nAvailable fruits in version '{data_version}':")
    for i, fruit in enumerate(available_fruits, 1):
        print(f"  {i}. {fruit}")

    print("\nOptions:")
    print("  - Press Enter to train all fruits (default)")
    print("  - Enter 'all' to train all fruits")
    print("  - Enter numbers separated by commas (e.g., 1,3,5)")
    print("  - Enter fruit names separated by commas (e.g., apple,banana)")

    selection = input("\nSelect fruits [default: all]: ").strip().lower()

    # Default to all fruits if empty
    if not selection or selection == 'all':
        return available_fruits

    # Parse selection
    selected_fruits = []

    # Try parsing as numbers
    if ',' in selection or selection.isdigit():
        try:
            indices = [int(x.strip()) - 1 for x in selection.split(',')]
            selected_fruits = [available_fruits[i] for i in indices
                             if 0 <= i < len(available_fruits)]
        except (ValueError, IndexError):
            pass

    # Try parsing as fruit names
    if not selected_fruits:
        fruit_names = [x.strip() for x in selection.split(',')]
        selected_fruits = [f for f in fruit_names if f in available_fruits]

    if not selected_fruits:
        print("Invalid selection. Please try again.")
        return select_fruits(data_version)

    return selected_fruits


def configure_training():
    """
    Prompt user for training parameters

    Returns:
        dict: Training configuration
    """
    print("\n" + "="*60)
    print("STEP 4: Configure Training Parameters")
    print("="*60)

    print("\nNote: Recommended values are shown as defaults.")

    config = {
        'epochs': get_user_input("Number of epochs", default=400, input_type=int),
        'batch_size': get_user_input("Batch size", default=32, input_type=int),
        'learning_rate': get_user_input("Learning rate", default=0.0002, input_type=float),
        'beta1': get_user_input("Beta1 for Adam optimizer", default=0.5, input_type=float),
        'latent_dim': get_user_input("Latent dimension (noise vector size)", default=100, input_type=int)
    }

    return config


def get_model_description(data_version):
    """
    Prompt user for model description

    Args:
        data_version (str): Dataset version being used

    Returns:
        str: User-provided model description
    """
    print("\n" + "="*60)
    print("STEP 5: Model Description")
    print("="*60)

    print("\nProvide a brief description of this model (1-3 sentences).")
    print("This will be saved in models/model_{name}/info/description.txt")
    print("\nExample: 'A more advanced model trained on a larger dataset with higher resolution images.'")
    print("Example: 'Baseline model for initial testing and comparison.'")

    description = input("\nModel description: ").strip()

    while not description:
        print("Description cannot be empty. Please provide a description.")
        description = input("\nModel description: ").strip()

    return description


def display_training_summary(data_version, model_name, fruits, config):
    """
    Display training summary before starting

    Args:
        data_version (str): Dataset version
        model_name (str): Model name
        fruits (list): List of fruits to train
        config (dict): Training configuration
    """
    print("\n" + "="*60)
    print("TRAINING SUMMARY")
    print("="*60)

    print(f"\nDataset Version: {data_version}")
    print(f"Model Name: {model_name}")
    print(f"Fruits to train: {', '.join(fruits)} ({len(fruits)} total)")

    print(f"\nTraining Configuration:")
    for key, value in config.items():
        print(f"  {key}: {value}")

    print(f"\nOutput directory: models/model_{model_name}/")
    print(f"\nEstimated training time per fruit: ~{config['epochs'] * 0.1:.1f} minutes")
    print(f"Total estimated time: ~{config['epochs'] * 0.1 * len(fruits):.1f} minutes")

    print("\n" + "="*60)


def main():
    """
    Main training function
    """
    print("="*60)
    print("Multi-Fruit GAN Training Script")
    print("="*60)

    # Step 1: Select dataset version
    data_version = select_data_version()

    # Auto-detect image resolution
    img_size = detect_image_resolution(data_version)

    # Step 2: Select model name
    model_name = select_model_name()

    # Step 3: Select fruits to train
    fruits = select_fruits(data_version)

    # Step 4: Configure training parameters
    config = configure_training()

    # Add image size to config
    config['img_size'] = img_size

    # Step 5: Get model description
    model_description = get_model_description(data_version)

    # Display summary
    display_training_summary(data_version, model_name, fruits, config)

    # Confirm before starting
    confirm = input("\nStart training? (yes/no): ").strip().lower()
    if confirm not in ['yes', 'y']:
        print("Training cancelled.")
        sys.exit(0)

    # Initialize trainer
    print("\nInitializing trainer...")
    trainer = MultiFruitGANTrainer(
        model_name=model_name,
        data_version=data_version,
        config=config
    )

    # Start training
    print("\nStarting training...")
    histories = trainer.train_all_fruits(fruits, model_description=model_description)

    # Final summary
    print("\n" + "="*60)
    print("TRAINING COMPLETED SUCCESSFULLY!")
    print("="*60)

    print(f"\nModels saved in: models/model_{model_name}/")
    print(f"Training histories saved in: models/model_{model_name}/info/")

    print("\nYou can now use generate_images.py to create images:")
    print(f"  python generate_images.py {model_name} {fruits[0]} --num-images 16")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\n\nError: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
