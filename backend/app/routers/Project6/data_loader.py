"""
Data Loader for GAN Training
Loads NPZ datasets with version selection
"""

import numpy as np
import os
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
import torch

# Script-relative directory management
SCRIPT_DIR = Path(__file__).parent
NPZ_DATA_DIR = SCRIPT_DIR / 'npzData'


class FruitDataset(Dataset):
    """
    Custom Dataset for loading fruit images from NPZ files
    """
    
    def __init__(self, data_arrays, normalize=True):
        """
        Initialize the dataset
        
        Args:
            data_arrays (list): List of numpy arrays from NPZ files
            normalize (bool): Whether to normalize images to [-1, 1]
        """
        # Concatenate all arrays
        self.data = np.concatenate(data_arrays, axis=0)
        self.normalize = normalize
        
        # Ensure data is float32
        self.data = self.data.astype(np.float32)
        
        # Normalize to [-1, 1] if requested (required for tanh output)
        if self.normalize:
            self.data = (self.data / 127.5) - 1.0
        
        # Auto-detect number of channels
        if len(self.data.shape) == 3:  # (N, H, W) - grayscale
            self.channels = 1
            # Add channel dimension
            self.data = np.expand_dims(self.data, axis=1)  # (N, 1, H, W)
        elif len(self.data.shape) == 4:  # (N, C, H, W)
            self.channels = self.data.shape[1]
        else:
            raise ValueError(f"Unexpected data shape: {self.data.shape}")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        """
        Get a single image
        
        Args:
            idx (int): Index of the image
            
        Returns:
            torch.Tensor: Image tensor
        """
        image = self.data[idx]
        
        # Convert to tensor
        image_tensor = torch.from_numpy(image).float()
        
        return image_tensor


def get_available_versions(npz_dir=None):
    """
    Get all available dataset versions from the npzData directory
    
    Args:
        npz_dir (str or Path): Path to npzData directory. If None, uses script-relative path.
        
    Returns:
        set: Set of unique version names (e.g., {'v1', 'v2', 'v3'})
    """
    if npz_dir is None:
        npz_dir = NPZ_DATA_DIR
    else:
        npz_dir = Path(npz_dir)
    
    npz_files = list(npz_dir.glob('*.npz'))
    versions = set()
    
    for file in npz_files:
        # Parse filename: fruit_type_version.npz
        # e.g., apple_v1.npz -> v1
        parts = file.stem.split('_')
        if len(parts) >= 2:
            version = parts[-1]  # Get last part as version
            versions.add(version)
    
    return sorted(versions)


def get_fruit_types_for_version(npz_dir=None, version='v1'):
    """
    Get all fruit types available for a specific version
    
    Args:
        npz_dir (str or Path): Path to npzData directory. If None, uses script-relative path.
        version (str): Version name (e.g., 'v1')
        
    Returns:
        list: List of fruit type names
    """
    if npz_dir is None:
        npz_dir = NPZ_DATA_DIR
    else:
        npz_dir = Path(npz_dir)
    
    pattern = f'*_{version}.npz'
    npz_files = list(npz_dir.glob(pattern))
    
    fruit_types = []
    for file in npz_files:
        # Parse filename: fruit_type_version.npz
        parts = file.stem.split('_')
        # Everything except the last part (which is version) is the fruit type
        fruit_type = '_'.join(parts[:-1])
        fruit_types.append(fruit_type)
    
    return sorted(fruit_types)


def load_dataset_for_version(version, npz_dir=None, selected_fruits=None):
    """
    Load all fruit datasets for a specific version
    
    Args:
        version (str): Version name (e.g., 'v1')
        npz_dir (str or Path): Path to npzData directory. If None, uses script-relative path.
        selected_fruits (list): Specific fruits to load. If None, load all.
        
    Returns:
        tuple: (dataset, fruit_types_loaded, channels)
    """
    if npz_dir is None:
        npz_dir = NPZ_DATA_DIR
    else:
        npz_dir = Path(npz_dir)
    
    available_fruits = get_fruit_types_for_version(npz_dir, version)
    
    if selected_fruits is not None:
        # Filter to only requested fruits
        fruits_to_load = [f for f in selected_fruits if f in available_fruits]
    else:
        fruits_to_load = available_fruits
    
    if not fruits_to_load:
        raise ValueError(
            f"No fruit datasets found for version '{version}'. "
            f"Available fruits: {available_fruits}"
        )
    
    print(f"\nLoading fruits for version '{version}':")
    data_arrays = []
    
    for fruit in fruits_to_load:
        file_path = npz_dir / f'{fruit}_{version}.npz'
        
        if file_path.exists():
            try:
                npz_data = np.load(file_path)
                # Get the first array from the npz file
                array_name = list(npz_data.keys())[0]
                images = npz_data[array_name]
                
                print(f"  - {fruit}: {images.shape[0]} images (shape: {images.shape})")
                data_arrays.append(images)
            except Exception as e:
                print(f"  - Error loading {fruit}: {str(e)}")
        else:
            print(f"  - {fruit}: File not found")
    
    total_images = sum(arr.shape[0] for arr in data_arrays)
    print(f"\nTotal images loaded: {total_images}")
    
    dataset = FruitDataset(data_arrays)
    channels = dataset.channels
    
    return dataset, fruits_to_load, channels


def create_data_loader(dataset, batch_size=32, shuffle=True, num_workers=0):
    """
    Create a DataLoader for the dataset
    
    Args:
        dataset (Dataset): The dataset
        batch_size (int): Batch size for training
        shuffle (bool): Whether to shuffle the data
        num_workers (int): Number of worker processes
        
    Returns:
        DataLoader: PyTorch DataLoader
    """
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers
    )
    return dataloader


if __name__ == "__main__":
    # Test the data loader
    # Get available versions
    versions = get_available_versions()
    print(f"Available versions: {versions}")
    
    # Load v1 dataset
    if 'v1' in versions:
        dataset, fruits = load_dataset_for_version('v1')
        print(f"\nLoaded fruits: {fruits}")
        print(f"Dataset size: {len(dataset)}")
        
        # Create dataloader
        dataloader = create_data_loader(dataset, batch_size=32)
        
        # Test a batch
        batch = next(iter(dataloader))
        print(f"Batch shape: {batch.shape}")
        print(f"Batch value range: [{batch.min():.3f}, {batch.max():.3f}]")
