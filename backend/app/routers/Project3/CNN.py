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
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Set up paths
DIR_PATH = os.path.dirname(os.path.realpath(__file__))
MODEL_DIR = os.path.join(DIR_PATH, "model")
CONFIG_PATH = os.path.join(DIR_PATH, "render_pricing_config.json")
DEFAULT_DATASET_PATH = os.path.join(DIR_PATH, "vehicles_dataset.npz")

# Create model directory if it doesn't exist
os.makedirs(MODEL_DIR, exist_ok=True)

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


def load_pricing_config():
    """Load pricing config from JSON file, with defaults if not found."""
    if os.path.exists(CONFIG_PATH):
        with open(CONFIG_PATH, 'r') as f:
            return json.load(f)
    else:
        # Return default config if file doesn't exist
        return {
            "fixed_monthly_cost": 19.0,
            "available_cpus": 64.0,
            "available_ram_gb": 512.0,
            "included_bandwidth_gb": 500.0,
            "build_pipeline_minutes": 500.0,
            "database_cost_per_gb": 0.25,
            "cost_per_cpu_per_month": 19.0 / 64.0,
            "cost_per_gb_ram_per_month": 19.0 / 512.0,
            "overage_bandwidth_cost_per_gb": 0.10,
            "overage_build_minutes_cost": 0.01,
            "additional_storage_cost_per_gb": 0.10,
        }


class RenderPricingConfig:
    """Render platform pricing configuration loaded from config file."""

    def __init__(self):
        # Load pricing from config file
        config = load_pricing_config()

        # Store loaded values
        self.fixed_monthly_cost = config.get("fixed_monthly_cost", 19.0)
        self.available_cpus = config.get("available_cpus", 64.0)
        self.available_ram_gb = config.get("available_ram_gb", 512.0)
        self.included_bandwidth_gb = config.get("included_bandwidth_gb", 500.0)
        self.build_pipeline_minutes = config.get("build_pipeline_minutes", 500.0)
        self.database_cost_per_gb = config.get("database_cost_per_gb", 0.25)

        # Calculate derived pricing values
        self.cost_per_cpu_per_month = self.fixed_monthly_cost / self.available_cpus
        self.cost_per_gb_ram_per_month = self.fixed_monthly_cost / self.available_ram_gb
        self.overage_bandwidth_cost_per_gb = config.get("overage_bandwidth_cost_per_gb", 0.10)
        self.overage_build_minutes_cost = config.get("overage_build_minutes_cost", 0.01)
        self.additional_storage_cost_per_gb = config.get("additional_storage_cost_per_gb", 0.10)


class CNNTrainingCostModel:
    """Cost model for CNN vehicle classifier training on Render"""

    def __init__(self, pricing_config: RenderPricingConfig = None):
        self.pricing = pricing_config or RenderPricingConfig()

        # Model specifications (CNN for vehicle classification)
        self.model_specs = {
            "input_shape": (128, 128, 1),
            "num_classes": 3,
            "conv_filters": [32, 64, 128],
            "dense_units": 128,
            "model_size_mb": 16.9,  # Approximate size for this architecture
            "dataset_size_gb": 0.026,  # 26MB compressed dataset
        }

        # Training specifications (defaults)
        self.training_specs = {
            "batch_size": 32,
            "epochs": 50,
            "learning_rate": 0.001,
            "estimated_training_hours": 1.0,  # Conservative estimate
            "early_stop_patience": 7,
        }

        # Compute specifications
        self.compute_specs = {
            "cpu_per_worker": 2.0,  # Conservative CPU allocation
            "ram_per_worker_gb": 4.0,  # Conservative RAM allocation
            "gpu_memory_gb": 4.0 if torch.cuda.is_available() else 0.0,
        }

    def calculate_training_cost(self, training_hours: float = None) -> Dict[str, float]:
        """Calculate total training cost breakdown"""
        if training_hours is None:
            training_hours = self.training_specs["estimated_training_hours"]

        # CPU cost (prorated monthly cost)
        cpu_cost = (self.compute_specs["cpu_per_worker"] * self.pricing.cost_per_cpu_per_month) * (training_hours / (24 * 30))

        # RAM cost (prorated monthly cost)
        ram_cost = (self.compute_specs["ram_per_worker_gb"] * self.pricing.cost_per_gb_ram_per_month) * (training_hours / (24 * 30))

        # Storage cost (model checkpoints, logs, etc.)
        storage_gb = self.model_specs["model_size_mb"] / 1024  # Convert MB to GB
        storage_cost = storage_gb * self.pricing.additional_storage_cost_per_gb * (training_hours / (24 * 30))

        # Data transfer cost (minimal for this use case)
        data_transfer_gb = 0.1  # Conservative estimate for logs/checkpoints
        data_transfer_cost = max(0, data_transfer_gb - self.pricing.included_bandwidth_gb) * self.pricing.overage_bandwidth_cost_per_gb

        # GPU cost (if applicable) - rough estimate
        gpu_cost = 0.0
        if self.compute_specs["gpu_memory_gb"] > 0:
            # Assume GPU costs ~$0.50-1.00 per hour for consumer GPU
            gpu_cost = training_hours * 0.75

        total_cost = cpu_cost + ram_cost + storage_cost + data_transfer_cost + gpu_cost

        return {
            "compute": cpu_cost,
            "memory": ram_cost,
            "storage": storage_cost,
            "data_transfer": data_transfer_cost,
            "gpu": gpu_cost,
            "total": total_cost,
            "training_hours": training_hours,
        }

    def calculate_cost_per_epoch(self, training_hours: float = None) -> Dict[str, float]:
        """Calculate cost per training epoch"""
        if training_hours is None:
            training_hours = self.training_specs["estimated_training_hours"]

        total_cost = self.calculate_training_cost(training_hours)["total"]
        epochs = self.training_specs["epochs"]

        cost_per_epoch = total_cost / epochs
        hours_per_epoch = training_hours / epochs
        cost_per_hour = total_cost / training_hours

        return {
            "cost_per_epoch": cost_per_epoch,
            "hours_per_epoch": hours_per_epoch,
            "cost_per_hour": cost_per_hour,
        }

    def generate_training_cost_report(self, actual_training_hours: float = None) -> Dict:
        """Generate comprehensive training cost report"""
        training_cost = self.calculate_training_cost(actual_training_hours)
        cost_per_epoch = self.calculate_cost_per_epoch(actual_training_hours)

        # Parameter scenarios for different configurations
        scenarios = {}
        base_cost = training_cost["total"]

        # Smaller model scenario
        small_model = CNNTrainingCostModel(self.pricing)
        small_model.model_specs["model_size_mb"] = 8.0
        small_model.training_specs["estimated_training_hours"] = 0.5
        scenarios["small_model"] = small_model.calculate_training_cost()["total"]

        # Larger model scenario
        large_model = CNNTrainingCostModel(self.pricing)
        large_model.model_specs["model_size_mb"] = 32.0
        large_model.training_specs["estimated_training_hours"] = 2.0
        scenarios["large_model"] = large_model.calculate_training_cost()["total"]

        return {
            "model_specs": self.model_specs,
            "training_specs": self.training_specs,
            "compute_specs": self.compute_specs,
            "training_cost": training_cost,
            "cost_per_epoch": cost_per_epoch,
            "parameter_scenarios": scenarios,
        }

    def get_cost_summary(self, actual_training_hours: float = None) -> Dict[str, str]:
        """Get formatted cost summary for display"""
        report = self.generate_training_cost_report(actual_training_hours)
        cost = report["training_cost"]

        return {
            "total_cost": ".4f",
            "compute_cost": ".4f",
            "memory_cost": ".4f",
            "storage_cost": ".4f",
            "training_hours": ".1f",
            "cost_per_hour": ".4f",
        }


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
        time.sleep(0.5)


class TrainingTimeEstimator:
    """
    Hybrid approach for estimating remaining training time.
    Uses EMA for batch timing initially, then switches to actual epoch times after warmup.
    """
    
    def __init__(self, total_epochs: int = None, warmup_epochs: int = 2, num_samples: int = None, batch_size: int = None, num_epochs: int = None):
        self.total_epochs = total_epochs or num_epochs
        self.warmup_epochs = warmup_epochs
        self.num_samples = num_samples
        self.batch_size = batch_size
        self.epoch_times = []
        self.batch_times = []
        self.ema_batch_time = None
        self.ema_alpha = 0.15
        self.validation_times = []
        
    def estimate_total_time(self) -> float:
        """
        Estimate total training time based on dataset size.
        Conservative estimate: 50ms per image per epoch + 10% validation overhead
        """
        if self.num_samples is None or self.batch_size is None or self.total_epochs is None:
            return 0  # Can't estimate without info
        
        # Rough estimate: 50ms per image per epoch (conservative for GPU)
        time_per_image_per_epoch = 0.05
        total_time = self.num_samples * time_per_image_per_epoch * self.total_epochs
        
        # Add validation overhead (roughly 10% of total)
        total_time *= 1.1
        
        return total_time
        
    def record_batch_time(self, batch_time: float):
        """Record a batch execution time and update EMA"""
        self.batch_times.append(batch_time)
        
        if self.ema_batch_time is None:
            self.ema_batch_time = batch_time
        else:
            self.ema_batch_time = (self.ema_alpha * batch_time) + ((1 - self.ema_alpha) * self.ema_batch_time)
    
    def record_epoch_time(self, epoch_time: float, validation_time: float):
        """Record full epoch time and validation time"""
        self.epoch_times.append(epoch_time)
        self.validation_times.append(validation_time)
    
    def estimate_remaining_time(self, current_epoch: int, batches_processed: int, total_batches: int) -> Dict[str, str]:
        """Estimate remaining training time using hybrid approach"""
        remaining_epochs = self.total_epochs - current_epoch - 1
        remaining_batches_this_epoch = total_batches - batches_processed
        
        if remaining_epochs < 0:
            remaining_epochs = 0
        
        # Strategy based on training progress
        if current_epoch < self.warmup_epochs or len(self.epoch_times) == 0:
            # Warmup phase: use batch-level EMA
            if self.ema_batch_time is None:
                return {'optimistic': 'N/A', 'realistic': 'N/A', 'pessimistic': 'N/A', 'remaining_epochs': remaining_epochs, 'in_warmup': True}
            
            this_epoch_estimate = (self.ema_batch_time * remaining_batches_this_epoch) + (self.validation_times[-1] if self.validation_times else self.ema_batch_time * 5)
            remaining_time = this_epoch_estimate + (self.ema_batch_time * total_batches + (self.validation_times[-1] if self.validation_times else self.ema_batch_time * 5)) * remaining_epochs
            
        else:
            # Post-warmup: use actual epoch times
            avg_epoch_time = sum(self.epoch_times) / len(self.epoch_times)
            
            if len(self.epoch_times) >= 2:
                recent_avg = sum(self.epoch_times[-2:]) / 2
                trend_factor = recent_avg / avg_epoch_time if avg_epoch_time > 0 else 1.0
            else:
                trend_factor = 1.0
            
            this_epoch_estimate = (self.ema_batch_time * remaining_batches_this_epoch) + (self.validation_times[-1] if self.validation_times else avg_epoch_time * 0.3)
            remaining_time = this_epoch_estimate + (avg_epoch_time * trend_factor * remaining_epochs)
        
        # Generate estimates with confidence intervals
        optimistic_time = remaining_time * 0.8
        realistic_time = remaining_time
        pessimistic_time = remaining_time * 1.2
        
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


class VehicleDataset(Dataset):
    """PyTorch Dataset for vehicle images"""
    
    def __init__(self, images: np.ndarray, labels: np.ndarray):
        # Convert to tensors
        self.images = torch.from_numpy(images).float()
        self.labels = torch.from_numpy(labels).long()
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        return self.images[idx], self.labels[idx]


class VehicleCNN(nn.Module):
    """
    CNN model for vehicle classification
    """
    
    def __init__(self, input_shape: Tuple[int, int, int] = (128, 128, 1), 
                 num_classes: int = 3, conv_filters: List[int] = None,
                 dense_units: int = 128, dropout: float = 0.0):
        super(VehicleCNN, self).__init__()
        
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.conv_filters = conv_filters or [32, 64, 128]
        self.dense_units = dense_units
        self.dropout = dropout
        
        # Input validation
        if len(input_shape) != 3:
            raise ValueError("input_shape must be (height, width, channels)")
        
        height, width, channels = input_shape
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(channels, self.conv_filters[0], kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(self.conv_filters[0], self.conv_filters[1], kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv3 = nn.Conv2d(self.conv_filters[1], self.conv_filters[2], kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Calculate flattened size after convolutions and pooling
        flat_height = height // 4
        flat_width = width // 4
        self.flatten_size = flat_height * flat_width * self.conv_filters[2]
        
        # Dense layers
        self.dense1 = nn.Linear(self.flatten_size, self.dense_units)
        self.dropout_layer = nn.Dropout(self.dropout)
        self.output = nn.Linear(self.dense_units, self.num_classes)
    
    def forward(self, x):
        # Conv block 1
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool1(x)
        
        # Conv block 2
        x = F.relu(self.conv3(x))
        x = self.pool2(x)
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # Dense layers
        x = F.relu(self.dense1(x))
        x = self.dropout_layer(x)
        x = self.output(x)
        
        return x
    
    def get_parameter_count(self) -> int:
        """Calculate total trainable parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def load_dataset(dataset_path: str = DEFAULT_DATASET_PATH):
    """Load dataset from .npz file and split into train/test sets"""
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset not found at: {dataset_path}")
    
    print(f"📂 Loading dataset from: {dataset_path}")
    data = np.load(dataset_path, allow_pickle=True)
    
    X = data['X'].astype(np.float32)
    y = data['y'].astype(np.float32)
    
    print(f"✓ Dataset loaded")
    print(f"  - Images shape (original): {X.shape}")
    print(f"  - Labels shape: {y.shape}")
    
    # Convert from NHWC (TensorFlow/NumPy) to NCHW (PyTorch) format
    # X shape: (N, 128, 128, 1) -> (N, 1, 128, 128)
    X = np.transpose(X, (0, 3, 1, 2))
    
    print(f"  - Images shape (PyTorch NCHW): {X.shape}")
    print(f"  - Image size: {X.shape[2]}x{X.shape[3]}, Channels: {X.shape[1]}")
    
    # Split into train/test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print(f"✓ Split into train/test (80/20)")
    print(f"  - Training samples: {len(X_train):,}")
    print(f"  - Testing samples: {len(X_test):,}")
    
    return X_train, X_test, y_train, y_test


def create_model(input_shape=(128, 128, 1), num_classes=3, conv_filters=None, 
                dense_units=128, dropout=0.0, learning_rate=0.001):
    """Create CNN model with Adam optimizer"""
    if conv_filters is None:
        conv_filters = [32, 64, 128]
    
    model = VehicleCNN(
        input_shape=input_shape,
        num_classes=num_classes,
        conv_filters=conv_filters,
        dense_units=dense_units,
        dropout=dropout
    )
    
    model = model.to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    total_params = model.get_parameter_count()
    
    print(f"✓ Model created with {total_params:,} parameters")
    print(f"  - Input shape: {input_shape}")
    print(f"  - Conv filters: {conv_filters}")
    print(f"  - Dense units: {dense_units}")
    print(f"  - Dropout: {dropout}")
    print(f"  - Learning rate: {learning_rate}")
    print(f"  - Device: {device}")
    
    return model, optimizer


def train_model(model, optimizer, X_train, y_train, X_test, y_test,
               batch_size=32, epochs=50, save_path=None, time_estimator=None):
    """Train the CNN model with detailed progress tracking, time estimation, and cost analysis"""
    global peak_memory_usage, training_active
    
    peak_memory_usage = 0
    training_active = True
    
    # Initialize cost model
    cost_model = CNNTrainingCostModel()
    
    # Start memory monitoring in background thread
    monitor_thread = threading.Thread(target=monitor_memory, daemon=True)
    monitor_thread.start()
    
    train_start_time = time.time()
    
    # Use provided estimator or create new one
    if time_estimator is None:
        time_estimator = TrainingTimeEstimator(total_epochs=epochs, warmup_epochs=2)
    
    # Create datasets and dataloaders
    train_dataset = VehicleDataset(X_train, y_train)
    test_dataset = VehicleDataset(X_test, y_test)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # Loss function and scheduler
    criterion = nn.CrossEntropyLoss()
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5,
                                                     patience=3, min_lr=1e-6)
    
    # Early stopping
    best_test_loss = float('inf')
    patience_counter = 0
    early_stop_patience = 7
    
    # Training history
    history = {
        'train_loss': [],
        'train_acc': [],
        'test_loss': [],
        'test_acc': [],
        'actual_training_seconds': 0,
        'actual_training_hours': 0,
        'peak_memory_gb': 0,
        'training_cost_breakdown': {},
        'cost_per_epoch': 0,
        'total_training_cost': 0
    }
    
    print(f"\nStarting training for {epochs} epochs...")
    print(f"Training samples: {len(X_train):,}")
    print(f"Batch size: {batch_size}")
    print(f"Total batches per epoch: {len(train_loader)}")
    
    for epoch in range(epochs):
        epoch_start_time = time.time()
        
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for batch_idx, (images, labels) in enumerate(train_loader):
            batch_start_time = time.time()
            images, labels = images.to(device), labels.to(device)
            
            # Handle one-hot encoded labels
            if labels.dim() > 1 and labels.shape[1] > 1:
                labels = torch.argmax(labels, dim=1)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            loss.backward()
            optimizer.step()
            
            # Statistics
            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
            
            # Record batch time
            batch_time = time.time() - batch_start_time
            time_estimator.record_batch_time(batch_time)
            
            if batch_idx % max(1, len(train_loader) // 5) == 0:
                print(f'  Epoch {epoch+1}/{epochs}, Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item():.4f}')
        
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
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                
                if labels.dim() > 1 and labels.shape[1] > 1:
                    labels = torch.argmax(labels, dim=1)
                
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                test_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                test_total += labels.size(0)
                test_correct += (predicted == labels).sum().item()
        
        validation_time = time.time() - validation_start_time
        
        # Calculate average testing metrics
        avg_test_loss = test_loss / len(test_loader)
        test_accuracy = 100 * test_correct / test_total
        
        # Store history
        history['train_loss'].append(avg_train_loss)
        history['train_acc'].append(train_accuracy)
        history['test_loss'].append(avg_test_loss)
        history['test_acc'].append(test_accuracy)
        
        # Record epoch time
        epoch_time = time.time() - epoch_start_time
        time_estimator.record_epoch_time(epoch_time, validation_time)
        
        # Estimate remaining time
        remaining_time_info = time_estimator.estimate_remaining_time(epoch, len(train_loader), len(train_loader))
        
        # Step the scheduler
        scheduler.step(avg_test_loss)
        
        # Early stopping logic
        if avg_test_loss < best_test_loss:
            best_test_loss = avg_test_loss
            patience_counter = 0
            if save_path:
                best_model_path = save_path.replace('.pth', '_best.pth')
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'input_shape': model.input_shape,
                    'num_classes': model.num_classes,
                    'conv_filters': model.conv_filters,
                    'dense_units': model.dense_units,
                    'dropout': model.dropout,
                    'best_epoch': epoch + 1,
                    'best_test_loss': best_test_loss,
                    'training_history': history
                }, best_model_path)
        else:
            patience_counter += 1
        
        print(f'Epoch {epoch+1}/{epochs}: Train Loss: {avg_train_loss:.4f}, Test Loss: {avg_test_loss:.4f}, Test Acc: {test_accuracy:.2f}%')
        print(f'  ⏱️  Epoch time: {epoch_time:.1f}s | Remaining: {remaining_time_info["realistic"]}')
    # Stop memory monitoring
    training_active = False
    monitor_thread.join(timeout=1)
    
    # Calculate actual training time
    train_end_time = time.time()
    actual_training_seconds = train_end_time - train_start_time
    actual_training_hours = actual_training_seconds / 3600
    
    history['actual_training_seconds'] = actual_training_seconds
    history['actual_training_hours'] = actual_training_hours
    history['peak_memory_gb'] = peak_memory_usage
    
    # Calculate training cost
    training_cost_breakdown = cost_model.calculate_training_cost(actual_training_hours)
    cost_per_epoch_info = cost_model.calculate_cost_per_epoch(actual_training_hours)
    
    history['training_cost_breakdown'] = training_cost_breakdown
    history['cost_per_epoch'] = cost_per_epoch_info['cost_per_epoch']
    history['total_training_cost'] = training_cost_breakdown['total']
    
    # Save final model with complete information
    if save_path:
        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'input_shape': model.input_shape,
            'num_classes': model.num_classes,
            'conv_filters': model.conv_filters,
            'dense_units': model.dense_units,
            'dropout': model.dropout,
            'training_history': history,
            'training_cost_breakdown': training_cost_breakdown,
            'total_training_cost': training_cost_breakdown['total'],
            'actual_training_hours': actual_training_hours,
            'peak_memory_gb': peak_memory_usage
        }, save_path)
        print(f"✓ Model saved to: {save_path}")
    
    print(f"\n📊 Training completed:")
    print(f"   Actual training time: {actual_training_hours:.2f} hours")
    print(f"   Peak memory usage: {peak_memory_usage:.2f} GB")
    print(f"   Total training cost: ${training_cost_breakdown['total']:.4f}")
    print(f"   Cost breakdown:")
    print(f"     - Compute: ${training_cost_breakdown['compute']:.4f}")
    print(f"     - Memory: ${training_cost_breakdown['memory']:.4f}")
    print(f"     - Storage: ${training_cost_breakdown['storage']:.4f}")
    print(f"     - Data transfer: ${training_cost_breakdown['data_transfer']:.4f}")
    print(f"     - GPU: ${training_cost_breakdown['gpu']:.4f}")
    print(f"   Cost per epoch: ${cost_per_epoch_info['cost_per_epoch']:.6f}")
    
    return history


def plot_metrics(history, save_dir=MODEL_DIR):
    """Plot training vs validation loss and accuracy"""
    plt.figure(figsize=(8, 6))
    plt.plot(history['train_acc'], label='Train Accuracy')
    plt.plot(history['test_acc'], label='Test Accuracy')
    plt.title("Model Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, "accuracy_plot.png"))
    plt.close()
    
    plt.figure(figsize=(8, 6))
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['test_loss'], label='Test Loss')
    plt.title("Model Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, "loss_plot.png"))
    plt.close()
    
    print(f"📊 Plots saved to {save_dir}")


def save_training_summary(actual_training_hours, peak_memory_usage, training_cost_breakdown, cost_per_epoch_info, save_dir=MODEL_DIR):
    """Save training summary information to a JSON file for API access"""
    summary_data = {
        "training_summary": {
            "actual_training_time_hours": round(actual_training_hours, 4),
            "actual_training_time_formatted": f"{actual_training_hours:.2f} hours",
            "peak_memory_usage_gb": round(peak_memory_usage, 2),
            "peak_memory_usage_formatted": f"{peak_memory_usage:.2f} GB",
            "total_training_cost": round(training_cost_breakdown['total'], 4),
            "total_training_cost_formatted": f"${training_cost_breakdown['total']:.4f}",
            "cost_breakdown": {
                "compute": round(training_cost_breakdown['compute'], 4),
                "compute_formatted": f"${training_cost_breakdown['compute']:.4f}",
                "memory": round(training_cost_breakdown['memory'], 4),
                "memory_formatted": f"${training_cost_breakdown['memory']:.4f}",
                "storage": round(training_cost_breakdown['storage'], 4),
                "storage_formatted": f"${training_cost_breakdown['storage']:.4f}",
                "data_transfer": round(training_cost_breakdown['data_transfer'], 4),
                "data_transfer_formatted": f"${training_cost_breakdown['data_transfer']:.4f}",
                "gpu": round(training_cost_breakdown['gpu'], 4),
                "gpu_formatted": f"${training_cost_breakdown['gpu']:.4f}"
            },
            "cost_per_epoch": round(cost_per_epoch_info['cost_per_epoch'], 6),
            "cost_per_epoch_formatted": f"${cost_per_epoch_info['cost_per_epoch']:.6f}",
            "timestamp": time.time(),
            "timestamp_formatted": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        }
    }

    # Save to JSON file
    json_path = os.path.join(save_dir, "training_summary.json")
    with open(json_path, 'w') as f:
        json.dump(summary_data, f, indent=2)

    print(f"📄 Training summary saved to: {json_path}")
    return json_path


def get_dataset_choice():
    """Let user choose which dataset to train on"""
    print("\n" + "="*60)
    print("DATASET SELECTION")
    print("="*60)
    
    # List available datasets
    datasets = []
    if os.path.exists(DEFAULT_DATASET_PATH):
        datasets.append(("1", "Default dataset (vehicles_dataset.npz)", DEFAULT_DATASET_PATH))
    
    # Check for other .npz files in the directory
    project_dir = os.path.dirname(DEFAULT_DATASET_PATH)
    if os.path.exists(project_dir):
        for file in os.listdir(project_dir):
            if file.endswith('.npz') and file != 'vehicles_dataset.npz':
                datasets.append((str(len(datasets) + 1), file, os.path.join(project_dir, file)))
    
    if not datasets:
        print("❌ No datasets found! Please run dataProcessor.py first.")
        return DEFAULT_DATASET_PATH
    
    print("\nAvailable datasets:")
    for idx, name, path in datasets:
        size_mb = os.path.getsize(path) / (1024 * 1024)
        print(f"  {idx}. {name} ({size_mb:.1f} MB)")
    
    # Get user choice
    while True:
        choice = input(f"\nSelect dataset (1-{len(datasets)}) [default: 1]: ").strip()
        if choice == "":
            choice = "1"
        
        try:
            choice_idx = int(choice) - 1
            if 0 <= choice_idx < len(datasets):
                selected = datasets[choice_idx][2]
                print(f"✓ Selected: {datasets[choice_idx][1]}")
                return selected
            else:
                print(f"❌ Invalid choice. Please select 1-{len(datasets)}")
        except ValueError:
            print(f"❌ Invalid input. Please enter a number 1-{len(datasets)}")


def get_model_choice():
    """Let user choose model configuration preset"""
    print("\n" + "="*60)
    print("MODEL CONFIGURATION")
    print("="*60)
    
    presets = {
        "1": {
            "name": "Small (Fast Training)",
            "conv_filters": [16, 32, 64],
            "dense_units": 64,
            "dropout": 0.3,
            "learning_rate": 0.001,
            "epochs": 30,
            "batch_size": 32
        },
        "2": {
            "name": "Default (Balanced)",
            "conv_filters": [32, 64, 128],
            "dense_units": 128,
            "dropout": 0.0,
            "learning_rate": 0.001,
            "epochs": 50,
            "batch_size": 32
        },
        "3": {
            "name": "Large (High Accuracy)",
            "conv_filters": [64, 128, 256],
            "dense_units": 256,
            "dropout": 0.5,
            "learning_rate": 0.0001,
            "epochs": 100,
            "batch_size": 16
        }
    }
    
    print("\nModel presets:")
    for key, config in presets.items():
        print(f"  {key}. {config['name']}")
        print(f"     - Conv filters: {config['conv_filters']}")
        print(f"     - Dense units: {config['dense_units']}")
        print(f"     - Epochs: {config['epochs']}, Batch size: {config['batch_size']}")
    
    print("  4. Custom")
    
    while True:
        choice = input("\nSelect configuration (1-4) [default: 2]: ").strip()
        if choice == "":
            choice = "2"
        
        if choice in presets:
            config = presets[choice]
            print(f"✓ Selected: {config['name']}")
            return config
        elif choice == "4":
            print("\n📝 Custom configuration:")
            config = {
                "name": "Custom",
                "conv_filters": [32, 64, 128],  # Default
                "dense_units": 128,
                "dropout": 0.0,
                "learning_rate": 0.001,
                "epochs": 50,
                "batch_size": 32
            }
            
            # Allow overrides
            print("(Press Enter to keep default values)")
            
            try:
                epochs_input = input(f"  Epochs [{config['epochs']}]: ").strip()
                if epochs_input:
                    config['epochs'] = int(epochs_input)
                
                batch_input = input(f"  Batch size [{config['batch_size']}]: ").strip()
                if batch_input:
                    config['batch_size'] = int(batch_input)
                
                lr_input = input(f"  Learning rate [{config['learning_rate']}]: ").strip()
                if lr_input:
                    config['learning_rate'] = float(lr_input)
                
                dropout_input = input(f"  Dropout [{config['dropout']}]: ").strip()
                if dropout_input:
                    config['dropout'] = float(dropout_input)
                
                print(f"✓ Custom configuration set")
                return config
            except ValueError:
                print("❌ Invalid input. Using defaults.")
                return presets["2"]
        else:
            print("❌ Invalid choice. Please select 1-4")


def main():
    """Main training pipeline with interactive features"""
    try:
        print("="*60)
        print("TRAINING VEHICLE CLASSIFIER CNN (PyTorch)")
        print("="*60)
        
        # Get dataset choice
        dataset_path = get_dataset_choice()
        
        # Get model configuration
        config = get_model_choice()
        
        # Load dataset
        print("\nLoading dataset...")
        X_train, X_test, y_train, y_test = load_dataset(dataset_path)
        
        # Estimate training time
        print("\n📊 Estimating training time...")
        estimator = TrainingTimeEstimator(
            num_samples=len(X_train),
            batch_size=config['batch_size'],
            num_epochs=config['epochs']
        )
        estimated_seconds = estimator.estimate_total_time()
        estimated_hours = estimated_seconds / 3600
        estimated_minutes = (estimated_seconds % 3600) / 60
        
        print(f"⏱️  Estimated training time: {estimated_hours:.1f} hours ({estimated_minutes:.0f} minutes)")
        
        # Estimate training cost
        print("\n💰 Estimating training cost...")
        cost_model = CNNTrainingCostModel()
        cost_summary = cost_model.get_cost_summary(estimated_hours)
        print(f"💵 Estimated training cost: ${cost_summary['total_cost']}")
        print(f"   Cost per hour: ${cost_summary['cost_per_hour']}")
        
        # Confirm or adjust
        proceed = input("\nProceed with training? (yes/no) [default: yes]: ").strip().lower()
        if proceed and proceed not in ['yes', 'y', '']:
            print("❌ Training cancelled by user")
            return
        
        # Create model
        print("\nCreating model...")
        model = VehicleCNN(
            input_shape=(128, 128, 1),
            num_classes=3,
            conv_filters=config['conv_filters'],
            dense_units=config['dense_units'],
            dropout=config['dropout']
        ).to(device)
        
        optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])
        
        print(f"✓ Model created with {sum(p.numel() for p in model.parameters()):,} parameters")
        print(f"  - Conv filters: {config['conv_filters']}")
        print(f"  - Dense units: {config['dense_units']}")
        print(f"  - Dropout: {config['dropout']}")
        print(f"  - Learning rate: {config['learning_rate']}")
        print(f"  - Device: {device}")
        
        # Train model with time estimation
        print("\nTraining model...")
        print(f"Starting training for {config['epochs']} epochs...")
        print(f"Training samples: {len(X_train):,}")
        print(f"Batch size: {config['batch_size']}")
        print(f"Total batches per epoch: {len(X_train) // config['batch_size']}")
        
        history = train_model(
            model, optimizer,
            X_train, y_train,
            X_test, y_test,
            batch_size=config['batch_size'],
            epochs=config['epochs'],
            save_path=os.path.join(MODEL_DIR, "vehicle_classifier_model.pth"),
            time_estimator=estimator
        )
        
        # Plot metrics
        print("\nGenerating plots...")
        plot_metrics(history)
        
        # Save training summary to JSON
        print("\nSaving training summary...")
        summary_path = save_training_summary(
            actual_training_hours=history['actual_training_hours'],
            peak_memory_usage=history['peak_memory_gb'],
            training_cost_breakdown=history['training_cost_breakdown'],
            cost_per_epoch_info={'cost_per_epoch': history['cost_per_epoch']}
        )
        
        print("\n🎉 Training completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Training failed: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
