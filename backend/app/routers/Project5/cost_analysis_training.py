"""
Cost analysis module for RNN text generator training on Render
Calculates training compute, storage, and operational costs
Adapted from projectRNN cost_analysis.py for Project5 training context
"""

import os
import json
from typing import Dict, List, Tuple
from datetime import datetime

FILE_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_FILE = os.path.join(FILE_DIR, "render_pricing_config.json")


def load_pricing_config():
    """Load pricing config from JSON file, with defaults if not found."""
    if os.path.exists(CONFIG_FILE):
        with open(CONFIG_FILE, 'r') as f:
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
        self.cost_per_cpu_per_month = config.get("cost_per_cpu_per_month", 19.0 / 64.0)
        self.cost_per_gb_ram_per_month = config.get("cost_per_gb_ram_per_month", 19.0 / 512.0)
        self.overage_bandwidth_cost_per_gb = config.get("overage_bandwidth_cost_per_gb", 0.10)
        self.overage_build_minutes_cost = config.get("overage_build_minutes_cost", 0.01)
        self.additional_storage_cost_per_gb = config.get("additional_storage_cost_per_gb", 0.10)

        # Store raw config for reference
        self.raw_config = config


class TrainingCostModel:
    """Cost model for LSTM text generator training on Render"""

    def __init__(self, pricing_config: RenderPricingConfig = None):
        self.pricing = pricing_config or RenderPricingConfig()

        # Model specifications for Spotify lyrics LSTM
        self.model_specs = {
            "vocab_size": 25000,
            "embedding_dim": 200,
            "hidden_size": 512,
            "num_layers": 3,
            "sequence_length": 50,
            "model_size_mb": 45.0,  # Approximate for 3-layer LSTM with embedding
            "dataset_size_gb": 0.020,  # 20MB lyrics preprocessed file
            "checkpoint_size_mb": 50.0,  # Saved model checkpoint
        }

        # Training specifications (from Model.py defaults)
        self.training_specs = {
            "batch_size": 128,
            "epochs": 25,
            "learning_rate": 0.005,
            "early_stop_patience": 7,
            "estimated_training_hours": 12.0,  # Estimated on standard GPU
            "estimated_gpu_hours": 12.0,
            "peak_memory_mb": 1024,  # LSTM with 128 batch size ~1GB
            "average_memory_mb": 768,
        }

        # Storage requirements
        self.storage_specs = {
            "dataset_gb": 0.020,
            "model_checkpoint_gb": 0.050,
            "logs_and_metrics_gb": 0.010,
            "total_storage_gb": 0.080,
        }

        # Compute specifications
        self.compute_specs = {
            "cpus_per_worker": 2.0,
            "ram_per_worker_gb": 8.0,
            "num_workers": 1,
            "total_cpu_hours": 2.0 * self.training_specs["estimated_training_hours"],
            "total_memory_gb_hours": 8.0 * self.training_specs["estimated_training_hours"],
        }

    def has_training_data(self) -> bool:
        """Check if model has been trained by looking for training metadata."""
        try:
            config_path = os.path.join(FILE_DIR, "render_pricing_config.json")
            if not os.path.exists(config_path):
                return False
            
            with open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
            
            # Check if actual training data exists in config
            return "actual_training_cost" in config and config.get("actual_training_cost") is not None
        except Exception:
            return False

    def calculate_training_cost(self, training_hours: float = None) -> Dict[str, float]:
        """
        Calculate one-time training cost
        
        Args:
            training_hours: Override default training hours if known
            
        Returns:
            Dict with compute, storage, data_transfer, and total costs
        """
        training_hours = training_hours or self.training_specs["estimated_training_hours"]

        # Compute cost: based on CPU-hours used
        cpu_hours = self.compute_specs["cpus_per_worker"] * training_hours
        compute_cost = cpu_hours * self.pricing.cost_per_cpu_per_month / (24 * 30)  # Convert monthly to hourly

        # Memory cost: based on GB-hours used
        memory_gb_hours = self.compute_specs["ram_per_worker_gb"] * training_hours
        memory_cost = memory_gb_hours * self.pricing.cost_per_gb_ram_per_month / (24 * 30)

        # Storage cost: for dataset + model checkpoints stored during training
        storage_cost = self.storage_specs["total_storage_gb"] * self.pricing.additional_storage_cost_per_gb

        # Data transfer cost: minimal for training (local operations)
        data_transfer_cost = 0.0

        total_cost = compute_cost + memory_cost + storage_cost + data_transfer_cost

        return {
            "compute": round(compute_cost, 4),
            "memory": round(memory_cost, 4),
            "storage": round(storage_cost, 4),
            "data_transfer": round(data_transfer_cost, 4),
            "total": round(total_cost, 4),
            "training_hours": training_hours,
        }

    def calculate_cost_per_epoch(self, training_hours: float = None) -> Dict[str, float]:
        """Calculate cost breakdown per epoch"""
        training_hours = training_hours or self.training_specs["estimated_training_hours"]
        cost_per_hour = self.calculate_training_cost(training_hours)["total"] / training_hours

        epochs = self.training_specs["epochs"]
        hours_per_epoch = training_hours / epochs

        return {
            "cost_per_hour": round(cost_per_hour, 6),
            "hours_per_epoch": round(hours_per_epoch, 2),
            "cost_per_epoch": round(cost_per_hour * hours_per_epoch, 4),
            "total_epochs": epochs,
        }

    def estimate_cost_with_parameters(
        self,
        batch_size: int = None,
        epochs: int = None,
        training_hours: float = None,
    ) -> Dict[str, float]:
        """
        Estimate cost with custom parameters
        
        Args:
            batch_size: Custom batch size (affects memory)
            epochs: Custom number of epochs
            training_hours: Custom training time estimate
            
        Returns:
            Cost breakdown with specified parameters
        """
        batch_size = batch_size or self.training_specs["batch_size"]
        epochs = epochs or self.training_specs["epochs"]
        training_hours = training_hours or self.training_specs["estimated_training_hours"]

        # Scale training hours based on epoch count (assumes roughly linear scaling)
        default_epochs = self.training_specs["epochs"]
        scaled_training_hours = training_hours * (epochs / default_epochs)

        cost = self.calculate_training_cost(scaled_training_hours)
        cost["batch_size"] = batch_size
        cost["epochs"] = epochs
        cost["scaled_training_hours"] = scaled_training_hours

        return cost

    def generate_training_cost_report(self, actual_training_hours: float = None) -> Dict:
        """Generate comprehensive training cost report"""
        # If no training has been performed, return report with zero costs
        if not self.has_training_data():
            return {
                "report_generated": datetime.now().isoformat(),
                "model_specs": self.model_specs,
                "training_specs": self.training_specs,
                "storage_specs": self.storage_specs,
                "compute_specs": self.compute_specs,
                "pricing_config": self.pricing.raw_config,
                "training_cost": {
                    "compute": 0.0,
                    "memory": 0.0,
                    "storage": 0.0,
                    "data_transfer": 0.0,
                    "total": 0.0,
                    "training_hours": 0.0,
                },
                "cost_per_epoch": {
                    "cost_per_hour": 0.0,
                    "hours_per_epoch": 0.0,
                    "cost_per_epoch": 0.0,
                    "total_epochs": 0,
                },
                "parameter_scenarios": {
                    "default": {"total": 0.0},
                    "batch_size_256": {"total": 0.0},
                    "batch_size_64": {"total": 0.0},
                    "epochs_50": {"total": 0.0},
                    "epochs_10": {"total": 0.0},
                },
            }
        
        training_cost = self.calculate_training_cost(actual_training_hours)
        cost_per_epoch = self.calculate_cost_per_epoch(actual_training_hours)

        # Alternative scenarios with different parameters
        scenarios = {
            "default": self.estimate_cost_with_parameters(),
            "batch_size_256": self.estimate_cost_with_parameters(batch_size=256),
            "batch_size_64": self.estimate_cost_with_parameters(batch_size=64),
            "epochs_50": self.estimate_cost_with_parameters(epochs=50),
            "epochs_10": self.estimate_cost_with_parameters(epochs=10),
        }

        return {
            "report_generated": datetime.now().isoformat(),
            "model_specs": self.model_specs,
            "training_specs": self.training_specs,
            "storage_specs": self.storage_specs,
            "compute_specs": self.compute_specs,
            "pricing_config": self.pricing.raw_config,
            "training_cost": training_cost,
            "cost_per_epoch": cost_per_epoch,
            "parameter_scenarios": scenarios,
        }

    def get_cost_summary(self, actual_training_hours: float = None) -> Dict[str, str]:
        """Get human-readable cost summary"""
        # If no training has been performed, return zero values
        if not self.has_training_data():
            return {
                "total_training_cost": "$0.0000",
                "compute_cost": "$0.0000",
                "memory_cost": "$0.0000",
                "storage_cost": "$0.0000",
                "training_hours": "0.00 hours",
                "cost_per_epoch": "$0.0000",
                "cost_per_hour": "$0.0000",
            }
        
        report = self.generate_training_cost_report(actual_training_hours)
        cost = report["training_cost"]

        return {
            "total_training_cost": f"${cost['total']:.4f}",
            "compute_cost": f"${cost['compute']:.4f}",
            "memory_cost": f"${cost['memory']:.4f}",
            "storage_cost": f"${cost['storage']:.4f}",
            "training_hours": f"{cost['training_hours']:.2f} hours",
            "cost_per_epoch": f"${report['cost_per_epoch']['cost_per_epoch']:.6f}",
            "cost_per_hour": f"${report['cost_per_epoch']['cost_per_hour']:.6f}",
        }


def calculate_batch_cost_estimate(batch_size: int, vocab_size: int = 25000) -> Dict[str, float]:
    """Quick utility function to estimate cost for a specific batch size"""
    pricing = RenderPricingConfig()
    model = TrainingCostModel(pricing)

    # Estimate memory usage (rough formula)
    # Memory ~ batch_size * sequence_length * vocab_size * bytes_per_float32
    # For LSTM: sequence_length=50, vocab_size=25k
    # Approximate: 50 * 25000 * 4 bytes = 5MB per sequence in batch
    # Plus model parameters: ~45MB
    estimated_memory_mb = 45 + (batch_size * 5)

    # Rough memory scaling factor
    default_memory_mb = model.compute_specs["ram_per_worker_gb"] * 1024
    memory_scale = estimated_memory_mb / default_memory_mb

    # Scale training time estimate (more memory = less batches = slightly faster per epoch, but more total data)
    # Conservative: assume roughly linear scaling
    training_hours_scaled = model.training_specs["estimated_training_hours"] * memory_scale

    return model.calculate_training_cost(training_hours_scaled)


if __name__ == "__main__":
    # Generate report and display summary
    pricing = RenderPricingConfig()
    model = TrainingCostModel(pricing)
    report = model.generate_training_cost_report()

    print("="*70)
    print("LSTM TEXT GENERATOR - TRAINING COST ANALYSIS REPORT")
    print("="*70)

    print("\n📊 MODEL SPECIFICATIONS:")
    print(f"   Vocab Size: {model.model_specs['vocab_size']:,}")
    print(f"   Embedding Dim: {model.model_specs['embedding_dim']}")
    print(f"   Hidden Size: {model.model_specs['hidden_size']}")
    print(f"   Num Layers: {model.model_specs['num_layers']}")
    print(f"   Model Size: ~{model.model_specs['model_size_mb']:.1f} MB")
    print(f"   Dataset Size: {model.model_specs['dataset_size_gb']:.3f} GB")

    print("\n⚙️  TRAINING SPECIFICATIONS:")
    print(f"   Batch Size: {model.training_specs['batch_size']}")
    print(f"   Epochs: {model.training_specs['epochs']}")
    print(f"   Learning Rate: {model.training_specs['learning_rate']}")
    print(f"   Early Stop Patience: {model.training_specs['early_stop_patience']}")
    print(f"   Estimated Training Time: {model.training_specs['estimated_training_hours']:.1f} hours")

    print("\n💰 TRAINING COST BREAKDOWN:")
    cost = report["training_cost"]
    print(f"   Compute Cost: ${cost['compute']:.4f}")
    print(f"   Memory Cost: ${cost['memory']:.4f}")
    print(f"   Storage Cost: ${cost['storage']:.4f}")
    print(f"   Data Transfer: ${cost['data_transfer']:.4f}")
    print(f"   {'─' * 40}")
    print(f"   TOTAL COST: ${cost['total']:.4f}")

    print("\n📈 COST PER EPOCH:")
    per_epoch = report["cost_per_epoch"]
    print(f"   Cost per Epoch: ${per_epoch['cost_per_epoch']:.6f}")
    print(f"   Hours per Epoch: {per_epoch['hours_per_epoch']:.2f}")
    print(f"   Cost per Hour: ${per_epoch['cost_per_hour']:.6f}")

    print("\n🎛️  PARAMETER VARIATIONS:")
    for scenario_name, scenario_cost in report["parameter_scenarios"].items():
        print(f"   {scenario_name}: ${scenario_cost['total']:.4f}")

    print("\n" + "="*70)

