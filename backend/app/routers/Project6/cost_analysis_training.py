"""
Cost analysis module for GAN training on Render
Calculates training compute, storage, and operational costs
Adapted from Project5 cost_analysis_training.py for GAN training context
"""

import os
import json
from typing import Dict
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


class GANTrainingCostModel:
    """Cost model for GAN training on Render"""

    def __init__(self, pricing_config: RenderPricingConfig = None):
        self.pricing = pricing_config or RenderPricingConfig()

    def calculate_training_cost(
        self,
        training_hours: float,
        peak_memory_gb: float,
        model_size_mb: float = 50.0,
        cpus_used: float = 2.0
    ) -> Dict[str, float]:
        """
        Calculate actual training cost based on real metrics

        Args:
            training_hours: Actual training time in hours
            peak_memory_gb: Peak memory usage in GB
            model_size_mb: Size of saved model in MB (default 50MB for GAN)
            cpus_used: Number of CPUs used (default 2.0)

        Returns:
            Dict with compute, memory, storage, and total costs
        """
        # Compute cost: based on CPU-hours used
        cpu_hours = cpus_used * training_hours
        compute_cost = cpu_hours * self.pricing.cost_per_cpu_per_month / (24 * 30)  # Convert monthly to hourly

        # Memory cost: based on GB-hours used
        memory_gb_hours = peak_memory_gb * training_hours
        memory_cost = memory_gb_hours * self.pricing.cost_per_gb_ram_per_month / (24 * 30)

        # Storage cost: for model checkpoints
        storage_gb = model_size_mb / 1024.0
        storage_cost = storage_gb * self.pricing.additional_storage_cost_per_gb

        # Data transfer cost: minimal for training (local operations)
        data_transfer_cost = 0.0

        total_cost = compute_cost + memory_cost + storage_cost + data_transfer_cost

        return {
            "compute": round(compute_cost, 6),
            "memory": round(memory_cost, 6),
            "storage": round(storage_cost, 6),
            "data_transfer": round(data_transfer_cost, 6),
            "total": round(total_cost, 6),
            "training_hours": round(training_hours, 4),
            "peak_memory_gb": round(peak_memory_gb, 4),
        }

    def calculate_cost_per_epoch(
        self,
        training_hours: float,
        total_epochs: int,
        peak_memory_gb: float,
        cpus_used: float = 2.0
    ) -> Dict[str, float]:
        """Calculate cost breakdown per epoch"""
        total_cost_data = self.calculate_training_cost(
            training_hours, peak_memory_gb, cpus_used=cpus_used
        )

        cost_per_hour = total_cost_data["total"] / training_hours if training_hours > 0 else 0
        hours_per_epoch = training_hours / total_epochs if total_epochs > 0 else 0

        return {
            "cost_per_hour": round(cost_per_hour, 6),
            "hours_per_epoch": round(hours_per_epoch, 4),
            "cost_per_epoch": round(cost_per_hour * hours_per_epoch, 6),
            "total_epochs": total_epochs,
        }

    def generate_training_cost_report(
        self,
        training_hours: float,
        peak_memory_gb: float,
        total_epochs: int,
        model_config: Dict,
        cpus_used: float = 2.0,
        model_size_mb: float = 50.0,
    ) -> Dict:
        """Generate comprehensive training cost report for GAN training"""

        training_cost = self.calculate_training_cost(
            training_hours, peak_memory_gb, model_size_mb, cpus_used
        )

        cost_per_epoch = self.calculate_cost_per_epoch(
            training_hours, total_epochs, peak_memory_gb, cpus_used
        )

        return {
            "report_generated": datetime.now().isoformat(),
            "model_config": model_config,
            "training_cost": training_cost,
            "cost_per_epoch": cost_per_epoch,
            "pricing_config": self.pricing.raw_config,
        }

    def get_cost_summary(
        self,
        training_hours: float,
        peak_memory_gb: float,
        total_epochs: int,
        cpus_used: float = 2.0
    ) -> Dict[str, str]:
        """Get human-readable cost summary"""

        cost_data = self.calculate_training_cost(training_hours, peak_memory_gb, cpus_used=cpus_used)
        cost_per_epoch_data = self.calculate_cost_per_epoch(
            training_hours, total_epochs, peak_memory_gb, cpus_used
        )

        return {
            "total_training_cost": f"${cost_data['total']:.6f}",
            "compute_cost": f"${cost_data['compute']:.6f}",
            "memory_cost": f"${cost_data['memory']:.6f}",
            "storage_cost": f"${cost_data['storage']:.6f}",
            "training_hours": f"{cost_data['training_hours']:.2f} hours",
            "peak_memory_gb": f"{cost_data['peak_memory_gb']:.2f} GB",
            "cost_per_epoch": f"${cost_per_epoch_data['cost_per_epoch']:.6f}",
            "cost_per_hour": f"${cost_per_epoch_data['cost_per_hour']:.6f}",
        }


def save_cost_report_to_file(report: Dict, output_path: str):
    """Save cost report to JSON file"""
    with open(output_path, 'w') as f:
        json.dump(report, f, indent=2)


if __name__ == "__main__":
    # Test the cost model
    pricing = RenderPricingConfig()
    model = GANTrainingCostModel(pricing)

    # Example: 2 hours of training with 4GB peak memory
    test_config = {
        'epochs': 400,
        'batch_size': 32,
        'learning_rate': 0.0002,
        'latent_dim': 100
    }

    report = model.generate_training_cost_report(
        training_hours=2.0,
        peak_memory_gb=4.0,
        total_epochs=400,
        model_config=test_config,
        cpus_used=2.0
    )

    print("="*70)
    print("GAN TRAINING - COST ANALYSIS REPORT")
    print("="*70)

    print("\n💰 TRAINING COST BREAKDOWN:")
    cost = report["training_cost"]
    print(f"   Compute Cost: ${cost['compute']:.6f}")
    print(f"   Memory Cost: ${cost['memory']:.6f}")
    print(f"   Storage Cost: ${cost['storage']:.6f}")
    print(f"   {'─' * 40}")
    print(f"   TOTAL COST: ${cost['total']:.6f}")

    print("\n📈 COST PER EPOCH:")
    per_epoch = report["cost_per_epoch"]
    print(f"   Cost per Epoch: ${per_epoch['cost_per_epoch']:.6f}")
    print(f"   Hours per Epoch: {per_epoch['hours_per_epoch']:.4f}")
    print(f"   Cost per Hour: ${per_epoch['cost_per_hour']:.6f}")

    print("\n" + "="*70)
