"""
Cost analysis module for Diffusion Model training on Render
Calculates training compute, storage, and operational costs
Adapted from Project6 cost_analysis_training.py for diffusion model training
"""

import os
import json
from typing import Dict
from datetime import datetime
from pathlib import Path

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
            "cost_per_cpu_per_month": 0.296875,
            "cost_per_gb_ram_per_month": 0.037109375,
            "overage_bandwidth_cost_per_gb": 0.10,
            "overage_build_minutes_cost": 0.01,
            "additional_storage_cost_per_gb": 0.10,
            "gpu_cost_per_hour": 0.50,
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
        self.cost_per_cpu_per_month = config.get("cost_per_cpu_per_month", 0.296875)
        self.cost_per_gb_ram_per_month = config.get("cost_per_gb_ram_per_month", 0.037109375)
        self.overage_bandwidth_cost_per_gb = config.get("overage_bandwidth_cost_per_gb", 0.10)
        self.overage_build_minutes_cost = config.get("overage_build_minutes_cost", 0.01)
        self.additional_storage_cost_per_gb = config.get("additional_storage_cost_per_gb", 0.10)
        self.gpu_cost_per_hour = config.get("gpu_cost_per_hour", 0.50)

        # Store raw config for reference
        self.raw_config = config


class DiffusionTrainingCostModel:
    """Cost model for Diffusion Model training on Render"""

    def __init__(self, pricing_config: RenderPricingConfig = None):
        self.pricing = pricing_config or RenderPricingConfig()

    def calculate_training_cost(
        self,
        training_hours: float,
        peak_memory_gb: float,
        model_size_mb: float = 100.0,
        cpus_used: float = 2.0,
        use_gpu: bool = True
    ) -> Dict[str, float]:
        """
        Calculate actual training cost based on real metrics

        Args:
            training_hours: Actual training time in hours
            peak_memory_gb: Peak memory usage in GB
            model_size_mb: Size of saved model in MB (default 100MB for diffusion)
            cpus_used: Number of CPUs used (default 2.0)
            use_gpu: Whether GPU was used (default True)

        Returns:
            Dict with compute, memory, storage, GPU, and total costs
        """
        # GPU cost (if used)
        if use_gpu:
            gpu_cost = training_hours * self.pricing.gpu_cost_per_hour
        else:
            gpu_cost = 0.0

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

        total_cost = compute_cost + memory_cost + storage_cost + data_transfer_cost + gpu_cost

        return {
            "compute": round(compute_cost, 6),
            "memory": round(memory_cost, 6),
            "storage": round(storage_cost, 6),
            "gpu": round(gpu_cost, 6),
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
        cpus_used: float = 2.0,
        use_gpu: bool = True
    ) -> Dict[str, float]:
        """Calculate cost breakdown per epoch"""
        total_cost_data = self.calculate_training_cost(
            training_hours, peak_memory_gb, cpus_used=cpus_used, use_gpu=use_gpu
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
        model_size_mb: float = 100.0,
        use_gpu: bool = True,
        model_name: str = "diffusion_model"
    ) -> Dict:
        """Generate comprehensive training cost report for Diffusion Model training"""

        training_cost = self.calculate_training_cost(
            training_hours, peak_memory_gb, model_size_mb, cpus_used, use_gpu
        )

        cost_per_epoch = self.calculate_cost_per_epoch(
            training_hours, total_epochs, peak_memory_gb, cpus_used, use_gpu
        )

        return {
            "report_generated": datetime.now().isoformat(),
            "model_name": model_name,
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
        cpus_used: float = 2.0,
        use_gpu: bool = True
    ) -> Dict[str, str]:
        """Get human-readable cost summary"""

        cost_data = self.calculate_training_cost(
            training_hours, peak_memory_gb, cpus_used=cpus_used, use_gpu=use_gpu
        )
        cost_per_epoch_data = self.calculate_cost_per_epoch(
            training_hours, total_epochs, peak_memory_gb, cpus_used, use_gpu
        )

        summary = {
            "total_training_cost": f"${cost_data['total']:.6f}",
            "compute_cost": f"${cost_data['compute']:.6f}",
            "memory_cost": f"${cost_data['memory']:.6f}",
            "storage_cost": f"${cost_data['storage']:.6f}",
            "training_hours": f"{cost_data['training_hours']:.2f} hours",
            "peak_memory_gb": f"{cost_data['peak_memory_gb']:.2f} GB",
            "cost_per_epoch": f"${cost_per_epoch_data['cost_per_epoch']:.6f}",
            "cost_per_hour": f"${cost_per_epoch_data['cost_per_hour']:.6f}",
        }

        if use_gpu:
            summary["gpu_cost"] = f"${cost_data['gpu']:.6f}"

        return summary


def save_cost_report_to_file(report: Dict, output_path: str):
    """Save cost report to JSON file"""
    with open(output_path, 'w') as f:
        json.dump(report, f, indent=2)


def generate_cost_report_from_training_history(
    model_dir: Path,
    model_name: str,
    cpus_used: float = 2.0,
    peak_memory_gb: float = 4.0,
    use_gpu: bool = True
) -> Dict:
    """
    Generate cost report from saved training history.

    Args:
        model_dir: Directory containing training_history.json and config.json
        model_name: Name of the model
        cpus_used: Number of CPUs used during training
        peak_memory_gb: Peak memory usage in GB
        use_gpu: Whether GPU was used

    Returns:
        Cost analysis report
    """
    # Load training history
    history_path = model_dir / "training_history.json"
    if not history_path.exists():
        raise FileNotFoundError(f"Training history not found: {history_path}")

    with open(history_path, 'r') as f:
        history = json.load(f)

    # Load config
    config_path = model_dir / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")

    with open(config_path, 'r') as f:
        config = json.load(f)

    # Calculate training hours from epoch times
    epoch_times = history.get('epoch_times', [])
    training_hours = sum(epoch_times) / 3600.0  # Convert seconds to hours

    # Get number of epochs
    total_epochs = len(history.get('train_loss', []))

    # Estimate model size (checkpoints)
    model_size_mb = 100.0  # Default estimate
    checkpoint_files = list(model_dir.glob("*.pth"))
    if checkpoint_files:
        # Use size of best model
        best_model_path = model_dir / "best_model.pth"
        if best_model_path.exists():
            model_size_mb = best_model_path.stat().st_size / (1024 * 1024)

    # Generate report
    cost_model = DiffusionTrainingCostModel()
    report = cost_model.generate_training_cost_report(
        training_hours=training_hours,
        peak_memory_gb=peak_memory_gb,
        total_epochs=total_epochs,
        model_config=config,
        cpus_used=cpus_used,
        model_size_mb=model_size_mb,
        use_gpu=use_gpu,
        model_name=model_name
    )

    # Save report
    report_path = model_dir / "cost_analysis_report.json"
    save_cost_report_to_file(report, str(report_path))

    return report


def print_cost_report(report: Dict):
    """Print formatted cost report to console"""
    print("\n" + "=" * 70)
    print("DIFFUSION MODEL TRAINING - COST ANALYSIS REPORT")
    print("=" * 70)

    print(f"\nModel: {report.get('model_name', 'Unknown')}")
    print(f"Report Generated: {report['report_generated']}")

    print("\n" + "-" * 70)
    print("TRAINING COST BREAKDOWN:")
    print("-" * 70)
    cost = report["training_cost"]
    print(f"   Compute Cost:       ${cost['compute']:.6f}")
    print(f"   Memory Cost:        ${cost['memory']:.6f}")
    print(f"   Storage Cost:       ${cost['storage']:.6f}")
    if cost.get('gpu', 0) > 0:
        print(f"   GPU Cost:           ${cost['gpu']:.6f}")
    print(f"   Data Transfer Cost: ${cost['data_transfer']:.6f}")
    print(f"   {'-' * 40}")
    print(f"   TOTAL COST:         ${cost['total']:.6f}")

    print("\n" + "-" * 70)
    print("TRAINING METRICS:")
    print("-" * 70)
    print(f"   Training Hours:     {cost['training_hours']:.2f} hours")
    print(f"   Peak Memory:        {cost['peak_memory_gb']:.2f} GB")

    print("\n" + "-" * 70)
    print("COST PER EPOCH:")
    print("-" * 70)
    per_epoch = report["cost_per_epoch"]
    print(f"   Cost per Epoch:     ${per_epoch['cost_per_epoch']:.6f}")
    print(f"   Hours per Epoch:    {per_epoch['hours_per_epoch']:.4f}")
    print(f"   Cost per Hour:      ${per_epoch['cost_per_hour']:.6f}")
    print(f"   Total Epochs:       {per_epoch['total_epochs']}")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    # Test the cost model
    pricing = RenderPricingConfig()
    model = DiffusionTrainingCostModel(pricing)

    # Example: 3 hours of training with 4GB peak memory, GPU enabled
    test_config = {
        'timesteps': 1000,
        'num_epochs': 100,
        'batch_size': 16,
        'learning_rate': 0.0001,
        'features': [64, 128, 256, 512],
        'time_emb_dim': 256
    }

    report = model.generate_training_cost_report(
        training_hours=3.0,
        peak_memory_gb=4.0,
        total_epochs=100,
        model_config=test_config,
        cpus_used=2.0,
        use_gpu=True,
        model_name="test_diffusion_model"
    )

    print_cost_report(report)
