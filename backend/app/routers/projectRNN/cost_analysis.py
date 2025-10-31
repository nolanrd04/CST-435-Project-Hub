"""
Cost analysis module for RNN deployment on Render
Calculates training, inference, and operational costs
"""

import os
import json
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple
import numpy as np

FILE_DIR = os.path.dirname(os.path.abspath(__file__))
VIZ_DIR = os.path.join(FILE_DIR, "visualizations")
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


class RNNCostModel:
    """Cost model for RNN deployment"""

    def __init__(self, pricing_config: RenderPricingConfig = None):
        self.pricing = pricing_config or RenderPricingConfig()

        # Model specifications (measured from training)
        self.model_params = {
            "vocab_size": 4000,  # Estimate based on Alice in Wonderland
            "sequence_length": 50,
            "embedding_dim": 100,
            "lstm_units": 150,
            "num_layers": 2,
            "model_size_mb": 5.2,  # Approximate PyTorch model size
            "dataset_size_gb": 0.001,  # Alice in Wonderland ~ 150 KB
        }

        # Runtime specifications
        self.runtime_specs = {
            "training_hours": 2.5,  # Estimated training time
            "inference_latency_ms": 150,  # Average per request
            "peak_memory_mb": 512,  # Memory during training
            "required_storage_gb": 10,  # Model + dataset + logs
        }

        # Deployment assumptions
        self.deployment_assumptions = {
            "uptime_percentage": 99.5,
            "requests_per_day_low": 100,
            "requests_per_day_medium": 10000,
            "requests_per_day_high": 1000000,
            "deployment_months": 12,
        }

    def calculate_training_cost(self) -> Dict[str, float]:
        """Calculate one-time training cost"""
        # For now, simplified: just use the fixed monthly cost
        # Training is assumed to be part of the subscription
        training_cost = self.pricing.fixed_monthly_cost

        return {
            "compute": round(training_cost, 2),
            "storage": 0.0,
            "data_transfer": 0.0,
            "total": round(training_cost, 2),
        }

    def calculate_monthly_inference_cost(self, requests_per_day: int) -> Dict[str, float]:
        """Calculate monthly inference cost for given request volume"""
        # Monthly cost is the fixed subscription cost
        monthly_cost = self.pricing.fixed_monthly_cost

        requests_per_month = requests_per_day * 30
        cost_per_inference = monthly_cost / requests_per_month if requests_per_month > 0 else 0

        return {
            "compute": round(monthly_cost, 2),
            "storage": 0.0,
            "data_transfer": 0.0,
            "total": round(monthly_cost, 2),
            "cost_per_inference": round(cost_per_inference, 6),
        }

    def generate_cost_report(self) -> Dict:
        """Generate comprehensive cost report"""
        training_cost = self.calculate_training_cost()

        scenarios = {}
        for scenario_name, requests_per_day in [
            ("low", self.deployment_assumptions["requests_per_day_low"]),
            ("medium", self.deployment_assumptions["requests_per_day_medium"]),
            ("high", self.deployment_assumptions["requests_per_day_high"]),
        ]:
            scenarios[scenario_name] = self.calculate_monthly_inference_cost(requests_per_day)

        # Annual cost calculations
        annual_costs = {}
        for scenario_name, monthly_data in scenarios.items():
            deployment_months = self.deployment_assumptions["deployment_months"]
            annual_costs[scenario_name] = round(training_cost["total"] + (monthly_data["total"] * deployment_months), 2)

        return {
            "model_specs": self.model_params,
            "runtime_specs": self.runtime_specs,
            "assumptions": self.deployment_assumptions,
            "pricing_config": self.pricing.raw_config,
            "training_cost": training_cost,
            "monthly_costs": scenarios,
            "annual_costs": annual_costs,
        }

    def update_pricing(self, pricing_dict: Dict):
        """Update pricing configuration from external source"""
        if "compute_hourly_rate" in pricing_dict:
            self.pricing.compute_hourly_rate = pricing_dict["compute_hourly_rate"]
        if "storage_per_gb_month" in pricing_dict:
            self.pricing.storage_pricing["standard"] = pricing_dict["storage_per_gb_month"]
        if "data_transfer_per_gb" in pricing_dict:
            self.pricing.data_transfer_cost_per_gb = pricing_dict["data_transfer_per_gb"]
        if "instance_type" in pricing_dict:
            instance = pricing_dict["instance_type"]
            if instance in self.pricing.compute_pricing:
                self.pricing.selected_instance = instance
                self.pricing.compute_hourly_rate = self.pricing.compute_pricing[instance]


class CostVisualizer:
    """Generate visualizations for cost analysis"""

    @staticmethod
    def plot_cost_breakdown(cost_report: Dict, save_path: str = VIZ_DIR):
        """Plot pie chart of cost breakdown"""
        os.makedirs(save_path, exist_ok=True)

        # Training cost breakdown
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        training_data = cost_report["training_cost"]
        training_labels = [k for k in training_data.keys() if k != "total"]
        training_values = [training_data[k] for k in training_labels]

        colors = ["#FF9999", "#66B2FF", "#99FF99"]
        ax1.pie(training_values, labels=training_labels, autopct="%1.1f%%", colors=colors, startangle=90)
        ax1.set_title("Training Cost Breakdown", fontsize=12, fontweight="bold")

        # Monthly inference cost breakdown (medium scenario)
        monthly_data = cost_report["monthly_costs"]["medium"]
        monthly_labels = [k for k in monthly_data.keys() if k not in ["total", "cost_per_inference"]]
        monthly_values = [monthly_data[k] for k in monthly_labels]

        ax2.pie(monthly_values, labels=monthly_labels, autopct="%1.1f%%", colors=colors, startangle=90)
        ax2.set_title("Monthly Inference Cost Breakdown (Medium Volume)", fontsize=12, fontweight="bold")

        plt.tight_layout()
        plt.savefig(f"{save_path}/cost_breakdown.png", dpi=300, bbox_inches="tight")
        plt.close()

    @staticmethod
    def plot_scaling_scenarios(cost_report: Dict, save_path: str = "visualizations"):
        """Plot cost scaling across different request volumes"""
        os.makedirs(save_path, exist_ok=True)

        scenarios = cost_report["monthly_costs"]
        scenario_names = list(scenarios.keys())
        compute_costs = [scenarios[s]["compute"] for s in scenario_names]
        storage_costs = [scenarios[s]["storage"] for s in scenario_names]
        transfer_costs = [scenarios[s]["data_transfer"] for s in scenario_names]
        total_costs = [scenarios[s]["total"] for s in scenario_names]

        x = np.arange(len(scenario_names))
        width = 0.2

        fig, ax = plt.subplots(figsize=(10, 6))

        ax.bar(x - 1.5*width, compute_costs, width, label="Compute", color="#FF9999")
        ax.bar(x - 0.5*width, storage_costs, width, label="Storage", color="#66B2FF")
        ax.bar(x + 0.5*width, transfer_costs, width, label="Data Transfer", color="#99FF99")
        ax.bar(x + 1.5*width, total_costs, width, label="Total", color="#FFD700")

        ax.set_xlabel("Scenario", fontweight="bold")
        ax.set_ylabel("Monthly Cost ($)", fontweight="bold")
        ax.set_title("Monthly Cost Scaling Across Request Volumes", fontsize=12, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels([s.capitalize() for s in scenario_names])
        ax.legend()
        ax.grid(axis="y", alpha=0.3)

        plt.tight_layout()
        plt.savefig(f"{save_path}/cost_scaling.png", dpi=300, bbox_inches="tight")
        plt.close()

    @staticmethod
    def plot_training_vs_inference(cost_report: Dict, save_path: str = "visualizations"):
        """Plot training cost vs annual inference costs for different scenarios"""
        os.makedirs(save_path, exist_ok=True)

        training_cost = cost_report["training_cost"]["total"]
        annual_costs = cost_report["annual_costs"]

        scenarios = list(annual_costs.keys())
        costs = list(annual_costs.values())

        fig, ax = plt.subplots(figsize=(10, 6))

        x = np.arange(len(scenarios))
        width = 0.35

        # Training cost (one-time, shown for comparison)
        ax.bar(x - width/2, [training_cost] * len(scenarios), width, label=f"Training (one-time: ${training_cost})", color="#FF9999")
        ax.bar(x + width/2, costs, width, label="Annual Operating Cost", color="#66B2FF")

        ax.set_xlabel("Scenario", fontweight="bold")
        ax.set_ylabel("Cost ($)", fontweight="bold")
        ax.set_title("Training vs Annual Operating Costs", fontsize=12, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels([s.capitalize() for s in scenarios])
        ax.legend()
        ax.grid(axis="y", alpha=0.3)

        # Add value labels on bars
        for i, (train, annual) in enumerate(zip([training_cost] * len(scenarios), costs)):
            ax.text(i - width/2, train + 5, f"${train:.2f}", ha="center", va="bottom", fontsize=9)
            ax.text(i + width/2, annual + 5, f"${annual:.2f}", ha="center", va="bottom", fontsize=9)

        plt.tight_layout()
        plt.savefig(f"{save_path}/training_vs_inference.png", dpi=300, bbox_inches="tight")
        plt.close()

    @staticmethod
    def plot_cost_per_inference(cost_report: Dict, save_path: str = "visualizations"):
        """Plot cost per inference across different volumes"""
        os.makedirs(save_path, exist_ok=True)

        scenarios = cost_report["monthly_costs"]
        scenario_names = list(scenarios.keys())
        cost_per_inf = [scenarios[s]["cost_per_inference"] * 1_000_000 for s in scenario_names]  # in micro-cents

        fig, ax = plt.subplots(figsize=(10, 6))

        colors = ["#FF9999", "#66B2FF", "#99FF99"]
        bars = ax.bar(scenario_names, cost_per_inf, color=colors)

        ax.set_xlabel("Scenario", fontweight="bold")
        ax.set_ylabel("Cost per Inference (millionths of a dollar)", fontweight="bold")
        ax.set_title("Cost per Inference by Request Volume", fontsize=12, fontweight="bold")
        ax.grid(axis="y", alpha=0.3)

        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f"${height/1_000_000:.6f}",
                   ha="center", va="bottom", fontsize=10)

        plt.tight_layout()
        plt.savefig(f"{save_path}/cost_per_inference.png", dpi=300, bbox_inches="tight")
        plt.close()


def generate_all_visualizations(save_path: str = "backend/app/routers/projectRNN/visualizations"):
    """Generate all cost analysis visualizations"""
    pricing = RenderPricingConfig()
    model = RNNCostModel(pricing)
    report = model.generate_cost_report()

    CostVisualizer.plot_cost_breakdown(report, save_path)
    CostVisualizer.plot_scaling_scenarios(report, save_path)
    CostVisualizer.plot_training_vs_inference(report, save_path)
    CostVisualizer.plot_cost_per_inference(report, save_path)

    return report


if __name__ == "__main__":
    # Generate report and visualizations
    pricing = RenderPricingConfig()
    model = RNNCostModel(pricing)
    report = model.generate_cost_report()

    print("=== RNN COST ANALYSIS REPORT ===\n")
    print(f"Training Cost: ${report['training_cost']['total']}")
    print(f"\nMonthly Costs by Scenario:")
    for scenario, costs in report["monthly_costs"].items():
        print(f"  {scenario.upper()}: ${costs['total']}/month (${costs['cost_per_inference']:.6f} per inference)")
    print(f"\nAnnual Costs by Scenario:")
    for scenario, cost in report["annual_costs"].items():
        print(f"  {scenario.upper()}: ${cost}")

    # Generate visualizations
    save_path = "visualizations"
    generate_all_visualizations(save_path)
    print(f"\n✅ Visualizations saved to {save_path}/")
