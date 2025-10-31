"""
Interactive script to configure Render pricing for cost analysis.
Run this script locally to input your custom Render pricing values.
The configuration is saved to a JSON file that the React app reads.
"""

import json
import os
from pathlib import Path


def get_pricing_input():
    """Get pricing inputs from user interactively."""

    print("\n" + "="*60)
    print("RENDER PRICING CONFIGURATION TOOL")
    print("="*60)
    print("\nEnter your Render.com pricing information.")
    print("(Press Enter to use default values shown in brackets)\n")

    config = {}

    # Subscription cost
    try:
        value = input("1. Monthly subscription cost (default: $19): $").strip()
        config['monthly_subscription_cost'] = float(value) if value else 19.0
    except ValueError:
        config['monthly_subscription_cost'] = 19.0
        print("   → Using default: $19")

    # Included bandwidth
    try:
        value = input("2. Included bandwidth per month in GB (default: 500): ").strip()
        config['included_bandwidth_gb'] = float(value) if value else 500.0
    except ValueError:
        config['included_bandwidth_gb'] = 500.0
        print("   → Using default: 500 GB")

    # Build pipeline minutes
    try:
        value = input("3. Build pipeline minutes per user per month (default: 500): ").strip()
        config['build_pipeline_minutes'] = float(value) if value else 500.0
    except ValueError:
        config['build_pipeline_minutes'] = 500.0
        print("   → Using default: 500 minutes")

    # Available CPUs
    try:
        value = input("4. Available CPUs (default: 64): ").strip()
        config['available_cpus'] = float(value) if value else 64.0
    except ValueError:
        config['available_cpus'] = 64.0
        print("   → Using default: 64 CPUs")

    # Available RAM
    try:
        value = input("5a. Available RAM in GB (default: 512): ").strip()
        config['available_ram_gb'] = float(value) if value else 512.0
    except ValueError:
        config['available_ram_gb'] = 512.0
        print("   → Using default: 512 GB")

    # Database cost per GB
    try:
        value = input("5b. Database cost per GB per month (default: $0.25): $").strip()
        config['database_cost_per_gb'] = float(value) if value else 0.25
    except ValueError:
        config['database_cost_per_gb'] = 0.25
        print("   → Using default: $0.25")

    return config


def calculate_derived_pricing(config):
    """Calculate derived pricing values from base inputs."""

    # Fixed subscription cost (Render's base price)
    config['fixed_monthly_cost'] = 19.0

    # Cost per GB of bandwidth used (over included amount)
    # Assume standard cloud overage pricing: $0.10 per GB
    config['overage_bandwidth_cost_per_gb'] = config['monthly_subscription_cost'] / config['included_bandwidth_gb']

    # Cost per minute of build pipeline (if over included)
    # Assume: $0.01 per minute of overage
    config['overage_build_minutes_cost'] = 0.01

    # Cost per CPU per month (derived from available CPUs)
    # $19/month divided evenly across available CPUs
    config['cost_per_cpu_per_month'] = 19.0 / config['available_cpus']

    # Cost per GB of RAM per month (derived from available RAM)
    # $19/month divided evenly across available RAM
    config['cost_per_gb_ram_per_month'] = 19.0 / config['available_ram_gb']

    # Storage cost (for model checkpoints, datasets, logs)
    # Assume: included in subscription tier, additional = $0.10/GB/month
    config['additional_storage_cost_per_gb'] = 0.10

    return config


def save_config(config, output_path):
    """Save configuration to JSON file, preserving existing values."""
    # Load existing config if it exists
    if os.path.exists(output_path):
        with open(output_path, 'r', encoding='utf-8') as f:
            existing_config = json.load(f)
    else:
        existing_config = {}

    # Merge new config with existing config
    merged_config = {**existing_config, **config}

    # Save merged config
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(merged_config, f, indent=2)

    return output_path


def display_config_summary(config):
    """Display a summary of the configuration."""

    print("\n" + "="*60)
    print("CONFIGURATION SUMMARY")
    print("="*60)
    print(f"\nFixed Monthly Subscription:     ${config['fixed_monthly_cost']:.2f}/month")
    print(f"Available CPUs:                 {config['available_cpus']:.0f}")
    print(f"Available RAM:                  {config['available_ram_gb']:.0f} GB")
    print(f"Included Bandwidth:             {config['included_bandwidth_gb']:.0f} GB/month")
    print(f"Build Pipeline Minutes:         {config['build_pipeline_minutes']:.0f} min/month")
    print(f"Database Cost:                  ${config['database_cost_per_gb']:.2f}/GB/month")

    print("\n" + "-"*60)
    print("DERIVED VALUES (calculated automatically)")
    print("-"*60)
    print(f"Cost per CPU per month:         ${config['cost_per_cpu_per_month']:.4f}")
    print(f"Cost per GB RAM per month:      ${config['cost_per_gb_ram_per_month']:.4f}")
    print(f"Overage Bandwidth Cost:         ${config['overage_bandwidth_cost_per_gb']:.2f}/GB")
    print(f"Overage Build Minutes Cost:     ${config['overage_build_minutes_cost']:.2f}/minute")
    print(f"Additional Storage Cost:        ${config['additional_storage_cost_per_gb']:.2f}/GB/month")
    print("\n")


def main():
    """Main script execution."""

    # Get user input
    config = get_pricing_input()

    # Calculate derived values
    config = calculate_derived_pricing(config)

    # Display summary
    display_config_summary(config)

    # Save to file
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_path = os.path.join(script_dir, "render_pricing_config.json")

    saved_path = save_config(config, output_path)

    print("✅ Configuration saved to:")
    print(f"   {saved_path}")
    print("\nThis file is read by the React app to display pricing.")
    print("Restart your React app to see the updated values.\n")


if __name__ == "__main__":
    main()
