# Cost Analysis Setup Guide

## Overview

The RNN cost analysis tool now uses a configuration-based approach:

1. **Configuration Script** (`configure_pricing.py`) - Run locally to set pricing values
2. **Backend** (`cost_analysis.py`) - Reads config and calculates costs
3. **Frontend** (`CostAnalysis.tsx`) - Displays pricing (read-only) and cost calculations

## Setup Instructions

### Step 1: Run the Configuration Script

The first time you set up the cost analysis, run the interactive configuration script:

```bash
cd backend
python app/routers/projectRNN/configure_pricing.py
```

This will prompt you for Render pricing values:

1. **Monthly subscription cost** (default: $19)
   - The fixed base cost of your Render subscription tier

2. **Included bandwidth per month** (default: 500 GB)
   - How much bandwidth is included in your subscription

3. **Build pipeline minutes per user per month** (default: 500)
   - CI/CD build minutes included in your plan

4. **Available CPUs** (default: 64)
   - Total number of CPUs available in your Render plan
   - The $19/month cost is divided evenly across these CPUs

5a. **Available RAM in GB** (default: 512)
   - Total amount of RAM available in your Render plan
   - The $19/month cost is divided evenly across this RAM

5b. **Database cost per GB per month** (default: $0.25)
   - Cost of persistent database storage

### Step 2: Configuration File Created

The script saves your configuration to:
```
backend/app/routers/projectRNN/render_pricing_config.json
```

This file contains:
- Your 5 input values
- Automatically calculated derived values (overage costs, etc.)

### Step 3: Start Backend and Frontend

Run your backend and frontend as normal:

```bash
# Terminal 1: Backend
cd backend
uvicorn main:app --reload

# Terminal 2: Frontend
cd frontend
npm start
```

### Step 4: View Cost Analysis

1. Navigate to the RNN Text Generator page
2. Click the "💰 Cost Analysis" tab
3. You'll see:
   - Your Render pricing configuration (read-only)
   - Cost breakdowns by scenario
   - Visualization graphs
   - Annual cost projections

## Updating Pricing Values

To change any pricing values, simply run the configuration script again:

```bash
python app/routers/projectRNN/configure_pricing.py
```

Enter new values when prompted, and the config file will be updated. The next time you refresh your React app, it will use the new values.

## Example Workflow

```bash
# Initial setup
python backend/app/routers/projectRNN/configure_pricing.py

# Output:
# ============================================================
# RENDER PRICING CONFIGURATION TOOL
# ============================================================
#
# 1. Monthly subscription cost (default: $19): $25
# 2. Included bandwidth per month in GB (default: 500): 750
# 3. Build pipeline minutes per user per month (default: 500): 750
# 4. Vertical scaling cost (64 CPUs & 512GB RAM) (default: $1000): $1200
# 5. Database cost per GB per month (default: $0.25): $0.30
#
# ============================================================
# CONFIGURATION SUMMARY
# ============================================================
# ...
# ✅ Configuration saved to:
#    /path/to/render_pricing_config.json
```

## Files Involved

| File | Purpose |
|------|---------|
| `configure_pricing.py` | Interactive script to set pricing values |
| `render_pricing_config.json` | Configuration file (created after running script) |
| `cost_analysis.py` | Backend module that loads config and calculates costs |
| `main.py` | FastAPI endpoints that serve cost data |
| `CostAnalysis.tsx` | React component that displays pricing and costs |

## Default Values

If the config file doesn't exist, the system uses these defaults:

```json
{
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
  "additional_storage_cost_per_gb": 0.10
}
```

**Note:** The `cost_per_cpu_per_month` and `cost_per_gb_ram_per_month` are automatically calculated by dividing the $19/month subscription cost across the available CPUs and RAM.

## Troubleshooting

**Q: I ran the config script but the React app still shows old values**
- Make sure you restarted your frontend (npm start) after updating the config
- The backend caches the config, so you may need to restart it too

**Q: The config file is missing**
- Just run the configuration script again - it will create a new one with defaults

**Q: I want to see the raw config values**
- The config file is at: `backend/app/routers/projectRNN/render_pricing_config.json`
- You can edit it directly with a text editor if needed
