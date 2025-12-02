import { getApiUrl } from '../getApiUrl.ts';

const API_BASE = '/project7';

// Types
export interface ModelInfo {
  model_name: string;
  config: {
    features: number[];
    time_emb_dim: number;
    timesteps: number;
    num_epochs: number;
    batch_size: number;
    learning_rate: number;
    sample_every: number;
    early_stopping_patience: number;
  };
  training_progress?: {
    current_epoch: number;
    total_epochs: number;
    best_val_loss: number;
    train_loss: number;
    val_loss: number;
  };
  model_parameters: number;
  image_size: number;
}

export interface ColorizeRequest {
  model_name: string;
  image: string; // Base64 encoded grayscale or color image
  num_inference_steps?: number; // Optional: reduce from 1000 for faster inference
}

export interface ColorizeResponse {
  success: boolean;
  grayscale_image: string; // Base64 encoded grayscale input
  colorized_image: string; // Base64 encoded RGB output
  model_name: string;
  inference_time_seconds: number;
}

export interface TrainingHistory {
  epochs: number[];
  train_loss: number[];
  val_loss: number[];
  epoch_times: number[];
}

export interface CostAnalysis {
  model_name: string;
  training_cost_breakdown: {
    compute_cost: number;
    memory_cost: number;
    storage_cost: number;
    total_cost: number;
  };
  cost_per_epoch: {
    avg_cost_per_epoch: number;
    total_epochs: number;
  };
  peak_memory_gb: number;
  training_hours: number;
  cpus_used: number;
  gpu_used: boolean;
}

// API Functions
export const project7API = {
  // List available models
  async listModels(): Promise<string[]> {
    const response = await fetch(`${getApiUrl()}${API_BASE}/models`);
    if (!response.ok) throw new Error('Failed to fetch models');
    return response.json();
  },

  // Get model information
  async getModelInfo(modelName: string): Promise<ModelInfo> {
    const response = await fetch(`${getApiUrl()}${API_BASE}/models/${modelName}/info`);
    if (!response.ok) throw new Error('Failed to fetch model info');
    return response.json();
  },

  // Colorize image
  async colorizeImage(request: ColorizeRequest): Promise<ColorizeResponse> {
    const response = await fetch(`${getApiUrl()}${API_BASE}/colorize`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(request),
    });
    if (!response.ok) {
      const error = await response.json();
      throw new Error(error.detail || 'Failed to colorize image');
    }
    return response.json();
  },

  // Get training history
  async getTrainingHistory(modelName: string): Promise<TrainingHistory> {
    const response = await fetch(`${getApiUrl()}${API_BASE}/models/${modelName}/training-history`);
    if (!response.ok) throw new Error('Failed to fetch training history');
    return response.json();
  },

  // Get cost analysis
  async getCostAnalysis(modelName: string): Promise<CostAnalysis> {
    const response = await fetch(`${getApiUrl()}${API_BASE}/models/${modelName}/cost-analysis`);
    if (!response.ok) throw new Error('Failed to fetch cost analysis');
    return response.json();
  },

  // Check if model exists
  async checkModelExists(modelName: string): Promise<boolean> {
    try {
      const models = await this.listModels();
      return models.includes(modelName);
    } catch {
      return false;
    }
  },
};
