import { getApiUrl } from '../getApiUrl.ts';

const API_BASE = '/project9';

// Types
export interface ModelInfo {
  model_name: string;
  description: string;
  training_config?: {
    model_name: string;
    image_size: number;
    batch_size: number;
    learning_rate: number;
    epochs: number;
    optimizer: string;
    loss_function: string;
    created_at: string;
  };
  image_size: number;
  total_parameters?: number;
  training_stats?: {
    total_epochs: number;
    best_val_loss: number;
    final_train_loss: number;
    final_val_loss: number;
  };
}

export interface ColorizeRequest {
  image: string; // Base64 encoded image
  model_name: string;
}

export interface ColorizeResponse {
  success: boolean;
  colorized_image: string; // Base64 encoded
  grayscale_image: string; // Base64 encoded
  model_name: string;
}

export interface TrainingCostAnalysis {
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
}

// API Functions
export const project9API = {
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

  // Get cost analysis
  async getCostAnalysis(modelName?: string): Promise<TrainingCostAnalysis> {
    const url = modelName
      ? `${getApiUrl()}${API_BASE}/cost-analysis/${modelName}`
      : `${getApiUrl()}${API_BASE}/cost-analysis`;
    const response = await fetch(url);
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
