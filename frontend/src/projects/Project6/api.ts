import { getApiUrl } from '../getApiUrl.ts';

const API_BASE = '/project6';

// Types
export interface ModelInfo {
  model_name: string;
  data_version: string;
  description: string;
  training_config?: {
    epochs: number;
    batch_size: number;
    learning_rate: number;
    image_resolution: number;
    image_count_per_fruit: number;
    image_size?: number;
  };
  training_stats?: {
    total_training_time_hours: number;
    peak_memory_gb: number;
    total_training_cost: number;
    avg_cost_per_fruit: number;
    avg_cost_per_epoch: number;
  };
  fruits: string[];
  model_architecture?: {
    image_size: number;
    latent_dim: number;
    channels: number;
    generator_layers: string;
    discriminator_layers: string;
  };
  total_parameters?: number;
  parameters_per_fruit?: Record<string, number>;
}

export interface GenerateImageRequest {
  model_name: string;
  fruit: string;
  num_images: number;
  seed?: number;
}

export interface GenerateImageResponse {
  success: boolean;
  images: string[]; // Base64 encoded images
  model_name: string;
  fruit: string;
  num_images: number;
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

export interface CreateModelRequest {
  model_name: string;
  data_version: string;
  description: string;
  create_new_data: boolean;
  override_existing?: boolean;
  // If creating new data:
  image_count?: number;
  image_resolution?: number;
  stroke_importance?: number;
  // Training parameters:
  epochs?: number;
  batch_size?: number;
  learning_rate?: number;
}

export interface CreateModelResponse {
  success: boolean;
  message: string;
  model_name: string;
  estimated_training_time?: number;
  warning?: string;
}

export interface TrainingStatus {
  is_training: boolean;
  current_fruit?: string;
  current_epoch?: number;
  total_epochs?: number;
  progress_percent?: number;
  estimated_time_remaining?: number;
  recent_logs?: string[];
}

// API Functions
export const project6API = {
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

  // Generate images
  async generateImages(request: GenerateImageRequest): Promise<GenerateImageResponse> {
    const response = await fetch(`${getApiUrl()}${API_BASE}/generate`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(request),
    });
    if (!response.ok) throw new Error('Failed to generate images');
    return response.json();
  },

  // Get training cost analysis
  async getTrainingCostAnalysis(modelName?: string): Promise<TrainingCostAnalysis> {
    const url = modelName
      ? `${getApiUrl()}${API_BASE}/cost-analysis/${modelName}`
      : `${getApiUrl()}${API_BASE}/cost-analysis`;
    const response = await fetch(url);
    if (!response.ok) throw new Error('Failed to fetch cost analysis');
    return response.json();
  },

  // Create new model
  async createModel(request: CreateModelRequest): Promise<CreateModelResponse> {
    const response = await fetch(`${getApiUrl()}${API_BASE}/models/create`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(request),
    });
    if (!response.ok) {
      const error = await response.json();
      throw new Error(error.message || 'Failed to create model');
    }
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

  // List available data versions
  async listDataVersions(): Promise<string[]> {
    const response = await fetch(`${getApiUrl()}${API_BASE}/data-versions`);
    if (!response.ok) throw new Error('Failed to fetch data versions');
    return response.json();
  },

  // Get training status (for real-time updates during training)
  async getTrainingStatus(): Promise<TrainingStatus> {
    const response = await fetch(`${getApiUrl()}${API_BASE}/training/status`);
    if (!response.ok) throw new Error('Failed to fetch training status');
    return response.json();
  },

  // Get training history for a specific fruit
  async getTrainingHistory(modelName: string, fruit: string): Promise<any> {
    const response = await fetch(
      `${getApiUrl()}${API_BASE}/models/${modelName}/training-history/${fruit}`
    );
    if (!response.ok) throw new Error('Failed to fetch training history');
    return response.json();
  },
};
