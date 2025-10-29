import axios from 'axios';

// Get API URL based on user preference stored in localStorage
const apiMode = typeof window !== 'undefined' ? localStorage.getItem('API_MODE') : null;
const API_BASE_URL = apiMode === 'local' ? 'http://localhost:8000' : 'https://cst-435-project-hub.onrender.com';

const api = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
});

export interface GenerateRequest {
  seed_text: string;
  num_words: number;
  temperature: number;
}

export interface GenerateResponse {
  generated_text: string;
  seed_text: string;
  num_words: number;
  temperature: number;
}

export interface ModelInfo {
  vocab_size: number;
  sequence_length: number;
  embedding_dim: number;
  lstm_units: number;
  num_layers: number;
}

export const generateText = async (request: GenerateRequest): Promise<GenerateResponse> => {
  const response = await api.post<GenerateResponse>('/generate', request);
  return response.data;
};

export const getModelInfo = async (): Promise<ModelInfo> => {
  const response = await api.get<ModelInfo>('/model/info');
  return response.data;
};

export const getArchitectureImage = (): string => {
  return `${API_BASE_URL}/visualizations/architecture`;
};

export const getTrainingHistoryImage = (): string => {
  return `${API_BASE_URL}/visualizations/training`;
};