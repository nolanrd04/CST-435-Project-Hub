import React, { useEffect, useState } from 'react';
import { getModelInfo, getArchitectureImage, getTrainingHistoryImage, ModelInfo } from '../projects/projectRNN/api';

const ModelVisualizer: React.FC = () => {
  const [modelInfo, setModelInfo] = useState<ModelInfo | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');

  useEffect(() => {
    const fetchModelInfo = async () => {
      try {
        const info = await getModelInfo();
        setModelInfo(info);
        setError('');
      } catch (err) {
        setError('Failed to load model information. Make sure the backend is running.');
        console.error('Failed to load model info:', err);
      } finally {
        setLoading(false);
      }
    };

    fetchModelInfo();
  }, []);

  if (loading) {
    return (
      <div className="max-w-6xl mx-auto flex items-center justify-center py-16">
        <div className="text-center">
          <div className="w-12 h-12 border-4 border-indigo-200 border-t-indigo-600 rounded-full animate-spin mx-auto mb-4"></div>
          <p className="text-gray-600 font-medium">Loading model information...</p>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="max-w-6xl mx-auto py-8">
        <div className="bg-red-50 border-l-4 border-red-500 rounded-lg p-6">
          <p className="text-red-700 font-medium">{error}</p>
        </div>
      </div>
    );
  }

  return (
    <div className="max-w-6xl mx-auto">
      {/* Header */}
      <div className="mb-10">
        <h2 className="text-4xl font-bold text-white mb-2">
          Model Architecture & Training
        </h2>
        <p className="text-indigo-100">
          Detailed information about the LSTM neural network architecture
        </p>
      </div>

      {/* Model Info Cards */}
      {modelInfo && (
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-10">
          {/* Vocabulary Size Card */}
          <div className="bg-white rounded-xl shadow-lg p-8 hover:shadow-xl transition-shadow">
            <div className="flex items-start justify-between">
              <div>
                <p className="text-sm font-semibold text-gray-500 uppercase tracking-wider">
                  Vocabulary Size
                </p>
                <p className="text-4xl font-bold text-indigo-600 mt-2">
                  {modelInfo.vocab_size.toLocaleString()}
                </p>
              </div>
              <div className="bg-indigo-100 p-3 rounded-lg">
                <svg
                  className="w-6 h-6 text-indigo-600"
                  fill="none"
                  stroke="currentColor"
                  viewBox="0 0 24 24"
                >
                  <path
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    strokeWidth={2}
                    d="M12 6V4m0 2a2 2 0 100 4m0-4a2 2 0 110 4m-6 8a2 2 0 100-4m0 4a2 2 0 110-4m0 4v2m0-6V4m6 6v10m6-2a2 2 0 100-4m0 4a2 2 0 110-4m0 4v2m0-6V4"
                  />
                </svg>
              </div>
            </div>
            <p className="text-xs text-gray-500 mt-4">
              Total unique words in the vocabulary
            </p>
          </div>

          {/* LSTM Units Card */}
          <div className="bg-white rounded-xl shadow-lg p-8 hover:shadow-xl transition-shadow">
            <div className="flex items-start justify-between">
              <div>
                <p className="text-sm font-semibold text-gray-500 uppercase tracking-wider">
                  LSTM Units
                </p>
                <p className="text-4xl font-bold text-purple-600 mt-2">
                  {modelInfo.lstm_units}
                </p>
              </div>
              <div className="bg-purple-100 p-3 rounded-lg">
                <svg
                  className="w-6 h-6 text-purple-600"
                  fill="none"
                  stroke="currentColor"
                  viewBox="0 0 24 24"
                >
                  <path
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    strokeWidth={2}
                    d="M9 19V6l12-3v13M9 19c0 1.105-1.343 2-3 2s-3-.895-3-2 1.343-2 3-2 3 .895 3 2zm12-3c0 1.105-1.343 2-3 2s-3-.895-3-2 1.343-2 3-2 3 .895 3 2zM9 10l12-3"
                  />
                </svg>
              </div>
            </div>
            <p className="text-xs text-gray-500 mt-4">
              Hidden units per LSTM layer
            </p>
          </div>

          {/* Layers Card */}
          <div className="bg-white rounded-xl shadow-lg p-8 hover:shadow-xl transition-shadow">
            <div className="flex items-start justify-between">
              <div>
                <p className="text-sm font-semibold text-gray-500 uppercase tracking-wider">
                  Number of Layers
                </p>
                <p className="text-4xl font-bold text-pink-600 mt-2">
                  {modelInfo.num_layers}
                </p>
              </div>
              <div className="bg-pink-100 p-3 rounded-lg">
                <svg
                  className="w-6 h-6 text-pink-600"
                  fill="none"
                  stroke="currentColor"
                  viewBox="0 0 24 24"
                >
                  <path
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    strokeWidth={2}
                    d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12"
                  />
                </svg>
              </div>
            </div>
            <p className="text-xs text-gray-500 mt-4">
              Stacked LSTM layers for complex learning
            </p>
          </div>
        </div>
      )}

      {/* Architecture Diagram */}
      <div className="bg-white rounded-xl shadow-lg p-8 mb-10">
        <h3 className="text-2xl font-bold text-gray-800 mb-6">
          Model Architecture Diagram
        </h3>
        <div className="bg-gray-50 rounded-lg p-4 overflow-auto">
          <img
            src={getArchitectureImage()}
            alt="Model Architecture"
            className="w-full rounded-lg border border-gray-200"
          />
        </div>
        <p className="text-sm text-gray-600 mt-4">
          Visual representation of the LSTM neural network structure with embedding layer,
          stacked LSTM layers, and output dense layer.
        </p>
      </div>

      {/* Training History */}
      <div className="bg-white rounded-xl shadow-lg p-8">
        <h3 className="text-2xl font-bold text-gray-800 mb-6">
          Training History & Performance
        </h3>
        <div className="bg-gray-50 rounded-lg p-4 overflow-auto">
          <img
            src={getTrainingHistoryImage()}
            alt="Training History"
            className="w-full rounded-lg border border-gray-200"
          />
        </div>
        <p className="text-sm text-gray-600 mt-4">
          Training and validation metrics over epochs showing model convergence and generalization.
        </p>
      </div>

      {/* Info Box */}
      <div className="mt-10 bg-gradient-to-r from-indigo-50 to-purple-50 rounded-xl border border-indigo-200 p-6">
        <h4 className="text-lg font-semibold text-gray-800 mb-3">About This Model</h4>
        <ul className="text-gray-700 space-y-2 text-sm">
          <li>✓ Temperature sampling enables controlled text generation creativity</li>
          <li>✓ Multi-layer LSTM architecture captures complex patterns in sequences</li>
          <li>✓ Embedding layer converts words to dense vector representations</li>
          <li>✓ Softmax output layer generates probability distributions for next words</li>
        </ul>
      </div>
    </div>
  );
};

export default ModelVisualizer;