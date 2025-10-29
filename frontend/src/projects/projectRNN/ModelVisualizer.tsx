import React, { useEffect, useState } from 'react';
import { getModelInfo, getArchitectureImage, getTrainingHistoryImage, ModelInfo } from './api.ts';

const ModelVisualizer: React.FC = () => {
  const [modelInfo, setModelInfo] = useState<ModelInfo | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const fetchModelInfo = async () => {
      try {
        const info = await getModelInfo();
        setModelInfo(info);
      } catch (err) {
        console.error('Failed to load model info');
      } finally {
        setLoading(false);
      }
    };

    fetchModelInfo();
  }, []);

  if (loading) {
    return <div className="text-center p-8">Loading model information...</div>;
  }

  return (
    <div className="max-w-6xl mx-auto p-6">
      <h2 className="text-3xl font-bold mb-6 text-indigo-600">
        Model Architecture & Training
      </h2>

      {/* Model Info Cards */}
      {modelInfo && (
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-8">
          <div className="bg-white p-6 rounded-lg shadow">
            <h3 className="text-sm font-medium text-gray-500">Vocabulary Size</h3>
            <p className="text-3xl font-bold text-indigo-600">{modelInfo.vocab_size.toLocaleString()}</p>
          </div>

          <div className="bg-white p-6 rounded-lg shadow">
            <h3 className="text-sm font-medium text-gray-500">LSTM Units</h3>
            <p className="text-3xl font-bold text-indigo-600">{modelInfo.lstm_units}</p>
          </div>

          <div className="bg-white p-6 rounded-lg shadow">
            <h3 className="text-sm font-medium text-gray-500">Number of Layers</h3>
            <p className="text-3xl font-bold text-indigo-600">{modelInfo.num_layers}</p>
          </div>
        </div>
      )}

      {/* Architecture Diagram */}
      <div className="bg-white rounded-lg shadow-lg p-6 mb-8">
        <h3 className="text-xl font-semibold mb-4">Model Architecture</h3>
        <img
          src={getArchitectureImage()}
          alt="Model Architecture"
          className="w-full rounded-lg"
        />
      </div>

      {/* Training History */}
      <div className="bg-white rounded-lg shadow-lg p-6">
        <h3 className="text-xl font-semibold mb-4">Training History</h3>
        <img
          src={getTrainingHistoryImage()}
          alt="Training History"
          className="w-full rounded-lg"
        />
      </div>
    </div>
  );
};

export default ModelVisualizer;