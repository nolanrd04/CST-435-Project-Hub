import React, { useState, useEffect, useCallback } from 'react';
import { AiOutlineInfoCircle, AiOutlineWarning } from 'react-icons/ai';
import { project7API, ModelInfo, TrainingHistory } from './api.ts';

function ModelInformation() {
  const [models, setModels] = useState<string[]>([]);
  const [selectedModel, setSelectedModel] = useState<string>('');
  const [modelInfo, setModelInfo] = useState<ModelInfo | null>(null);
  const [trainingHistory, setTrainingHistory] = useState<TrainingHistory | null>(null);
  const [loading, setLoading] = useState<boolean>(false);
  const [error, setError] = useState<string>('');

  const loadTrainingHistory = useCallback(async (modelName: string) => {
    try {
      const history = await project7API.getTrainingHistory(modelName);
      setTrainingHistory(history);
    } catch (err) {
      console.error('Failed to load training history:', err);
      setTrainingHistory(null);
    }
  }, []);

  const loadModelInfo = useCallback(async (modelName: string) => {
    setLoading(true);
    setError('');
    try {
      const info = await project7API.getModelInfo(modelName);
      setModelInfo(info);
      loadTrainingHistory(modelName);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to load model info');
    } finally {
      setLoading(false);
    }
  }, [loadTrainingHistory]);

  const loadModels = useCallback(async () => {
    try {
      const modelList = await project7API.listModels();
      setModels(modelList);
      if (modelList.length > 0) {
        setSelectedModel(modelList[0]);
        loadModelInfo(modelList[0]);
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to load models');
    }
  }, [loadModelInfo]);

  useEffect(() => {
    loadModels();
  }, [loadModels]);

  const handleModelChange = (modelName: string) => {
    setSelectedModel(modelName);
    loadModelInfo(modelName);
  };

  return (
    <div style={{ padding: '20px' }}>
      <h2
        style={{
          marginBottom: '30px',
          display: 'flex',
          alignItems: 'center',
          gap: '10px',
        }}
      >
        <span
          style={{
            background: 'linear-gradient(135deg, #667eea, #764ba2)',
            color: 'white',
            width: '40px',
            height: '40px',
            borderRadius: '50%',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            fontSize: '20px',
          }}
        >
          <AiOutlineInfoCircle size={24} />
        </span>
        Model Information
      </h2>

      {/* Error Message */}
      {error && (
        <div
          style={{
            backgroundColor: '#fee',
            border: '2px solid #f99',
            borderRadius: '8px',
            padding: '15px',
            marginBottom: '20px',
            color: '#c33',
            display: 'flex',
            alignItems: 'center',
            gap: '10px',
          }}
        >
          <AiOutlineWarning size={20} />
          {error}
        </div>
      )}

      {/* Model Selection */}
      <div style={{ marginBottom: '30px' }}>
        <label
          style={{
            display: 'block',
            fontSize: '14px',
            fontWeight: 'bold',
            marginBottom: '10px',
            color: '#333',
          }}
        >
          Select Model
        </label>
        {models.length === 0 ? (
          <p style={{ color: '#999', fontStyle: 'italic' }}>
            No models available. Please train a model first.
          </p>
        ) : (
          <select
            value={selectedModel}
            onChange={(e) => handleModelChange(e.target.value)}
            style={{
              width: '100%',
              maxWidth: '400px',
              padding: '12px',
              fontSize: '14px',
              border: '2px solid #e0e0e0',
              borderRadius: '8px',
              backgroundColor: 'white',
              cursor: 'pointer',
            }}
          >
            {models.map((model) => (
              <option key={model} value={model}>
                {model}
              </option>
            ))}
          </select>
        )}
      </div>

      {loading && (
        <div style={{ textAlign: 'center', padding: '40px', color: '#667eea' }}>
          Loading model information...
        </div>
      )}

      {!loading && modelInfo && (
        <>
          {/* Model Configuration */}
          <div
            style={{
              background: 'white',
              borderRadius: '12px',
              padding: '24px',
              marginBottom: '30px',
              boxShadow: '0 2px 8px rgba(0,0,0,0.1)',
            }}
          >
            <h3 style={{ marginTop: 0, color: '#667eea', display: 'flex', alignItems: 'center', gap: '10px' }}>
              <AiOutlineInfoCircle size={24} />
              Model Configuration
            </h3>

            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(250px, 1fr))', gap: '20px' }}>
              <div>
                <div style={{ fontWeight: 'bold', color: '#666', marginBottom: '5px' }}>Model Name</div>
                <div style={{ fontSize: '18px', color: '#333' }}>{modelInfo.model_name}</div>
              </div>

              <div>
                <div style={{ fontWeight: 'bold', color: '#666', marginBottom: '5px' }}>Parameters</div>
                <div style={{ fontSize: '18px', color: '#333' }}>
                  {(modelInfo.model_parameters / 1_000_000).toFixed(2)}M
                </div>
              </div>

              <div>
                <div style={{ fontWeight: 'bold', color: '#666', marginBottom: '5px' }}>Image Size</div>
                <div style={{ fontSize: '18px', color: '#333' }}>{modelInfo.image_size}x{modelInfo.image_size}</div>
              </div>

              <div>
                <div style={{ fontWeight: 'bold', color: '#666', marginBottom: '5px' }}>U-Net Features</div>
                <div style={{ fontSize: '18px', color: '#333' }}>[{modelInfo.config.features.join(', ')}]</div>
              </div>

              <div>
                <div style={{ fontWeight: 'bold', color: '#666', marginBottom: '5px' }}>Timesteps</div>
                <div style={{ fontSize: '18px', color: '#333' }}>{modelInfo.config.timesteps}</div>
              </div>

              <div>
                <div style={{ fontWeight: 'bold', color: '#666', marginBottom: '5px' }}>Batch Size</div>
                <div style={{ fontSize: '18px', color: '#333' }}>{modelInfo.config.batch_size}</div>
              </div>

              <div>
                <div style={{ fontWeight: 'bold', color: '#666', marginBottom: '5px' }}>Learning Rate</div>
                <div style={{ fontSize: '18px', color: '#333' }}>{modelInfo.config.learning_rate}</div>
              </div>

              <div>
                <div style={{ fontWeight: 'bold', color: '#666', marginBottom: '5px' }}>Total Epochs</div>
                <div style={{ fontSize: '18px', color: '#333' }}>{modelInfo.config.num_epochs}</div>
              </div>
            </div>
          </div>

          {/* Training Progress */}
          {modelInfo.training_progress && (
            <div
              style={{
                background: 'white',
                borderRadius: '12px',
                padding: '24px',
                marginBottom: '30px',
                boxShadow: '0 2px 8px rgba(0,0,0,0.1)',
              }}
            >
              <h3 style={{ marginTop: 0, color: '#667eea' }}>Training Progress</h3>

              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', gap: '20px' }}>
                <div>
                  <div style={{ fontWeight: 'bold', color: '#666', marginBottom: '5px' }}>Current Epoch</div>
                  <div style={{ fontSize: '24px', color: '#667eea', fontWeight: 'bold' }}>
                    {modelInfo.training_progress.current_epoch} / {modelInfo.training_progress.total_epochs}
                  </div>
                </div>

                <div>
                  <div style={{ fontWeight: 'bold', color: '#666', marginBottom: '5px' }}>Training Loss</div>
                  <div style={{ fontSize: '24px', color: '#333' }}>
                    {modelInfo.training_progress.train_loss.toFixed(6)}
                  </div>
                </div>

                <div>
                  <div style={{ fontWeight: 'bold', color: '#666', marginBottom: '5px' }}>Validation Loss</div>
                  <div style={{ fontSize: '24px', color: '#333' }}>
                    {modelInfo.training_progress.val_loss.toFixed(6)}
                  </div>
                </div>

                <div>
                  <div style={{ fontWeight: 'bold', color: '#666', marginBottom: '5px' }}>Best Val Loss</div>
                  <div style={{ fontSize: '24px', color: '#3c3' }}>
                    {modelInfo.training_progress.best_val_loss.toFixed(6)}
                  </div>
                </div>
              </div>
            </div>
          )}

          {/* Training Loss History */}
          {trainingHistory && trainingHistory.train_loss.length > 0 && (
            <div
              style={{
                background: 'white',
                borderRadius: '12px',
                padding: '24px',
                marginBottom: '30px',
                boxShadow: '0 2px 8px rgba(0,0,0,0.1)',
              }}
            >
              <h3 style={{ marginTop: 0, color: '#667eea', display: 'flex', alignItems: 'center', gap: '10px' }}>
                <AiOutlineInfoCircle size={24} />
                Training Loss Summary
              </h3>

              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', gap: '20px', marginBottom: '20px' }}>
                <div>
                  <div style={{ fontWeight: 'bold', color: '#666', marginBottom: '5px' }}>Latest Train Loss</div>
                  <div style={{ fontSize: '24px', color: '#667eea', fontWeight: 'bold' }}>
                    {trainingHistory.train_loss[trainingHistory.train_loss.length - 1].toFixed(6)}
                  </div>
                </div>

                <div>
                  <div style={{ fontWeight: 'bold', color: '#666', marginBottom: '5px' }}>Latest Val Loss</div>
                  <div style={{ fontSize: '24px', color: '#f093fb', fontWeight: 'bold' }}>
                    {trainingHistory.val_loss[trainingHistory.val_loss.length - 1].toFixed(6)}
                  </div>
                </div>

                <div>
                  <div style={{ fontWeight: 'bold', color: '#666', marginBottom: '5px' }}>Best Train Loss</div>
                  <div style={{ fontSize: '24px', color: '#3c3', fontWeight: 'bold' }}>
                    {Math.min(...trainingHistory.train_loss).toFixed(6)}
                  </div>
                </div>

                <div>
                  <div style={{ fontWeight: 'bold', color: '#666', marginBottom: '5px' }}>Best Val Loss</div>
                  <div style={{ fontSize: '24px', color: '#3c3', fontWeight: 'bold' }}>
                    {Math.min(...trainingHistory.val_loss).toFixed(6)}
                  </div>
                </div>
              </div>

              <div style={{ fontSize: '14px', color: '#666' }}>
                Total epochs trained: {trainingHistory.epochs.length}
              </div>
            </div>
          )}
        </>
      )}
    </div>
  );
}

export default ModelInformation;
