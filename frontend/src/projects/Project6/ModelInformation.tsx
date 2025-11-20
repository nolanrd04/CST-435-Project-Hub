import React, { useState, useEffect, useCallback } from 'react';
import { AiOutlineInfoCircle, AiOutlineWarning, AiOutlineLineChart } from 'react-icons/ai';
import { project6API, ModelInfo } from './api.ts';

const FRUITS = ['apple', 'banana', 'blackberry', 'grape', 'pear', 'strawberry', 'watermelon'];

function ModelInformation() {
  const [models, setModels] = useState<string[]>([]);
  const [selectedModel, setSelectedModel] = useState<string>('');
  const [modelInfo, setModelInfo] = useState<ModelInfo | null>(null);
  const [selectedFruit, setSelectedFruit] = useState<string>('apple');
  const [trainingHistory, setTrainingHistory] = useState<any | null>(null);
  const [loading, setLoading] = useState<boolean>(false);
  const [error, setError] = useState<string>('');

  const loadTrainingHistory = useCallback(async (modelName: string, fruit: string) => {
    try {
      const history = await project6API.getTrainingHistory(modelName, fruit);
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
      const info = await project6API.getModelInfo(modelName);
      setModelInfo(info);
      // Load training history for first fruit
      loadTrainingHistory(modelName, selectedFruit);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to load model info');
    } finally {
      setLoading(false);
    }
  }, [selectedFruit, loadTrainingHistory]);

  const loadModels = useCallback(async () => {
    try {
      const modelList = await project6API.listModels();
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

  const handleFruitChange = (fruit: string) => {
    setSelectedFruit(fruit);
    if (selectedModel) {
      loadTrainingHistory(selectedModel, fruit);
    }
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
            No models available. Please create a model first.
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

      {/* Loading State */}
      {loading && (
        <div style={{ textAlign: 'center', padding: '40px' }}>
          <p style={{ fontSize: '18px', color: '#667eea' }}>Loading model information...</p>
        </div>
      )}

      {/* Model Info Display */}
      {!loading && modelInfo && (
        <>
          {/* Basic Information */}
          <div
            style={{
              backgroundColor: '#f8f9ff',
              border: '2px solid #667eea',
              borderRadius: '12px',
              padding: '25px',
              marginBottom: '20px',
            }}
          >
            <h3 style={{ margin: '0 0 15px 0', color: '#667eea' }}>Model Details</h3>
            <div style={{ lineHeight: '1.8', color: '#333' }}>
              <p style={{ margin: '8px 0' }}>
                <strong>Model Name:</strong> {modelInfo.model_name}
              </p>
              <p style={{ margin: '8px 0' }}>
                <strong>Data Version:</strong> {modelInfo.data_version}
              </p>
              <p style={{ margin: '8px 0' }}>
                <strong>Description:</strong> {modelInfo.description}
              </p>
              <p style={{ margin: '8px 0' }}>
                <strong>Trained Fruits:</strong> {modelInfo.fruits.join(', ')}
              </p>
            </div>
          </div>

          {/* Model Architecture */}
          {modelInfo.model_architecture && (
            <div
              style={{
                backgroundColor: '#f0f4ff',
                border: '2px solid #4f46e5',
                borderRadius: '12px',
                padding: '25px',
                marginBottom: '20px',
              }}
            >
              <h3 style={{ margin: '0 0 15px 0', color: '#4f46e5' }}>Model Architecture</h3>
              <div
                style={{
                  display: 'grid',
                  gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))',
                  gap: '15px',
                  marginBottom: '15px',
                }}
              >
                <div>
                  <div style={{ fontSize: '13px', color: '#666' }}>Image Size</div>
                  <div style={{ fontSize: '20px', fontWeight: 'bold', color: '#4f46e5', marginTop: '5px' }}>
                    {modelInfo.model_architecture.image_size}x{modelInfo.model_architecture.image_size}
                  </div>
                </div>
                <div>
                  <div style={{ fontSize: '13px', color: '#666' }}>Latent Dimension</div>
                  <div style={{ fontSize: '20px', fontWeight: 'bold', color: '#4f46e5', marginTop: '5px' }}>
                    {modelInfo.model_architecture.latent_dim}
                  </div>
                </div>
                <div>
                  <div style={{ fontSize: '13px', color: '#666' }}>Channels</div>
                  <div style={{ fontSize: '20px', fontWeight: 'bold', color: '#4f46e5', marginTop: '5px' }}>
                    {modelInfo.model_architecture.channels} (Grayscale)
                  </div>
                </div>
              </div>
              <div style={{ marginTop: '15px' }}>
                <div style={{ fontSize: '13px', color: '#666', marginBottom: '5px' }}>Generator Architecture</div>
                <div style={{ fontSize: '14px', color: '#333', padding: '10px', backgroundColor: 'white', borderRadius: '6px', border: '1px solid #e0e7ff' }}>
                  {modelInfo.model_architecture.generator_layers}
                </div>
              </div>
              <div style={{ marginTop: '10px' }}>
                <div style={{ fontSize: '13px', color: '#666', marginBottom: '5px' }}>Discriminator Architecture</div>
                <div style={{ fontSize: '14px', color: '#333', padding: '10px', backgroundColor: 'white', borderRadius: '6px', border: '1px solid #e0e7ff' }}>
                  {modelInfo.model_architecture.discriminator_layers}
                </div>
              </div>
            </div>
          )}

          {/* Parameter Counts */}
          {modelInfo.total_parameters && (
            <div
              style={{
                backgroundColor: '#fef3c7',
                border: '2px solid #f59e0b',
                borderRadius: '12px',
                padding: '25px',
                marginBottom: '20px',
              }}
            >
              <h3 style={{ margin: '0 0 15px 0', color: '#f59e0b' }}>Model Parameters</h3>
              <div style={{ marginBottom: '15px' }}>
                <div style={{ fontSize: '13px', color: '#666', marginBottom: '5px' }}>Total Parameters (All Fruits)</div>
                <div style={{ fontSize: '28px', fontWeight: 'bold', color: '#f59e0b' }}>
                  {modelInfo.total_parameters.toLocaleString()}
                </div>
              </div>

              {modelInfo.parameters_per_fruit && (
                <div>
                  <div style={{ fontSize: '13px', color: '#666', marginBottom: '10px' }}>Parameters per Fruit (Generator + Discriminator)</div>
                  <div
                    style={{
                      display: 'grid',
                      gridTemplateColumns: 'repeat(auto-fill, minmax(150px, 1fr))',
                      gap: '10px',
                    }}
                  >
                    {Object.entries(modelInfo.parameters_per_fruit).map(([fruit, count]) => (
                      <div
                        key={fruit}
                        style={{
                          backgroundColor: 'white',
                          padding: '12px',
                          borderRadius: '8px',
                          border: '1px solid #fbbf24',
                          textAlign: 'center',
                        }}
                      >
                        <div style={{ fontSize: '12px', color: '#666', textTransform: 'capitalize', marginBottom: '5px' }}>
                          {fruit}
                        </div>
                        <div style={{ fontSize: '16px', fontWeight: 'bold', color: '#f59e0b' }}>
                          {count.toLocaleString()}
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </div>
          )}

          {/* Training Configuration */}
          {modelInfo.training_config && (
            <div
              style={{
                backgroundColor: '#f8f9ff',
                border: '2px solid #667eea',
                borderRadius: '12px',
                padding: '25px',
                marginBottom: '20px',
              }}
            >
              <h3 style={{ margin: '0 0 15px 0', color: '#667eea' }}>Training Configuration</h3>
              <div
                style={{
                  display: 'grid',
                  gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))',
                  gap: '15px',
                }}
              >
                <div>
                  <div style={{ fontSize: '13px', color: '#666' }}>Epochs</div>
                  <div style={{ fontSize: '20px', fontWeight: 'bold', color: '#667eea', marginTop: '5px' }}>
                    {modelInfo.training_config.epochs}
                  </div>
                </div>
                <div>
                  <div style={{ fontSize: '13px', color: '#666' }}>Batch Size</div>
                  <div style={{ fontSize: '20px', fontWeight: 'bold', color: '#667eea', marginTop: '5px' }}>
                    {modelInfo.training_config.batch_size}
                  </div>
                </div>
                <div>
                  <div style={{ fontSize: '13px', color: '#666' }}>Learning Rate</div>
                  <div style={{ fontSize: '20px', fontWeight: 'bold', color: '#667eea', marginTop: '5px' }}>
                    {modelInfo.training_config.learning_rate}
                  </div>
                </div>
                <div>
                  <div style={{ fontSize: '13px', color: '#666' }}>Image Resolution</div>
                  <div style={{ fontSize: '20px', fontWeight: 'bold', color: '#667eea', marginTop: '5px' }}>
                    {modelInfo.training_config.image_resolution}x
                    {modelInfo.training_config.image_resolution}
                  </div>
                </div>
                <div>
                  <div style={{ fontSize: '13px', color: '#666' }}>Images per Fruit</div>
                  <div style={{ fontSize: '20px', fontWeight: 'bold', color: '#667eea', marginTop: '5px' }}>
                    {modelInfo.training_config.image_count_per_fruit}
                  </div>
                </div>
              </div>
            </div>
          )}

          {/* Training Statistics */}
          {modelInfo.training_stats && (
            <div
              style={{
                backgroundColor: '#f0fff4',
                border: '2px solid #48bb78',
                borderRadius: '12px',
                padding: '25px',
                marginBottom: '20px',
              }}
            >
              <h3 style={{ margin: '0 0 15px 0', color: '#48bb78' }}>Training Statistics</h3>
              <div
                style={{
                  display: 'grid',
                  gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))',
                  gap: '15px',
                }}
              >
                <div>
                  <div style={{ fontSize: '13px', color: '#666' }}>Total Training Time</div>
                  <div style={{ fontSize: '20px', fontWeight: 'bold', color: '#48bb78', marginTop: '5px' }}>
                    {modelInfo.training_stats.total_training_time_hours.toFixed(2)} hours
                  </div>
                </div>
                <div>
                  <div style={{ fontSize: '13px', color: '#666' }}>Peak Memory</div>
                  <div style={{ fontSize: '20px', fontWeight: 'bold', color: '#48bb78', marginTop: '5px' }}>
                    {modelInfo.training_stats.peak_memory_gb.toFixed(2)} GB
                  </div>
                </div>
                <div>
                  <div style={{ fontSize: '13px', color: '#666' }}>Total Cost</div>
                  <div style={{ fontSize: '20px', fontWeight: 'bold', color: '#48bb78', marginTop: '5px' }}>
                    ${modelInfo.training_stats.total_training_cost.toFixed(4)}
                  </div>
                </div>
                <div>
                  <div style={{ fontSize: '13px', color: '#666' }}>Avg Cost per Fruit</div>
                  <div style={{ fontSize: '20px', fontWeight: 'bold', color: '#48bb78', marginTop: '5px' }}>
                    ${modelInfo.training_stats.avg_cost_per_fruit.toFixed(4)}
                  </div>
                </div>
                <div>
                  <div style={{ fontSize: '13px', color: '#666' }}>Avg Cost per Epoch</div>
                  <div style={{ fontSize: '20px', fontWeight: 'bold', color: '#48bb78', marginTop: '5px' }}>
                    ${modelInfo.training_stats.avg_cost_per_epoch.toFixed(6)}
                  </div>
                </div>
              </div>
            </div>
          )}

          {/* Fruit-Specific Training History */}
          <div
            style={{
              backgroundColor: '#fff5f0',
              border: '2px solid #ff7b29',
              borderRadius: '12px',
              padding: '25px',
            }}
          >
            <h3
              style={{
                margin: '0 0 15px 0',
                color: '#ff7b29',
                display: 'flex',
                alignItems: 'center',
                gap: '10px',
              }}
            >
              <AiOutlineLineChart size={24} />
              Training History by Fruit
            </h3>

            {/* Fruit Selection */}
            <div style={{ marginBottom: '20px' }}>
              <label style={{ display: 'block', fontSize: '14px', fontWeight: 'bold', marginBottom: '10px' }}>
                Select Fruit:
              </label>
              <div
                style={{
                  display: 'grid',
                  gridTemplateColumns: 'repeat(auto-fill, minmax(120px, 1fr))',
                  gap: '10px',
                }}
              >
                {FRUITS.map((fruit) => (
                  <button
                    key={fruit}
                    onClick={() => handleFruitChange(fruit)}
                    style={{
                      padding: '10px',
                      border: selectedFruit === fruit ? '3px solid #ff7b29' : '2px solid #e0e0e0',
                      backgroundColor: selectedFruit === fruit ? '#fff5f0' : 'white',
                      borderRadius: '8px',
                      cursor: 'pointer',
                      fontWeight: selectedFruit === fruit ? 'bold' : 'normal',
                      color: selectedFruit === fruit ? '#ff7b29' : '#666',
                      textTransform: 'capitalize',
                      fontSize: '13px',
                    }}
                  >
                    {fruit}
                  </button>
                ))}
              </div>
            </div>

            {/* Training History Data */}
            {trainingHistory ? (
              <div>
                <div
                  style={{
                    backgroundColor: 'white',
                    padding: '20px',
                    borderRadius: '8px',
                    border: '1px solid #e0e0e0',
                    marginBottom: '15px',
                  }}
                >
                  <h4 style={{ margin: '0 0 10px 0', color: '#ff7b29', textTransform: 'capitalize' }}>
                    {selectedFruit} Training Metrics
                  </h4>
                  <div style={{ fontSize: '14px', lineHeight: '1.8', color: '#333' }}>
                    {trainingHistory.training_time_seconds && (
                      <p style={{ margin: '5px 0' }}>
                        <strong>Training Time:</strong>{' '}
                        {(trainingHistory.training_time_seconds / 60).toFixed(2)} minutes
                      </p>
                    )}
                    {trainingHistory.peak_memory_gb && (
                      <p style={{ margin: '5px 0' }}>
                        <strong>Peak Memory:</strong> {trainingHistory.peak_memory_gb.toFixed(2)} GB
                      </p>
                    )}
                    {trainingHistory.cost_summary && (
                      <>
                        <p style={{ margin: '5px 0' }}>
                          <strong>Total Cost:</strong> ${trainingHistory.cost_summary.total_cost.toFixed(4)}
                        </p>
                        <p style={{ margin: '5px 0' }}>
                          <strong>Cost per Epoch:</strong> $
                          {trainingHistory.cost_summary.cost_per_epoch.toFixed(6)}
                        </p>
                      </>
                    )}
                  </div>
                </div>

                {/* Loss Graph (Simple Text Display) */}
                {trainingHistory.losses && trainingHistory.losses.length > 0 && (
                  <div
                    style={{
                      backgroundColor: 'white',
                      padding: '20px',
                      borderRadius: '8px',
                      border: '1px solid #e0e0e0',
                    }}
                  >
                    <h4 style={{ margin: '0 0 10px 0', color: '#ff7b29' }}>Loss Values</h4>
                    <div
                      style={{
                        maxHeight: '200px',
                        overflowY: 'auto',
                        fontSize: '12px',
                        fontFamily: 'monospace',
                      }}
                    >
                      <div style={{ display: 'grid', gridTemplateColumns: '1fr 2fr 2fr', gap: '10px', fontWeight: 'bold', borderBottom: '1px solid #e0e0e0', paddingBottom: '8px', marginBottom: '8px' }}>
                        <div>Epoch</div>
                        <div>Gen Loss</div>
                        <div>Disc Loss</div>
                      </div>
                      {trainingHistory.losses.map((loss: any, idx: number) => (
                        <div
                          key={idx}
                          style={{
                            display: 'grid',
                            gridTemplateColumns: '1fr 2fr 2fr',
                            gap: '10px',
                            padding: '4px 0',
                            borderBottom: '1px solid #f0f0f0',
                          }}
                        >
                          <div>{loss.epoch}</div>
                          <div>{loss.generator_loss.toFixed(4)}</div>
                          <div>{loss.discriminator_loss.toFixed(4)}</div>
                        </div>
                      ))}
                    </div>
                  </div>
                )}
              </div>
            ) : (
              <p style={{ color: '#999', fontStyle: 'italic', textAlign: 'center', padding: '20px' }}>
                No training history available for {selectedFruit}
              </p>
            )}
          </div>
        </>
      )}
    </div>
  );
}

export default ModelInformation;
