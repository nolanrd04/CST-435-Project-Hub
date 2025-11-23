import React, { useState, useEffect, useCallback } from 'react';
import { AiOutlineDollar, AiOutlineWarning } from 'react-icons/ai';
import { project6API, TrainingCostAnalysis as CostData } from './api.ts';

function TrainingCostAnalysis() {
  const [models, setModels] = useState<string[]>([]);
  const [selectedModel, setSelectedModel] = useState<string>('');
  const [costData, setCostData] = useState<CostData | null>(null);
  const [loading, setLoading] = useState<boolean>(false);
  const [error, setError] = useState<string>('');

  const loadCostData = useCallback(async (modelName: string) => {
    setLoading(true);
    setError('');
    try {
      const data = await project6API.getTrainingCostAnalysis(modelName);
      setCostData(data);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to load cost analysis');
    } finally {
      setLoading(false);
    }
  }, []);

  const loadModels = useCallback(async () => {
    try {
      const modelList = await project6API.listModels();
      setModels(modelList);
      if (modelList.length > 0) {
        setSelectedModel(modelList[0]);
        // Auto-load cost data for first model
        loadCostData(modelList[0]);
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to load models');
    }
  }, [loadCostData]);

  // Load available models on mount
  useEffect(() => {
    loadModels();
  }, [loadModels]);

  const handleModelChange = (modelName: string) => {
    setSelectedModel(modelName);
    loadCostData(modelName);
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
          <AiOutlineDollar size={24} />
        </span>
        Training Cost Analysis
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
          <p style={{ fontSize: '18px', color: '#667eea' }}>Loading cost analysis...</p>
        </div>
      )}

      {/* Cost Data Display */}
      {!loading && costData && (
        <>
          {/* Cost Summary Cards */}
          <div
            style={{
              display: 'grid',
              gridTemplateColumns: 'repeat(auto-fit, minmax(250px, 1fr))',
              gap: '15px',
              marginBottom: '30px',
            }}
          >
            {/* Total Training Cost */}
            <div
              style={{
                backgroundColor: '#f8f9ff',
                border: '2px solid #667eea',
                borderRadius: '12px',
                padding: '20px',
                textAlign: 'center',
              }}
            >
              <div style={{ fontSize: '14px', color: '#666', marginBottom: '8px' }}>
                Total Training Cost
              </div>
              <div style={{ fontSize: '28px', fontWeight: 'bold', color: '#667eea' }}>
                ${(costData.training_cost_breakdown?.total_cost ?? 0).toFixed(4)}
              </div>
              <div style={{ fontSize: '12px', color: '#999', marginTop: '8px' }}>
                {(costData.training_hours ?? 0).toFixed(2)} hours
              </div>
            </div>

            {/* Cost Per Epoch */}
            <div
              style={{
                backgroundColor: '#f8f9ff',
                border: '2px solid #667eea',
                borderRadius: '12px',
                padding: '20px',
                textAlign: 'center',
              }}
            >
              <div style={{ fontSize: '14px', color: '#666', marginBottom: '8px' }}>
                Cost Per Epoch
              </div>
              <div style={{ fontSize: '28px', fontWeight: 'bold', color: '#667eea' }}>
                ${(costData.cost_per_epoch?.avg_cost_per_epoch ?? 0).toFixed(6)}
              </div>
              <div style={{ fontSize: '12px', color: '#999', marginTop: '8px' }}>
                For {costData.cost_per_epoch?.total_epochs ?? 0} epochs
              </div>
            </div>

            {/* Peak Memory */}
            <div
              style={{
                backgroundColor: '#f8f9ff',
                border: '2px solid #667eea',
                borderRadius: '12px',
                padding: '20px',
                textAlign: 'center',
              }}
            >
              <div style={{ fontSize: '14px', color: '#666', marginBottom: '8px' }}>
                Peak Memory Usage
              </div>
              <div style={{ fontSize: '28px', fontWeight: 'bold', color: '#667eea' }}>
                {(costData.peak_memory_gb ?? 0).toFixed(2)} GB
              </div>
              <div style={{ fontSize: '12px', color: '#999', marginTop: '8px' }}>
                Maximum allocated
              </div>
            </div>
          </div>

          {/* Cost Breakdown */}
          <div
            style={{
              backgroundColor: '#f8f9ff',
              border: '2px solid #667eea',
              borderRadius: '12px',
              padding: '20px',
              marginBottom: '20px',
            }}
          >
            <h3 style={{ margin: '0 0 15px 0', color: '#667eea' }}>Cost Breakdown</h3>
            <div
              style={{
                display: 'grid',
                gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))',
                gap: '15px',
              }}
            >
              <div>
                <div style={{ fontSize: '13px', color: '#666' }}>Compute Cost</div>
                <div
                  style={{
                    fontSize: '20px',
                    fontWeight: 'bold',
                    color: '#667eea',
                    marginTop: '5px',
                  }}
                >
                  ${(costData.training_cost_breakdown?.compute_cost ?? 0).toFixed(4)}
                </div>
              </div>
              <div>
                <div style={{ fontSize: '13px', color: '#666' }}>Memory Cost</div>
                <div
                  style={{
                    fontSize: '20px',
                    fontWeight: 'bold',
                    color: '#667eea',
                    marginTop: '5px',
                  }}
                >
                  ${(costData.training_cost_breakdown?.memory_cost ?? 0).toFixed(4)}
                </div>
              </div>
              <div>
                <div style={{ fontSize: '13px', color: '#666' }}>Storage Cost</div>
                <div
                  style={{
                    fontSize: '20px',
                    fontWeight: 'bold',
                    color: '#667eea',
                    marginTop: '5px',
                  }}
                >
                  ${(costData.training_cost_breakdown?.storage_cost ?? 0).toFixed(4)}
                </div>
              </div>
            </div>
          </div>

          {/* Training Information */}
          <div
            style={{
              backgroundColor: '#f0f4ff',
              border: '2px solid #667eea',
              borderRadius: '12px',
              padding: '20px',
            }}
          >
            <h3 style={{ margin: '0 0 15px 0', color: '#667eea' }}>Training Information</h3>
            <div style={{ lineHeight: '1.8', color: '#333' }}>
              <p style={{ margin: '8px 0' }}>
                <strong>Total Training Time:</strong>{' '}
                {Math.floor(costData.training_hours ?? 0)} hours{' '}
                {Math.round(((costData.training_hours ?? 0) % 1) * 60)} minutes
              </p>
              <p style={{ margin: '8px 0' }}>
                <strong>Total Epochs:</strong> {costData.cost_per_epoch?.total_epochs ?? 0}
              </p>
              <p style={{ margin: '8px 0' }}>
                <strong>Cost per Hour:</strong> $
                {(
                  (costData.training_cost_breakdown?.total_cost ?? 0) / (costData.training_hours ?? 1)
                ).toFixed(6)}
              </p>
            </div>
          </div>
        </>
      )}

      {/* No Data State */}
      {!loading && !costData && models.length > 0 && (
        <div
          style={{
            textAlign: 'center',
            padding: '40px',
            color: '#999',
            fontStyle: 'italic',
          }}
        >
          No cost analysis data available for this model.
        </div>
      )}
    </div>
  );
}

export default TrainingCostAnalysis;
