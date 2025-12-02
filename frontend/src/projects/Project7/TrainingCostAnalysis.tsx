import React, { useState, useEffect } from 'react';
import { AiOutlineDollarCircle, AiOutlineWarning } from 'react-icons/ai';
import { project7API, CostAnalysis } from './api.ts';

function TrainingCostAnalysis() {
  const [models, setModels] = useState<string[]>([]);
  const [selectedModel, setSelectedModel] = useState<string>('');
  const [costAnalysis, setCostAnalysis] = useState<CostAnalysis | null>(null);
  const [loading, setLoading] = useState<boolean>(false);
  const [error, setError] = useState<string>('');

  useEffect(() => {
    loadModels();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const loadModels = async () => {
    try {
      const modelList = await project7API.listModels();
      setModels(modelList);
      if (modelList.length > 0) {
        setSelectedModel(modelList[0]);
        loadCostAnalysis(modelList[0]);
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to load models');
    }
  };

  const loadCostAnalysis = async (modelName: string) => {
    setLoading(true);
    setError('');
    try {
      const analysis = await project7API.getCostAnalysis(modelName);
      setCostAnalysis(analysis);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to load cost analysis');
    } finally {
      setLoading(false);
    }
  };

  const handleModelChange = (modelName: string) => {
    setSelectedModel(modelName);
    loadCostAnalysis(modelName);
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
          <AiOutlineDollarCircle size={24} />
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
          Loading cost analysis...
        </div>
      )}

      {!loading && costAnalysis && (
        <>
          {/* Total Cost Summary */}
          <div
            style={{
              background: 'linear-gradient(135deg, #667eea, #764ba2)',
              borderRadius: '16px',
              padding: '32px',
              marginBottom: '30px',
              color: 'white',
              boxShadow: '0 4px 16px rgba(102, 126, 234, 0.3)',
            }}
          >
            <div style={{ fontSize: '14px', opacity: 0.9, marginBottom: '8px' }}>Total Training Cost</div>
            <div style={{ fontSize: '48px', fontWeight: 'bold' }}>
              ${costAnalysis.training_cost_breakdown.total_cost.toFixed(2)}
            </div>
            <div style={{ fontSize: '14px', opacity: 0.9, marginTop: '16px' }}>
              {costAnalysis.training_hours.toFixed(2)} hours of training •{' '}
              {costAnalysis.cost_per_epoch.total_epochs} epochs
            </div>
          </div>

          {/* Cost Breakdown */}
          <div
            style={{
              background: 'white',
              borderRadius: '12px',
              padding: '24px',
              marginBottom: '30px',
              boxShadow: '0 2px 8px rgba(0,0,0,0.1)',
            }}
          >
            <h3 style={{ marginTop: 0, color: '#667eea' }}>Cost Breakdown</h3>

            <div style={{ display: 'grid', gap: '16px' }}>
              {/* Compute Cost */}
              <div
                style={{
                  display: 'flex',
                  justifyContent: 'space-between',
                  alignItems: 'center',
                  padding: '16px',
                  background: '#f9f9ff',
                  borderRadius: '8px',
                }}
              >
                <div>
                  <div style={{ fontWeight: 'bold', color: '#333' }}>Compute Cost</div>
                  <div style={{ fontSize: '12px', color: '#666' }}>CPU/GPU processing time</div>
                </div>
                <div style={{ fontSize: '24px', fontWeight: 'bold', color: '#667eea' }}>
                  ${costAnalysis.training_cost_breakdown.compute_cost.toFixed(2)}
                </div>
              </div>

              {/* Memory Cost */}
              <div
                style={{
                  display: 'flex',
                  justifyContent: 'space-between',
                  alignItems: 'center',
                  padding: '16px',
                  background: '#f9f9ff',
                  borderRadius: '8px',
                }}
              >
                <div>
                  <div style={{ fontWeight: 'bold', color: '#333' }}>Memory Cost</div>
                  <div style={{ fontSize: '12px', color: '#666' }}>
                    RAM/VRAM usage ({costAnalysis.peak_memory_gb.toFixed(2)} GB peak)
                  </div>
                </div>
                <div style={{ fontSize: '24px', fontWeight: 'bold', color: '#764ba2' }}>
                  ${costAnalysis.training_cost_breakdown.memory_cost.toFixed(2)}
                </div>
              </div>

              {/* Storage Cost */}
              <div
                style={{
                  display: 'flex',
                  justifyContent: 'space-between',
                  alignItems: 'center',
                  padding: '16px',
                  background: '#f9f9ff',
                  borderRadius: '8px',
                }}
              >
                <div>
                  <div style={{ fontWeight: 'bold', color: '#333' }}>Storage Cost</div>
                  <div style={{ fontSize: '12px', color: '#666' }}>Model checkpoints and data</div>
                </div>
                <div style={{ fontSize: '24px', fontWeight: 'bold', color: '#f093fb' }}>
                  ${costAnalysis.training_cost_breakdown.storage_cost.toFixed(2)}
                </div>
              </div>
            </div>
          </div>

          {/* Per-Epoch Analysis */}
          <div
            style={{
              background: 'white',
              borderRadius: '12px',
              padding: '24px',
              marginBottom: '30px',
              boxShadow: '0 2px 8px rgba(0,0,0,0.1)',
            }}
          >
            <h3 style={{ marginTop: 0, color: '#667eea' }}>Per-Epoch Analysis</h3>

            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', gap: '20px' }}>
              <div>
                <div style={{ fontWeight: 'bold', color: '#666', marginBottom: '5px' }}>Cost Per Epoch</div>
                <div style={{ fontSize: '24px', color: '#333', fontWeight: 'bold' }}>
                  ${costAnalysis.cost_per_epoch.avg_cost_per_epoch.toFixed(4)}
                </div>
              </div>

              <div>
                <div style={{ fontWeight: 'bold', color: '#666', marginBottom: '5px' }}>Total Epochs</div>
                <div style={{ fontSize: '24px', color: '#333', fontWeight: 'bold' }}>
                  {costAnalysis.cost_per_epoch.total_epochs}
                </div>
              </div>

              <div>
                <div style={{ fontWeight: 'bold', color: '#666', marginBottom: '5px' }}>Time Per Epoch</div>
                <div style={{ fontSize: '24px', color: '#333', fontWeight: 'bold' }}>
                  {(costAnalysis.training_hours * 60 / costAnalysis.cost_per_epoch.total_epochs).toFixed(2)} min
                </div>
              </div>
            </div>
          </div>

          {/* System Resources */}
          <div
            style={{
              background: 'white',
              borderRadius: '12px',
              padding: '24px',
              boxShadow: '0 2px 8px rgba(0,0,0,0.1)',
            }}
          >
            <h3 style={{ marginTop: 0, color: '#667eea' }}>System Resources</h3>

            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', gap: '20px' }}>
              <div>
                <div style={{ fontWeight: 'bold', color: '#666', marginBottom: '5px' }}>CPUs Used</div>
                <div style={{ fontSize: '24px', color: '#333' }}>{costAnalysis.cpus_used.toFixed(1)}</div>
              </div>

              <div>
                <div style={{ fontWeight: 'bold', color: '#666', marginBottom: '5px' }}>GPU Used</div>
                <div style={{ fontSize: '24px', color: '#333' }}>{costAnalysis.gpu_used ? 'Yes' : 'No'}</div>
              </div>

              <div>
                <div style={{ fontWeight: 'bold', color: '#666', marginBottom: '5px' }}>Peak Memory</div>
                <div style={{ fontSize: '24px', color: '#333' }}>{costAnalysis.peak_memory_gb.toFixed(2)} GB</div>
              </div>

              <div>
                <div style={{ fontWeight: 'bold', color: '#666', marginBottom: '5px' }}>Training Time</div>
                <div style={{ fontSize: '24px', color: '#333' }}>{costAnalysis.training_hours.toFixed(2)} hrs</div>
              </div>
            </div>
          </div>

          {/* Note */}
          <div
            style={{
              marginTop: '30px',
              padding: '16px',
              background: '#fff9e6',
              border: '1px solid #ffe066',
              borderRadius: '8px',
              fontSize: '14px',
              color: '#666',
            }}
          >
            <strong>Note:</strong> Cost estimates are based on cloud computing pricing for similar GPU/CPU
            configurations. Actual costs may vary depending on your infrastructure.
          </div>
        </>
      )}
    </div>
  );
}

export default TrainingCostAnalysis;
