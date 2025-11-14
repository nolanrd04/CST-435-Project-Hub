import React, { useState, useEffect, useCallback } from 'react';
import { AiOutlineDollar, AiOutlineWarning } from 'react-icons/ai';
import { getApiUrl } from '../getApiUrl.ts';

interface CostReport {
  pricing_config: Record<string, any>;
}

interface GenerateTextResponse {
  generated_text: string;
  seed_text: string;
  num_words: number;
  temperature: number;
  query_cost: number;
}

function CostAnalysis() {
  const [costReport, setCostReport] = useState<CostReport | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');

  const [seedText, setSeedText] = useState('');
  const [numWords, setNumWords] = useState(50);
  const [temperature, setTemperature] = useState(1.0);
  const [response, setResponse] = useState<GenerateTextResponse | null>(null);
  const [textLoading, setTextLoading] = useState(false);
  const [textError, setTextError] = useState('');

  const API_BASE_URL = getApiUrl();

  const fetchCostAnalysis = useCallback(async () => {
    setLoading(true);
    setError('');
    try {
      const response = await fetch(`${API_BASE_URL}/cost-analysis/report`);
      if (!response.ok) throw new Error('Failed to fetch cost analysis');
      const data: CostReport = await response.json();
      setCostReport(data);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'An error occurred');
    } finally {
      setLoading(false);
    }
  }, [API_BASE_URL]);

  useEffect(() => {
    fetchCostAnalysis();
  }, [fetchCostAnalysis]);

  const handleGenerateText = async () => {
    setTextLoading(true);
    setTextError('');
    try {
      const response = await fetch(`${API_BASE_URL}/generate-text`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ seed_text: seedText, num_words: numWords, temperature }),
      });

      if (!response.ok) throw new Error('Failed to generate text');

      const data: GenerateTextResponse = await response.json();
      setResponse(data);
    } catch (err) {
      setTextError(err instanceof Error ? err.message : 'An error occurred');
    } finally {
      setTextLoading(false);
    }
  };

  if (loading) {
    return <div style={{ textAlign: 'center', padding: '40px' }}>Loading cost analysis...</div>;
  }

  if (error) {
    return (
      <div style={{ padding: '20px', color: 'red', backgroundColor: '#ffe6e6', borderRadius: '8px', display: 'flex', alignItems: 'center', gap: '8px' }}>
        <AiOutlineWarning size={16} />
        {error}
      </div>
    );
  }

  if (!costReport) {
    return <div>No cost data available</div>;
  }

  return (
    <div style={{ maxWidth: '900px', margin: '0 auto', padding: '20px' }}>
      <h2 style={{ marginBottom: '30px', display: 'flex', alignItems: 'center', gap: '10px' }}>
        <span style={{
          background: 'linear-gradient(135deg, #f093fb, #f5576c)',
          color: 'white',
          width: '40px',
          height: '40px',
          borderRadius: '50%',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          fontSize: '20px'
        }}>
          <AiOutlineDollar size={20} />
        </span>
        RNN Deployment Cost Analysis
      </h2>

      {/* Cost Values Display */}
      <div style={{
        backgroundColor: '#f0f9ff',
        border: '2px solid #0ea5e9',
        borderRadius: '12px',
        padding: '30px',
        marginBottom: '30px'
      }}>
        <h3 style={{ margin: '0 0 25px 0', color: '#0ea5e9' }}>Render Pricing Configuration</h3>

        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(280px, 1fr))', gap: '20px' }}>
          <div style={{
            backgroundColor: 'white',
            padding: '20px',
            borderRadius: '8px',
            border: '1px solid #e0e7ff'
          }}>
            <p style={{ margin: '0 0 10px 0', color: '#666', fontSize: '0.9em', fontWeight: '600' }}>
              Cost per CPU per month
            </p>
            <p style={{ margin: 0, fontSize: '1.8em', fontWeight: 'bold', color: '#0ea5e9' }}>
              ${costReport.pricing_config.cost_per_cpu_per_month?.toFixed(4) || '0.2969'}
            </p>
          </div>

          <div style={{
            backgroundColor: 'white',
            padding: '20px',
            borderRadius: '8px',
            border: '1px solid #e0e7ff'
          }}>
            <p style={{ margin: '0 0 10px 0', color: '#666', fontSize: '0.9em', fontWeight: '600' }}>
              Cost per GB RAM per month
            </p>
            <p style={{ margin: 0, fontSize: '1.8em', fontWeight: 'bold', color: '#0ea5e9' }}>
              ${costReport.pricing_config.cost_per_gb_ram_per_month?.toFixed(4) || '0.0371'}
            </p>
          </div>

          <div style={{
            backgroundColor: 'white',
            padding: '20px',
            borderRadius: '8px',
            border: '1px solid #e0e7ff'
          }}>
            <p style={{ margin: '0 0 10px 0', color: '#666', fontSize: '0.9em', fontWeight: '600' }}>
              Overage Bandwidth Cost
            </p>
            <p style={{ margin: 0, fontSize: '1.8em', fontWeight: 'bold', color: '#0ea5e9' }}>
              ${costReport.pricing_config.overage_bandwidth_cost_per_gb?.toFixed(2) || '0.10'}/GB
            </p>
          </div>

          <div style={{
            backgroundColor: 'white',
            padding: '20px',
            borderRadius: '8px',
            border: '1px solid #e0e7ff'
          }}>
            <p style={{ margin: '0 0 10px 0', color: '#666', fontSize: '0.9em', fontWeight: '600' }}>
              Overage Build Minutes Cost
            </p>
            <p style={{ margin: 0, fontSize: '1.8em', fontWeight: 'bold', color: '#0ea5e9' }}>
              ${costReport.pricing_config.overage_build_minutes_cost?.toFixed(2) || '0.01'}/min
            </p>
          </div>

          <div style={{
            backgroundColor: 'white',
            padding: '20px',
            borderRadius: '8px',
            border: '1px solid #e0e7ff'
          }}>
            <p style={{ margin: '0 0 10px 0', color: '#666', fontSize: '0.9em', fontWeight: '600' }}>
              Additional Storage Cost
            </p>
            <p style={{ margin: 0, fontSize: '1.8em', fontWeight: 'bold', color: '#0ea5e9' }}>
              ${costReport.pricing_config.additional_storage_cost_per_gb?.toFixed(2) || '0.10'}/GB/mo
            </p>
          </div>
        </div>

        <p style={{ margin: '20px 0 0 0', color: '#666', fontSize: '0.9em' }}>
          To update these values, run:<br/>
          <code style={{ backgroundColor: '#e0e7ff', padding: '4px 8px', borderRadius: '4px', fontFamily: 'monospace' }}>
            python backend/app/routers/projectRNN/configure_pricing.py
          </code>
        </p>
      </div>

      {/* Local Training Cost Display */}
      <div style={{
        backgroundColor: '#f0fff5ff',
        border: '2px solid #0ee950ff',
        borderRadius: '12px',
        padding: '30px',
        marginBottom: '30px'
      }}>
        

        <h3 style={{ margin: '0 0 25px 0', color: '#0ee953ff' }}>Local Model Training Cost</h3>

        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(500px, 1fr))', gap: '20px' }}>
          <div style={{
            backgroundColor: 'white',
            padding: '20px',
            borderRadius: '8px',
            border: '1px solid #e0e7ff'
          }}>
            <p style={{ margin: '0 0 10px 0', color: '#666', fontSize: '0.9em', fontWeight: '600' }}>
              Training Cost In USD Per Hour
            </p>
            <p style={{ margin: 0, fontSize: '1.8em', fontWeight: 'bold', color: '#0dbe2dff' }}>
              ${costReport.pricing_config.local_training_cost?.toFixed(4) || '0'}
            </p>
          </div>

          <div style={{
            backgroundColor: 'white',
            padding: '20px',
            borderRadius: '8px',
            border: '1px solid #e0e7ff'
          }}>
            <p style={{ margin: '0 0 10px 0', color: '#666', fontSize: '0.9em', fontWeight: '600' }}>
              Training Cost Calculation:
            </p>
            <p style={{ margin: 0, fontSize: '1.8em', fontWeight: 'bold', color: '#0dbe2dff' }}>
              Cost = (compute cost/hour * training hours) + (storage cost * (model file size MB / 1024))
            </p>
          </div>
        </div>

        <p style={{ margin: '20px 0 0 0', color: '#666', fontSize: '0.9em' }}>
          To update these values, run:<br/>
          <code style={{ backgroundColor: '#e0ffeaff', padding: '4px 8px', borderRadius: '4px', fontFamily: 'monospace' }}>
            python backend/app/routers/projectRNN/configure_pricing.py
          </code>
        </p>
      </div>

      {/* Text Generation Cost Display */}
      <div style={{
        backgroundColor: '#fff5f0',
        border: '2px solid #ff7b29',
        borderRadius: '12px',
        padding: '30px',
        marginBottom: '30px'
      }}>
        <h3 style={{ margin: '0 0 25px 0', color: '#ff7b29' }}>Text Generation Cost Analysis</h3>

        <div style={{ marginBottom: '20px' }}>
          <label>
            Seed Text:
            <input
              type="text"
              value={seedText}
              onChange={(e) => setSeedText(e.target.value)}
              style={{ marginLeft: '10px', padding: '5px', width: '300px' }}
            />
          </label>
        </div>

        <div style={{ marginBottom: '20px' }}>
          <label>
            Number of Words:
            <input
              type="number"
              value={numWords}
              onChange={(e) => setNumWords(Number(e.target.value))}
              style={{ marginLeft: '10px', padding: '5px', width: '100px' }}
            />
          </label>
        </div>

        <div style={{ marginBottom: '20px' }}>
          <label>
            Temperature:
            <input
              type="number"
              value={temperature}
              onChange={(e) => setTemperature(Number(e.target.value))}
              style={{ marginLeft: '10px', padding: '5px', width: '100px' }}
            />
          </label>
        </div>

        <button onClick={handleGenerateText} style={{ padding: '10px 20px', backgroundColor: '#ff7b29', color: 'white', border: 'none', borderRadius: '5px' }}>
          Generate Text
        </button>

        {textLoading && <p>Loading...</p>}

        {textError && <p style={{ color: 'red' }}>{textError}</p>}

        {response && (
          <div style={{ marginTop: '20px', padding: '20px', border: '1px solid #ccc', borderRadius: '5px' }}>
            <h3>Generated Text</h3>
            <p>{response.generated_text}</p>

            <h3>Query Cost</h3>
            <p>${response.query_cost.toFixed(4)}</p>
          </div>
        )}
      </div>

    </div>
  );
}

export default CostAnalysis;
