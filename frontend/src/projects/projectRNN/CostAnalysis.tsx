import React, { useState, useEffect } from 'react';

interface CostReport {
  pricing_config: Record<string, any>;
}

function CostAnalysis() {
  const [costReport, setCostReport] = useState<CostReport | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');

  const apiMode = typeof window !== 'undefined' ? localStorage.getItem('API_MODE') : null;
  const API_BASE_URL = apiMode === 'local' ? 'http://localhost:8000' : 'https://cst-435-project-hub.onrender.com';

  useEffect(() => {
    fetchCostAnalysis();
  }, []);

  const fetchCostAnalysis = async () => {
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
  };

  if (loading) {
    return <div style={{ textAlign: 'center', padding: '40px' }}>Loading cost analysis...</div>;
  }

  if (error) {
    return (
      <div style={{ padding: '20px', color: 'red', backgroundColor: '#ffe6e6', borderRadius: '8px' }}>
        ❌ {error}
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
          💰
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
    </div>
  );
}

export default CostAnalysis;
