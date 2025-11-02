import React, { useState, useEffect } from 'react';

interface CostSummary {
  total_training_cost: string;
  compute_cost: string;
  memory_cost: string;
  storage_cost: string;
  training_hours: string;
  cost_per_epoch: string;
  cost_per_hour: string;
}

interface TrainingCostReport {
  model_specs: Record<string, any>;
  training_specs: Record<string, any>;
  storage_specs: Record<string, any>;
  training_cost: Record<string, number>;
  cost_per_epoch: Record<string, number>;
  parameter_scenarios: Record<string, Record<string, number>>;
}

interface ActualCost {
  has_training_data: boolean;
  message?: string;
  actual_training_cost?: string;
  actual_training_hours?: string;
  peak_memory_gb?: string;
}

interface LyricVariation {
  variation: number;
  generated_text: string;
  word_count: number;
}

interface GenerateLyricsResponse {
  success: boolean;
  seed_text?: string;
  generated_text?: string;
  variations?: LyricVariation[];
  error?: string;
  parameters?: {
    max_length: number;
    temperature: number;
    top_k: number;
    num_variations: number;
  };
  model_info?: {
    vocab_size: number;
    embedding_dim: number;
    hidden_size: number;
    num_layers: number;
  };
}

function RNN({ activeTab: initialTab }: { activeTab?: string }) {
  const [activeTab, setActiveTab] = useState(initialTab || 'generator');
  const [costSummary, setCostSummary] = useState<CostSummary | null>(null);
  const [costReport, setCostReport] = useState<TrainingCostReport | null>(null);
  const [actualCost, setActualCost] = useState<ActualCost | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Lyric generation state
  const [seedText, setSeedText] = useState('');
  const [maxLength, setMaxLength] = useState(50);
  const [temperature, setTemperature] = useState(1.0);
  const [topK, setTopK] = useState(50);
  const [numVariations, setNumVariations] = useState(1);
  const [generatedLyrics, setGeneratedLyrics] = useState<GenerateLyricsResponse | null>(null);
  const [generationLoading, setGenerationLoading] = useState(false);
  const [generationError, setGenerationError] = useState<string | null>(null);

  // Fetch cost analysis on component mount
  useEffect(() => {
    if (activeTab === 'training-cost') {
      fetchCostAnalysis();
    }
  }, [activeTab]);

  const fetchCostAnalysis = async () => {
    setLoading(true);
    setError(null);
    try {
      const apiMode = localStorage.getItem('API_MODE');
      const apiUrl = apiMode === 'local' ? 'http://localhost:8000' : 'https://cst-435-project-hub.onrender.com';
      
      const summaryResponse = await fetch(`${apiUrl}/project5/cost-analysis/summary`);
      const reportResponse = await fetch(`${apiUrl}/project5/cost-analysis/report`);
      const actualCostResponse = await fetch(`${apiUrl}/project5/cost-analysis/actual`);

      if (!summaryResponse.ok || !reportResponse.ok) {
        throw new Error('Failed to fetch cost analysis data');
      }

      const summary = await summaryResponse.json();
      const report = await reportResponse.json();
      const actual = await actualCostResponse.json();

      setCostSummary(summary);
      setCostReport(report);
      setActualCost(actual);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'An error occurred');
      console.error('Error fetching cost analysis:', err);
    } finally {
      setLoading(false);
    }
  };

  const generateLyrics = async () => {
    setGenerationLoading(true);
    setGenerationError(null);
    try {
      if (!seedText.trim()) {
        setGenerationError('Please enter seed text to generate lyrics');
        setGenerationLoading(false);
        return;
      }

      const apiMode = localStorage.getItem('API_MODE');
      const apiUrl = apiMode === 'local' ? 'http://localhost:8000' : 'https://cst-435-project-hub.onrender.com';
      
      const response = await fetch(`${apiUrl}/project5/generate-lyrics`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({
          seed_text: seedText,
          max_length: maxLength,
          temperature: temperature,
          top_k: topK,
          num_variations: numVariations
        })
      });

      if (!response.ok) {
        throw new Error(`Server error: ${response.status}`);
      }

      const data = await response.json();
      setGeneratedLyrics(data);

      if (!data.success) {
        setGenerationError(data.error || 'Failed to generate lyrics');
      }
    } catch (err) {
      setGenerationError(err instanceof Error ? err.message : 'An error occurred');
      console.error('Error generating lyrics:', err);
    } finally {
      setGenerationLoading(false);
    }
  };

  return (
  <div style={{ maxWidth: '900px', margin: '0 auto' }}>
      {/* Tab Navigation */}
      <div style={{
        display: 'flex',
        gap: '10px',
        marginBottom: '20px',
        borderBottom: '2px solid #e5e7eb',
        flexWrap: 'wrap'
      }}>
        <button
          onClick={() => setActiveTab('song-generator')}
          style={{
            padding: '12px 24px',
            backgroundColor: activeTab === 'song-generator' ? '#667eea' : 'transparent',
            color: activeTab === 'song-generator' ? 'white' : '#666',
            border: 'none',
            borderRadius: '8px 8px 0 0',
            cursor: 'pointer',
            fontWeight: activeTab === 'song-generator' ? 'bold' : 'normal',
            fontSize: '16px',
            transition: 'all 0.3s ease'
          }}
        >
          Song Generator
        </button>
        <button
          onClick={() => setActiveTab('training-cost')}
          style={{
            padding: '12px 24px',
            backgroundColor: activeTab === 'training-cost' ? '#667eea' : 'transparent',
            color: activeTab === 'training-cost' ? 'white' : '#666',
            border: 'none',
            borderRadius: '8px 8px 0 0',
            cursor: 'pointer',
            fontWeight: activeTab === 'training-cost' ? 'bold' : 'normal',
            fontSize: '16px',
            transition: 'all 0.3s ease'
          }}
        >
          Training Cost Analysis
        </button>
        <button
          onClick={() => setActiveTab('requirements')}
          style={{
            padding: '12px 24px',
            backgroundColor: activeTab === 'requirements' ? '#667eea' : 'transparent',
            color: activeTab === 'requirements' ? 'white' : '#666',
            border: 'none',
            borderRadius: '8px 8px 0 0',
            cursor: 'pointer',
            fontWeight: activeTab === 'requirements' ? 'bold' : 'normal',
            fontSize: '16px',
            transition: 'all 0.3s ease'
          }}
        >
          Description and Requirements
        </button>
      </div>

      {activeTab === 'generator' && (
        <div>
          <h2>Text Generator</h2>
          <p>This is the text generator tab content.</p>
        </div>
      )}
      {activeTab === 'song-generator' && (
        <div style={{ padding: '20px' }}>
          <h2 style={{ marginBottom: '30px', display: 'flex', alignItems: 'center', gap: '10px' }}>
            <span style={{
              background: 'linear-gradient(135deg, #667eea, #764ba2)',
              color: 'white',
              width: '40px',
              height: '40px',
              borderRadius: '50%',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              fontSize: '24px'
            }}>
              🎵
            </span>
            AI Song Lyric Generator
          </h2>

          {generationError && (
            <div style={{
              backgroundColor: '#fee',
              border: '2px solid #f99',
              borderRadius: '8px',
              padding: '15px',
              marginBottom: '20px',
              color: '#c33'
            }}>
              <strong>Error:</strong> {generationError}
            </div>
          )}

          {/* Input Section */}
          <div style={{
            backgroundColor: '#f8f9ff',
            border: '2px solid #667eea',
            borderRadius: '12px',
            padding: '25px',
            marginBottom: '20px'
          }}>
            <h3 style={{ margin: '0 0 20px 0', color: '#667eea' }}>Create Your Song</h3>
            
            {/* Seed Text Input */}
            <div style={{ marginBottom: '20px' }}>
              <label style={{ display: 'block', fontSize: '14px', color: '#666', marginBottom: '8px', fontWeight: 'bold' }}>
                Seed Text (starting lyrics)
              </label>
              <input
                type="text"
                value={seedText}
                onChange={(e) => setSeedText(e.target.value)}
                placeholder="e.g., 'love is', 'dancing in the', 'when the sun'"
                style={{
                  width: '100%',
                  padding: '12px',
                  fontSize: '14px',
                  border: '2px solid #e0e0e0',
                  borderRadius: '8px',
                  fontFamily: 'inherit',
                  boxSizing: 'border-box'
                }}
              />
              <div style={{ fontSize: '12px', color: '#999', marginTop: '5px' }}>
                {seedText.length}/500 characters • {seedText.split(/\s+/).filter(w => w).length} words
              </div>
            </div>

            {/* Controls Grid */}
            <div style={{
              display: 'grid',
              gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))',
              gap: '20px',
              marginBottom: '20px'
            }}>
              {/* Max Length */}
              <div>
                <label style={{ display: 'block', fontSize: '14px', color: '#666', marginBottom: '8px', fontWeight: 'bold' }}>
                  Max Words to Generate
                </label>
                <input
                  type="number"
                  min="1"
                  max="200"
                  value={maxLength}
                  onChange={(e) => setMaxLength(Math.max(1, Math.min(200, parseInt(e.target.value) || 50)))}
                  style={{
                    width: '100%',
                    padding: '12px',
                    fontSize: '14px',
                    border: '2px solid #e0e0e0',
                    borderRadius: '8px',
                    boxSizing: 'border-box'
                  }}
                />
                <div style={{ fontSize: '12px', color: '#999', marginTop: '5px' }}>Current: {maxLength}</div>
              </div>

              {/* Temperature */}
              <div>
                <label style={{ display: 'block', fontSize: '14px', color: '#666', marginBottom: '8px', fontWeight: 'bold' }}>
                  Temperature (Creativity)
                </label>
                <input
                  type="number"
                  min="0.1"
                  max="3.0"
                  step="0.1"
                  value={temperature}
                  onChange={(e) => setTemperature(Math.max(0.1, Math.min(3.0, parseFloat(e.target.value) || 1.0)))}
                  style={{
                    width: '100%',
                    padding: '12px',
                    fontSize: '14px',
                    border: '2px solid #e0e0e0',
                    borderRadius: '8px',
                    boxSizing: 'border-box'
                  }}
                />
                <div style={{ fontSize: '12px', color: '#999', marginTop: '5px' }}>
                  {temperature < 0.5 ? 'Deterministic' : temperature > 1.5 ? 'Very Random' : 'Balanced'}
                </div>
              </div>

              {/* Top-K */}
              <div>
                <label style={{ display: 'block', fontSize: '14px', color: '#666', marginBottom: '8px', fontWeight: 'bold' }}>
                  Top-K (Word Diversity)
                </label>
                <input
                  type="number"
                  min="1"
                  max="1000"
                  value={topK}
                  onChange={(e) => setTopK(Math.max(1, Math.min(1000, parseInt(e.target.value) || 50)))}
                  style={{
                    width: '100%',
                    padding: '12px',
                    fontSize: '14px',
                    border: '2px solid #e0e0e0',
                    borderRadius: '8px',
                    boxSizing: 'border-box'
                  }}
                />
                <div style={{ fontSize: '12px', color: '#999', marginTop: '5px' }}>Current: {topK}</div>
              </div>

              {/* Variations */}
              <div>
                <label style={{ display: 'block', fontSize: '14px', color: '#666', marginBottom: '8px', fontWeight: 'bold' }}>
                  Generate Variations
                </label>
                <input
                  type="number"
                  min="1"
                  max="5"
                  value={numVariations}
                  onChange={(e) => setNumVariations(Math.max(1, Math.min(5, parseInt(e.target.value) || 1)))}
                  style={{
                    width: '100%',
                    padding: '12px',
                    fontSize: '14px',
                    border: '2px solid #e0e0e0',
                    borderRadius: '8px',
                    boxSizing: 'border-box'
                  }}
                />
                <div style={{ fontSize: '12px', color: '#999', marginTop: '5px' }}>1-5 variations</div>
              </div>
            </div>

            {/* Generate Button */}
            <button
              onClick={generateLyrics}
              disabled={generationLoading || !seedText.trim()}
              style={{
                width: '100%',
                padding: '14px 24px',
                fontSize: '16px',
                fontWeight: 'bold',
                backgroundColor: generationLoading || !seedText.trim() ? '#ccc' : '#667eea',
                color: 'white',
                border: 'none',
                borderRadius: '8px',
                cursor: generationLoading || !seedText.trim() ? 'not-allowed' : 'pointer',
                transition: 'all 0.3s ease'
              }}
            >
              {generationLoading ? 'Generating...' : 'Generate Song Lyrics'}
            </button>
          </div>

          {/* Results Section */}
          {generatedLyrics && generatedLyrics.success && (
            <div style={{
              backgroundColor: '#f0fff4',
              border: '2px solid #48bb78',
              borderRadius: '12px',
              padding: '25px'
            }}>
              <h3 style={{ margin: '0 0 20px 0', color: '#48bb78', display: 'flex', alignItems: 'center', gap: '10px' }}>
                ✅ Generated Lyrics
              </h3>

              {/* Primary Generated Text */}
              {generatedLyrics.generated_text && (
                <div style={{
                  backgroundColor: 'white',
                  border: '2px solid #e0e0e0',
                  borderRadius: '8px',
                  padding: '20px',
                  marginBottom: '20px',
                  lineHeight: '1.8',
                  fontSize: '16px',
                  fontStyle: 'italic',
                  color: '#333'
                }}>
                  <strong style={{ color: '#48bb78' }}>Seed:</strong> {generatedLyrics.seed_text}
                  <div style={{ marginTop: '15px', color: '#666' }}>
                    {generatedLyrics.generated_text}
                  </div>
                </div>
              )}

              {/* Variations */}
              {generatedLyrics.variations && generatedLyrics.variations.length > 1 && (
                <div style={{ marginBottom: '20px' }}>
                  <h4 style={{ margin: '0 0 15px 0', color: '#48bb78' }}>📚 Other Variations:</h4>
                  <div style={{
                    display: 'grid',
                    gridTemplateColumns: 'repeat(auto-fit, minmax(250px, 1fr))',
                    gap: '15px'
                  }}>
                    {generatedLyrics.variations.map((variation, idx) => (
                      idx > 0 && (
                        <div key={idx} style={{
                          backgroundColor: 'white',
                          border: '1px solid #e0e0e0',
                          borderRadius: '8px',
                          padding: '15px',
                          fontSize: '14px',
                          lineHeight: '1.6'
                        }}>
                          <div style={{ fontSize: '12px', color: '#999', marginBottom: '8px' }}>
                            Variation {variation.variation} • {variation.word_count} words
                          </div>
                          <div style={{ color: '#666', fontStyle: 'italic' }}>
                            {variation.generated_text}
                          </div>
                        </div>
                      )
                    ))}
                  </div>
                </div>
              )}

              {/* Model Info */}
              {generatedLyrics.model_info && (
                <div style={{
                  backgroundColor: 'rgba(72, 187, 120, 0.1)',
                  border: '1px solid #48bb78',
                  borderRadius: '8px',
                  padding: '15px',
                  fontSize: '13px',
                  color: '#666'
                }}>
                  <strong style={{ color: '#48bb78' }}>Model Configuration:</strong>
                  <div style={{ marginTop: '8px' }}>
                    Vocab Size: {generatedLyrics.model_info.vocab_size?.toLocaleString()} • 
                    Embedding: {generatedLyrics.model_info.embedding_dim}D • 
                    Hidden: {generatedLyrics.model_info.hidden_size} • 
                    Layers: {generatedLyrics.model_info.num_layers}
                  </div>
                </div>
              )}
            </div>
          )}
        </div>
      )}
      {activeTab === 'training-cost' && (
        <div style={{ padding: '20px' }}>
          <h2 style={{ marginBottom: '30px', display: 'flex', alignItems: 'center', gap: '10px' }}>
            <span style={{
              background: 'linear-gradient(135deg, #667eea, #764ba2)',
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
            Training Cost Analysis
          </h2>

          {loading && (
            <div style={{ textAlign: 'center', padding: '40px' }}>
              <p style={{ fontSize: '18px', color: '#667eea' }}>Loading cost analysis...</p>
            </div>
          )}

          {error && (
            <div style={{
              backgroundColor: '#fee',
              border: '2px solid #f99',
              borderRadius: '8px',
              padding: '15px',
              marginBottom: '20px',
              color: '#c33'
            }}>
              <strong>Error:</strong> {error}
            </div>
          )}

          {costSummary && !loading && (
            <>
              {/* Cost Summary Cards */}
              <div style={{
                display: 'grid',
                gridTemplateColumns: 'repeat(auto-fit, minmax(250px, 1fr))',
                gap: '15px',
                marginBottom: '30px'
              }}>
                {/* Total Training Cost */}
                <div style={{
                  backgroundColor: '#f8f9ff',
                  border: '2px solid #667eea',
                  borderRadius: '12px',
                  padding: '20px',
                  textAlign: 'center'
                }}>
                  <div style={{ fontSize: '14px', color: '#666', marginBottom: '8px' }}>Total Training Cost</div>
                  <div style={{ fontSize: '28px', fontWeight: 'bold', color: '#667eea' }}>
                    {costSummary.total_training_cost}
                  </div>
                  <div style={{ fontSize: '12px', color: '#999', marginTop: '8px' }}>
                    {costSummary.training_hours}
                  </div>
                </div>

                {/* Cost Per Epoch */}
                <div style={{
                  backgroundColor: '#f8f9ff',
                  border: '2px solid #667eea',
                  borderRadius: '12px',
                  padding: '20px',
                  textAlign: 'center'
                }}>
                  <div style={{ fontSize: '14px', color: '#666', marginBottom: '8px' }}>Cost Per Epoch</div>
                  <div style={{ fontSize: '28px', fontWeight: 'bold', color: '#667eea' }}>
                    {costSummary.cost_per_epoch}
                  </div>
                  <div style={{ fontSize: '12px', color: '#999', marginTop: '8px' }}>
                    For 25 epochs
                  </div>
                </div>

                {/* Cost Per Hour */}
                <div style={{
                  backgroundColor: '#f8f9ff',
                  border: '2px solid #667eea',
                  borderRadius: '12px',
                  padding: '20px',
                  textAlign: 'center'
                }}>
                  <div style={{ fontSize: '14px', color: '#666', marginBottom: '8px' }}>Cost Per Hour</div>
                  <div style={{ fontSize: '28px', fontWeight: 'bold', color: '#667eea' }}>
                    {costSummary.cost_per_hour}
                  </div>
                  <div style={{ fontSize: '12px', color: '#999', marginTop: '8px' }}>
                    Training rate
                  </div>
                </div>
              </div>

              {/* Cost Breakdown */}
              <div style={{
                backgroundColor: '#f8f9ff',
                border: '2px solid #667eea',
                borderRadius: '12px',
                padding: '20px',
                marginBottom: '20px'
              }}>
                <h3 style={{ margin: '0 0 15px 0', color: '#667eea' }}>Cost Breakdown</h3>
                <div style={{
                  display: 'grid',
                  gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))',
                  gap: '15px'
                }}>
                  <div>
                    <div style={{ fontSize: '13px', color: '#666' }}>Compute Cost</div>
                    <div style={{ fontSize: '20px', fontWeight: 'bold', color: '#667eea', marginTop: '5px' }}>
                      {costSummary.compute_cost}
                    </div>
                  </div>
                  <div>
                    <div style={{ fontSize: '13px', color: '#666' }}>Memory Cost</div>
                    <div style={{ fontSize: '20px', fontWeight: 'bold', color: '#667eea', marginTop: '5px' }}>
                      {costSummary.memory_cost}
                    </div>
                  </div>
                  <div>
                    <div style={{ fontSize: '13px', color: '#666' }}>Storage Cost</div>
                    <div style={{ fontSize: '20px', fontWeight: 'bold', color: '#667eea', marginTop: '5px' }}>
                      {costSummary.storage_cost}
                    </div>
                  </div>
                </div>
              </div>

              {/* Actual vs Estimated Cost */}
              {actualCost && (
                <div style={{
                  backgroundColor: actualCost.has_training_data ? '#f0fff4' : '#fffaf0',
                  border: `2px solid ${actualCost.has_training_data ? '#48bb78' : '#ed8936'}`,
                  borderRadius: '12px',
                  padding: '20px',
                  marginBottom: '20px'
                }}>
                  <h3 style={{ 
                    margin: '0 0 15px 0', 
                    color: actualCost.has_training_data ? '#48bb78' : '#ed8936'
                  }}>
                    {actualCost.has_training_data ? 'Actual Training Metrics' : 'Estimated Training Metrics'}
                  </h3>
                  {actualCost.has_training_data ? (
                    <div style={{
                      display: 'grid',
                      gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))',
                      gap: '15px'
                    }}>
                      <div>
                        <div style={{ fontSize: '13px', color: '#666' }}>Actual Cost</div>
                        <div style={{ fontSize: '20px', fontWeight: 'bold', color: '#48bb78', marginTop: '5px' }}>
                          {actualCost.actual_training_cost}
                        </div>
                      </div>
                      <div>
                        <div style={{ fontSize: '13px', color: '#666' }}>Actual Time</div>
                        <div style={{ fontSize: '20px', fontWeight: 'bold', color: '#48bb78', marginTop: '5px' }}>
                          {actualCost.actual_training_hours}
                        </div>
                      </div>
                      <div>
                        <div style={{ fontSize: '13px', color: '#666' }}>Peak Memory</div>
                        <div style={{ fontSize: '20px', fontWeight: 'bold', color: '#48bb78', marginTop: '5px' }}>
                          {actualCost.peak_memory_gb}
                        </div>
                      </div>
                    </div>
                  ) : (
                    <div style={{ fontSize: '14px', color: '#666', fontStyle: 'italic' }}>
                      {actualCost.message}
                    </div>
                  )}
                </div>
              )}

              {/* Model Specifications */}
              {costReport && (
                <>
                  <div style={{
                    backgroundColor: '#f8f9ff',
                    border: '2px solid #667eea',
                    borderRadius: '12px',
                    padding: '20px',
                    marginBottom: '20px'
                  }}>
                    <h3 style={{ margin: '0 0 15px 0', color: '#667eea' }}>Model Specifications</h3>
                    <div style={{
                      display: 'grid',
                      gridTemplateColumns: 'repeat(auto-fit, minmax(180px, 1fr))',
                      gap: '12px'
                    }}>
                      <div>
                        <div style={{ fontSize: '12px', color: '#666' }}>Vocabulary Size</div>
                        <div style={{ fontSize: '18px', fontWeight: 'bold', marginTop: '4px' }}>
                          {costReport.model_specs.vocab_size.toLocaleString()}
                        </div>
                      </div>
                      <div>
                        <div style={{ fontSize: '12px', color: '#666' }}>Embedding Dimension</div>
                        <div style={{ fontSize: '18px', fontWeight: 'bold', marginTop: '4px' }}>
                          {costReport.model_specs.embedding_dim}
                        </div>
                      </div>
                      <div>
                        <div style={{ fontSize: '12px', color: '#666' }}>Hidden Size</div>
                        <div style={{ fontSize: '18px', fontWeight: 'bold', marginTop: '4px' }}>
                          {costReport.model_specs.hidden_size}
                        </div>
                      </div>
                      <div>
                        <div style={{ fontSize: '12px', color: '#666' }}>Layers</div>
                        <div style={{ fontSize: '18px', fontWeight: 'bold', marginTop: '4px' }}>
                          {costReport.model_specs.num_layers}
                        </div>
                      </div>
                      <div>
                        <div style={{ fontSize: '12px', color: '#666' }}>Dataset Size</div>
                        <div style={{ fontSize: '18px', fontWeight: 'bold', marginTop: '4px' }}>
                          {(costReport.model_specs.dataset_size_gb * 1024).toFixed(1)} MB
                        </div>
                      </div>
                      <div>
                        <div style={{ fontSize: '12px', color: '#666' }}>Model Size</div>
                        <div style={{ fontSize: '18px', fontWeight: 'bold', marginTop: '4px' }}>
                          {costReport.model_specs.model_size_mb.toFixed(1)} MB
                        </div>
                      </div>
                    </div>
                  </div>

                  {/* Training Configuration */}
                  <div style={{
                    backgroundColor: '#f8f9ff',
                    border: '2px solid #667eea',
                    borderRadius: '12px',
                    padding: '20px'
                  }}>
                    <h3 style={{ margin: '0 0 15px 0', color: '#667eea' }}>Training Configuration</h3>
                    <div style={{
                      display: 'grid',
                      gridTemplateColumns: 'repeat(auto-fit, minmax(180px, 1fr))',
                      gap: '12px'
                    }}>
                      <div>
                        <div style={{ fontSize: '12px', color: '#666' }}>Batch Size</div>
                        <div style={{ fontSize: '18px', fontWeight: 'bold', marginTop: '4px' }}>
                          {costReport.training_specs.batch_size}
                        </div>
                      </div>
                      <div>
                        <div style={{ fontSize: '12px', color: '#666' }}>Epochs</div>
                        <div style={{ fontSize: '18px', fontWeight: 'bold', marginTop: '4px' }}>
                          {costReport.training_specs.epochs}
                        </div>
                      </div>
                      <div>
                        <div style={{ fontSize: '12px', color: '#666' }}>Learning Rate</div>
                        <div style={{ fontSize: '18px', fontWeight: 'bold', marginTop: '4px' }}>
                          {costReport.training_specs.learning_rate}
                        </div>
                      </div>
                      <div>
                        <div style={{ fontSize: '12px', color: '#666' }}>Est. Training Time</div>
                        <div style={{ fontSize: '18px', fontWeight: 'bold', marginTop: '4px' }}>
                          {costReport.training_specs.estimated_training_hours.toFixed(1)}h
                        </div>
                      </div>
                    </div>
                  </div>
                </>
              )}
            </>
          )}
        </div>
      )}
      {activeTab === 'requirements' && (
        <div style={{ padding: '20px' }}>
          <h2 style={{ marginBottom: '30px', display: 'flex', alignItems: 'center', gap: '10px' }}>
            <span style={{
              background: 'linear-gradient(135deg, #667eea, #764ba2)',
              color: 'white',
              width: '40px',
              height: '40px',
              borderRadius: '50%',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              fontSize: '20px'
            }}>
              📋
            </span>
            Project Requirements
          </h2>

          {/* Requirement 1 */}
          <div style={{
            backgroundColor: '#f8f9ff',
            border: '2px solid #667eea',
            borderRadius: '12px',
            padding: '20px',
            marginBottom: '20px'
          }}>
            <h3 style={{ margin: '0 0 15px 0', color: '#667eea' }}>1. Prepare the Data Set</h3>
            <div style={{ marginLeft: '20px', lineHeight: '1.8', color: '#333' }}>
              <div>Remove punctuation.</div>
              <div>Split strings into lists of individual words.</div>
              <div>Convert the individual words into integers (using a tokenizer).</div>
            </div>
          </div>

          {/* Requirement 2 */}
          <div style={{
            backgroundColor: '#f8f9ff',
            border: '2px solid #667eea',
            borderRadius: '12px',
            padding: '20px',
            marginBottom: '20px'
          }}>
            <h3 style={{ margin: '0 0 15px 0', color: '#667eea' }}>2. Create Feature and Labels from Sequences</h3>
            <div style={{ marginLeft: '20px', lineHeight: '1.8', color: '#333' }}>
              <div>Set the number of words as a parameter.</div>
              <div>Select a subset of the data to be used as a training set.</div>
              <div>Decide on a training method:
                <div style={{ marginLeft: '20px', marginTop: '8px' }}>
                  <li>Use words 1 through n as features and the n+1 word as the label.</li>
                  <li>Use words 2 through n+1 as features and the n+2 word as the label.</li>
                  <li>Use words 3 through n+2 as features and the n+3 word as the label.</li>
                </div>
              </div>
            </div>
          </div>

          {/* Requirement 3 */}
          <div style={{
            backgroundColor: '#f8f9ff',
            border: '2px solid #667eea',
            borderRadius: '12px',
            padding: '20px',
            marginBottom: '20px'
          }}>
            <h3 style={{ margin: '0 0 15px 0', color: '#667eea' }}>3. Build an LSTM Model with Embedding and Dense Layers</h3>
            <p style={{ marginTop: 0, marginBottom: '12px', fontStyle: 'italic', color: '#666', fontSize: '0.95em' }}>Avoid using Tensorflow:</p>
            <ul style={{ marginLeft: '20px', lineHeight: '1.8', margin: 0, color: '#333' }}>
              <div><strong>Embedding Layer:</strong> Maps each input word to a 100-dimensional vector with optional pretrained weights.</div>
              <div><strong>Masking Layer:</strong> Masks words without pretrained embeddings (represented as zeros).</div>
              <div><strong>LSTM Layer:</strong> LSTM cells with dropout to prevent overfitting. Returns sequences only if using 2+ layers.</div>
              <div><strong>Dense Layer:</strong> Fully connected layer with ReLU activation for additional representational capacity.</div>
              <div><strong>Dropout Layer:</strong> Prevents overfitting to the training data.</div>
              <div><strong>Output Dense Layer:</strong> Produces probability for every word in vocabulary using softmax.</div>
            </ul>
          </div>

          {/* Requirement 4 */}
          <div style={{
            backgroundColor: '#f8f9ff',
            border: '2px solid #667eea',
            borderRadius: '12px',
            padding: '20px',
            marginBottom: '20px'
          }}>
            <h3 style={{ margin: '0 0 15px 0', color: '#667eea' }}>4. Compile the Model with the Adam Optimizer</h3>
            <div style={{ marginLeft: '20px', lineHeight: '1.8', color: '#333' }}>
              <div>Load pretrained embeddings from the GloVe algorithm (trained on Wikipedia texts).</div>
              <div>Assign 100-dimensional vectors to each word in the vocabulary.</div>
              <div>Explore embeddings using cosine similarity between vectors.</div>
            </div>
          </div>

          {/* Requirement 5 */}
          <div style={{
            backgroundColor: '#f8f9ff',
            border: '2px solid #667eea',
            borderRadius: '12px',
            padding: '20px',
            marginBottom: '20px'
          }}>
            <h3 style={{ margin: '0 0 15px 0', color: '#667eea' }}>5. Train the Model to Predict the Next Word in the Sequence</h3>
            <ul style={{ marginLeft: '20px', lineHeight: '1.8', margin: 0, color: '#333' }}>
              <div><strong>Model Checkpoint:</strong> Saves the best model (by validation loss) on disk for later use.</div>
              <div><strong>Early Stopping:</strong> Halts training when validation loss stops decreasing.</div>
            </ul>
          </div>
        </div>
      )}
    </div>
  );
}

export default RNN;
