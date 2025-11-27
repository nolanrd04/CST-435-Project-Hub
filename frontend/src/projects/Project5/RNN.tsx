import React, { useState, useEffect } from 'react';
import {
  AiOutlineAudio,
  AiOutlineDollar,
  AiOutlineFileText,
  AiOutlineYoutube,
  AiOutlineCheckCircle,
  AiOutlineBook
} from 'react-icons/ai';
import { getApiUrl } from '../getApiUrl.ts';

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
  const [activeTab, setActiveTab] = useState(initialTab || 'generator' || 'youtube' || 'song-generator' || 'training-cost' || 'requirements');
  const [costSummary, setCostSummary] = useState<CostSummary | null>(null);
  const [costReport, setCostReport] = useState<TrainingCostReport | null>(null);
  const [actualCost, setActualCost] = useState<ActualCost | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Model selection state
  const [selectedModel, setSelectedModel] = useState('lyrics_model');
  const AVAILABLE_MODELS = [
    { id: 'lyrics_model', name: 'Large Model (lyrics_model)', description: 'Full model - Better quality - CAN\'T RUN ON CLOUD!' },
    { id: 'lyrics_model_2', name: 'Small Model (lyrics_model_2)', description: 'Smaller model - Lower quality' }
  ];

  // Lyric generation state
  const [seedText, setSeedText] = useState('');
  const [maxLength, setMaxLength] = useState(50);
  const [temperature, setTemperature] = useState(1.0);
  const [topK, setTopK] = useState(50);
  const [numVariations, setNumVariations] = useState(1);
  const [generatedLyrics, setGeneratedLyrics] = useState<GenerateLyricsResponse | null>(null);
  const [generationLoading, setGenerationLoading] = useState(false);
  const [generationError, setGenerationError] = useState<string | null>(null);

  // Model info state
  const [modelInfo, setModelInfo] = useState<any | null>(null);
  const [modelInfoLoading, setModelInfoLoading] = useState(false);

  // Fetch cost analysis on component mount
  useEffect(() => {
    if (activeTab === 'training-cost') {
      fetchCostAnalysis();
    } else if (activeTab === 'requirements') {
      fetchModelInfo();
    }
  }, [activeTab]);

  const fetchCostAnalysis = async () => {
    setLoading(true);
    setError(null);
    try {
  // Resolve API base URL via user preference helper
  const apiUrl = getApiUrl();

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

  const fetchModelInfo = async () => {
    setModelInfoLoading(true);
    try {
  // Resolve API base URL via user preference helper
  const apiUrl = getApiUrl();

  const response = await fetch(`${apiUrl}/project5/lyric-generator-info`);
      
      if (!response.ok) {
        throw new Error('Failed to fetch model info');
      }

      const info = await response.json();
      setModelInfo(info);
    } catch (err) {
      console.error('Error fetching model info:', err);
    } finally {
      setModelInfoLoading(false);
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

  // Resolve API base URL via user preference helper
  const apiUrl = getApiUrl();

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
          num_variations: numVariations,
          model: selectedModel
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
    <div style={{ maxWidth: '1200px', margin: '0 auto', padding: '20px' }}>
      {/* Header */}
      <div style={{ marginBottom: '30px', textAlign: 'center' }}>
        <h1
          style={{
            fontSize: '36px',
            fontWeight: 'bold',
            background: 'linear-gradient(135deg, #667eea, #764ba2)',
            WebkitBackgroundClip: 'text',
            WebkitTextFillColor: 'transparent',
            marginBottom: '10px',
          }}
        >
          Project 5: RNN Song Lyric Generator
        </h1>
        <p style={{ color: '#666', fontSize: '16px' }}>
          LSTM-based text generation trained on song lyrics
        </p>
      </div>

      {/* Tab Navigation */}
      <div style={{
        display: 'flex',
        gap: '10px',
        marginBottom: '30px',
        borderBottom: '2px solid #e5e7eb',
        flexWrap: 'wrap',
        overflowX: 'auto',
      }}>
        <button
          onClick={() => setActiveTab('song-generator')}
          style={{
            padding: '12px 20px',
            backgroundColor: activeTab === 'song-generator' ? '#667eea' : 'transparent',
            color: activeTab === 'song-generator' ? 'white' : '#666',
            border: 'none',
            borderRadius: '8px 8px 0 0',
            cursor: 'pointer',
            fontWeight: activeTab === 'song-generator' ? 'bold' : 'normal',
            fontSize: '15px',
            transition: 'all 0.3s ease',
            display: 'flex',
            alignItems: 'center',
            gap: '8px',
            whiteSpace: 'nowrap',
          }}
        >
          <AiOutlineAudio size={20} />
          Song Generator
        </button>
        <button
          onClick={() => setActiveTab('training-cost')}
          style={{
            padding: '12px 20px',
            backgroundColor: activeTab === 'training-cost' ? '#667eea' : 'transparent',
            color: activeTab === 'training-cost' ? 'white' : '#666',
            border: 'none',
            borderRadius: '8px 8px 0 0',
            cursor: 'pointer',
            fontWeight: activeTab === 'training-cost' ? 'bold' : 'normal',
            fontSize: '15px',
            transition: 'all 0.3s ease',
            display: 'flex',
            alignItems: 'center',
            gap: '8px',
            whiteSpace: 'nowrap',
          }}
        >
          <AiOutlineDollar size={20} />
          Training Cost Analysis
        </button>
        <button
          onClick={() => setActiveTab('requirements')}
          style={{
            padding: '12px 20px',
            backgroundColor: activeTab === 'requirements' ? '#667eea' : 'transparent',
            color: activeTab === 'requirements' ? 'white' : '#666',
            border: 'none',
            borderRadius: '8px 8px 0 0',
            cursor: 'pointer',
            fontWeight: activeTab === 'requirements' ? 'bold' : 'normal',
            fontSize: '15px',
            transition: 'all 0.3s ease',
            display: 'flex',
            alignItems: 'center',
            gap: '8px',
            whiteSpace: 'nowrap',
          }}
        >
          <AiOutlineFileText size={20} />
          Description and Requirements
        </button>
        <button
          onClick={() => setActiveTab('youtube')}
          style={{
            padding: '12px 20px',
            backgroundColor: activeTab === 'youtube' ? '#667eea' : 'transparent',
            color: activeTab === 'youtube' ? 'white' : '#666',
            border: 'none',
            borderRadius: '8px 8px 0 0',
            cursor: 'pointer',
            fontWeight: activeTab === 'youtube' ? 'bold' : 'normal',
            fontSize: '15px',
            transition: 'all 0.3s ease',
            display: 'flex',
            alignItems: 'center',
            gap: '8px',
            whiteSpace: 'nowrap',
          }}
        >
          <AiOutlineYoutube size={20} />
          YouTube Showcase
        </button>
      </div>

      {/* Tab Content */}
      <div style={{ minHeight: '500px' }}>
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
              <AiOutlineAudio size={24} />
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
            
            {/* Model Selection */}
            <div style={{ marginBottom: '20px', padding: '15px', backgroundColor: 'white', borderRadius: '8px', border: '2px solid #e0e0e0' }}>
              <label style={{ display: 'block', fontSize: '14px', color: '#666', marginBottom: '12px', fontWeight: 'bold' }}>
                Select Model
              </label>
              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', gap: '12px' }}>
                {AVAILABLE_MODELS.map((model) => (
                  <button
                    key={model.id}
                    onClick={() => setSelectedModel(model.id)}
                    style={{
                      padding: '12px 16px',
                      border: selectedModel === model.id ? '3px solid #667eea' : '2px solid #e0e0e0',
                      backgroundColor: selectedModel === model.id ? '#f0f4ff' : 'white',
                      borderRadius: '8px',
                      cursor: 'pointer',
                      textAlign: 'left',
                      transition: 'all 0.3s ease',
                      fontWeight: selectedModel === model.id ? 'bold' : 'normal',
                      color: selectedModel === model.id ? '#667eea' : '#666'
                    }}
                  >
                    <div style={{ fontSize: '15px', marginBottom: '4px' }}>{model.name}</div>
                    <div style={{ fontSize: '12px', opacity: 0.7 }}>{model.description}</div>
                  </button>
                ))}
              </div>
            </div>
            
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
                <AiOutlineCheckCircle size={24} />
                Generated Lyrics
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
                  <h4 style={{ margin: '0 0 15px 0', color: '#48bb78', display: 'flex', alignItems: 'center', gap: '8px' }}>
                    <AiOutlineBook size={20} />
                    Other Variations:
                  </h4>
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
              <AiOutlineDollar size={20} />
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

              {costReport && (
                <>
                  

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
          {/* Description */}
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
              <AiOutlineFileText size={20} />
            </span>
            Project Description
          </h2>

          {/* Model Architecture & Statistics */}
          {modelInfoLoading && (
            <div style={{
              backgroundColor: '#f8f9ff',
              border: '2px solid #667eea',
              borderRadius: '12px',
              padding: '20px',
              marginBottom: '20px',
              textAlign: 'center',
              color: '#667eea'
            }}>
              Loading model information...
            </div>
          )}

          {modelInfo && modelInfo.success && (
            <div style={{
              backgroundColor: '#f0f4ff',
              border: '2px solid #667eea',
              borderRadius: '12px',
              padding: '20px',
              marginBottom: '20px'
            }}>
              <h3 style={{ margin: '0 0 15px 0', color: '#667eea', display: 'flex', alignItems: 'center', gap: '8px' }}>
                Model Architecture & Statistics
              </h3>
              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(250px, 1fr))', gap: '15px' }}>
                {/* Parameter Count */}
                <div style={{
                  backgroundColor: 'white',
                  border: '1px solid #e0e0e0',
                  borderRadius: '8px',
                  padding: '15px',
                  boxShadow: '0 2px 4px rgba(0,0,0,0.1)'
                }}>
                  <div style={{ fontSize: '14px', color: '#666', marginBottom: '5px' }}>Total Parameters</div>
                  <div style={{ fontSize: '24px', fontWeight: 'bold', color: '#667eea' }}>
                    {modelInfo.parameter_count?.toLocaleString() || 'N/A'}
                  </div>
                </div>

                {/* Vocabulary Size */}
                <div style={{
                  backgroundColor: 'white',
                  border: '1px solid #e0e0e0',
                  borderRadius: '8px',
                  padding: '15px',
                  boxShadow: '0 2px 4px rgba(0,0,0,0.1)'
                }}>
                  <div style={{ fontSize: '14px', color: '#666', marginBottom: '5px' }}>Vocabulary Size</div>
                  <div style={{ fontSize: '24px', fontWeight: 'bold', color: '#764ba2' }}>
                    {modelInfo.vocabulary_size?.toLocaleString() || 'N/A'}
                  </div>
                </div>

                {/* Embedding Dimension */}
                <div style={{
                  backgroundColor: 'white',
                  border: '1px solid #e0e0e0',
                  borderRadius: '8px',
                  padding: '15px',
                  boxShadow: '0 2px 4px rgba(0,0,0,0.1)'
                }}>
                  <div style={{ fontSize: '14px', color: '#666', marginBottom: '5px' }}>Embedding Dimension</div>
                  <div style={{ fontSize: '24px', fontWeight: 'bold', color: '#48bb78' }}>
                    {modelInfo.model_architecture?.embedding_dim || 'N/A'}
                  </div>
                </div>

                {/* Hidden Size */}
                <div style={{
                  backgroundColor: 'white',
                  border: '1px solid #e0e0e0',
                  borderRadius: '8px',
                  padding: '15px',
                  boxShadow: '0 2px 4px rgba(0,0,0,0.1)'
                }}>
                  <div style={{ fontSize: '14px', color: '#666', marginBottom: '5px' }}>Hidden Size</div>
                  <div style={{ fontSize: '24px', fontWeight: 'bold', color: '#ed8936' }}>
                    {modelInfo.model_architecture?.hidden_size || 'N/A'}
                  </div>
                </div>

                {/* Number of Layers */}
                <div style={{
                  backgroundColor: 'white',
                  border: '1px solid #e0e0e0',
                  borderRadius: '8px',
                  padding: '15px',
                  boxShadow: '0 2px 4px rgba(0,0,0,0.1)'
                }}>
                  <div style={{ fontSize: '14px', color: '#666', marginBottom: '5px' }}>Number of Layers</div>
                  <div style={{ fontSize: '24px', fontWeight: 'bold', color: '#f56565' }}>
                    {modelInfo.model_architecture?.num_layers || 'N/A'}
                  </div>
                </div>

                {/* Device */}
                <div style={{
                  backgroundColor: 'white',
                  border: '1px solid #e0e0e0',
                  borderRadius: '8px',
                  padding: '15px',
                  boxShadow: '0 2px 4px rgba(0,0,0,0.1)'
                }}>
                  <div style={{ fontSize: '14px', color: '#666', marginBottom: '5px' }}>Device</div>
                  <div style={{ fontSize: '14px', fontWeight: 'bold', color: '#38a169' }}>
                    {modelInfo.device || 'N/A'}
                  </div>
                </div>
              </div>
            </div>
          )}

          {/* General Description */}
          <div style={{
            backgroundColor: '#f8f9ff',
            border: '2px solid #667eea',
            borderRadius: '12px',
            padding: '20px',
            marginBottom: '20px'
          }}>
            <h3 style={{ margin: '0 0 15px 0', color: '#667eea' }}>General Description</h3>
            <div style={{ marginLeft: '20px', lineHeight: '1.8', color: '#333' }}>
              <div>This Project uses a Recurrent Neural Network (RNN) architecture to generate lyrics. 
                This model is trained on a large variety of song lyrics in order to generate new lyrics
                from a given seed text. The initial dataset, downloaded from Kagglehub at 
                <code style={{ fontFamily: 'monospace', backgroundColor: '#f5f5f5', padding: '2px 6px', borderRadius: '3px' }}>
                  kagglehub.dataset_download("devdope/900k-spotify")
                </code>
                , contains over 900,000 song lyrics from various artists and genres. We trimmed down the dataset to only include lyrics
                and converted the json to a txt file that capped out at 20mb. This means that the data was greatly limited for training,
                due to device hardware and time limitations.
              </div>
            </div>
          </div>

          {/* Pipeline */}
          <div style={{
            backgroundColor: '#f8f9ff',
            border: '2px solid #667eea',
            borderRadius: '12px',
            padding: '20px',
            marginBottom: '20px'
          }}>
            <h3 style={{ margin: '0 0 15px 0', color: '#667eea' }}>Project Pipeline</h3>
            <div style={{ marginLeft: '20px', lineHeight: '1.8', color: '#333' }}>
              <div>1. Run DataPreprocessor.py to download the kaggle dataset. This will download the json, extract the text, preprocess it, and save the "lyrics_preprocessed.txt" file for training.</div>
              <div>2. Run configure_pricing.py if using a different hosting service than Render. We are using Render so this step can be skipped. This file will create a json file of numbers to use for the cost analysis.</div>
              <div>3. Run train_model.py to train the LSTM model on the preprocessed lyrics dataset. This will generate a .pth model for generating song lyrics. This file will also handle the training cost calculations.</div>              
            </div>
          </div>

          {/* Errors and Issues */}
          <div style={{
            backgroundColor: '#f8f9ff',
            border: '2px solid #667eea',
            borderRadius: '12px',
            padding: '20px',
            marginBottom: '20px'
          }}>
            <h3 style={{ margin: '0 0 15px 0', color: '#667eea' }}>Possible Issues</h3>
            <div style={{ marginLeft: '20px', lineHeight: '1.8', color: '#333' }}>
              <div>There are some issues when it comes to the output of this dataset. The first issue is that the vocabulary of training was severely limited due to hardware limitations of our computers.
                This causes some tokens (words) to be unrecognizeable during generation which can lead to nonsensical lyrics. Another issue is that because we don't use transformers, grammar management
                is very poor which also can lead to nonsensical lyrics.
                The final issue to discuss is the dataset itself. The dataset contains song lyrics, which means a lot of people were freely expressing themselves and were not concerened about proper english sentences.
                That means the text generator might not generate coherent thoughts.
                In order to improve the model we could increase the training dataset, increase the vocabulary, and add grammar management (such as no repeated words, 
                nouns and verbs in correct order, etc.).
              </div>
            </div>
          </div>


          {/* Requirements */}
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
              <AiOutlineFileText size={20} />
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
      {/* YouTube showcase */}
      {activeTab === 'youtube' && (
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
                background: 'linear-gradient(135deg, #FF0000, #CC0000)',
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
              <AiOutlineYoutube size={24} />
            </span>
            YouTube Showcase
          </h2>

          <div
            style={{
              backgroundColor: '#f8f9ff',
              border: '2px solid #667eea',
              borderRadius: '12px',
              padding: '40px',
              maxWidth: '900px',
              margin: '0 auto',
            }}
          >
            {/* YouTube Embed */}
            <div
              style={{
                position: 'relative',
                width: '100%',
                paddingBottom: '56.25%', // 16:9 aspect ratio
                height: 0,
                overflow: 'hidden',
                borderRadius: '8px',
                boxShadow: '0 4px 6px rgba(0, 0, 0, 0.1)',
              }}
            >
              <iframe
                src="https://www.youtube.com/embed/v3rsRGiNJSA"
                title="Project 5: Song Lyric Generator Demonstration"
                style={{
                  position: 'absolute',
                  top: 0,
                  left: 0,
                  width: '100%',
                  height: '100%',
                  border: 'none',
                  borderRadius: '8px',
                }}
                allowFullScreen
                allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture"
              />
            </div>
          </div>
        </div>
      )}
      </div>
    </div>
  );
}

export default RNN;
