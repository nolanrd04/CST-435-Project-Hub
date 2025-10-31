import React, { useState } from 'react';
import CostAnalysis from './CostAnalysis.tsx';

function TextGenerator() {
  const [activeTab, setActiveTab] = useState<'generator' | 'cost'>('generator');
  const [seedText, setSeedText] = useState('');
  const [numWords, setNumWords] = useState(50);
  const [temperature, setTemperature] = useState(1.0);
  const [generatedText, setGeneratedText] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');

  const handleGenerate = async () => {
    if (!seedText.trim()) {
      setError('Please enter some seed text');
      return;
    }

    setLoading(true);
    setError('');
    try {
      const apiMode = localStorage.getItem('API_MODE');
      const apiUrl = apiMode === 'local' ? 'http://localhost:8000' : 'https://cst-435-project-hub.onrender.com';
      const response = await fetch(`${apiUrl}/generate-text`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          seed_text: seedText,
          num_words: numWords,
          temperature: temperature,
        }),
      });

      if (!response.ok) {
        throw new Error('Failed to generate text');
      }

      const data = await response.json();
      setGeneratedText(data.generated_text);
      setError('');
    } catch (err) {
      setError(err instanceof Error ? err.message : 'An error occurred');
      console.error('Error generating text:', err);
    } finally {
      setLoading(false);
    }
  };

  const handleClear = () => {
    setSeedText('');
    setGeneratedText('');
    setError('');
  };

  const handleCopy = () => {
    navigator.clipboard.writeText(generatedText);
    alert('Text copied to clipboard!');
  };

  const getTemperatureDescription = (temp: number) => {
    if (temp <= 0.5) return 'Conservative - More predictable';
    if (temp <= 1.0) return 'Balanced - Mix of predictability & creativity';
    if (temp <= 1.5) return 'Creative - More varied output';
    return 'Wild - Maximum randomness';
  };

  return (
    <div style={{ maxWidth: '900px', margin: '0 auto' }}>
      {/* Tab Navigation */}
      <div style={{
        display: 'flex',
        gap: '10px',
        marginBottom: '20px',
        borderBottom: '2px solid #e5e7eb'
      }}>
        <button
          onClick={() => setActiveTab('generator')}
          style={{
            padding: '12px 24px',
            backgroundColor: activeTab === 'generator' ? '#667eea' : 'transparent',
            color: activeTab === 'generator' ? 'white' : '#666',
            border: 'none',
            borderRadius: '8px 8px 0 0',
            cursor: 'pointer',
            fontWeight: activeTab === 'generator' ? 'bold' : 'normal',
            fontSize: '16px',
            transition: 'all 0.3s ease'
          }}
        >
          ✨ Text Generator
        </button>
        <button
          onClick={() => setActiveTab('cost')}
          style={{
            padding: '12px 24px',
            backgroundColor: activeTab === 'cost' ? '#667eea' : 'transparent',
            color: activeTab === 'cost' ? 'white' : '#666',
            border: 'none',
            borderRadius: '8px 8px 0 0',
            cursor: 'pointer',
            fontWeight: activeTab === 'cost' ? 'bold' : 'normal',
            fontSize: '16px',
            transition: 'all 0.3s ease'
          }}
        >
          💰 Cost Analysis
        </button>
      </div>

      {/* Generator Tab */}
      {activeTab === 'generator' && (
    <div className="form" style={{ maxWidth: '700px', margin: '0 auto' }}>
      <h2 className="title" style={{ marginBottom: '30px', display: 'flex', alignItems: 'center', gap: '10px' }}>
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
          ✨
        </span>
        Generate New Text
      </h2>

      {/* Project Description */}
      <div style={{
        backgroundColor: '#f0f4ff',
        border: '2px solid #667eea',
        borderRadius: '12px',
        padding: '16px',
        marginBottom: '25px',
        color: '#2d3748',
        lineHeight: '1.6',
        fontSize: '14px'
      }}>
        <p style={{ margin: 0 }}>
          <strong>About this project:</strong> This RNN (Recurrent Neural Network) uses LSTM (Long Short-Term Memory) cells to generate text based on a seed phrase.
          The model learns patterns from training data and can generate creative text continuations. Adjust the temperature to control creativity -
          lower values produce more predictable text, while higher values produce more random and creative output. This model was trained on "Alice in Wonderland," 
          and has a low accuracy score due to hardware limitations.
        </p>
      </div>

      {/* Error Message */}
      {error && (
        <div className="error" style={{ marginBottom: '20px', backgroundColor: '#fed7d7', border: '1px solid #fc8181', borderLeft: '4px solid #e53e3e', borderRadius: '8px' }}>
          ⚠️ {error}
        </div>
      )}

      {/* Seed Text Section */}
      <div className="form-group">
        <label style={{ display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '10px' }}>
          <span>📝</span>
          Seed Text <span style={{ color: '#e53e3e' }}>*</span>
        </label>
        <textarea
          value={seedText}
          onChange={(e) => setSeedText(e.target.value)}
          placeholder="Enter some starting text to generate from... (e.g., 'The future of AI')"
          disabled={loading}
          className="textarea"
          rows={5}
          style={{
            borderRadius: '12px',
            borderColor: loading ? '#cbd5e0' : '#ccc',
            backgroundColor: loading ? '#f7fafc' : '#ffffff'
          }}
        />
        <p style={{ fontSize: '12px', color: '#718096', marginTop: '8px' }}>
          💡 The longer and more descriptive your seed text, the better the generated output
        </p>
      </div>

      {/* Controls Grid */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(280px, 1fr))', gap: '20px', marginBottom: '25px' }}>
        {/* Word Count */}
        <div style={{
          background: 'linear-gradient(to bottom right, #dbeafe, #e0e7ff)',
          borderRadius: '12px',
          padding: '20px'
        }}>
          <label style={{ display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '12px', fontWeight: '600', color: '#2d3748' }}>
            <span style={{ fontSize: '18px' }}>📊</span>
            Number of Words
          </label>
          <div style={{ display: 'flex', alignItems: 'center', gap: '15px' }}>
            <input
              type="range"
              min="10"
              max="200"
              value={numWords}
              onChange={(e) => setNumWords(Number(e.target.value))}
              disabled={loading}
              style={{
                flex: 1,
                height: '8px',
                borderRadius: '8px',
                cursor: loading ? 'not-allowed' : 'pointer',
                opacity: loading ? 0.5 : 1,
                background: `linear-gradient(to right, #93c5fd 0%, #6366f1 ${(numWords / 200) * 100}%, #e5e7eb ${(numWords / 200) * 100}%, #e5e7eb 100%)`
              }}
            />
            <div style={{
              background: 'white',
              border: '2px solid #818cf8',
              borderRadius: '8px',
              padding: '8px 12px',
              minWidth: '60px',
              textAlign: 'center',
              fontWeight: 'bold',
              color: '#667eea'
            }}>
              {numWords}
            </div>
          </div>
          <p style={{ fontSize: '12px', color: '#4a5568', marginTop: '8px' }}>Range: 10 - 200 words</p>
        </div>

        {/* Temperature */}
        <div style={{
          background: 'linear-gradient(to bottom right, #fce7f3, #fae8ff)',
          borderRadius: '12px',
          padding: '20px'
        }}>
          <label style={{ display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '12px', fontWeight: '600', color: '#2d3748' }}>
            <span style={{ fontSize: '18px' }}>🔥</span>
            Temperature (Creativity)
          </label>
          <div style={{ display: 'flex', alignItems: 'center', gap: '15px' }}>
            <input
              type="range"
              min="0.1"
              max="2.0"
              step="0.1"
              value={temperature}
              onChange={(e) => setTemperature(Number(e.target.value))}
              disabled={loading}
              style={{
                flex: 1,
                height: '8px',
                borderRadius: '8px',
                cursor: loading ? 'not-allowed' : 'pointer',
                opacity: loading ? 0.5 : 1,
                background: `linear-gradient(to right, #e9d5ff 0%, #f472b6 ${((temperature - 0.1) / 1.9) * 100}%, #e5e7eb ${((temperature - 0.1) / 1.9) * 100}%, #e5e7eb 100%)`
              }}
            />
            <div style={{
              background: 'white',
              border: '2px solid #c084fc',
              borderRadius: '8px',
              padding: '8px 12px',
              minWidth: '60px',
              textAlign: 'center',
              fontWeight: 'bold',
              color: '#9333ea'
            }}>
              {temperature.toFixed(1)}
            </div>
          </div>
          <p style={{ fontSize: '12px', color: '#4a5568', marginTop: '8px' }}>
            {getTemperatureDescription(temperature)}
          </p>
        </div>
      </div>

      {/* Action Buttons */}
      <div style={{ display: 'flex', gap: '12px', justifyContent: 'flex-end', marginTop: '30px', marginBottom: '20px' }}>
        <button
          onClick={handleClear}
          disabled={loading || !seedText}
          className="button secondary"
          style={{
            opacity: loading || !seedText ? 0.5 : 1,
            cursor: loading || !seedText ? 'not-allowed' : 'pointer'
          }}
        >
          ✕ Clear
        </button>
        <button
          onClick={handleGenerate}
          disabled={loading || !seedText.trim()}
          className="button"
          style={{
            opacity: loading || !seedText.trim() ? 0.5 : 1,
            cursor: loading || !seedText.trim() ? 'not-allowed' : 'pointer'
          }}
        >
          {loading ? (
            <>
              <span style={{ display: 'inline-block', marginRight: '8px' }}>⏳</span>
              Generating...
            </>
          ) : (
            <>
              <span style={{ display: 'inline-block', marginRight: '8px' }}>✨</span>
              Generate Text
            </>
          )}
        </button>
      </div>

      {/* Output Section */}
      {generatedText && (
        <div className="output">
          <h3 style={{ marginBottom: '20px', display: 'flex', alignItems: 'center', gap: '10px' }}>
            <span style={{ fontSize: '24px' }}>🎯</span>
            Generated Text
            <span style={{
              fontSize: '11px',
              background: '#dcfce7',
              color: '#166534',
              padding: '4px 12px',
              borderRadius: '20px',
              fontWeight: 'bold',
              marginLeft: 'auto'
            }}>
              ✓ Complete
            </span>
          </h3>

          <div className="generated-text" style={{
            background: 'linear-gradient(to bottom right, #f7fafc, #edf2f7)',
            borderColor: '#c7d2fe',
            marginBottom: '20px',
            borderRadius: '12px'
          }}>
            {generatedText}
          </div>

          <div style={{ display: 'flex', gap: '10px', flexWrap: 'wrap' }}>
            <button
              onClick={handleCopy}
              className="copy-button"
              style={{
                padding: '12px 20px',
                backgroundColor: '#e0e7ff',
                color: '#4f46e5',
                border: 'none',
                borderRadius: '8px',
                fontWeight: '600',
                cursor: 'pointer'
              }}
            >
              📋 Copy to Clipboard
            </button>
            <button
              onClick={() => setGeneratedText('')}
              style={{
                padding: '12px 20px',
                backgroundColor: '#f3f4f6',
                color: '#4b5563',
                border: 'none',
                borderRadius: '8px',
                fontWeight: '600',
                cursor: 'pointer'
              }}
            >
              🔄 Generate Again
            </button>
          </div>
        </div>
      )}
    </div>
      )}

      {/* Cost Analysis Tab */}
      {activeTab === 'cost' && <CostAnalysis />}
    </div>
  );
}

export default TextGenerator;