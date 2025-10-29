import React, { useState } from 'react';

function SentimentAnalyzer() {
  const [reviewText, setReviewText] = useState('');
  const [result, setResult] = useState<any>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [showProcessed, setShowProcessed] = useState(false);

  const handleAnalyze = async () => {
    if (!reviewText.trim()) {
      setError('Please enter a hotel review');
      return;
    }

    setLoading(true);
    setError('');
    try {
      const response = await fetch('http://localhost:8000/analyze-sentiment', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          review_text: reviewText,
        }),
      });

      if (!response.ok) {
        throw new Error('Failed to analyze sentiment');
      }

      const data = await response.json();
      setResult(data);
      setError('');
    } catch (err) {
      setError(err instanceof Error ? err.message : 'An error occurred');
      console.error('Error analyzing sentiment:', err);
    } finally {
      setLoading(false);
    }
  };

  const handleClear = () => {
    setReviewText('');
    setResult(null);
    setError('');
    setShowProcessed(false);
  };

  const getSentimentColor = (sentiment: string) => {
    if (sentiment.includes('Positive')) return '#10b981';
    if (sentiment.includes('Negative')) return '#ef4444';
    return '#f59e0b';
  };

  const getSentimentBadgeColor = (classification: string) => {
    if (classification === 'Positive') {
      return { bg: '#dcfce7', color: '#166534' };
    } else {
      return { bg: '#fee2e2', color: '#991b1b' };
    }
  };

  return (
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
          💬
        </span>
        Sentiment Analyzer
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
          <strong>About this project:</strong> This NLP sentiment analyzer uses a trained Logistic Regression model
          with TF-IDF vectorization to classify hotel reviews as positive or negative. It also provides word-level
          sentiment scores using TF-IDF weighted analysis to show which words contribute to the overall sentiment.
        </p>
      </div>

      {/* Error Message */}
      {error && (
        <div className="error" style={{ marginBottom: '20px', backgroundColor: '#fed7d7', border: '1px solid #fc8181', borderLeft: '4px solid #e53e3e', borderRadius: '8px' }}>
          ⚠️ {error}
        </div>
      )}

      {/* Review Input Section */}
      <div className="form-group">
        <label style={{ display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '10px' }}>
          <span>✍️</span>
          Hotel Review <span style={{ color: '#e53e3e' }}>*</span>
        </label>
        <textarea
          value={reviewText}
          onChange={(e) => setReviewText(e.target.value)}
          placeholder="Enter a hotel review to analyze its sentiment... (e.g., 'The hotel was amazing! Great staff and clean rooms.')"
          disabled={loading}
          className="textarea"
          rows={6}
          style={{
            borderRadius: '12px',
            borderColor: loading ? '#cbd5e0' : '#ccc',
            backgroundColor: loading ? '#f7fafc' : '#ffffff'
          }}
        />
        <p style={{ fontSize: '12px', color: '#718096', marginTop: '8px' }}>
          💡 The model works best with complete sentences describing hotel experiences (rooms, staff, service, etc.)
        </p>
      </div>

      {/* Action Buttons */}
      <div style={{ display: 'flex', gap: '12px', justifyContent: 'flex-end', marginTop: '30px', marginBottom: '20px' }}>
        <button
          onClick={handleClear}
          disabled={loading || !reviewText}
          className="button secondary"
          style={{
            opacity: loading || !reviewText ? 0.5 : 1,
            cursor: loading || !reviewText ? 'not-allowed' : 'pointer'
          }}
        >
          ✕ Clear
        </button>
        <button
          onClick={handleAnalyze}
          disabled={loading || !reviewText.trim()}
          className="button"
          style={{
            opacity: loading || !reviewText.trim() ? 0.5 : 1,
            cursor: loading || !reviewText.trim() ? 'not-allowed' : 'pointer'
          }}
        >
          {loading ? (
            <>
              <span style={{ display: 'inline-block', marginRight: '8px' }}>⏳</span>
              Analyzing...
            </>
          ) : (
            <>
              <span style={{ display: 'inline-block', marginRight: '8px' }}>🔍</span>
              Analyze Sentiment
            </>
          )}
        </button>
      </div>

      {/* Results Section */}
      {result && (
        <div className="output">
          <h3 style={{ marginBottom: '20px', display: 'flex', alignItems: 'center', gap: '10px' }}>
            <span style={{ fontSize: '24px' }}>📊</span>
            Analysis Results
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

          {/* Classification Badge */}
          <div style={{ marginBottom: '20px' }}>
            <p style={{ color: '#718096', fontSize: '12px', marginBottom: '8px' }}>Classification:</p>
            <div style={{
              display: 'inline-block',
              padding: '12px 24px',
              borderRadius: '8px',
              fontWeight: 'bold',
              fontSize: '16px',
              ...getSentimentBadgeColor(result.classification)
            }}>
              {result.classification === 'Positive' ? '👍 Positive Review' : '👎 Negative Review'}
            </div>
          </div>

          {/* Main Metrics */}
          <div style={{
            display: 'grid',
            gridTemplateColumns: '1fr 1fr',
            gap: '15px',
            marginBottom: '20px'
          }}>
            {/* Confidence */}
            <div style={{
              background: '#f0f4ff',
              border: '2px solid #667eea',
              borderRadius: '12px',
              padding: '16px'
            }}>
              <p style={{ color: '#718096', fontSize: '12px', margin: '0 0 8px 0' }}>Confidence</p>
              <p style={{ color: '#667eea', fontSize: '24px', fontWeight: 'bold', margin: 0 }}>
                {(result.confidence * 100).toFixed(1)}%
              </p>
            </div>

            {/* Sentiment Score */}
            <div style={{
              background: '#fff5f0',
              border: '2px solid #ea580c',
              borderRadius: '12px',
              padding: '16px'
            }}>
              <p style={{ color: '#718096', fontSize: '12px', margin: '0 0 8px 0' }}>Sentiment Score</p>
              <p style={{ color: '#ea580c', fontSize: '24px', fontWeight: 'bold', margin: 0 }}>
                {result.sentiment_score.toFixed(2)}/5.0
              </p>
            </div>
          </div>

          {/* Sentiment Label and Details */}
          <div style={{
            background: 'linear-gradient(to bottom right, #f7fafc, #edf2f7)',
            borderRadius: '12px',
            padding: '20px',
            border: '2px solid #c7d2fe',
            marginBottom: '20px'
          }}>
            <p style={{ color: '#718096', fontSize: '12px', margin: '0 0 8px 0' }}>Sentiment Category:</p>
            <p style={{
              fontSize: '18px',
              fontWeight: 'bold',
              margin: 0,
              color: getSentimentColor(result.sentiment_label)
            }}>
              {result.sentiment_label}
            </p>
          </div>

          {/* Probability Bars */}
          <div style={{ marginBottom: '20px' }}>
            <h4 style={{ color: '#2d3748', fontWeight: 'bold', marginBottom: '12px' }}>Classification Probabilities:</h4>
            <div style={{ marginBottom: '15px' }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '5px' }}>
                <span style={{ fontWeight: '600', color: '#2d3748' }}>👍 Positive Probability</span>
                <span style={{ fontWeight: 'bold', color: '#10b981' }}>
                  {(result.positive_probability * 100).toFixed(1)}%
                </span>
              </div>
              <div style={{
                width: '100%',
                backgroundColor: '#e5e7eb',
                borderRadius: '8px',
                height: '10px',
                overflow: 'hidden'
              }}>
                <div
                  style={{
                    background: 'linear-gradient(to right, #10b981, #059669)',
                    height: '100%',
                    borderRadius: '8px',
                    width: `${result.positive_probability * 100}%`,
                    transition: 'width 0.5s ease'
                  }}
                ></div>
              </div>
            </div>

            <div>
              <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '5px' }}>
                <span style={{ fontWeight: '600', color: '#2d3748' }}>👎 Negative Probability</span>
                <span style={{ fontWeight: 'bold', color: '#ef4444' }}>
                  {(result.negative_probability * 100).toFixed(1)}%
                </span>
              </div>
              <div style={{
                width: '100%',
                backgroundColor: '#e5e7eb',
                borderRadius: '8px',
                height: '10px',
                overflow: 'hidden'
              }}>
                <div
                  style={{
                    background: 'linear-gradient(to right, #ef4444, #dc2626)',
                    height: '100%',
                    borderRadius: '8px',
                    width: `${result.negative_probability * 100}%`,
                    transition: 'width 0.5s ease'
                  }}
                ></div>
              </div>
            </div>
          </div>

          {/* Processed Text Toggle */}
          <div style={{
            background: '#f0f4ff',
            borderRadius: '12px',
            padding: '16px',
            marginBottom: '20px',
            borderLeft: '4px solid #667eea'
          }}>
            <button
              onClick={() => setShowProcessed(!showProcessed)}
              style={{
                background: 'none',
                border: 'none',
                color: '#667eea',
                fontWeight: 'bold',
                cursor: 'pointer',
                fontSize: '14px',
                padding: 0
              }}
            >
              {showProcessed ? '▼' : '▶'} Processed Text (after cleaning)
            </button>
            {showProcessed && (
              <p style={{
                marginTop: '12px',
                color: '#2d3748',
                fontStyle: 'italic',
                fontSize: '13px',
                lineHeight: '1.5'
              }}>
                {result.processed_text}
              </p>
            )}
          </div>

          {/* Info Box */}
          <div style={{
            backgroundColor: '#dbeafe',
            borderLeft: '4px solid #3b82f6',
            borderRadius: '8px',
            padding: '12px',
            marginBottom: '15px',
            fontSize: '13px',
            color: '#1e40af'
          }}>
            <p style={{ margin: 0 }}>
              <strong>Note:</strong> This model was trained on TripAdvisor hotel reviews and uses word-level sentiment
              analysis combined with logistic regression classification. The sentiment score (1-5) represents the
              average sentiment of words found in the review.
            </p>
          </div>

          {/* Action Button */}
          <button
            onClick={handleClear}
            style={{
              padding: '12px 20px',
              backgroundColor: '#e0e7ff',
              color: '#4f46e5',
              border: 'none',
              borderRadius: '8px',
              fontWeight: '600',
              cursor: 'pointer',
              width: '100%'
            }}
          >
            🔄 Analyze Another Review
          </button>
        </div>
      )}
    </div>
  );
}

export default SentimentAnalyzer;
