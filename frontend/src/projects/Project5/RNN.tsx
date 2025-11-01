import React, { useState } from 'react';

function RNN() {
  const [activeTab, setActiveTab] = useState('generator');

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
          RNN Text Generator
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
          Description and Requirements
        </button>
      </div>

      {activeTab === 'generator' && (
        <div>
          <h2>Text Generator</h2>
          <p>This is the text generator tab content.</p>
        </div>
      )}
      {activeTab === 'cost' && (
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
