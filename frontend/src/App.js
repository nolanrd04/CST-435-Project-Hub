import React, { useState, useEffect } from 'react';
import { BrowserRouter as Router, Route, Routes, Link } from 'react-router-dom';
import TextGenerator from './projects/projectRNN/TextGenerator.tsx';
import ImageClassifier from './projects/Project3/ImageClassifier.tsx';
import SentimentAnalyzer from './projects/Project4/SentimentAnalyzer.tsx';
import './App.css';

function App() {
  const [useLocalAPI, setUseLocalAPI] = useState(() => {
    const saved = localStorage.getItem('useLocalAPI');
    return saved ? JSON.parse(saved) : false;
  });

  useEffect(() => {
    localStorage.setItem('useLocalAPI', JSON.stringify(useLocalAPI));
    localStorage.setItem('API_MODE', useLocalAPI ? 'local' : 'deployed');
  }, [useLocalAPI]);

  return (
    <Router>
      <div className="App">
        <header className="App-header">
          <h1>CST435 Project Hub</h1>
          <p className="subtitle">Nolan and John's one-stop shop for all CST435 projects</p>

          {/* API Mode Toggle */}
          <div style={{ marginBottom: '20px' }}>
            <label style={{
              display: 'inline-flex',
              alignItems: 'center',
              gap: '10px',
              backgroundColor: '#FFD700',
              color: '#000',
              padding: '12px 20px',
              borderRadius: '8px',
              fontWeight: 'bold',
              cursor: 'pointer',
              fontSize: '16px'
            }}>
              <input
                type="checkbox"
                checked={useLocalAPI}
                onChange={(e) => setUseLocalAPI(e.target.checked)}
                style={{ width: '20px', height: '20px', cursor: 'pointer' }}
              />
              <span>
                Use localhost API instead of cloud API
              </span>
            </label>
          </div>

          <nav>
            <ul>
              <li><Link to="/">RNN Text Generator - Nolan's personal</Link></li>
              <li><Link to="/image-classifier">Project 3: Image Classifier</Link></li>
              <li><Link to="/sentiment-analyzer">Project 4: Sentiment Analyzer</Link></li>
            </ul>
          </nav>
        </header>
        <main>
          <Routes>
            <Route path="/" element={<TextGenerator />} />
            <Route path="/text-generator" element={<TextGenerator />} />
            <Route path="/image-classifier" element={<ImageClassifier />} />
            <Route path="/sentiment-analyzer" element={<SentimentAnalyzer />} />
          </Routes>
        </main>
      </div>
    </Router>
  );
}

export default App;