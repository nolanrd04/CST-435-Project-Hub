import React, { useState, useEffect, Suspense } from 'react';
import { BrowserRouter as Router, Route, Routes, Link } from 'react-router-dom';
import './App.css';

// Lazy load all project components for better performance
const TextGenerator = React.lazy(() =>
  import('./projects/projectRNN/TextGenerator.tsx').then(module => ({ default: module.default }))
);
const ImageClassifier = React.lazy(() =>
  import('./projects/Project3/ImageClassifier.tsx').then(module => ({ default: module.default }))
);
const SentimentAnalyzer = React.lazy(() =>
  import('./projects/Project4/SentimentAnalyzer.tsx').then(module => ({ default: module.default }))
);
const RNN = React.lazy(() =>
  import('./projects/Project5/RNN.tsx').then(module => ({ default: module.default }))
);
const GeneticAlgorithmPage = React.lazy(() =>
  import('./projects/projectGA/GeneticAlgorithm.tsx').then(module => ({ default: module.default }))
);

// Loading component shown while chunks are downloading
function LoadingSpinner() {
  return (
    <div style={{
      display: 'flex',
      justifyContent: 'center',
      alignItems: 'center',
      minHeight: '400px',
      fontSize: '18px',
      color: '#999'
    }}>
      <span>⏳ Loading project...</span>
    </div>
  );
}

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
              backgroundColor: '#00000095',
              color: '#d4d4d4ff',
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
            <ul style={{
              display: 'flex',
              gap: '12px',
              listStyle: 'none',
              padding: 0,
              margin: 0,
              justifyContent: 'center',
              flexWrap: 'wrap'
            }}>
              <li>
                <Link to="/" className="button-link-to-projects">
                  Text Generator (Nolan's Personal)
                </Link>

              </li>
              <li>
                <Link to="/genetic-algorithm" className="button-link-to-projects">
                  Genetic Algorithm (Nolan's Personal)
                </Link>
              </li>
              <li>
                <Link to="/image-classifier" className="button-link-to-projects">
                  Project3: Image Classifier
                </Link>
              </li>
              <li>
                <Link to="/sentiment-analyzer" className="button-link-to-projects">
                  Project4: Sentiment Analyzer
                </Link>
              </li>
              <li>
                <Link to="/Project5" className="button-link-to-projects">
                  Project5: RNN
                </Link>
              </li>
            </ul>
          </nav>
        </header>
        <main>
          <Suspense fallback={<LoadingSpinner />}>
            <Routes>
              <Route path="/" element={<TextGenerator />} />
              <Route path="/text-generator" element={<TextGenerator />} />
              <Route path="/image-classifier" element={<ImageClassifier />} />
              <Route path="/sentiment-analyzer" element={<SentimentAnalyzer />} />
              <Route path="/Project5" element={<RNN activeTab="song-generator" />} />
              <Route path="/genetic-algorithm" element={<GeneticAlgorithmPage />} />
            </Routes>
          </Suspense>
        </main>
      </div>
    </Router>
  );
}

export default App;