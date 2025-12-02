import React, { useState, useEffect, Suspense } from 'react';
import { BrowserRouter as Router, Route, Routes, Link, useLocation } from 'react-router-dom';
import { FaHome } from 'react-icons/fa';
import './App.css';

// Import HomePage
import HomePage from './components/HomePage.tsx';

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
const Project6 = React.lazy(() =>
  import('./projects/Project6/Project6.tsx').then(module => ({ default: module.default }))
);
const Project7 = React.lazy(() =>
  import('./projects/Project7/Project7.tsx').then(module => ({ default: module.default }))
);

// Loading component shown while chunks are downloading
function LoadingSpinner() {
  return (
    <div style={{
      display: 'flex',
      flexDirection: 'column',
      justifyContent: 'center',
      alignItems: 'center',
      minHeight: '400px',
      gap: '20px'
    }}>
      <div style={{
        width: '60px',
        height: '60px',
        border: '4px solid #f3f3f3',
        borderTop: '4px solid #667eea',
        borderRadius: '50%',
        animation: 'spin 1s linear infinite'
      }}></div>
      <span style={{
        fontSize: '18px',
        color: '#667eea',
        fontWeight: '600'
      }}>Loading project...</span>
      <style>{`
        @keyframes spin {
          0% { transform: rotate(0deg); }
          100% { transform: rotate(360deg); }
        }
      `}</style>
    </div>
  );
}

// Navigation Header Component
function NavHeader({ useLocalAPI, setUseLocalAPI }: { useLocalAPI: boolean; setUseLocalAPI: (val: boolean) => void }) {
  const location = useLocation();
  const isHomePage = location.pathname === '/' || location.pathname === '/home';

  if (isHomePage) return null;

  return (
    <div className="nav-header">
      <div className="nav-container">
        <Link to="/home" className="nav-home-btn">
          <FaHome size={20} />
          <span>Home</span>
        </Link>

        <label className="api-toggle">
          <input
            type="checkbox"
            checked={useLocalAPI}
            onChange={(e) => setUseLocalAPI(e.target.checked)}
          />
          <span className="api-toggle-label">
            Local API
          </span>
        </label>
      </div>
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
        <NavHeader useLocalAPI={useLocalAPI} setUseLocalAPI={setUseLocalAPI} />
        <main>
          <Suspense fallback={<LoadingSpinner />}>
            <Routes>
              <Route path="/" element={<HomePage />} />
              <Route path="/home" element={<HomePage />} />
              <Route path="/text-generator" element={<TextGenerator />} />
              <Route path="/image-classifier" element={<ImageClassifier />} />
              <Route path="/sentiment-analyzer" element={<SentimentAnalyzer />} />
              <Route path="/Project5" element={<RNN activeTab="song-generator" />} />
              <Route path="/genetic-algorithm" element={<GeneticAlgorithmPage />} />
              <Route path="/project6" element={<Project6 />} />
              <Route path="/project7" element={<Project7 />} />
            </Routes>
          </Suspense>
        </main>
      </div>
    </Router>
  );
}

export default App;