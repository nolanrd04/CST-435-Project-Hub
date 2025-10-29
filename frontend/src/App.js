import React from 'react';
import { BrowserRouter as Router, Route, Routes, Link } from 'react-router-dom';
import TextGenerator from './projects/projectRNN/TextGenerator.tsx';
import ImageClassifier from './projects/Project3/ImageClassifier.tsx';
import SentimentAnalyzer from './projects/Project4/SentimentAnalyzer.tsx';
import './App.css';

function App() {
  return (
    <Router>
      <div className="App">
        <header className="App-header">
          <h1>CST435 Project Hub</h1>
          <p className="subtitle">Nolan and John's one-stop shop for all CST435 projects</p>
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