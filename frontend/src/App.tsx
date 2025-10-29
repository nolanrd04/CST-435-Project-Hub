import React, { useState } from 'react';
import TextGenerator from './components/TextGenerator';
import ImageClassifier from './components/ImageClassifier';
import SentimentAnalyzer from './projects/Project4/SentimentAnalyzer';

type ProjectType = 'rnn' | 'cnn' | 'sentiment';

function App() {
  const [activeProject, setActiveProject] = useState<ProjectType>('rnn');

  const projects = [
    { id: 'rnn', name: 'RNN Text Generator', icon: '✨', description: 'Generate text with LSTM' },
    { id: 'cnn', name: 'Image Classifier', icon: '🖼️', description: 'Classify vehicle images' },
    { id: 'sentiment', name: 'Sentiment Analyzer', icon: '💬', description: 'Analyze hotel reviews' },
  ];

  return (
    <div className="min-h-screen bg-gradient-to-br from-indigo-500 to-purple-600">
      <div className="container mx-auto py-8">
        {/* Header */}
        <header className="text-center mb-10">
          <h1 className="text-5xl font-bold text-white mb-2">
            AI Hub
          </h1>
          <p className="text-xl text-indigo-100">
            Explore Neural Networks & Machine Learning
          </p>
        </header>

        {/* Project Navigation */}
        <div className="mb-10">
          <div className="flex flex-wrap justify-center gap-4">
            {projects.map((project) => (
              <button
                key={project.id}
                onClick={() => setActiveProject(project.id as ProjectType)}
                className={`px-6 py-4 rounded-xl font-semibold transition-all transform hover:scale-105 active:scale-95 ${
                  activeProject === project.id
                    ? 'bg-white text-indigo-600 shadow-lg'
                    : 'bg-white bg-opacity-20 text-white hover:bg-opacity-30 backdrop-blur'
                }`}
              >
                <div className="text-2xl mb-1">{project.icon}</div>
                <div className="font-bold text-sm">{project.name}</div>
                <div className="text-xs opacity-80">{project.description}</div>
              </button>
            ))}
          </div>
        </div>

        {/* Project Content */}
        <div className="mb-12">
          {activeProject === 'rnn' && <TextGenerator />}
          {activeProject === 'cnn' && <ImageClassifier />}
          {activeProject === 'sentiment' && <SentimentAnalyzer />}
        </div>

        {/* Footer */}
        <footer className="text-center mt-12 text-white">
          <p>CST-435 / AIT-204: AI & Machine Learning Projects</p>
        </footer>
      </div>
    </div>
  );
}

export default App;