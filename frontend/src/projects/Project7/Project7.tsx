import React, { useState, lazy, Suspense } from 'react';
import {
  AiOutlinePicture,
  AiOutlineDollar,
  AiOutlineInfoCircle,
  AiOutlineYoutube,
} from 'react-icons/ai';

// Lazy load all tab components for performance
const ImagePainter = lazy(() => import('./ImagePainter.tsx'));
const TrainingCostAnalysis = lazy(() => import('./TrainingCostAnalysis.tsx'));
const ModelInformation = lazy(() => import('./ModelInformation.tsx'));
const YouTubeLink = lazy(() => import('./YouTubeLink.tsx'));

type TabType = 'image-painter' | 'cost-analysis' | 'model-info' | 'youtube';

function LoadingSpinner() {
  return (
    <div
      style={{
        display: 'flex',
        justifyContent: 'center',
        alignItems: 'center',
        minHeight: '300px',
        fontSize: '16px',
        color: '#667eea',
      }}
    >
      <span>Loading...</span>
    </div>
  );
}

function Project7() {
  const [activeTab, setActiveTab] = useState<TabType>('image-painter');

  const tabs = [
    {
      id: 'image-painter' as TabType,
      label: 'Image Colorization',
      icon: <AiOutlinePicture size={20} />,
    },
    {
      id: 'model-info' as TabType,
      label: 'Model Information',
      icon: <AiOutlineInfoCircle size={20} />,
    },
    {
      id: 'cost-analysis' as TabType,
      label: 'Training Cost Analysis',
      icon: <AiOutlineDollar size={20} />,
    },
    {
      id: 'youtube' as TabType,
      label: 'YouTube',
      icon: <AiOutlineYoutube size={20} />,
    },
  ];

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
          Project 7: Diffusion Image Colorization
        </h1>
        <p style={{ color: '#666', fontSize: '16px' }}>
          Transform grayscale images to color using a conditional diffusion model trained on Tiny ImageNet.
        </p>
      </div>

      {/* Tabs */}
      <div
        style={{
          display: 'flex',
          gap: '10px',
          borderBottom: '2px solid #e0e0e0',
          marginBottom: '30px',
          overflowX: 'auto',
          flexWrap: 'wrap',
        }}
      >
        {tabs.map((tab) => (
          <button
            key={tab.id}
            onClick={() => setActiveTab(tab.id)}
            style={{
              padding: '12px 20px',
              fontSize: '14px',
              fontWeight: activeTab === tab.id ? 'bold' : 'normal',
              color: activeTab === tab.id ? '#667eea' : '#666',
              background: 'none',
              border: 'none',
              borderBottom: activeTab === tab.id ? '3px solid #667eea' : '3px solid transparent',
              cursor: 'pointer',
              transition: 'all 0.3s',
              display: 'flex',
              alignItems: 'center',
              gap: '8px',
              whiteSpace: 'nowrap',
            }}
            onMouseEnter={(e) => {
              if (activeTab !== tab.id) {
                e.currentTarget.style.color = '#667eea';
              }
            }}
            onMouseLeave={(e) => {
              if (activeTab !== tab.id) {
                e.currentTarget.style.color = '#666';
              }
            }}
          >
            {tab.icon}
            {tab.label}
          </button>
        ))}
      </div>

      {/* Tab Content */}
      <div>
        <Suspense fallback={<LoadingSpinner />}>
          {activeTab === 'image-painter' && <ImagePainter />}
          {activeTab === 'cost-analysis' && <TrainingCostAnalysis />}
          {activeTab === 'model-info' && <ModelInformation />}
          {activeTab === 'youtube' && <YouTubeLink />}
        </Suspense>
      </div>
    </div>
  );
}

export default Project7;
