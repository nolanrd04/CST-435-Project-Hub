import React, { useState, lazy, Suspense } from 'react';
import {
  AiOutlineCamera,
  AiOutlineDollar,
  AiOutlinePlusCircle,
  AiOutlineFileText,
  AiOutlineInfoCircle,
  AiOutlineYoutube,
} from 'react-icons/ai';

// Lazy load all tab components for performance
const ImageGenerator = lazy(() => import('./ImageGenerator.tsx'));
const TrainingCostAnalysis = lazy(() => import('./TrainingCostAnalysis.tsx'));
const CreateModel = lazy(() => import('./CreateModel.tsx'));
const ProjectDescription = lazy(() => import('./ProjectDescription.tsx'));
const ModelInformation = lazy(() => import('./ModelInformation.tsx'));
const YouTubeLink = lazy(() => import('./YouTubeLink.tsx'));

type TabType =
  | 'image-generator'
  | 'cost-analysis'
  | 'create-model'
  | 'description'
  | 'model-info'
  | 'youtube';

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

function Project6() {
  const [activeTab, setActiveTab] = useState<TabType>('image-generator');

  const tabs = [
    {
      id: 'image-generator' as TabType,
      label: 'Image Generator',
      icon: <AiOutlineCamera size={20} />,
    },
    {
      id: 'cost-analysis' as TabType,
      label: 'Training Cost Analysis',
      icon: <AiOutlineDollar size={20} />,
    },
    {
      id: 'create-model' as TabType,
      label: 'Create New Model',
      icon: <AiOutlinePlusCircle size={20} />,
    },
    {
      id: 'description' as TabType,
      label: 'Project Description',
      icon: <AiOutlineFileText size={20} />,
    },
    {
      id: 'model-info' as TabType,
      label: 'Model Information',
      icon: <AiOutlineInfoCircle size={20} />,
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
          Project 6: GAN Image Generator
        </h1>
        <p style={{ color: '#666', fontSize: '16px' }}>
          Generate computer drawn images of fruits. Trained on human-drawn images of fruits.
        </p>
      </div>

      {/* Tab Navigation */}
      <div
        style={{
          display: 'flex',
          gap: '10px',
          marginBottom: '30px',
          borderBottom: '2px solid #e5e7eb',
          flexWrap: 'wrap',
          overflowX: 'auto',
        }}
      >
        {tabs.map((tab) => (
          <button
            key={tab.id}
            onClick={() => setActiveTab(tab.id)}
            style={{
              padding: '12px 20px',
              backgroundColor: activeTab === tab.id ? '#667eea' : 'transparent',
              color: activeTab === tab.id ? 'white' : '#666',
              border: 'none',
              borderRadius: '8px 8px 0 0',
              cursor: 'pointer',
              fontWeight: activeTab === tab.id ? 'bold' : 'normal',
              fontSize: '15px',
              transition: 'all 0.3s ease',
              display: 'flex',
              alignItems: 'center',
              gap: '8px',
              whiteSpace: 'nowrap',
            }}
          >
            {tab.icon}
            {tab.label}
          </button>
        ))}
      </div>

      {/* Tab Content - Only render active tab */}
      <div style={{ minHeight: '500px' }}>
        <Suspense fallback={<LoadingSpinner />}>
          {activeTab === 'image-generator' && <ImageGenerator />}
          {activeTab === 'cost-analysis' && <TrainingCostAnalysis />}
          {activeTab === 'create-model' && <CreateModel />}
          {activeTab === 'description' && <ProjectDescription />}
          {activeTab === 'model-info' && <ModelInformation />}
          {activeTab === 'youtube' && <YouTubeLink />}
        </Suspense>
      </div>
    </div>
  );
}

export default Project6;
