import React, { useState, useEffect } from 'react';
import { AiOutlineCamera, AiOutlineWarning, AiOutlineCheckCircle } from 'react-icons/ai';
import { project6API } from './api.ts';

const FRUITS = ['apple', 'banana', 'blackberry', 'grape', 'pear', 'strawberry', 'watermelon'];

function ImageGenerator() {
  const [models, setModels] = useState<string[]>([]);
  const [selectedModel, setSelectedModel] = useState<string>('');
  const [selectedFruit, setSelectedFruit] = useState<string>('apple');
  const [numImages, setNumImages] = useState<number>(16);
  const [generatedImages, setGeneratedImages] = useState<string[]>([]);
  const [loading, setLoading] = useState<boolean>(false);
  const [loadingModels, setLoadingModels] = useState<boolean>(true);
  const [error, setError] = useState<string>('');
  const [success, setSuccess] = useState<string>('');

  // Load available models on mount
  useEffect(() => {
    loadModels();
  }, []);

  const loadModels = async () => {
    setLoadingModels(true);
    setError('');
    try {
      const modelList = await project6API.listModels();
      setModels(modelList);
      if (modelList.length > 0) {
        setSelectedModel(modelList[0]);
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to load models');
    } finally {
      setLoadingModels(false);
    }
  };

  const handleGenerate = async () => {
    if (!selectedModel) {
      setError('Please select a model');
      return;
    }

    setLoading(true);
    setError('');
    setSuccess('');
    setGeneratedImages([]);

    try {
      const response = await project6API.generateImages({
        model_name: selectedModel,
        fruit: selectedFruit,
        num_images: numImages,
      });

      if (response.success) {
        setGeneratedImages(response.images);
        setSuccess(`Successfully generated ${response.images.length} ${selectedFruit} images!`);
      } else {
        setError('Failed to generate images');
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : 'An error occurred while generating images');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div style={{ padding: '20px' }}>
      <h2
        style={{
          marginBottom: '30px',
          display: 'flex',
          alignItems: 'center',
          gap: '10px',
        }}
      >
        <span
          style={{
            background: 'linear-gradient(135deg, #667eea, #764ba2)',
            color: 'white',
            width: '40px',
            height: '40px',
            borderRadius: '50%',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            fontSize: '20px',
          }}
        >
          <AiOutlineCamera size={24} />
        </span>
        Generate Fruit Images
      </h2>

      {/* Error Message */}
      {error && (
        <div
          style={{
            backgroundColor: '#fee',
            border: '2px solid #f99',
            borderRadius: '8px',
            padding: '15px',
            marginBottom: '20px',
            color: '#c33',
            display: 'flex',
            alignItems: 'center',
            gap: '10px',
          }}
        >
          <AiOutlineWarning size={20} />
          {error}
        </div>
      )}

      {/* Success Message */}
      {success && (
        <div
          style={{
            backgroundColor: '#d4edda',
            border: '2px solid #28a745',
            borderRadius: '8px',
            padding: '15px',
            marginBottom: '20px',
            color: '#155724',
            display: 'flex',
            alignItems: 'center',
            gap: '10px',
          }}
        >
          <AiOutlineCheckCircle size={20} />
          {success}
        </div>
      )}

      {/* Control Panel */}
      <div
        style={{
          backgroundColor: '#f8f9ff',
          border: '2px solid #667eea',
          borderRadius: '12px',
          padding: '25px',
          marginBottom: '30px',
        }}
      >
        <h3 style={{ margin: '0 0 20px 0', color: '#667eea' }}>Configuration</h3>

        {/* Model Selection */}
        <div style={{ marginBottom: '20px' }}>
          <label
            style={{
              display: 'block',
              fontSize: '14px',
              fontWeight: 'bold',
              marginBottom: '10px',
              color: '#333',
            }}
          >
            Select Model
          </label>
          {loadingModels ? (
            <p style={{ color: '#666', fontStyle: 'italic' }}>Loading models...</p>
          ) : models.length === 0 ? (
            <p style={{ color: '#999', fontStyle: 'italic' }}>
              No models available. Please create a model first.
            </p>
          ) : (
            <select
              value={selectedModel}
              onChange={(e) => setSelectedModel(e.target.value)}
              style={{
                width: '100%',
                padding: '12px',
                fontSize: '14px',
                border: '2px solid #e0e0e0',
                borderRadius: '8px',
                backgroundColor: 'white',
                cursor: 'pointer',
              }}
            >
              {models.map((model) => (
                <option key={model} value={model}>
                  {model}
                </option>
              ))}
            </select>
          )}
        </div>

        {/* Fruit Selection */}
        <div style={{ marginBottom: '20px' }}>
          <label
            style={{
              display: 'block',
              fontSize: '14px',
              fontWeight: 'bold',
              marginBottom: '10px',
              color: '#333',
            }}
          >
            Select Fruit
          </label>
          <div
            style={{
              display: 'grid',
              gridTemplateColumns: 'repeat(auto-fill, minmax(120px, 1fr))',
              gap: '10px',
            }}
          >
            {FRUITS.map((fruit) => (
              <button
                key={fruit}
                onClick={() => setSelectedFruit(fruit)}
                style={{
                  padding: '12px',
                  border:
                    selectedFruit === fruit ? '3px solid #667eea' : '2px solid #e0e0e0',
                  backgroundColor: selectedFruit === fruit ? '#f0f4ff' : 'white',
                  borderRadius: '8px',
                  cursor: 'pointer',
                  fontWeight: selectedFruit === fruit ? 'bold' : 'normal',
                  color: selectedFruit === fruit ? '#667eea' : '#666',
                  textTransform: 'capitalize',
                  transition: 'all 0.3s ease',
                }}
              >
                {fruit}
              </button>
            ))}
          </div>
        </div>

        {/* Number of Images */}
        <div style={{ marginBottom: '20px' }}>
          <label
            style={{
              display: 'block',
              fontSize: '14px',
              fontWeight: 'bold',
              marginBottom: '10px',
              color: '#333',
            }}
          >
            Number of Images: {numImages}
          </label>
          <input
            type="range"
            min="1"
            max="32"
            value={numImages}
            onChange={(e) => setNumImages(Number(e.target.value))}
            style={{
              width: '100%',
              height: '8px',
              borderRadius: '8px',
              cursor: 'pointer',
            }}
          />
          <div
            style={{
              display: 'flex',
              justifyContent: 'space-between',
              fontSize: '12px',
              color: '#999',
              marginTop: '5px',
            }}
          >
            <span>1</span>
            <span>16</span>
            <span>32</span>
          </div>
        </div>

        {/* Generate Button */}
        <button
          onClick={handleGenerate}
          disabled={loading || !selectedModel || models.length === 0}
          style={{
            width: '100%',
            padding: '14px 24px',
            fontSize: '16px',
            fontWeight: 'bold',
            backgroundColor:
              loading || !selectedModel || models.length === 0 ? '#ccc' : '#667eea',
            color: 'white',
            border: 'none',
            borderRadius: '8px',
            cursor:
              loading || !selectedModel || models.length === 0 ? 'not-allowed' : 'pointer',
            transition: 'all 0.3s ease',
          }}
        >
          {loading ? 'Generating Images...' : 'Generate Images'}
        </button>
      </div>

      {/* Generated Images Display */}
      {generatedImages.length > 0 && (
        <div
          style={{
            backgroundColor: '#f0fff4',
            border: '2px solid #48bb78',
            borderRadius: '12px',
            padding: '25px',
          }}
        >
          <h3
            style={{
              margin: '0 0 20px 0',
              color: '#48bb78',
              display: 'flex',
              alignItems: 'center',
              gap: '10px',
            }}
          >
            <AiOutlineCheckCircle size={24} />
            Generated {selectedFruit} Images
          </h3>

          <div
            style={{
              display: 'grid',
              gridTemplateColumns: 'repeat(auto-fill, minmax(150px, 1fr))',
              gap: '15px',
            }}
          >
            {generatedImages.map((img, idx) => (
              <div
                key={idx}
                style={{
                  backgroundColor: 'white',
                  border: '2px solid #e0e0e0',
                  borderRadius: '8px',
                  padding: '10px',
                  textAlign: 'center',
                }}
              >
                <img
                  src={`data:image/png;base64,${img}`}
                  alt={`Generated ${selectedFruit} ${idx + 1}`}
                  style={{
                    width: '100%',
                    height: 'auto',
                    borderRadius: '4px',
                    imageRendering: 'crisp-edges',
                  }}
                />
                <div style={{ fontSize: '12px', color: '#999', marginTop: '8px' }}>
                  Image {idx + 1}
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Loading Overlay */}
      {loading && (
        <div
          style={{
            position: 'fixed',
            top: 0,
            left: 0,
            right: 0,
            bottom: 0,
            backgroundColor: 'rgba(0, 0, 0, 0.5)',
            display: 'flex',
            justifyContent: 'center',
            alignItems: 'center',
            zIndex: 1000,
          }}
        >
          <div
            style={{
              backgroundColor: 'white',
              padding: '30px 50px',
              borderRadius: '12px',
              textAlign: 'center',
            }}
          >
            <div
              style={{
                fontSize: '18px',
                fontWeight: 'bold',
                marginBottom: '10px',
                color: '#667eea',
              }}
            >
              Generating {selectedFruit} images...
            </div>
            <div style={{ fontSize: '14px', color: '#666' }}>Please wait</div>
          </div>
        </div>
      )}
    </div>
  );
}

export default ImageGenerator;
