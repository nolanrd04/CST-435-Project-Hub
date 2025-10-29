import React, { useState, useRef } from 'react';

function ImageClassifier() {
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [preview, setPreview] = useState<string | null>(null);
  const [classification, setClassification] = useState<any>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleFileSelect = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (file) {
      // Validate file type
      if (!['image/png', 'image/jpeg', 'image/jpg'].includes(file.type)) {
        setError('Please upload a PNG or JPG image');
        return;
      }

      setSelectedFile(file);
      setError('');

      // Create preview
      const reader = new FileReader();
      reader.onload = (e) => {
        setPreview(e.target?.result as string);
      };
      reader.readAsDataURL(file);
    }
  };

  const handleClassify = async () => {
    if (!selectedFile) {
      setError('Please select an image first');
      return;
    }

    setLoading(true);
    setError('');
    try {
      const formData = new FormData();
      formData.append('file', selectedFile);

      const apiMode = localStorage.getItem('API_MODE');
      const apiUrl = apiMode === 'local' ? 'http://localhost:8000' : 'https://cst-435-project-hub.onrender.com';
      const response = await fetch(`${apiUrl}/classify-image`, {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        throw new Error('Failed to classify image');
      }

      const result = await response.json();
      setClassification(result);
      setError('');
    } catch (err) {
      setError(err instanceof Error ? err.message : 'An error occurred');
      console.error('Error classifying image:', err);
    } finally {
      setLoading(false);
    }
  };

  const handleClear = () => {
    setSelectedFile(null);
    setPreview(null);
    setClassification(null);
    setError('');
    if (fileInputRef.current) {
      fileInputRef.current.value = '';
    }
  };

  const getConfidenceColor = (confidence: number) => {
    if (confidence >= 0.8) return 'bg-green-100 text-green-700';
    if (confidence >= 0.6) return 'bg-yellow-100 text-yellow-700';
    return 'bg-orange-100 text-orange-700';
  };

  const getVehicleEmoji = (vehicleType: string) => {
    switch (vehicleType.toLowerCase()) {
      case 'car':
        return '🚗';
      case 'airplane':
        return '✈️';
      case 'motorbike':
        return '🏍️';
      default:
        return '🚙';
    }
  };

  return (
    <div className="form">
      <h2 className="title" style={{ marginBottom: '20px' }}>🖼️ Vehicle Image Classifier</h2>

      {/* Project Description */}
      <div style={{
        backgroundColor: '#fff5f0',
        border: '2px solid #ea580c',
        borderRadius: '12px',
        padding: '16px',
        marginBottom: '25px',
        color: '#2d3748',
        lineHeight: '1.6',
        fontSize: '14px'
      }}>
        <p style={{ margin: 0 }}>
          <strong>About this project:</strong> This CNN (Convolutional Neural Network) classifier is trained to identify three types of vehicles:
          cars, airplanes, and motorbikes. The model uses convolutional layers to extract visual features, max pooling to reduce dimensionality,
          and dense layers for classification. Upload any vehicle image to get instant predictions with confidence scores. Please note all though
          there is a lot of training data, the data is relatively old so newer styles of vehicles may give strange results.
        </p>
      </div>

      {/* Error Message */}
      {error && (
        <div className="error">⚠️ {error}</div>
      )}

      {/* Upload Area */}
      <div
        className="form-group"
        onClick={() => fileInputRef.current?.click()}
        style={{
          border: '3px dashed #667eea',
          borderRadius: '12px',
          padding: '40px 20px',
          textAlign: 'center',
          cursor: 'pointer',
          backgroundColor: '#f7fafc',
          transition: 'all 0.3s ease',
        }}
        onMouseOver={(e) => {
          (e.currentTarget as HTMLDivElement).style.backgroundColor = '#edf2f7';
        }}
        onMouseOut={(e) => {
          (e.currentTarget as HTMLDivElement).style.backgroundColor = '#f7fafc';
        }}
      >
        <input
          ref={fileInputRef}
          type="file"
          accept="image/png,image/jpeg,image/jpg"
          onChange={handleFileSelect}
          style={{ display: 'none' }}
        />
        <div style={{ fontSize: '40px', marginBottom: '10px' }}>📸</div>
        <p style={{ fontSize: '16px', fontWeight: 'bold', marginBottom: '5px' }}>
          Click to upload or drag and drop
        </p>
        <p style={{ fontSize: '12px', color: '#718096' }}>
          PNG or JPG (Max 5MB)
        </p>
      </div>

      {/* Preview Section */}
      {preview && (
        <div className="form-group">
          <label>Preview:</label>
          <div style={{
            backgroundColor: '#f7fafc',
            borderRadius: '12px',
            overflow: 'hidden',
            border: '2px solid #e2e8f0',
            marginBottom: '10px'
          }}>
            <img
              src={preview}
              alt="Preview"
              style={{
                width: '100%',
                maxHeight: '300px',
                objectFit: 'contain'
              }}
            />
          </div>
          <p style={{ fontSize: '12px', color: '#718096', marginTop: '5px' }}>
            {selectedFile?.name} ({((selectedFile?.size || 0) / 1024 / 1024).toFixed(2)}) MB
          </p>
        </div>
      )}

      {/* Action Buttons */}
      <div style={{ display: 'flex', gap: '10px', justifyContent: 'flex-end', marginTop: '20px' }}>
        <button
          onClick={handleClear}
          disabled={loading || !selectedFile}
          className="button secondary"
        >
          ✕ Clear
        </button>
        <button
          onClick={handleClassify}
          disabled={loading || !selectedFile}
          className="button"
          style={{
            opacity: loading || !selectedFile ? 0.5 : 1,
            cursor: loading || !selectedFile ? 'not-allowed' : 'pointer',
          }}
        >
          {loading ? (
            <>
              <span style={{ display: 'inline-block', marginRight: '5px' }}>⏳</span>
              Classifying...
            </>
          ) : (
            <>
              <span style={{ display: 'inline-block', marginRight: '5px' }}>🔍</span>
              Classify
            </>
          )}
        </button>
      </div>

      {/* Classification Results */}
      {classification && (
        <div className="output">
          <h3 style={{ marginBottom: '20px' }}>✓ Classification Result</h3>

          {/* Main Prediction */}
          <div className="generated-text" style={{
            textAlign: 'center',
            backgroundColor: '#f7fafc',
            borderColor: '#c7d2fe',
            marginBottom: '20px'
          }}>
            <div style={{ fontSize: '60px', marginBottom: '15px' }}>
              {getVehicleEmoji(classification.predicted_class)}
            </div>
            <h4 style={{
              fontSize: '32px',
              fontWeight: 'bold',
              color: '#2d3748',
              marginBottom: '15px',
              textTransform: 'capitalize'
            }}>
              {classification.predicted_class}
            </h4>
            <div style={{
              display: 'inline-block',
              padding: '8px 16px',
              borderRadius: '20px',
              fontWeight: 'bold',
              fontSize: '14px',
            }}
            className={getConfidenceColor(classification.confidence)}
            >
              Confidence: {(classification.confidence * 100).toFixed(1)}%
            </div>
          </div>

          {/* Class Probabilities */}
          <div style={{ marginBottom: '20px' }}>
            <h5 style={{ fontSize: '16px', fontWeight: 'bold', color: '#2d3748', marginBottom: '15px' }}>
              Class Probabilities:
            </h5>
            <div style={{ display: 'grid', gap: '15px' }}>
              {Object.entries(classification.class_probabilities).map(
                ([className, probability]: [string, any]) => (
                  <div key={className}>
                    <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '5px' }}>
                      <span style={{ fontWeight: '600', color: '#2d3748', textTransform: 'capitalize' }}>
                        {getVehicleEmoji(className)} {className}
                      </span>
                      <span style={{ fontSize: '12px', fontWeight: 'bold', color: '#4a5568' }}>
                        {(probability * 100).toFixed(1)}%
                      </span>
                    </div>
                    <div style={{
                      width: '100%',
                      backgroundColor: '#e2e8f0',
                      borderRadius: '8px',
                      height: '8px',
                      overflow: 'hidden'
                    }}>
                      <div
                        style={{
                          background: 'linear-gradient(to right, #667eea, #764ba2)',
                          height: '100%',
                          borderRadius: '8px',
                          transition: 'width 0.5s ease',
                          width: `${probability * 100}%`,
                        }}
                      ></div>
                    </div>
                  </div>
                )
              )}
            </div>
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
            <p>
              <strong>Note:</strong> This model was trained on older vehicle images (cars, airplanes, and motorbikes).
              It may perform better with images from similar time periods.
            </p>
          </div>

          {/* Action Button */}
          <button
            onClick={handleClear}
            className="button secondary"
            style={{ width: '100%' }}
          >
            🔄 Classify Another
          </button>
        </div>
      )}
    </div>
  );
}

export default ImageClassifier;
