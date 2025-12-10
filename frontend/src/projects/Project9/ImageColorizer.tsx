import React, { useState, useEffect, useRef } from 'react';
import { AiOutlinePicture, AiOutlineWarning, AiOutlineCheckCircle, AiOutlineCloudUpload } from 'react-icons/ai';
import { project9API } from './api.ts';

function ImageColorizer() {
  const [models, setModels] = useState<string[]>([]);
  const [selectedModel, setSelectedModel] = useState<string>('');
  const [uploadedImage, setUploadedImage] = useState<string | null>(null);
  const [grayscalePreview, setGrayscalePreview] = useState<string | null>(null);
  const [colorizedImage, setColorizedImage] = useState<string | null>(null);
  const [loading, setLoading] = useState<boolean>(false);
  const [loadingModels, setLoadingModels] = useState<boolean>(true);
  const [error, setError] = useState<string>('');
  const [success, setSuccess] = useState<string>('');

  const fileInputRef = useRef<HTMLInputElement>(null);

  // Load available models on mount
  useEffect(() => {
    loadModels();
  }, []);

  const loadModels = async () => {
    setLoadingModels(true);
    setError('');
    try {
      const modelList = await project9API.listModels();
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

  const resizeImageTo256x256 = (img: HTMLImageElement): string => {
    const canvas = document.createElement('canvas');
    canvas.width = 256;
    canvas.height = 256;
    const ctx = canvas.getContext('2d')!;

    // Draw image scaled to 256x256
    ctx.drawImage(img, 0, 0, 256, 256);

    return canvas.toDataURL('image/png');
  };

  const convertToGrayscale = (img: HTMLImageElement): string => {
    const canvas = document.createElement('canvas');
    canvas.width = 256;
    canvas.height = 256;
    const ctx = canvas.getContext('2d')!;

    // Draw resized image
    ctx.drawImage(img, 0, 0, 256, 256);

    // Get image data and convert to grayscale
    const imageData = ctx.getImageData(0, 0, 256, 256);
    const data = imageData.data;

    for (let i = 0; i < data.length; i += 4) {
      const avg = (data[i] + data[i + 1] + data[i + 2]) / 3;
      data[i] = avg;     // R
      data[i + 1] = avg; // G
      data[i + 2] = avg; // B
    }

    ctx.putImageData(imageData, 0, 0);
    return canvas.toDataURL('image/png');
  };

  const processFile = (file: File) => {
    // Check file type
    if (!file.type.startsWith('image/')) {
      setError('Please upload an image file');
      return;
    }

    const reader = new FileReader();
    reader.onload = (e) => {
      const img = new Image();
      img.onload = () => {
        // Resize to 256x256 and convert to grayscale
        const resized = resizeImageTo256x256(img);
        const grayscale = convertToGrayscale(img);

        setUploadedImage(resized);
        setGrayscalePreview(grayscale);
        setColorizedImage(null);
        setError('');
        setSuccess('');
      };
      img.src = e.target?.result as string;
    };
    reader.readAsDataURL(file);
  };

  const handleFileUpload = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (!file) return;
    processFile(file);
  };

  const handleDragOver = (event: React.DragEvent<HTMLDivElement>) => {
    event.preventDefault();
    event.stopPropagation();
  };

  const handleDrop = (event: React.DragEvent<HTMLDivElement>) => {
    event.preventDefault();
    event.stopPropagation();

    const file = event.dataTransfer.files?.[0];
    if (file) {
      processFile(file);
    }
  };

  const handleColorize = async () => {
    if (!selectedModel) {
      setError('Please select a model');
      return;
    }

    if (!grayscalePreview) {
      setError('Please upload an image first');
      return;
    }

    setLoading(true);
    setError('');
    setSuccess('');
    setColorizedImage(null);

    try {
      const response = await project9API.colorizeImage({
        model_name: selectedModel,
        image: grayscalePreview.split(',')[1], // Remove data:image/png;base64, prefix
      });

      if (response.success) {
        setColorizedImage(`data:image/png;base64,${response.colorized_image}`);
        setSuccess('Successfully colorized fruit image!');
      } else {
        setError('Failed to colorize image');
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : 'An error occurred while colorizing');
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
          <AiOutlinePicture size={24} />
        </span>
        Fruit Image Colorization
      </h2>

      {/* Error Message */}
      {error && (
        <div
          style={{
            background: '#fee',
            border: '1px solid #fcc',
            borderRadius: '8px',
            padding: '12px 16px',
            marginBottom: '20px',
            display: 'flex',
            alignItems: 'center',
            gap: '10px',
          }}
        >
          <AiOutlineWarning size={20} color="#c33" />
          <span style={{ color: '#c33' }}>{error}</span>
        </div>
      )}

      {/* Success Message */}
      {success && (
        <div
          style={{
            background: '#efe',
            border: '1px solid #cfc',
            borderRadius: '8px',
            padding: '12px 16px',
            marginBottom: '20px',
            display: 'flex',
            alignItems: 'center',
            gap: '10px',
          }}
        >
          <AiOutlineCheckCircle size={20} color="#3c3" />
          <span style={{ color: '#3c3' }}>{success}</span>
        </div>
      )}

      {/* Model Selection */}
      <div style={{ marginBottom: '30px' }}>
        <label
          style={{
            display: 'block',
            marginBottom: '8px',
            fontWeight: 'bold',
            color: '#333',
          }}
        >
          Select Model
        </label>
        {loadingModels ? (
          <div style={{ padding: '10px', color: '#666' }}>Loading models...</div>
        ) : models.length === 0 ? (
          <div style={{ padding: '10px', color: '#c33' }}>
            No models found. Please train a model first using the training script.
          </div>
        ) : (
          <select
            value={selectedModel}
            onChange={(e) => setSelectedModel(e.target.value)}
            style={{
              width: '100%',
              padding: '12px',
              fontSize: '16px',
              borderRadius: '8px',
              border: '2px solid #ddd',
              background: 'white',
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

      {/* Upload Section */}
      <div style={{ marginBottom: '30px' }}>
        <label
          style={{
            display: 'block',
            marginBottom: '8px',
            fontWeight: 'bold',
            color: '#333',
          }}
        >
          Upload Fruit Image
        </label>
        <div
          onClick={() => fileInputRef.current?.click()}
          onDragOver={handleDragOver}
          onDrop={handleDrop}
          style={{
            border: '2px dashed #667eea',
            borderRadius: '12px',
            padding: '40px',
            textAlign: 'center',
            cursor: 'pointer',
            background: '#f9f9ff',
            transition: 'all 0.3s',
          }}
          onMouseEnter={(e) => {
            e.currentTarget.style.background = '#f0f0ff';
            e.currentTarget.style.borderColor = '#764ba2';
          }}
          onMouseLeave={(e) => {
            e.currentTarget.style.background = '#f9f9ff';
            e.currentTarget.style.borderColor = '#667eea';
          }}
        >
          <AiOutlineCloudUpload size={48} color="#667eea" style={{ marginBottom: '10px' }} />
          <div style={{ fontSize: '16px', color: '#667eea', marginBottom: '5px' }}>
            Click to upload or drag and drop a fruit image
          </div>
          <div style={{ fontSize: '14px', color: '#999' }}>
            Image will be automatically resized to 256x256 and converted to grayscale
          </div>
        </div>
        <input
          ref={fileInputRef}
          type="file"
          accept="image/*"
          onChange={handleFileUpload}
          style={{ display: 'none' }}
        />
      </div>

      {/* Image Preview Section */}
      {(uploadedImage || grayscalePreview || colorizedImage) && (
        <div style={{ marginBottom: '30px' }}>
          <div
            style={{
              display: 'grid',
              gridTemplateColumns: 'repeat(auto-fit, minmax(250px, 1fr))',
              gap: '20px',
            }}
          >
            {/* Original Uploaded */}
            {uploadedImage && (
              <div>
                <h4 style={{ marginBottom: '10px', color: '#333' }}>Original (256x256)</h4>
                <div
                  style={{
                    background: '#f5f5f5',
                    borderRadius: '8px',
                    padding: '10px',
                    display: 'flex',
                    justifyContent: 'center',
                    alignItems: 'center',
                  }}
                >
                  <img
                    src={uploadedImage}
                    alt="Original"
                    style={{
                      width: '256px',
                      height: '256px',
                      border: '2px solid #ddd',
                      borderRadius: '4px',
                    }}
                  />
                </div>
              </div>
            )}

            {/* Grayscale */}
            {grayscalePreview && (
              <div>
                <h4 style={{ marginBottom: '10px', color: '#333' }}>Grayscale Input</h4>
                <div
                  style={{
                    background: '#f5f5f5',
                    borderRadius: '8px',
                    padding: '10px',
                    display: 'flex',
                    justifyContent: 'center',
                    alignItems: 'center',
                  }}
                >
                  <img
                    src={grayscalePreview}
                    alt="Grayscale"
                    style={{
                      width: '256px',
                      height: '256px',
                      border: '2px solid #ddd',
                      borderRadius: '4px',
                    }}
                  />
                </div>
              </div>
            )}

            {/* Colorized */}
            {colorizedImage && (
              <div>
                <h4 style={{ marginBottom: '10px', color: '#333' }}>
                  Colorized Output
                </h4>
                <div
                  style={{
                    background: '#f5f5f5',
                    borderRadius: '8px',
                    padding: '10px',
                    display: 'flex',
                    justifyContent: 'center',
                    alignItems: 'center',
                  }}
                >
                  <img
                    src={colorizedImage}
                    alt="Colorized"
                    style={{
                      width: '256px',
                      height: '256px',
                      border: '2px solid #667eea',
                      borderRadius: '4px',
                    }}
                  />
                </div>
              </div>
            )}
          </div>
        </div>
      )}

      {/* Colorize Button */}
      <button
        onClick={handleColorize}
        disabled={loading || !grayscalePreview || !selectedModel}
        style={{
          width: '100%',
          padding: '16px',
          fontSize: '18px',
          fontWeight: 'bold',
          color: 'white',
          background: loading || !grayscalePreview || !selectedModel
            ? '#ccc'
            : 'linear-gradient(135deg, #667eea, #764ba2)',
          border: 'none',
          borderRadius: '12px',
          cursor: loading || !grayscalePreview || !selectedModel ? 'not-allowed' : 'pointer',
          transition: 'all 0.3s',
        }}
        onMouseEnter={(e) => {
          if (!loading && grayscalePreview && selectedModel) {
            e.currentTarget.style.transform = 'translateY(-2px)';
            e.currentTarget.style.boxShadow = '0 6px 20px rgba(102, 126, 234, 0.4)';
          }
        }}
        onMouseLeave={(e) => {
          e.currentTarget.style.transform = 'translateY(0)';
          e.currentTarget.style.boxShadow = 'none';
        }}
      >
        {loading ? 'Colorizing...' : 'Colorize Image'}
      </button>

      {/* Info Section */}
      <div
        style={{
          marginTop: '30px',
          padding: '20px',
          background: '#f9f9ff',
          borderRadius: '12px',
          border: '1px solid #e0e0ff',
        }}
      >
        <h4 style={{ marginTop: 0, color: '#667eea' }}>How it works:</h4>
        <div style={{ color: '#666', lineHeight: '1.8' }}>
          <div>• Upload any fruit image (it will be resized to 256x256 pixels)</div>
          <div>• The image is automatically converted to grayscale</div>
          <div>• Our U-Net model colorizes the fruit using learned patterns</div>
          <div>• Best results with strawberries, oranges, bananas, blackberries, and pineapples</div>
          <div>• The model was trained specifically on fruit images for optimal colorization</div>
        </div>
      </div>
    </div>
  );
}

export default ImageColorizer;
