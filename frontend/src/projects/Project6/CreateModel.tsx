import React, { useState, useEffect } from 'react';
import {
  AiOutlinePlusCircle,
  AiOutlineWarning,
  AiOutlineCheckCircle,
  AiOutlineInfoCircle,
} from 'react-icons/ai';
import { project6API } from './api.ts';
import { getApiUrl } from '../getApiUrl.ts';

const FRUITS = ['apple', 'banana', 'blackberry', 'grape', 'pear', 'strawberry', 'watermelon'];

function CreateModel() {
  // Check if using localhost API
  const apiUrl = getApiUrl();
  const isLocalhost = apiUrl.includes('localhost') || apiUrl.includes('127.0.0.1');

  // Form state
  const [modelName, setModelName] = useState<string>('');
  const [description, setDescription] = useState<string>('');
  const [createNewData, setCreateNewData] = useState<boolean>(false);
  const [selectedDataVersion, setSelectedDataVersion] = useState<string>('');
  const [dataVersions, setDataVersions] = useState<string[]>([]);

  // New data parameters
  const [imageCount, setImageCount] = useState<number>(100);
  const [imageResolution, setImageResolution] = useState<number>(64);
  const [strokeImportance, setStrokeImportance] = useState<number>(5);

  // Training parameters
  const [epochs, setEpochs] = useState<number>(50);
  const [batchSize, setBatchSize] = useState<number>(32);
  const [learningRate, setLearningRate] = useState<number>(0.0002);

  // UI state
  const [loading, setLoading] = useState<boolean>(false);
  const [error, setError] = useState<string>('');
  const [success, setSuccess] = useState<string>('');
  const [warning, setWarning] = useState<string>('');
  const [estimatedTime, setEstimatedTime] = useState<number | null>(null);
  const [showConfirmOverride, setShowConfirmOverride] = useState<boolean>(false);

  useEffect(() => {
    loadDataVersions();
  }, []);

  const loadDataVersions = async () => {
    try {
      const versions = await project6API.listDataVersions();
      setDataVersions(versions);
      if (versions.length > 0) {
        setSelectedDataVersion(versions[0]);
      }
    } catch (err) {
      console.error('Failed to load data versions:', err);
    }
  };

  const calculateEstimatedTime = () => {
    // Rough estimation: ~1 minute per epoch per fruit
    const timePerFruit = epochs * 1; // 1 minute per epoch
    const totalMinutes = timePerFruit * FRUITS.length;
    return totalMinutes;
  };

  const handleSubmit = async () => {
    // Validation
    if (!modelName.trim()) {
      setError('Please enter a model name');
      return;
    }

    if (!description.trim()) {
      setError('Please enter a model description');
      return;
    }

    if (!createNewData && !selectedDataVersion) {
      setError('Please select a data version');
      return;
    }

    setLoading(true);
    setError('');
    setSuccess('');
    setWarning('');

    try {
      const request = {
        model_name: modelName,
        data_version: createNewData ? modelName : selectedDataVersion,
        description: description,
        create_new_data: createNewData,
        override_existing: showConfirmOverride, // Send override flag if user confirmed
        ...(createNewData && {
          image_count: imageCount,
          image_resolution: imageResolution,
          stroke_importance: strokeImportance,
        }),
        epochs: epochs,
        batch_size: batchSize,
        learning_rate: learningRate,
      };

      // Reset confirmation flag after sending
      setShowConfirmOverride(false);

      const response = await project6API.createModel(request);

      if (response.success) {
        setSuccess(response.message);
        if (response.estimated_training_time) {
          setEstimatedTime(response.estimated_training_time);
        }
        // Reset form
        setModelName('');
        setDescription('');
        setCreateNewData(false);
      } else {
        // Check if error is about model existing
        if (response.message.includes('already exists')) {
          setWarning(`Model "${modelName}" already exists. Click "Train Model" again to override.`);
          setShowConfirmOverride(true);
        } else {
          setError(response.message);
        }
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to create model');
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
          <AiOutlinePlusCircle size={24} />
        </span>
        Create New GAN Model
      </h2>

      {/* Localhost Warning */}
      {!isLocalhost && (
        <div
          style={{
            backgroundColor: '#fef3c7',
            border: '2px solid #f59e0b',
            borderRadius: '8px',
            padding: '20px',
            marginBottom: '20px',
            color: '#92400e',
          }}
        >
          <div style={{ display: 'flex', alignItems: 'center', gap: '10px', marginBottom: '10px' }}>
            <AiOutlineWarning size={24} />
            <strong style={{ fontSize: '16px' }}>Training Not Available on Remote Server</strong>
          </div>
          <p style={{ margin: '5px 0', fontSize: '14px', lineHeight: '1.6' }}>
            GAN model training is computationally intensive and requires significant GPU/CPU resources.
            Training can only be performed on a local machine.
          </p>
          <p style={{ margin: '10px 0 0 0', fontSize: '14px', lineHeight: '1.6' }}>
            <strong>To train models:</strong> Switch to localhost API mode in settings or run the backend locally
            at <code style={{ backgroundColor: '#fff', padding: '2px 6px', borderRadius: '4px' }}>http://localhost:8000</code>
          </p>
        </div>
      )}

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

      {/* Warning Message */}
      {warning && (
        <div
          style={{
            backgroundColor: '#fff3cd',
            border: '2px solid #ffc107',
            borderRadius: '8px',
            padding: '15px',
            marginBottom: '20px',
            color: '#856404',
            display: 'flex',
            alignItems: 'center',
            gap: '10px',
          }}
        >
          <AiOutlineWarning size={20} />
          {warning}
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
          }}
        >
          <div style={{ display: 'flex', alignItems: 'center', gap: '10px', marginBottom: '10px' }}>
            <AiOutlineCheckCircle size={20} />
            <strong>{success}</strong>
          </div>
          {estimatedTime && (
            <div style={{ fontSize: '14px', marginLeft: '30px' }}>
              Estimated training time: {Math.floor(estimatedTime / 60)} hours{' '}
              {estimatedTime % 60} minutes
            </div>
          )}
        </div>
      )}

      {/* Model Name */}
      <div
        style={{
          backgroundColor: '#f8f9ff',
          border: '2px solid #667eea',
          borderRadius: '12px',
          padding: '25px',
          marginBottom: '20px',
        }}
      >
        <h3 style={{ margin: '0 0 20px 0', color: '#667eea' }}>Model Configuration</h3>

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
            Model Name (Data Version) <span style={{ color: '#e53e3e' }}>*</span>
          </label>
          <input
            type="text"
            value={modelName}
            onChange={(e) => setModelName(e.target.value)}
            placeholder="e.g., model_v1, my_model, attempt_2"
            style={{
              width: '100%',
              padding: '12px',
              fontSize: '14px',
              border: '2px solid #e0e0e0',
              borderRadius: '8px',
            }}
          />
          <div style={{ fontSize: '12px', color: '#666', marginTop: '5px' }}>
            This name will be used for both the model and data version if creating new data
          </div>
        </div>

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
            Model Description <span style={{ color: '#e53e3e' }}>*</span>
          </label>
          <textarea
            value={description}
            onChange={(e) => setDescription(e.target.value)}
            placeholder="Describe your model (e.g., purpose, expected improvements, etc.)"
            rows={4}
            style={{
              width: '100%',
              padding: '12px',
              fontSize: '14px',
              border: '2px solid #e0e0e0',
              borderRadius: '8px',
              fontFamily: 'inherit',
            }}
          />
        </div>
      </div>

      {/* Data Source */}
      <div
        style={{
          backgroundColor: '#f8f9ff',
          border: '2px solid #667eea',
          borderRadius: '12px',
          padding: '25px',
          marginBottom: '20px',
        }}
      >
        <h3 style={{ margin: '0 0 20px 0', color: '#667eea' }}>Data Source</h3>

        <div style={{ marginBottom: '20px' }}>
          <label style={{ display: 'flex', alignItems: 'center', gap: '10px', cursor: 'pointer' }}>
            <input
              type="radio"
              checked={!createNewData}
              onChange={() => setCreateNewData(false)}
              style={{ width: '20px', height: '20px', cursor: 'pointer' }}
            />
            <span style={{ fontSize: '14px', fontWeight: 'bold' }}>
              Use Existing Data Version
            </span>
          </label>
          {!createNewData && (
            <div style={{ marginLeft: '30px', marginTop: '10px' }}>
              <select
                value={selectedDataVersion}
                onChange={(e) => setSelectedDataVersion(e.target.value)}
                style={{
                  padding: '10px',
                  fontSize: '14px',
                  border: '2px solid #e0e0e0',
                  borderRadius: '8px',
                  cursor: 'pointer',
                  minWidth: '200px',
                }}
              >
                {dataVersions.length === 0 ? (
                  <option value="">No data versions available</option>
                ) : (
                  dataVersions.map((version) => (
                    <option key={version} value={version}>
                      {version}
                    </option>
                  ))
                )}
              </select>
            </div>
          )}
        </div>

        <div>
          <label style={{ display: 'flex', alignItems: 'center', gap: '10px', cursor: 'pointer' }}>
            <input
              type="radio"
              checked={createNewData}
              onChange={() => setCreateNewData(true)}
              style={{ width: '20px', height: '20px', cursor: 'pointer' }}
            />
            <span style={{ fontSize: '14px', fontWeight: 'bold' }}>Create New Data</span>
          </label>
          {createNewData && (
            <div
              style={{
                marginLeft: '30px',
                marginTop: '15px',
                padding: '15px',
                backgroundColor: 'white',
                borderRadius: '8px',
                border: '1px solid #e0e0e0',
              }}
            >
              <div style={{ marginBottom: '15px' }}>
                <label
                  style={{
                    display: 'block',
                    fontSize: '13px',
                    fontWeight: 'bold',
                    marginBottom: '8px',
                  }}
                >
                  Images per Fruit: {imageCount}
                </label>
                <input
                  type="range"
                  min="50"
                  max="500"
                  step="50"
                  value={imageCount}
                  onChange={(e) => setImageCount(Number(e.target.value))}
                  style={{ width: '100%' }}
                />
                <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '11px', color: '#999' }}>
                  <span>50</span>
                  <span>500</span>
                </div>
              </div>

              <div style={{ marginBottom: '15px' }}>
                <label
                  style={{
                    display: 'block',
                    fontSize: '13px',
                    fontWeight: 'bold',
                    marginBottom: '8px',
                  }}
                >
                  Image Resolution: {imageResolution}x{imageResolution}
                </label>
                <select
                  value={imageResolution}
                  onChange={(e) => setImageResolution(Number(e.target.value))}
                  style={{
                    width: '100%',
                    padding: '10px',
                    border: '2px solid #e0e0e0',
                    borderRadius: '8px',
                  }}
                >
                  <option value={32}>32x32 (Fast, Lower Quality)</option>
                  <option value={64}>64x64 (Balanced)</option>
                  <option value={128}>128x128 (Slow, Higher Quality)</option>
                </select>
              </div>

              <div style={{ marginBottom: '15px' }}>
                <label
                  style={{
                    display: 'block',
                    fontSize: '13px',
                    fontWeight: 'bold',
                    marginBottom: '8px',
                  }}
                >
                  Stroke Importance: {strokeImportance}
                </label>
                <input
                  type="range"
                  min="1"
                  max="10"
                  value={strokeImportance}
                  onChange={(e) => setStrokeImportance(Number(e.target.value))}
                  style={{ width: '100%' }}
                />
                <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '11px', color: '#999' }}>
                  <span>1 (Simple)</span>
                  <span>10 (Detailed)</span>
                </div>
              </div>

              <div
                style={{
                  backgroundColor: '#f0f4ff',
                  padding: '10px',
                  borderRadius: '6px',
                  fontSize: '12px',
                  color: '#666',
                  display: 'flex',
                  alignItems: 'start',
                  gap: '8px',
                }}
              >
                <AiOutlineInfoCircle size={16} style={{ marginTop: '2px', flexShrink: 0 }} />
                <div>
                  Data will be created for all 7 fruits automatically. Total images:{' '}
                  {imageCount * FRUITS.length}
                </div>
              </div>
            </div>
          )}
        </div>
      </div>

      {/* Training Parameters */}
      <div
        style={{
          backgroundColor: '#f8f9ff',
          border: '2px solid #667eea',
          borderRadius: '12px',
          padding: '25px',
          marginBottom: '20px',
        }}
      >
        <h3 style={{ margin: '0 0 20px 0', color: '#667eea' }}>Training Parameters</h3>

        <div
          style={{
            display: 'grid',
            gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))',
            gap: '20px',
          }}
        >
          <div>
            <label style={{ display: 'block', fontSize: '13px', fontWeight: 'bold', marginBottom: '8px' }}>
              Epochs
            </label>
            <input
              type="number"
              min="10"
              max="1000"
              value={epochs}
              onChange={(e) => setEpochs(Number(e.target.value))}
              style={{
                width: '100%',
                padding: '10px',
                border: '2px solid #e0e0e0',
                borderRadius: '8px',
              }}
            />
          </div>

          <div>
            <label style={{ display: 'block', fontSize: '13px', fontWeight: 'bold', marginBottom: '8px' }}>
              Batch Size
            </label>
            <input
              type="number"
              min="8"
              max="128"
              value={batchSize}
              onChange={(e) => setBatchSize(Number(e.target.value))}
              style={{
                width: '100%',
                padding: '10px',
                border: '2px solid #e0e0e0',
                borderRadius: '8px',
              }}
            />
          </div>

          <div>
            <label style={{ display: 'block', fontSize: '13px', fontWeight: 'bold', marginBottom: '8px' }}>
              Learning Rate
            </label>
            <input
              type="number"
              min="0.00001"
              max="0.01"
              step="0.00001"
              value={learningRate}
              onChange={(e) => setLearningRate(Number(e.target.value))}
              style={{
                width: '100%',
                padding: '10px',
                border: '2px solid #e0e0e0',
                borderRadius: '8px',
              }}
            />
          </div>
        </div>
      </div>

      {/* Estimated Stats */}
      <div
        style={{
          backgroundColor: '#fffaf0',
          border: '2px solid #ff7b29',
          borderRadius: '12px',
          padding: '20px',
          marginBottom: '20px',
        }}
      >
        <h3 style={{ margin: '0 0 15px 0', color: '#ff7b29' }}>Estimated Training Statistics</h3>
        <div style={{ fontSize: '14px', color: '#666', lineHeight: '1.8' }}>
          <p style={{ margin: '8px 0' }}>
            <strong>Total Fruits:</strong> {FRUITS.length}
          </p>
          <p style={{ margin: '8px 0' }}>
            <strong>Epochs per Fruit:</strong> {epochs}
          </p>
          <p style={{ margin: '8px 0' }}>
            <strong>Estimated Time:</strong> ~{calculateEstimatedTime()} minutes (
            {Math.floor(calculateEstimatedTime() / 60)} hours {calculateEstimatedTime() % 60} minutes)
          </p>
          <p style={{ margin: '8px 0' }}>
            <strong>Note:</strong> Training occurs for all 7 fruits automatically
          </p>
        </div>
      </div>

      {/* Submit Button */}
      <button
        onClick={handleSubmit}
        disabled={loading || !isLocalhost}
        style={{
          width: '100%',
          padding: '16px 24px',
          fontSize: '18px',
          fontWeight: 'bold',
          backgroundColor: loading || !isLocalhost ? '#ccc' : '#667eea',
          color: 'white',
          border: 'none',
          borderRadius: '12px',
          cursor: loading || !isLocalhost ? 'not-allowed' : 'pointer',
          transition: 'all 0.3s ease',
          opacity: !isLocalhost ? 0.6 : 1,
        }}
      >
        {loading ? 'Creating and Training Model...' : !isLocalhost ? 'Training Disabled (Remote Server)' : 'Train Model'}
      </button>

      {!isLocalhost && (
        <p style={{ textAlign: 'center', marginTop: '10px', fontSize: '13px', color: '#666', fontStyle: 'italic' }}>
          Training is only available when connected to a local backend server
        </p>
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
            backgroundColor: 'rgba(0, 0, 0, 0.7)',
            display: 'flex',
            justifyContent: 'center',
            alignItems: 'center',
            zIndex: 1000,
          }}
        >
          <div
            style={{
              backgroundColor: 'white',
              padding: '40px 60px',
              borderRadius: '12px',
              textAlign: 'center',
              maxWidth: '500px',
            }}
          >
            <div style={{ fontSize: '20px', fontWeight: 'bold', marginBottom: '15px', color: '#667eea' }}>
              Training GAN Model
            </div>
            <div style={{ fontSize: '14px', color: '#666', marginBottom: '10px' }}>
              This may take a while. Training all 7 fruits...
            </div>
            <div style={{ fontSize: '12px', color: '#999' }}>
              Estimated time: ~{calculateEstimatedTime()} minutes
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

export default CreateModel;
