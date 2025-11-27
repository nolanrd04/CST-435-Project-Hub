import React, { useState, useRef, useCallback } from 'react';
import {
  AiOutlineCamera,
  AiOutlineFileText,
  AiOutlineDollar,
  AiOutlineYoutube,
  AiOutlineSearch,
  AiOutlineReload,
  AiOutlineClose,
  AiOutlineWarning,
  AiOutlineCheckCircle,
  AiOutlineBarChart,
  AiOutlineDesktop,
  AiOutlineCloud,
  AiOutlineHdd,
  AiOutlineWifi,
  AiOutlineThunderbolt
} from 'react-icons/ai';
import { MdDirectionsCar, MdAirplaneTicket, MdTwoWheeler } from 'react-icons/md';
import { getApiUrl } from '../getApiUrl.ts';

function ImageClassifier({ activeTab: initialTab }: { activeTab?: string }) {
  const [activeTab, setActiveTab] = useState(initialTab || 'classifier');
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [preview, setPreview] = useState<string | null>(null);
  const [classification, setClassification] = useState<any>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [modelSource, setModelSource] = useState<'local' | 'huggingface'>('local');
  const [trainingSummary, setTrainingSummary] = useState<any>(null);
  const [summaryLoading, setSummaryLoading] = useState(false);
  const [summaryError, setSummaryError] = useState('');
  const [isDragging, setIsDragging] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const processFile = (file: File) => {
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
  };

  const handleFileSelect = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (file) {
      processFile(file);
    }
  };

  const handleDragOver = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(true);
  };

  const handleDragEnter = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(true);
  };

  const handleDragLeave = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(false);
  };

  const handleDrop = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(false);

    const files = e.dataTransfer.files;
    if (files && files.length > 0) {
      processFile(files[0]);
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

  const apiUrl = getApiUrl();
  const response = await fetch(`${apiUrl}/classify-image?model_source=${modelSource}`, {
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

  const fetchTrainingSummary = useCallback(async () => {
    setSummaryLoading(true);
    setSummaryError('');
    try {
      const apiUrl = getApiUrl();
      const response = await fetch(`${apiUrl}/project3/training-summary`);

      if (!response.ok) {
        throw new Error('Failed to fetch training summary');
      }

      const result = await response.json();
      setTrainingSummary(result);
    } catch (err) {
      setSummaryError(err instanceof Error ? err.message : 'Failed to load training summary');
      console.error('Error fetching training summary:', err);
    } finally {
      setSummaryLoading(false);
    }
  }, []);

  // Fetch training summary when cost analysis tab is selected
  React.useEffect(() => {
    if (activeTab === 'cost-analysis' && !trainingSummary && !summaryLoading) {
      fetchTrainingSummary();
    }
  }, [activeTab, trainingSummary, summaryLoading, fetchTrainingSummary]);

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

  const getVehicleIcon = (vehicleType: string) => {
    switch (vehicleType.toLowerCase()) {
      case 'car':
        return <MdDirectionsCar size={24} />;
      case 'airplane':
        return <MdAirplaneTicket size={24} />;
      case 'motorbike':
        return <MdTwoWheeler size={24} />;
      default:
        return <MdDirectionsCar size={24} />;
    }
  };

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
          Project 3: Vehicle Image Classifier
        </h1>
        <p style={{ color: '#666', fontSize: '16px' }}>
          CNN-based classifier for cars, airplanes, and motorbikes
        </p>
      </div>

      {/* Tab Navigation */}
      <div style={{
        display: 'flex',
        gap: '10px',
        marginBottom: '30px',
        borderBottom: '2px solid #e5e7eb',
        flexWrap: 'wrap',
        overflowX: 'auto',
      }}>
        <button
          onClick={() => setActiveTab('classifier')}
          style={{
            padding: '12px 20px',
            backgroundColor: activeTab === 'classifier' ? '#667eea' : 'transparent',
            color: activeTab === 'classifier' ? 'white' : '#666',
            border: 'none',
            borderRadius: '8px 8px 0 0',
            cursor: 'pointer',
            fontWeight: activeTab === 'classifier' ? 'bold' : 'normal',
            fontSize: '15px',
            transition: 'all 0.3s ease',
            display: 'flex',
            alignItems: 'center',
            gap: '8px',
            whiteSpace: 'nowrap',
          }}
        >
          <AiOutlineCamera size={20} />
          Vehicle Classifier
        </button>
        <button
          onClick={() => setActiveTab('description')}
          style={{
            padding: '12px 20px',
            backgroundColor: activeTab === 'description' ? '#667eea' : 'transparent',
            color: activeTab === 'description' ? 'white' : '#666',
            border: 'none',
            borderRadius: '8px 8px 0 0',
            cursor: 'pointer',
            fontWeight: activeTab === 'description' ? 'bold' : 'normal',
            fontSize: '15px',
            transition: 'all 0.3s ease',
            display: 'flex',
            alignItems: 'center',
            gap: '8px',
            whiteSpace: 'nowrap',
          }}
        >
          <AiOutlineFileText size={20} />
          Project Description
        </button>
        <button
          onClick={() => setActiveTab('cost-analysis')}
          style={{
            padding: '12px 20px',
            backgroundColor: activeTab === 'cost-analysis' ? '#667eea' : 'transparent',
            color: activeTab === 'cost-analysis' ? 'white' : '#666',
            border: 'none',
            borderRadius: '8px 8px 0 0',
            cursor: 'pointer',
            fontWeight: activeTab === 'cost-analysis' ? 'bold' : 'normal',
            fontSize: '15px',
            transition: 'all 0.3s ease',
            display: 'flex',
            alignItems: 'center',
            gap: '8px',
            whiteSpace: 'nowrap',
          }}
        >
          <AiOutlineDollar size={20} />
          Cost Analysis
        </button>
        <button
          onClick={() => setActiveTab('youtube')}
          style={{
            padding: '12px 20px',
            backgroundColor: activeTab === 'youtube' ? '#667eea' : 'transparent',
            color: activeTab === 'youtube' ? 'white' : '#666',
            border: 'none',
            borderRadius: '8px 8px 0 0',
            cursor: 'pointer',
            fontWeight: activeTab === 'youtube' ? 'bold' : 'normal',
            fontSize: '15px',
            transition: 'all 0.3s ease',
            display: 'flex',
            alignItems: 'center',
            gap: '8px',
            whiteSpace: 'nowrap',
          }}
        >
          <AiOutlineYoutube size={20} />
          YouTube Link
        </button>
      </div>

      {/* Tab Content */}
      <div style={{ minHeight: '500px' }}>
        {activeTab === 'classifier' && (
        <div>
          <h2 className="title" style={{ marginBottom: '20px', display: 'flex', alignItems: 'center', gap: '8px' }}>
            <AiOutlineCamera size={28} />
            Vehicle Image Classifier
          </h2>
        </div>
      )}



      {/*}
      <div style={{
        backgroundColor: '#ffebebff',
        border: '2px solid #f56565',
        borderRadius: '12px',
        padding: '16px',
        marginBottom: '25px',
        color: '#d13939ff',
        lineHeight: '1.6',
        fontSize: '14px'
      }}>
        <p style={{ margin: 0 }}>
          WARNING: Model is too large to run on the cloud. Run a local API to use this project.
        </p>
      </div>*/}

      {activeTab === 'youtube' && (
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
                background: 'linear-gradient(135deg, #FF0000, #CC0000)',
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
              <AiOutlineYoutube size={24} />
            </span>
            YouTube Showcase
          </h2>

          <div
            style={{
              backgroundColor: '#f8f9ff',
              border: '2px solid #667eea',
              borderRadius: '12px',
              padding: '40px',
              maxWidth: '900px',
              margin: '0 auto',
            }}
          >
            {/* YouTube Embed */}
            <div
              style={{
                position: 'relative',
                width: '100%',
                paddingBottom: '56.25%', // 16:9 aspect ratio
                height: 0,
                overflow: 'hidden',
                borderRadius: '8px',
                boxShadow: '0 4px 6px rgba(0, 0, 0, 0.1)',
              }}
            >
              <iframe
                src="https://www.youtube.com/embed/1YkOgaKalUI"
                title="Project 3: Vehicle Image Classifier Demonstration"
                style={{
                  position: 'absolute',
                  top: 0,
                  left: 0,
                  width: '100%',
                  height: '100%',
                  border: 'none',
                  borderRadius: '8px',
                }}
                allowFullScreen
                allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture"
              />
            </div>
          </div>
        </div>
      )}

      {activeTab === 'description' && (
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
      )}

      {activeTab === 'cost-analysis' && (
        <div>
          <h2 style={{ marginBottom: '20px', display: 'flex', alignItems: 'center', gap: '10px' }}>
            <AiOutlineDollar size={28} />
            Training Cost Analysis
          </h2>

          {summaryError && (
            <div style={{
              backgroundColor: '#fee2e2',
              border: '1px solid #fca5a5',
              borderRadius: '8px',
              padding: '12px',
              marginBottom: '20px',
              color: '#dc2626',
              display: 'flex',
              alignItems: 'center',
              gap: '10px'
            }}>
              <AiOutlineWarning size={20} />
              {summaryError}
            </div>
          )}

          {summaryLoading ? (
            <div style={{
              textAlign: 'center',
              padding: '40px',
              backgroundColor: '#f8fafc',
              borderRadius: '12px',
              border: '2px solid #e2e8f0'
            }}>
              <div style={{ fontSize: '24px', marginBottom: '10px' }}>⏳</div>
              <p>Loading training cost analysis...</p>
            </div>
          ) : trainingSummary ? (
            trainingSummary.has_training_data ? (
              <div>
                {/* Training Overview */}
                <div style={{
                  backgroundColor: '#f0f9ff',
                  border: '2px solid #0ea5e9',
                  borderRadius: '12px',
                  padding: '20px',
                  marginBottom: '20px'
                }}>
                  <h3 style={{ marginTop: 0, color: '#0c4a6e', display: 'flex', alignItems: 'center', gap: '8px' }}>
                    <AiOutlineBarChart size={24} />
                    Training Overview
                  </h3>
                  <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', gap: '15px' }}>
                    <div style={{ textAlign: 'center' }}>
                      <div style={{ fontSize: '24px', fontWeight: 'bold', color: '#0369a1' }}>
                        {trainingSummary.training_summary.actual_training_time_formatted}
                      </div>
                      <div style={{ fontSize: '12px', color: '#64748b' }}>Training Time</div>
                    </div>
                    <div style={{ textAlign: 'center' }}>
                      <div style={{ fontSize: '24px', fontWeight: 'bold', color: '#0369a1' }}>
                        {trainingSummary.training_summary.peak_memory_usage_formatted}
                      </div>
                      <div style={{ fontSize: '12px', color: '#64748b' }}>Peak Memory</div>
                    </div>
                    <div style={{ textAlign: 'center' }}>
                      <div style={{ fontSize: '24px', fontWeight: 'bold', color: '#0369a1' }}>
                        {trainingSummary.training_summary.total_training_cost_formatted}
                      </div>
                      <div style={{ fontSize: '12px', color: '#64748b' }}>Total Cost</div>
                    </div>
                  </div>
                </div>

                {/* Cost Breakdown */}
                <div style={{
                  backgroundColor: '#f0fdf4',
                  border: '2px solid #22c55e',
                  borderRadius: '12px',
                  padding: '20px',
                  marginBottom: '20px'
                }}>
                  <h3 style={{ marginTop: 0, color: '#166534', display: 'flex', alignItems: 'center', gap: '8px' }}>
                    <AiOutlineDollar size={24} />
                    Cost Breakdown
                  </h3>
                  <div style={{ display: 'grid', gap: '12px' }}>
                    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                      <span style={{ fontWeight: '500', display: 'flex', alignItems: 'center', gap: '8px' }}>
                        <AiOutlineDesktop size={16} />
                        Compute:
                      </span>
                      <span style={{ fontWeight: 'bold', color: '#166534' }}>
                        {trainingSummary.training_summary.cost_breakdown.compute_formatted}
                      </span>
                    </div>
                    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                      <span style={{ fontWeight: '500', display: 'flex', alignItems: 'center', gap: '8px' }}>
                        <AiOutlineCloud size={16} />
                        Memory:
                      </span>
                      <span style={{ fontWeight: 'bold', color: '#166534' }}>
                        {trainingSummary.training_summary.cost_breakdown.memory_formatted}
                      </span>
                    </div>
                    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                      <span style={{ fontWeight: '500', display: 'flex', alignItems: 'center', gap: '8px' }}>
                        <AiOutlineHdd size={16} />
                        Storage:
                      </span>
                      <span style={{ fontWeight: 'bold', color: '#166534' }}>
                        {trainingSummary.training_summary.cost_breakdown.storage_formatted}
                      </span>
                    </div>
                    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                      <span style={{ fontWeight: '500', display: 'flex', alignItems: 'center', gap: '8px' }}>
                        <AiOutlineWifi size={16} />
                        Data Transfer:
                      </span>
                      <span style={{ fontWeight: 'bold', color: '#166534' }}>
                        {trainingSummary.training_summary.cost_breakdown.data_transfer_formatted}
                      </span>
                    </div>
                    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                      <span style={{ fontWeight: '500', display: 'flex', alignItems: 'center', gap: '8px' }}>
                        <AiOutlineThunderbolt size={16} />
                        GPU:
                      </span>
                      <span style={{ fontWeight: 'bold', color: '#166534' }}>
                        {trainingSummary.training_summary.cost_breakdown.gpu_formatted}
                      </span>
                    </div>
                    <hr style={{ border: 'none', borderTop: '1px solid #bbf7d0', margin: '8px 0' }} />
                    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                      <span style={{ fontWeight: 'bold', fontSize: '16px', display: 'flex', alignItems: 'center', gap: '8px' }}>
                        <AiOutlineDollar size={16} />
                        Total Cost:
                      </span>
                      <span style={{ fontWeight: 'bold', fontSize: '16px', color: '#166534' }}>
                        {trainingSummary.training_summary.total_training_cost_formatted}
                      </span>
                    </div>
                  </div>
                </div>

                {/* Additional Metrics */}
                <div style={{
                  backgroundColor: '#fef3c7',
                  border: '2px solid #f59e0b',
                  borderRadius: '12px',
                  padding: '20px'
                }}>
                  <h3 style={{ marginTop: 0, color: '#92400e', display: 'flex', alignItems: 'center', gap: '8px' }}>
                    <AiOutlineBarChart size={24} />
                    Additional Metrics
                  </h3>
                  <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', gap: '15px' }}>
                    <div>
                      <div style={{ fontSize: '18px', fontWeight: 'bold', color: '#92400e' }}>
                        {trainingSummary.training_summary.cost_per_epoch_formatted}
                      </div>
                      <div style={{ fontSize: '12px', color: '#a16207' }}>Cost per Epoch</div>
                    </div>
                    <div>
                      <div style={{ fontSize: '18px', fontWeight: 'bold', color: '#92400e' }}>
                        {new Date(trainingSummary.training_summary.timestamp * 1000).toLocaleString()}
                      </div>
                      <div style={{ fontSize: '12px', color: '#a16207' }}>Training Completed</div>
                    </div>
                  </div>
                </div>

                {/* Refresh Button */}
                <div style={{ textAlign: 'center', marginTop: '20px' }}>
                  <button
                    onClick={fetchTrainingSummary}
                    disabled={summaryLoading}
                    style={{
                      padding: '10px 20px',
                      backgroundColor: summaryLoading ? '#cbd5e1' : '#3b82f6',
                      color: 'white',
                      border: 'none',
                      borderRadius: '8px',
                      cursor: summaryLoading ? 'not-allowed' : 'pointer',
                      fontSize: '14px',
                      display: 'flex',
                      alignItems: 'center',
                      gap: '8px',
                      margin: '0 auto'
                    }}
                  >
                    <AiOutlineReload size={16} className={summaryLoading ? 'animate-spin' : ''} />
                    {summaryLoading ? 'Refreshing...' : 'Refresh Data'}
                  </button>
                </div>
              </div>
            ) : (
              <div style={{
                textAlign: 'center',
                padding: '40px',
                backgroundColor: '#f8fafc',
                borderRadius: '12px',
                border: '2px solid #e2e8f0'
              }}>
                <div style={{ fontSize: '48px', marginBottom: '20px', display: 'flex', justifyContent: 'center' }}>
                  <AiOutlineBarChart size={48} />
                </div>
                <h3 style={{ color: '#64748b', marginBottom: '10px' }}>No Training Data Available</h3>
                <p style={{ color: '#94a3b8', marginBottom: '20px' }}>
                  {trainingSummary.message}
                </p>
                <button
                  onClick={fetchTrainingSummary}
                  style={{
                    padding: '10px 20px',
                    backgroundColor: '#3b82f6',
                    color: 'white',
                    border: 'none',
                    borderRadius: '8px',
                    cursor: 'pointer',
                    fontSize: '14px',
                    display: 'flex',
                    alignItems: 'center',
                    gap: '8px',
                    margin: '0 auto'
                  }}
                >
                  <AiOutlineReload size={16} />
                  Try Again
                </button>
              </div>
            )
          ) : null}
        </div>
      )}

      {/* Error Message */}
      {activeTab === 'classifier' && error && (
        <div className="error" style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
          <AiOutlineWarning size={20} />
          {error}
        </div>
      )}

      {/* Upload Area */}
      {activeTab === 'classifier' && (
      <div
        className="form-group"
        onClick={() => fileInputRef.current?.click()}
        onDragOver={handleDragOver}
        onDragEnter={handleDragEnter}
        onDragLeave={handleDragLeave}
        onDrop={handleDrop}
        style={{
          border: isDragging ? '3px dashed #4c51bf' : '3px dashed #667eea',
          borderRadius: '12px',
          padding: '40px 20px',
          textAlign: 'center',
          cursor: 'pointer',
          backgroundColor: isDragging ? '#e6fffa' : '#f7fafc',
          transition: 'all 0.3s ease',
        }}
        onMouseOver={(e) => {
          if (!isDragging) {
            (e.currentTarget as HTMLDivElement).style.backgroundColor = '#edf2f7';
          }
        }}
        onMouseOut={(e) => {
          if (!isDragging) {
            (e.currentTarget as HTMLDivElement).style.backgroundColor = '#f7fafc';
          }
        }}
      >
        <input
          ref={fileInputRef}
          type="file"
          accept="image/png,image/jpeg,image/jpg"
          onChange={handleFileSelect}
          style={{ display: 'none' }}
        />
        <div style={{ marginBottom: '10px', display: 'flex', justifyContent: 'center' }}>
          <AiOutlineCamera size={48} />
        </div>
        <p style={{ fontSize: '16px', fontWeight: 'bold', marginBottom: '5px' }}>
          Click to upload or drag and drop
        </p>
        <p style={{ fontSize: '12px', color: '#718096' }}>
          PNG or JPG (Max 5MB)
        </p>
      </div>
      )}

      {/* Preview Section */}
      {activeTab === 'classifier' && preview && (
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
      {activeTab === 'classifier' && (
      <div style={{ display: 'flex', gap: '10px', justifyContent: 'flex-end', marginTop: '20px' }}>
        <button
          onClick={handleClear}
          disabled={loading || !selectedFile}
          className="button secondary"
          style={{ display: 'flex', alignItems: 'center', gap: '8px' }}
        >
          <AiOutlineClose size={18} />
          Clear
        </button>
        <button
          onClick={handleClassify}
          disabled={loading || !selectedFile}
          className="button"
          style={{
            opacity: loading || !selectedFile ? 0.5 : 1,
            cursor: loading || !selectedFile ? 'not-allowed' : 'pointer',
            display: 'flex',
            alignItems: 'center',
            gap: '8px'
          }}
        >
          {loading ? (
            <>
              <AiOutlineReload size={18} className="animate-spin" />
              Classifying...
            </>
          ) : (
            <>
              <AiOutlineSearch size={18} />
              Classify
            </>
          )}
        </button>
      </div>
      )}

      {/* Classification Results */}
      {activeTab === 'classifier' && classification && (
        <div className="output">
          <h3 style={{ marginBottom: '20px', display: 'flex', alignItems: 'center', gap: '10px' }}>
            <AiOutlineCheckCircle size={24} style={{ color: '#22c55e' }} />
            Classification Result
          </h3>

          {/* Main Prediction */}
          <div className="generated-text" style={{
            textAlign: 'center',
            backgroundColor: '#f7fafc',
            borderColor: '#c7d2fe',
            marginBottom: '20px'
          }}>
            <div style={{ marginBottom: '15px', display: 'flex', justifyContent: 'center' }}>
              <div style={{ fontSize: '60px' }}>
                {getVehicleIcon(classification.predicted_class)}
              </div>
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
                      <span style={{ fontWeight: '600', color: '#2d3748', textTransform: 'capitalize', display: 'flex', alignItems: 'center', gap: '8px' }}>
                        {getVehicleIcon(className)}
                        {className}
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

          {/* Model Source Info Box */}
          <div style={{
            backgroundColor: '#dbeafe',
            borderLeft: '4px solid #3b82f6',
            borderRadius: '8px',
            padding: '12px',
            marginBottom: '15px',
            fontSize: '13px',
            color: '#1e40af'
          }}>
            <p style={{ margin: '0 0 8px 0' }}>
              <strong>Model Source:</strong> {classification.model_source === 'local' ? 'Local (Trained)' : 'HuggingFace (Pre-trained)'}
            </p>
            <p style={{ margin: 0 }}>
              <strong>Note:</strong> This model was trained on older vehicle images (cars, airplanes, and motorbikes).
              It may perform better with images from similar time periods.
            </p>
          </div>

          {/* Action Button */}
          <button
            onClick={handleClear}
            className="button secondary"
            style={{ width: '100%', display: 'flex', alignItems: 'center', justifyContent: 'center', gap: '8px' }}
          >
            <AiOutlineReload size={18} />
            Classify Another
          </button>
        </div>
      )}
      </div>
    </div>
  );
}

export default ImageClassifier;
