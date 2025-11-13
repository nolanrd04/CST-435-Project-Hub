import React, { useState, useRef } from 'react';
import { getApiUrl } from '../projects/getApiUrl.ts';

function ImageClassifier() {
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [preview, setPreview] = useState<string | null>(null);
  const [classification, setClassification] = useState<any>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [modelSource, setModelSource] = useState<'local' | 'huggingface'>('local');
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
    <div className="max-w-3xl mx-auto">
      {/* Upload Card */}
      <div className="bg-white rounded-2xl shadow-2xl p-8 mb-8">
        <h2 className="text-3xl font-bold text-gray-800 mb-8 flex items-center gap-3">
          <span className="bg-gradient-to-r from-indigo-600 to-purple-600 text-white w-10 h-10 rounded-full flex items-center justify-center text-lg">
            🖼️
          </span>
          Vehicle Image Classifier
        </h2>

        {/* Model Selection Toggle */}
        <div className="mb-8 p-4 bg-blue-50 border-2 border-blue-200 rounded-xl">
          <p className="text-sm font-bold text-blue-900 mb-3">🤖 Model Selection</p>
          <div className="flex gap-6">
            <label className="flex items-center gap-3 cursor-pointer">
              <input
                type="radio"
                value="local"
                checked={modelSource === 'local'}
                onChange={(e) => setModelSource(e.target.value as 'local')}
                disabled={loading}
                className="w-5 h-5 cursor-pointer"
              />
              <span className={`font-medium ${modelSource === 'local' ? 'text-blue-900 font-bold' : 'text-blue-700'}`}>
                Local Model (Trained)
              </span>
            </label>
            <label className="flex items-center gap-3 cursor-pointer">
              <input
                type="radio"
                value="huggingface"
                checked={modelSource === 'huggingface'}
                onChange={(e) => setModelSource(e.target.value as 'huggingface')}
                disabled={loading}
                className="w-5 h-5 cursor-pointer"
              />
              <span className={`font-medium ${modelSource === 'huggingface' ? 'text-blue-900 font-bold' : 'text-blue-700'}`}>
                HuggingFace (cloud) Model
              </span>
            </label>
          </div>
          <p className="text-xs text-blue-800 mt-2">
            {modelSource === 'local' 
              ? '✓ Using your locally trained model' 
              : '✓ Using pre-trained model from HuggingFace'}
          </p>
        </div>

        {/* Error Message */}
        {error && (
          <div className="mb-6 p-4 bg-red-50 border-l-4 border-red-500 rounded-lg">
            <p className="text-red-700 font-medium">⚠️ {error}</p>
          </div>
        )}

        {/* Upload Area */}
        <div className="mb-8">
          <div
            onClick={() => fileInputRef.current?.click()}
            className="border-3 border-dashed border-indigo-300 rounded-xl p-8 text-center cursor-pointer hover:bg-indigo-50 transition-colors"
          >
            <input
              ref={fileInputRef}
              type="file"
              accept="image/png,image/jpeg,image/jpg"
              onChange={handleFileSelect}
              className="hidden"
            />
            <div className="text-5xl mb-3">📸</div>
            <p className="text-lg font-semibold text-gray-800 mb-2">
              Click to upload or drag and drop
            </p>
            <p className="text-sm text-gray-600">
              PNG or JPG (Max 5MB)
            </p>
          </div>
        </div>

        {/* Preview Section */}
        {preview && (
          <div className="mb-8">
            <p className="text-sm font-semibold text-gray-700 mb-3">Preview:</p>
            <div className="bg-gray-100 rounded-xl overflow-hidden border-2 border-gray-200">
              <img
                src={preview}
                alt="Preview"
                className="w-full max-h-64 object-contain"
              />
            </div>
            <p className="text-xs text-gray-600 mt-2">
              {selectedFile?.name} ({((selectedFile?.size || 0) / 1024 / 1024).toFixed(2)} MB)
            </p>
          </div>
        )}

        {/* Action Buttons */}
        <div className="flex gap-4 justify-end">
          <button
            onClick={handleClear}
            disabled={loading || !selectedFile}
            className="px-6 py-3 border-2 border-gray-300 text-gray-700 font-semibold rounded-lg hover:bg-gray-50 transition-all disabled:opacity-50 disabled:cursor-not-allowed hover:shadow-md"
          >
            ✕ Clear
          </button>
          <button
            onClick={handleClassify}
            disabled={loading || !selectedFile}
            className="px-8 py-3 bg-gradient-to-r from-indigo-600 to-purple-600 text-white font-semibold rounded-lg hover:shadow-lg transition-all disabled:bg-gray-400 disabled:cursor-not-allowed disabled:shadow-none flex items-center gap-2 hover:scale-105 active:scale-95"
          >
            {loading ? (
              <>
                <div className="w-5 h-5 border-2 border-white border-t-transparent rounded-full animate-spin"></div>
                Classifying...
              </>
            ) : (
              <>
                <span>🔍</span>
                Classify
              </>
            )}
          </button>
        </div>
      </div>

      {/* Classification Results */}
      {classification && (
        <div className="bg-white rounded-2xl shadow-2xl p-8 animate-in fade-in duration-300">
          <div className="flex items-center justify-between mb-8">
            <h3 className="text-2xl font-bold text-gray-800 flex items-center gap-2">
              <span className="text-3xl">✓</span>
              Classification Result
            </h3>
            <span className="text-xs bg-green-100 text-green-700 px-3 py-1 rounded-full font-semibold">
              ✓ Complete
            </span>
          </div>

          {/* Main Prediction */}
          <div className="bg-gradient-to-br from-indigo-50 via-purple-50 to-pink-50 rounded-xl p-8 mb-8 border-2 border-indigo-200">
            <div className="text-center">
              <div className="text-7xl mb-4">
                {getVehicleEmoji(classification.predicted_class)}
              </div>
              <h4 className="text-4xl font-bold text-gray-800 mb-3 capitalize">
                {classification.predicted_class}
              </h4>
              <div className={`inline-block px-4 py-2 rounded-full font-semibold ${getConfidenceColor(classification.confidence)}`}>
                Confidence: {(classification.confidence * 100).toFixed(1)}%
              </div>
            </div>
          </div>

          {/* Class Probabilities */}
          <div className="mb-8">
            <h5 className="text-lg font-bold text-gray-800 mb-4">Class Probabilities:</h5>
            <div className="space-y-3">
              {Object.entries(classification.class_probabilities).map(
                ([className, probability]: [string, any]) => (
                  <div key={className}>
                    <div className="flex items-center justify-between mb-1">
                      <span className="font-semibold text-gray-700 capitalize">
                        {getVehicleEmoji(className)} {className}
                      </span>
                      <span className="text-sm font-bold text-gray-600">
                        {(probability * 100).toFixed(1)}%
                      </span>
                    </div>
                    <div className="w-full bg-gray-200 rounded-full h-2 overflow-hidden">
                      <div
                        className="bg-gradient-to-r from-indigo-600 to-purple-600 h-full rounded-full transition-all duration-500"
                        style={{
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
          <div className="bg-blue-50 border-l-4 border-blue-500 rounded-lg p-4 mb-6">
            <p className="text-blue-700 text-sm">
              <strong>Note:</strong> This model was trained on older vehicle images (cars, airplanes, and motorbikes).
              It may perform better with images from similar time periods and may struggle with modern vehicle designs.
            </p>
          </div>

          {/* Action Buttons */}
          <div className="flex gap-3">
            <button
              onClick={handleClear}
              className="px-4 py-3 bg-indigo-100 text-indigo-700 font-semibold rounded-lg hover:bg-indigo-200 transition-all flex items-center gap-2 hover:shadow-md"
            >
              🔄 Classify Another
            </button>
          </div>
        </div>
      )}
    </div>
  );
}

export default ImageClassifier;
