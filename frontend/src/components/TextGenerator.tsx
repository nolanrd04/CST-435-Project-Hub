import React, { useState } from 'react';

function TextGenerator() {
  const [seedText, setSeedText] = useState('');
  const [numWords, setNumWords] = useState(50);
  const [temperature, setTemperature] = useState(1.0);
  const [generatedText, setGeneratedText] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');

  const handleGenerate = async () => {
    if (!seedText.trim()) {
      setError('Please enter some seed text');
      return;
    }

    setLoading(true);
    setError('');
    try {
      const response = await fetch('http://localhost:8000/generate-text', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          seed_text: seedText,
          num_words: numWords,
          temperature: temperature,
        }),
      });

      if (!response.ok) {
        throw new Error('Failed to generate text');
      }

      const data = await response.json();
      setGeneratedText(data.generated_text);
      setError('');
    } catch (err) {
      setError(err instanceof Error ? err.message : 'An error occurred');
      console.error('Error generating text:', err);
    } finally {
      setLoading(false);
    }
  };

  const handleClear = () => {
    setSeedText('');
    setGeneratedText('');
    setError('');
  };

  return (
    <div className="max-w-4xl mx-auto">
      {/* Form Card */}
      <div className="bg-white rounded-xl shadow-lg p-8 mb-8">
        <h2 className="text-3xl font-bold text-gray-800 mb-6">Generate Text</h2>

        {/* Error Message */}
        {error && (
          <div className="mb-6 p-4 bg-red-50 border-l-4 border-red-500 rounded">
            <p className="text-red-700 font-medium">{error}</p>
          </div>
        )}

        {/* Seed Text Section */}
        <div className="mb-6">
          <label className="block text-sm font-semibold text-gray-700 mb-2">
            Seed Text
            <span className="text-red-500 ml-1">*</span>
          </label>
          <textarea
            value={seedText}
            onChange={(e) => setSeedText(e.target.value)}
            placeholder="Enter some starting text to generate from..."
            disabled={loading}
            className="w-full px-4 py-3 border-2 border-gray-300 rounded-lg focus:outline-none focus:border-indigo-500 focus:ring-2 focus:ring-indigo-200 transition-all disabled:bg-gray-100 disabled:cursor-not-allowed resize-none"
            rows={4}
          />
          <p className="text-xs text-gray-500 mt-1">
            Minimum 1 word required
          </p>
        </div>

        {/* Input Controls Section */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mb-8">
          {/* Word Count */}
          <div>
            <label className="block text-sm font-semibold text-gray-700 mb-2">
              Number of Words
            </label>
            <div className="flex items-center gap-3">
              <input
                type="range"
                min="10"
                max="200"
                value={numWords}
                onChange={(e) => setNumWords(Number(e.target.value))}
                disabled={loading}
                className="flex-1 h-2 bg-gray-300 rounded-lg appearance-none cursor-pointer disabled:cursor-not-allowed disabled:opacity-50"
              />
              <input
                type="number"
                min="10"
                max="200"
                value={numWords}
                onChange={(e) => setNumWords(Number(e.target.value))}
                disabled={loading}
                className="w-20 px-3 py-2 border-2 border-gray-300 rounded-lg focus:outline-none focus:border-indigo-500 text-center font-semibold disabled:bg-gray-100 disabled:cursor-not-allowed"
              />
            </div>
          </div>

          {/* Temperature */}
          <div>
            <label className="block text-sm font-semibold text-gray-700 mb-2">
              Temperature <span className="text-gray-500 font-normal">(Creativity)</span>
            </label>
            <div className="flex items-center gap-3">
              <input
                type="range"
                min="0.1"
                max="2.0"
                step="0.1"
                value={temperature}
                onChange={(e) => setTemperature(Number(e.target.value))}
                disabled={loading}
                className="flex-1 h-2 bg-gray-300 rounded-lg appearance-none cursor-pointer disabled:cursor-not-allowed disabled:opacity-50"
              />
              <input
                type="number"
                min="0.1"
                max="2.0"
                step="0.1"
                value={temperature}
                onChange={(e) => setTemperature(Number(e.target.value))}
                disabled={loading}
                className="w-20 px-3 py-2 border-2 border-gray-300 rounded-lg focus:outline-none focus:border-indigo-500 text-center font-semibold disabled:bg-gray-100 disabled:cursor-not-allowed"
              />
            </div>
            <p className="text-xs text-gray-500 mt-1">
              Lower = More predictable | Higher = More creative
            </p>
          </div>
        </div>

        {/* Buttons */}
        <div className="flex gap-4 justify-end">
          <button
            onClick={handleClear}
            disabled={loading || !seedText}
            className="px-6 py-3 border-2 border-gray-300 text-gray-700 font-semibold rounded-lg hover:bg-gray-50 transition-all disabled:opacity-50 disabled:cursor-not-allowed"
          >
            Clear
          </button>
          <button
            onClick={handleGenerate}
            disabled={loading || !seedText.trim()}
            className="px-8 py-3 bg-indigo-600 text-white font-semibold rounded-lg hover:bg-indigo-700 transition-all disabled:bg-gray-400 disabled:cursor-not-allowed flex items-center gap-2"
          >
            {loading ? (
              <>
                <div className="w-5 h-5 border-2 border-white border-t-transparent rounded-full animate-spin"></div>
                Generating...
              </>
            ) : (
              'Generate Text'
            )}
          </button>
        </div>
      </div>

      {/* Output Section */}
      {generatedText && (
        <div className="bg-white rounded-xl shadow-lg p-8 animate-in fade-in duration-300">
          <h3 className="text-2xl font-bold text-gray-800 mb-4">Generated Text</h3>
          <div className="bg-gradient-to-br from-indigo-50 to-purple-50 rounded-lg p-6 border-2 border-indigo-200">
            <p className="text-gray-800 leading-relaxed whitespace-pre-wrap text-base">
              {generatedText}
            </p>
          </div>
          <button
            onClick={() => {
              navigator.clipboard.writeText(generatedText);
              alert('Text copied to clipboard!');
            }}
            className="mt-4 px-4 py-2 bg-indigo-100 text-indigo-700 font-semibold rounded-lg hover:bg-indigo-200 transition-all"
          >
            Copy to Clipboard
          </button>
        </div>
      )}
    </div>
  );
}

export default TextGenerator;