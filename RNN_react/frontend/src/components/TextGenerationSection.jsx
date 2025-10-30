import { useState, useEffect } from 'react';
import { Sparkles, AlertCircle, Loader2 } from 'lucide-react';
import { generateText } from '../services/api';

/**
 * TextGenerationSection Component
 * Handles text generation input and displays results
 */
function TextGenerationSection({ lstmStatus, selectedSeedText, onGenerate }) {
  const [seedText, setSeedText] = useState('');
  const [numWords, setNumWords] = useState(30);
  const [temperature, setTemperature] = useState(0.7);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);

  // Update seed text when example is selected
  useEffect(() => {
    if (selectedSeedText) {
      setSeedText(selectedSeedText);
    }
  }, [selectedSeedText]);

  const handleGenerate = async () => {
    if (!seedText.trim()) {
      setError('Please enter seed text');
      return;
    }

    if (!lstmStatus?.model_loaded) {
      setError('LSTM model not loaded. Please train the model first.');
      return;
    }

    setLoading(true);
    setError(null);
    setResult(null);

    try {
      const data = await generateText(seedText, numWords, temperature);
      setResult(data);
      if (onGenerate) onGenerate(data);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  const handleKeyPress = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleGenerate();
    }
  };

  return (
    <div className="bg-white rounded-xl shadow-lg p-6 border border-purple-100">
      {/* Header */}
      <div className="flex items-center gap-3 mb-6">
        <div className="p-2 bg-purple-100 rounded-lg">
          <Sparkles className="w-6 h-6 text-purple-600" />
        </div>
        <div>
          <h2 className="text-2xl font-bold text-gray-800">Text Generation</h2>
          <p className="text-sm text-gray-600">LSTM-powered next-word prediction</p>
        </div>
      </div>

      {/* Model Status Warning */}
      {!lstmStatus?.model_loaded && (
        <div className="mb-4 p-3 bg-amber-50 border border-amber-200 rounded-lg flex items-start gap-2">
          <AlertCircle className="w-5 h-5 text-amber-600 mt-0.5 flex-shrink-0" />
          <div className="text-sm text-amber-800">
            <p className="font-semibold">Model not loaded</p>
            <p>Run <code className="bg-amber-100 px-1 rounded">python backend/example_pipeline.py</code> to train the LSTM model.</p>
          </div>
        </div>
      )}

      {/* Seed Text Input */}
      <div className="mb-4">
        <label className="block text-sm font-medium text-gray-700 mb-2">
          Seed Text
          <span className="ml-2 text-xs text-gray-500 font-normal">
            (Starting words that the AI will continue)
          </span>
        </label>
        <input
          type="text"
          value={seedText}
          onChange={(e) => setSeedText(e.target.value)}
          onKeyPress={handleKeyPress}
          placeholder="Enter starting text (e.g., 'the quick brown')"
          className="w-full px-4 py-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-purple-500 focus:border-transparent transition-all"
          disabled={loading || !lstmStatus?.model_loaded}
        />
        <p className="mt-1 text-xs text-gray-500">
          💡 The model reads your seed text and predicts what comes next, one word at a time
        </p>
      </div>

      {/* Number of Words Slider */}
      <div className="mb-4">
        <label className="block text-sm font-medium text-gray-700 mb-2">
          Number of Words: <span className="text-purple-600 font-semibold">{numWords}</span>
          <span className="ml-2 text-xs text-gray-500 font-normal">
            (How many words to add after your seed)
          </span>
        </label>
        <input
          type="range"
          min="10"
          max="70"
          value={numWords}
          onChange={(e) => setNumWords(parseInt(e.target.value))}
          className="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer accent-purple-600"
          disabled={loading || !lstmStatus?.model_loaded}
        />
        <div className="flex justify-between text-xs text-gray-500 mt-1">
          <span>10 words</span>
          <span>70 words</span>
        </div>
      </div>

      {/* Temperature Slider */}
      <div className="mb-6">
        <label className="block text-sm font-medium text-gray-700 mb-2">
          Temperature (Creativity Level): <span className="text-purple-600 font-semibold">{temperature.toFixed(1)}</span>
          <span className="text-xs text-gray-500 ml-2">
            {temperature < 0.5 ? '(Very predictable)' : temperature < 0.8 ? '(Balanced)' : '(Creative)'}
          </span>
        </label>
        <input
          type="range"
          min="0.1"
          max="2.0"
          step="0.1"
          value={temperature}
          onChange={(e) => setTemperature(parseFloat(e.target.value))}
          className="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer accent-purple-600"
          disabled={loading || !lstmStatus?.model_loaded}
        />
        <div className="flex justify-between text-xs text-gray-500 mt-1">
          <span>0.1 (Safe)</span>
          <span>1.0 (Balanced)</span>
          <span>2.0 (Wild)</span>
        </div>
        <p className="mt-2 text-xs text-gray-500">
          💡 Lower = more predictable (follows training data closely) | Higher = more surprising (takes risks)
        </p>
      </div>

      {/* Generate Button */}
      <button
        onClick={handleGenerate}
        disabled={loading || !lstmStatus?.model_loaded || !seedText.trim()}
        className="w-full bg-gradient-to-r from-purple-600 to-indigo-600 text-white font-semibold py-3 px-6 rounded-lg hover:from-purple-700 hover:to-indigo-700 disabled:from-gray-400 disabled:to-gray-500 disabled:cursor-not-allowed transition-all duration-200 flex items-center justify-center gap-2"
      >
        {loading ? (
          <>
            <Loader2 className="w-5 h-5 animate-spin" />
            Generating...
          </>
        ) : (
          <>
            <Sparkles className="w-5 h-5" />
            Generate Text
          </>
        )}
      </button>

      {/* Error Display */}
      {error && (
        <div className="mt-4 p-3 bg-red-50 border border-red-200 rounded-lg flex items-start gap-2">
          <AlertCircle className="w-5 h-5 text-red-600 mt-0.5 flex-shrink-0" />
          <p className="text-sm text-red-800">{error}</p>
        </div>
      )}

      {/* Result Display */}
      {result && (
        <div className="mt-6 p-4 bg-gradient-to-br from-purple-50 to-indigo-50 rounded-lg border border-purple-200">
          <h3 className="text-sm font-semibold text-gray-700 mb-2">Generated Text:</h3>
          <p className="text-gray-800 leading-relaxed mb-3 text-lg">
            <span className="text-purple-700 font-medium bg-purple-100 px-1 rounded" title="Your seed text">
              {result.seed_text}
            </span>
            <span className="text-gray-900 bg-indigo-50 px-1 rounded ml-1" title="AI-generated continuation">
              {result.generated_text.slice(result.seed_text.length)}
            </span>
          </p>
          <div className="flex items-center justify-between text-xs text-gray-600 pt-2 border-t border-purple-200">
            <div>
              <span className="inline-block w-3 h-3 bg-purple-100 rounded mr-1"></span>
              Your seed text
            </div>
            <div>
              <span className="inline-block w-3 h-3 bg-indigo-50 rounded mr-1"></span>
              AI generated ({result.num_words_generated} word{result.num_words_generated !== 1 ? 's' : ''})
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

export default TextGenerationSection;
