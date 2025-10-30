import { useEffect, useState } from 'react';
import { Brain, CheckCircle, XCircle, RefreshCw } from 'lucide-react';
import { getLSTMStatus } from '../services/api';

/**
 * LSTMStatusCard Component
 * Displays LSTM model status and configuration
 */
function LSTMStatusCard({ onStatusUpdate }) {
  const [status, setStatus] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    loadStatus();
  }, []);

  const loadStatus = async () => {
    setLoading(true);
    setError(null);
    try {
      const data = await getLSTMStatus();
      setStatus(data);
      if (onStatusUpdate) onStatusUpdate(data);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="bg-white rounded-xl shadow-lg p-6 border border-purple-100">
      {/* Header */}
      <div className="flex items-center justify-between mb-4">
        <div className="flex items-center gap-2">
          <Brain className="w-5 h-5 text-purple-600" />
          <h3 className="text-lg font-bold text-gray-800">LSTM Model</h3>
        </div>
        <button
          onClick={loadStatus}
          disabled={loading}
          className="p-1 hover:bg-gray-100 rounded-lg transition-colors"
          title="Refresh status"
        >
          <RefreshCw className={`w-4 h-4 text-gray-600 ${loading ? 'animate-spin' : ''}`} />
        </button>
      </div>

      {/* Status Display */}
      {loading && !status ? (
        <div className="py-4 text-center text-gray-500">Loading...</div>
      ) : error ? (
        <div className="p-3 bg-red-50 border border-red-200 rounded-lg">
          <p className="text-sm text-red-800">{error}</p>
        </div>
      ) : status ? (
        <div className="space-y-3">
          {/* Model Loaded Status */}
          <div className="flex items-center justify-between p-3 bg-gray-50 rounded-lg">
            <span className="text-sm font-medium text-gray-700">Status</span>
            <div className="flex items-center gap-2">
              {status.model_loaded ? (
                <>
                  <CheckCircle className="w-4 h-4 text-green-600" />
                  <span className="text-sm font-semibold text-green-700">Loaded</span>
                </>
              ) : (
                <>
                  <XCircle className="w-4 h-4 text-red-600" />
                  <span className="text-sm font-semibold text-red-700">Not Loaded</span>
                </>
              )}
            </div>
          </div>

          {/* Configuration Details */}
          {status.model_loaded && (
            <>
              <div className="flex items-center justify-between p-3 bg-gray-50 rounded-lg">
                <span className="text-sm font-medium text-gray-700">Window Size</span>
                <span className="text-sm font-semibold text-gray-900">
                  {status.window_size || 'N/A'}
                </span>
              </div>
              <div className="flex items-center justify-between p-3 bg-gray-50 rounded-lg">
                <span className="text-sm font-medium text-gray-700">Vocab Size</span>
                <span className="text-sm font-semibold text-gray-900">
                  {status.vocab_size?.toLocaleString() || 'N/A'}
                </span>
              </div>
            </>
          )}

          {/* Additional Info */}
          {status.model_info?.message && !status.model_loaded && (
            <div className="p-3 bg-amber-50 border border-amber-200 rounded-lg">
              <p className="text-xs text-amber-800">{status.model_info.message}</p>
            </div>
          )}
        </div>
      ) : null}

      {/* Help Text */}
      <div className="mt-4 pt-4 border-t border-gray-200">
        <p className="text-xs text-gray-500">
          🔧 To train the model, run:
        </p>
        <code className="block mt-1 text-xs bg-gray-100 px-2 py-1 rounded">
          python backend/example_pipeline.py
        </code>
      </div>
    </div>
  );
}

export default LSTMStatusCard;
