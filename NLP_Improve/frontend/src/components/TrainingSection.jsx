import React, { useState, useEffect } from 'react'
import { artifacts } from '../services/api'

export default function ModelSelector({ selectedBackend, onBackendChange }){
  const [art, setArt] = useState(null)
  const [loading, setLoading] = useState(false)

  useEffect(() => {
    loadArtifacts()
  }, [])

  const loadArtifacts = async () => {
    setLoading(true)
    try {
      const a = await artifacts()
      setArt(a)
    } catch(e) {
      console.error('Failed to load artifacts:', e)
    } finally {
      setLoading(false)
    }
  }

  const hasSklearn = art?.artifacts?.sklearn_model
  const hasKeras = art?.artifacts?.keras_model
  const availableModels = art?.available_models || []

  return (
    <section className="glass rounded-2xl p-6 hover-lift">
      <div className="flex items-center gap-3 mb-4">
        <div className="w-10 h-10 bg-gradient-to-br from-pink-500 to-rose-500 rounded-lg flex items-center justify-center text-white text-xl">
          �
        </div>
        <h3 className="text-xl font-bold text-gray-800">Model Selection</h3>
      </div>
      
      <p className="text-sm text-gray-600 mb-4">
        Choose which pre-trained model to use for predictions. Both models are trained on the same data with optimized hyperparameters.
      </p>

      <div className="space-y-3">
        {/* Sklearn Model */}
        <button
          onClick={() => onBackendChange('sklearn')}
          disabled={!hasSklearn}
          className={`w-full p-4 rounded-xl text-left transition-all ${
            selectedBackend === 'sklearn'
              ? 'bg-gradient-to-r from-green-500 to-emerald-500 text-white shadow-lg scale-105'
              : hasSklearn
              ? 'bg-white border-2 border-gray-200 hover:border-green-300 hover:shadow-md'
              : 'bg-gray-100 border-2 border-gray-200 opacity-50 cursor-not-allowed'
          }`}
        >
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-3">
              <span className="text-2xl">📊</span>
              <div>
                <div className="font-bold">Sklearn (LogisticRegression)</div>
                <div className={`text-sm ${selectedBackend === 'sklearn' ? 'text-green-100' : 'text-gray-600'}`}>
                  Fast, accurate, optimized with GridSearchCV
                </div>
              </div>
            </div>
            {selectedBackend === 'sklearn' && <span className="text-2xl">✓</span>}
            {!hasSklearn && <span className="text-sm text-red-500">Not trained</span>}
          </div>
          {hasSklearn && art?.metrics?.sklearn && (
            <div className="mt-3 pt-3 border-t border-white/30">
              <div className="text-sm grid grid-cols-2 gap-2">
                <div className={selectedBackend === 'sklearn' ? 'text-green-100' : 'text-gray-600'}>
                  Accuracy: <span className="font-bold">{(art.metrics.sklearn.accuracy * 100).toFixed(2)}%</span>
                </div>
                {art.metrics.sklearn.best_params?.C && (
                  <div className={selectedBackend === 'sklearn' ? 'text-green-100' : 'text-gray-600'}>
                    C: <span className="font-bold">{art.metrics.sklearn.best_params.C}</span>
                  </div>
                )}
              </div>
            </div>
          )}
        </button>

        {/* Keras Model */}
        <button
          onClick={() => onBackendChange('keras')}
          disabled={!hasKeras}
          className={`w-full p-4 rounded-xl text-left transition-all ${
            selectedBackend === 'keras'
              ? 'bg-gradient-to-r from-indigo-500 to-purple-500 text-white shadow-lg scale-105'
              : hasKeras
              ? 'bg-white border-2 border-gray-200 hover:border-purple-300 hover:shadow-md'
              : 'bg-gray-100 border-2 border-gray-200 opacity-50 cursor-not-allowed'
          }`}
        >
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-3">
              <span className="text-2xl">🧠</span>
              <div>
                <div className="font-bold">Keras (Neural Network)</div>
                <div className={`text-sm ${selectedBackend === 'keras' ? 'text-purple-100' : 'text-gray-600'}`}>
                  Deep learning, multi-layer architecture
                </div>
              </div>
            </div>
            {selectedBackend === 'keras' && <span className="text-2xl">✓</span>}
            {!hasKeras && <span className="text-sm text-red-500">Not trained</span>}
          </div>
          {hasKeras && art?.metrics?.keras && (
            <div className="mt-3 pt-3 border-t border-white/30">
              <div className="text-sm grid grid-cols-2 gap-2">
                <div className={selectedBackend === 'keras' ? 'text-purple-100' : 'text-gray-600'}>
                  Accuracy: <span className="font-bold">{(art.metrics.keras.accuracy * 100).toFixed(2)}%</span>
                </div>
                {art.metrics.keras.epochs_trained && (
                  <div className={selectedBackend === 'keras' ? 'text-purple-100' : 'text-gray-600'}>
                    Epochs: <span className="font-bold">{art.metrics.keras.epochs_trained}</span>
                  </div>
                )}
              </div>
            </div>
          )}
        </button>
      </div>

      {/* Status */}
      {!hasSklearn && !hasKeras && (
        <div className="mt-4 p-4 bg-yellow-50 border-2 border-yellow-200 rounded-xl">
          <div className="flex items-start gap-3">
            <span className="text-2xl">⚠️</span>
            <div className="flex-1">
              <div className="font-bold text-yellow-800">Models Not Found</div>
              <div className="text-sm text-yellow-700 mt-1">
                Please train the models first by running:<br/>
                <code className="bg-yellow-100 px-2 py-1 rounded mt-1 inline-block">
                  python -m backend.train_models
                </code>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Refresh button */}
      <button
        onClick={loadArtifacts}
        disabled={loading}
        className="mt-4 w-full px-4 py-2 bg-gradient-to-r from-blue-500 to-cyan-500 text-white rounded-xl font-semibold hover:from-blue-600 hover:to-cyan-600 disabled:opacity-50 transition-all"
      >
        {loading ? '🔄 Checking...' : '🔄 Refresh Model Status'}
      </button>
    </section>
  )
}
