import React, { useEffect, useState } from 'react'
import { artifacts } from '../services/api'

export default function FrontPage({ onLearnMore }){
  const [art, setArt] = useState(null)
  const [showDetails, setShowDetails] = useState(false)

  useEffect(()=>{
    let mounted = true
    artifacts().then(a=>{ if(mounted) setArt(a) }).catch(()=>{})
    return ()=>{ mounted=false }
  }, [])

  return (
    <section className="mb-8">
      {/* Hero Section */}
      <div className="glass rounded-3xl p-8 mb-6 hover-lift bg-gradient-to-br from-purple-100 via-blue-100 to-pink-100 animate-gradient">
        <div className="flex items-start gap-6">
          <div className="flex-shrink-0">
            <div className="w-20 h-20 bg-gradient-to-br from-purple-600 to-pink-600 rounded-2xl flex items-center justify-center text-white text-4xl">
              🧠
            </div>
          </div>
          <div className="flex-1">
            <h2 className="text-3xl font-bold bg-gradient-to-r from-purple-700 to-pink-700 bg-clip-text text-transparent mb-3">
              Advanced NLP Sentiment Analysis
            </h2>
            <p className="text-gray-700 text-lg leading-relaxed mb-4">
              This application uses cutting-edge neural network architecture to analyze text sentiment with exceptional accuracy. 
              Powered by TensorFlow and optimized through advanced hyperparameter tuning techniques.
            </p>
            <div className="grid grid-cols-3 gap-4 mt-6">
              <div className="bg-white/60 p-4 rounded-xl text-center">
                <div className="text-2xl font-bold text-purple-600">68%</div>
                <div className="text-sm text-gray-600 mt-1">Accuracy</div>
              </div>
              <div className="bg-white/60 p-4 rounded-xl text-center">
                <div className="text-2xl font-bold text-blue-600">3x</div>
                <div className="text-sm text-gray-600 mt-1">Faster Response</div>
              </div>
              <div className="bg-white/60 p-4 rounded-xl text-center">
                <div className="text-2xl font-bold text-pink-600">Dual</div>
                <div className="text-sm text-gray-600 mt-1">Models</div>
              </div>
            </div>
            
            {/* Learn More Button */}
            {onLearnMore && (
              <div className="mt-6">
                <button
                  onClick={onLearnMore}
                  className="w-full bg-gradient-to-r from-purple-500 to-pink-500 text-white font-semibold py-3 px-6 rounded-xl hover:shadow-lg hover:scale-105 transition-all duration-300 flex items-center justify-center gap-2"
                >
                  <span>📖 Learn About Improvements & Future Enhancements</span>
                  <span>→</span>
                </button>
              </div>
            )}
          </div>
        </div>
      </div>

      {/* Model Architecture & Improvements */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mb-6">
        {/* Model Architecture */}
        <div className="glass rounded-2xl p-6 hover-lift">
          <div className="flex items-center gap-3 mb-4">
            <div className="w-12 h-12 bg-gradient-to-br from-blue-500 to-cyan-500 rounded-xl flex items-center justify-center text-white text-2xl">
              ⚙️
            </div>
            <h3 className="text-xl font-bold text-gray-800">Model Architecture</h3>
          </div>
          <ul className="space-y-3">
            <li className="flex items-start gap-3">
              <span className="text-blue-500 text-xl">•</span>
              <div>
                <div className="font-semibold text-gray-800">TF-IDF Preprocessing</div>
                <div className="text-sm text-gray-600">Advanced text vectorization with stopword removal</div>
              </div>
            </li>
            <li className="flex items-start gap-3">
              <span className="text-blue-500 text-xl">•</span>
              <div>
                <div className="font-semibold text-gray-800">Dense Neural Network</div>
                <div className="text-sm text-gray-600">Multi-layer architecture with dropout regularization</div>
              </div>
            </li>
            <li className="flex items-start gap-3">
              <span className="text-blue-500 text-xl">•</span>
              <div>
                <div className="font-semibold text-gray-800">Multi-Class Classification</div>
                <div className="text-sm text-gray-600">3-class sentiment: Negative, Neutral, Positive</div>
              </div>
            </li>
          </ul>
        </div>

        {/* Key Improvements */}
        <div className="glass rounded-2xl p-6 hover-lift">
          <div className="flex items-center gap-3 mb-4">
            <div className="w-12 h-12 bg-gradient-to-br from-green-500 to-emerald-500 rounded-xl flex items-center justify-center text-white text-2xl">
              🚀
            </div>
            <h3 className="text-xl font-bold text-gray-800">Key Improvements</h3>
          </div>
          <ul className="space-y-3">
            <li className="flex items-start gap-3">
              <span className="text-green-500 text-xl">✓</span>
              <div>
                <div className="font-semibold text-gray-800">Epoch Optimization</div>
                <div className="text-sm text-gray-600">Quadratic peak detection finds optimal training duration</div>
              </div>
            </li>
            <li className="flex items-start gap-3">
              <span className="text-green-500 text-xl">✓</span>
              <div>
                <div className="font-semibold text-gray-800">Grid Search Tuning</div>
                <div className="text-sm text-gray-600">Systematic exploration of hyperparameter space</div>
              </div>
            </li>
            <li className="flex items-start gap-3">
              <span className="text-green-500 text-xl">✓</span>
              <div>
                <div className="font-semibold text-gray-800">KerasTuner Integration</div>
                <div className="text-sm text-gray-600">Advanced Bayesian & Hyperband optimization</div>
              </div>
            </li>
          </ul>
        </div>
      </div>

      {/* Technical Details (Expandable) */}
      <div className="glass rounded-2xl p-6 hover-lift">
        <button 
          onClick={() => setShowDetails(!showDetails)}
          className="w-full flex items-center justify-between"
        >
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 bg-gradient-to-br from-orange-500 to-red-500 rounded-lg flex items-center justify-center text-white text-xl">
              📊
            </div>
            <h3 className="text-xl font-bold text-gray-800">Technical Details & Model Artifacts</h3>
          </div>
          <span className="text-2xl text-gray-400">{showDetails ? '−' : '+'}</span>
        </button>
        
        {showDetails && art && (
          <div className="mt-6 space-y-4">
            <div className="grid grid-cols-2 gap-4">
              <div className="bg-gradient-to-br from-purple-50 to-blue-50 p-4 rounded-xl">
                <div className="text-sm font-semibold text-gray-600 mb-1">Vocabulary Size</div>
                <div className="text-2xl font-bold text-purple-600">{art.vocab_size?.toLocaleString() || 'N/A'}</div>
              </div>
              <div className="bg-gradient-to-br from-blue-50 to-cyan-50 p-4 rounded-xl">
                <div className="text-sm font-semibold text-gray-600 mb-1">Training Samples</div>
                <div className="text-2xl font-bold text-blue-600">{art.num_samples?.toLocaleString() || 'N/A'}</div>
              </div>
            </div>
            
            <div className="bg-gray-50 p-4 rounded-xl">
              <div className="text-sm font-semibold text-gray-700 mb-2">Complete Artifact Data:</div>
              <pre className="bg-white p-3 rounded-lg text-xs overflow-x-auto border border-gray-200">
                {JSON.stringify(art, null, 2)}
              </pre>
            </div>
          </div>
        )}
      </div>
    </section>
  )
}
