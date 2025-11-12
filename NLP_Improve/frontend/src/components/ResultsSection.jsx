import React from 'react'

export default function ResultsSection({ result }){
  if(!result) return null
  if(result.error) return (
    <div className="glass rounded-2xl p-6 bg-gradient-to-br from-red-50 to-orange-50 border-2 border-red-200">
      <div className="flex items-center gap-3">
        <span className="text-3xl">❌</span>
        <div>
          <div className="font-bold text-red-800">Error</div>
          <div className="text-red-600">{result.error}</div>
        </div>
      </div>
    </div>
  )

  // Handle 3-class sentiment: 0=negative, 1=neutral, 2=positive
  const sentiment = result.sentiment || 'unknown'
  const label = result.label
  const prob = typeof result.probability === 'number' ? result.probability : (parseFloat(result.probability) || 0)
  const percent = Math.round((prob || 0) * 100)
  
  // Determine colors and emoji based on sentiment
  const sentimentConfig = {
    negative: { 
      colors: 'from-red-50 to-orange-50',
      badgeColors: 'from-red-500 to-orange-500',
      textColor: 'text-red-600',
      emoji: '😞',
      label: 'Negative Sentiment'
    },
    neutral: {
      colors: 'from-gray-50 to-slate-50',
      badgeColors: 'from-gray-500 to-slate-500',
      textColor: 'text-gray-600',
      emoji: '😐',
      label: 'Neutral Sentiment'
    },
    positive: {
      colors: 'from-green-50 to-emerald-50',
      badgeColors: 'from-green-500 to-emerald-500',
      textColor: 'text-green-600',
      emoji: '😊',
      label: 'Positive Sentiment'
    },
    unknown: {
      colors: 'from-purple-50 to-pink-50',
      badgeColors: 'from-purple-500 to-pink-500',
      textColor: 'text-purple-600',
      emoji: '❓',
      label: 'Unknown'
    }
  }
  
  const config = sentimentConfig[sentiment] || sentimentConfig.unknown

  return (
    <section className={`glass rounded-2xl p-6 hover-lift bg-gradient-to-br ${config.colors}`}>
      <div className="flex items-center gap-3 mb-6">
        <div className={`w-12 h-12 rounded-xl flex items-center justify-center text-white text-2xl bg-gradient-to-br ${config.badgeColors}`}>
          {config.emoji}
        </div>
        <div>
          <h3 className="text-xl font-bold text-gray-800">Prediction Result</h3>
          <div className={`text-sm font-semibold ${config.textColor}`}>
            {config.label}
          </div>
        </div>
      </div>

      <div className="space-y-4">
        {/* Confidence Score */}
        <div className="bg-white/60 p-4 rounded-xl">
          <div className="flex items-center justify-between mb-3">
            <span className="text-sm font-semibold text-gray-700">Confidence Score</span>
            <span className={`text-2xl font-bold ${config.textColor}`}>
              {percent}%
            </span>
          </div>
          <div className="relative w-full bg-gray-200 rounded-full h-4 overflow-hidden">
            <div 
              style={{width: `${percent}%`}} 
              className={`h-full transition-all duration-500 bg-gradient-to-r ${config.badgeColors}`}
            ></div>
          </div>
          <div className="flex justify-between mt-2 text-xs text-gray-500">
            <span>0%</span>
            <span>50%</span>
            <span>100%</span>
          </div>
        </div>

        {/* Sentiment Display */}
        <div className="grid grid-cols-3 gap-2">
          <div className={`p-3 rounded-xl ${sentiment === 'negative' ? 'bg-red-100 ring-2 ring-red-400' : 'bg-white/60'}`}>
            <div className="text-xs text-gray-600 mb-1">Negative</div>
            <div className={`text-lg font-bold ${sentiment === 'negative' ? 'text-red-700' : 'text-gray-400'}`}>
              {sentiment === 'negative' ? '✓' : ''}
            </div>
          </div>
          <div className={`p-3 rounded-xl ${sentiment === 'neutral' ? 'bg-gray-100 ring-2 ring-gray-400' : 'bg-white/60'}`}>
            <div className="text-xs text-gray-600 mb-1">Neutral</div>
            <div className={`text-lg font-bold ${sentiment === 'neutral' ? 'text-gray-700' : 'text-gray-400'}`}>
              {sentiment === 'neutral' ? '✓' : ''}
            </div>
          </div>
          <div className={`p-3 rounded-xl ${sentiment === 'positive' ? 'bg-green-100 ring-2 ring-green-400' : 'bg-white/60'}`}>
            <div className="text-xs text-gray-600 mb-1">Positive</div>
            <div className={`text-lg font-bold ${sentiment === 'positive' ? 'text-green-700' : 'text-gray-400'}`}>
              {sentiment === 'positive' ? '✓' : ''}
            </div>
          </div>
        </div>

        {/* Backend Info */}
        <div className="pt-3 border-t border-gray-200 text-center">
          <div className="text-xs text-gray-500">
            Powered by <span className="font-semibold text-purple-600">{result.backend}</span> model
          </div>
        </div>
      </div>
    </section>
  )
}
