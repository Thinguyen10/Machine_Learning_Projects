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

  const prob = typeof result.probability === 'number' ? result.probability : (parseFloat(result.probability) || 0)
  const percent = Math.round((prob || 0) * 100)
  const isPositive = prob >= 0.5

  return (
    <section className={`glass rounded-2xl p-6 hover-lift bg-gradient-to-br ${isPositive ? 'from-green-50 to-emerald-50' : 'from-red-50 to-orange-50'}`}>
      <div className="flex items-center gap-3 mb-6">
        <div className={`w-12 h-12 rounded-xl flex items-center justify-center text-white text-2xl ${isPositive ? 'bg-gradient-to-br from-green-500 to-emerald-500' : 'bg-gradient-to-br from-red-500 to-orange-500'}`}>
          {isPositive ? '😊' : '😞'}
        </div>
        <div>
          <h3 className="text-xl font-bold text-gray-800">Prediction Result</h3>
          <div className={`text-sm font-semibold ${isPositive ? 'text-green-600' : 'text-red-600'}`}>
            {result.label === 1 || result.label === '1' ? 'Positive Sentiment' : (result.label === 0 || result.label === '0' ? 'Negative Sentiment' : String(result.label))}
          </div>
        </div>
      </div>

      <div className="space-y-4">
        {/* Confidence Score */}
        <div className="bg-white/60 p-4 rounded-xl">
          <div className="flex items-center justify-between mb-3">
            <span className="text-sm font-semibold text-gray-700">Confidence Score</span>
            <span className={`text-2xl font-bold ${isPositive ? 'text-green-600' : 'text-red-600'}`}>
              {percent}%
            </span>
          </div>
          <div className="relative w-full bg-gray-200 rounded-full h-4 overflow-hidden">
            <div 
              style={{width: `${percent}%`}} 
              className={`h-full transition-all duration-500 ${isPositive ? 'bg-gradient-to-r from-green-400 to-emerald-500' : 'bg-gradient-to-r from-red-400 to-orange-500'}`}
            ></div>
          </div>
          <div className="flex justify-between mt-2 text-xs text-gray-500">
            <span>Negative</span>
            <span>Neutral</span>
            <span>Positive</span>
          </div>
        </div>

        {/* Sentiment Breakdown */}
        <div className="grid grid-cols-2 gap-3">
          <div className={`p-3 rounded-xl ${isPositive ? 'bg-green-100' : 'bg-white/60'}`}>
            <div className="text-xs text-gray-600 mb-1">Positive</div>
            <div className={`text-xl font-bold ${isPositive ? 'text-green-700' : 'text-gray-400'}`}>
              {percent}%
            </div>
          </div>
          <div className={`p-3 rounded-xl ${!isPositive ? 'bg-red-100' : 'bg-white/60'}`}>
            <div className="text-xs text-gray-600 mb-1">Negative</div>
            <div className={`text-xl font-bold ${!isPositive ? 'text-red-700' : 'text-gray-400'}`}>
              {100 - percent}%
            </div>
          </div>
        </div>

        {/* Backend Info */}
        <div className="pt-3 border-t border-gray-200 text-center">
          <div className="text-xs text-gray-500">
            Powered by <span className="font-semibold text-purple-600">{result.backend}</span>
          </div>
        </div>
      </div>
    </section>
  )
}
