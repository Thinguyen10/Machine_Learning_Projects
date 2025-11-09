import React, { useState } from 'react'
import { train, artifacts } from '../services/api'

export default function TrainingSection({ setArtifacts, setTrainMetrics }){
  const [loading, setLoading] = useState(false)
  const [status, setStatus] = useState(null)

  const handleTrain = async (backend='sklearn')=>{
    setLoading(true)
    setStatus({ loading: true, message: `Training ${backend} model...` })
    try{
      const res = await train({ backend })
      setStatus({ success: true, metrics: res.metrics })
      if(setTrainMetrics) setTrainMetrics(res.metrics)
      // refresh artifacts
      try{
        const a = await artifacts()
        if(setArtifacts) setArtifacts(a)
      }catch(e){}
    }catch(e){
      setStatus({ success: false, error: e.message || String(e) })
    }finally{
      setLoading(false)
    }
  }

  const checkArtifacts = async ()=>{
    try{
      const a = await artifacts()
      setStatus({ artifacts: a })
      if(setArtifacts) setArtifacts(a)
    }catch(e){
      setStatus({ success: false, error: e.message || String(e) })
    }
  }

  return (
    <section className="glass rounded-2xl p-6 hover-lift">
      <div className="flex items-center gap-3 mb-4">
        <div className="w-10 h-10 bg-gradient-to-br from-pink-500 to-rose-500 rounded-lg flex items-center justify-center text-white text-xl">
          🎓
        </div>
        <h3 className="text-xl font-bold text-gray-800">Train Model</h3>
      </div>
      
      <div className="grid grid-cols-3 gap-3">
        <button 
          className="px-4 py-3 bg-gradient-to-r from-green-500 to-emerald-500 text-white rounded-xl font-semibold hover:from-green-600 hover:to-emerald-600 disabled:opacity-50 transition-all shadow-lg hover:shadow-xl transform hover:scale-105 text-sm"
          onClick={() => handleTrain('sklearn')} 
          disabled={loading}
        >
          📊 Sklearn
        </button>
        <button 
          className="px-4 py-3 bg-gradient-to-r from-indigo-500 to-purple-500 text-white rounded-xl font-semibold hover:from-indigo-600 hover:to-purple-600 disabled:opacity-50 transition-all shadow-lg hover:shadow-xl transform hover:scale-105 text-sm"
          onClick={() => handleTrain('keras')} 
          disabled={loading}
        >
          🧠 Keras
        </button>
        <button 
          className="px-4 py-3 bg-gradient-to-r from-blue-500 to-cyan-500 text-white rounded-xl font-semibold hover:from-blue-600 hover:to-cyan-600 transition-all shadow-lg hover:shadow-xl transform hover:scale-105 text-sm"
          onClick={checkArtifacts}
        >
          🔍 Check
        </button>
      </div>

      {status && (
        <div className="mt-4">
          {status.loading && (
            <div className="p-4 bg-gradient-to-r from-blue-50 to-cyan-50 border-2 border-blue-200 rounded-xl">
              <div className="flex items-center gap-3">
                <span className="text-2xl animate-spin">⚙️</span>
                <span className="font-semibold text-blue-700">{status.message}</span>
              </div>
            </div>
          )}
          
          {status.error && (
            <div className="p-4 bg-gradient-to-br from-red-50 to-orange-50 border-2 border-red-200 rounded-xl">
              <div className="flex items-center gap-3">
                <span className="text-2xl">❌</span>
                <div>
                  <div className="font-bold text-red-800">Error</div>
                  <div className="text-red-600 text-sm">{status.error}</div>
                </div>
              </div>
            </div>
          )}
          
          {status.metrics && (
            <div className="p-4 bg-gradient-to-br from-green-50 to-emerald-50 border-2 border-green-200 rounded-xl">
              <div className="flex items-center gap-2 mb-3">
                <span className="text-2xl">✅</span>
                <span className="font-bold text-green-800">Training Complete!</span>
              </div>
              <div className="bg-white/60 p-3 rounded-lg">
                <div className="text-sm font-semibold text-gray-700 mb-2">Metrics:</div>
                <pre className="text-xs bg-gray-50 p-3 rounded border border-gray-200 overflow-x-auto">
                  {JSON.stringify(status.metrics, null, 2)}
                </pre>
              </div>
            </div>
          )}
          
          {status.artifacts && (
            <div className="p-4 bg-gradient-to-br from-purple-50 to-pink-50 border-2 border-purple-200 rounded-xl">
              <div className="flex items-center gap-2 mb-3">
                <span className="text-2xl">📦</span>
                <span className="font-bold text-purple-800">Artifacts Loaded</span>
              </div>
              <div className="bg-white/60 p-3 rounded-lg">
                <pre className="text-xs bg-gray-50 p-3 rounded border border-gray-200 overflow-x-auto">
                  {JSON.stringify(status.artifacts, null, 2)}
                </pre>
              </div>
            </div>
          )}
        </div>
      )}
    </section>
  )
}
