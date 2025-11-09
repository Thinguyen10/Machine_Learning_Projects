import React, { useState } from 'react'
import { predict, transform } from '../services/api'

export default function InputSection({ text, setText, onResult }){
  const [loading, setLoading] = useState(false)
  const [preview, setPreview] = useState(null)

  const handleAnalyze = async ()=>{
    setLoading(true)
    try{
      const res = await predict(text)
      onResult(res)
    }catch(e){
      onResult({ error: e.message || String(e) })
    }finally{
      setLoading(false)
    }
  }

  const handlePreview = async ()=>{
    if(!text || !text.trim()) return
    try{
      const p = await transform(text)
      setPreview(p)
    }catch(e){
      setPreview({ error: e.message || String(e) })
    }
  }

  return (
    <section className="glass rounded-2xl p-6 hover-lift">
      <div className="flex items-center gap-3 mb-4">
        <div className="w-10 h-10 bg-gradient-to-br from-purple-500 to-pink-500 rounded-lg flex items-center justify-center text-white text-xl">
          ✍️
        </div>
        <h3 className="text-xl font-bold text-gray-800">Analyze Text</h3>
      </div>
      
      <textarea 
        id="nlp-input" 
        rows={6} 
        className="w-full p-4 border-2 border-purple-200 rounded-xl focus:border-purple-500 focus:ring-2 focus:ring-purple-200 transition-all outline-none resize-none" 
        value={text} 
        onChange={(e)=>setText(e.target.value)} 
        placeholder="Type or paste any text here to analyze its sentiment...&#10;&#10;Example: 'This product exceeded my expectations! Highly recommended.'"
      />
      
      <div className="mt-4 flex gap-3">
        <button 
          className="flex-1 px-6 py-3 bg-gradient-to-r from-purple-600 to-pink-600 text-white rounded-xl font-semibold hover:from-purple-700 hover:to-pink-700 disabled:opacity-50 disabled:cursor-not-allowed transition-all shadow-lg hover:shadow-xl transform hover:scale-105" 
          onClick={handleAnalyze} 
          disabled={loading||!text.trim()}
        >
          {loading ? (
            <span className="flex items-center justify-center gap-2">
              <span className="animate-spin">⚙️</span> Analyzing...
            </span>
          ) : (
            <span className="flex items-center justify-center gap-2">
              🔍 Analyze Sentiment
            </span>
          )}
        </button>
        <button 
          className="px-6 py-3 bg-gradient-to-r from-blue-500 to-cyan-500 text-white rounded-xl font-semibold hover:from-blue-600 hover:to-cyan-600 disabled:opacity-50 disabled:cursor-not-allowed transition-all shadow-lg hover:shadow-xl transform hover:scale-105" 
          onClick={handlePreview} 
          disabled={!text.trim()}
        >
          👁️ Preview
        </button>
      </div>

      {preview && (
        <div className="mt-4 p-4 bg-gradient-to-br from-blue-50 to-cyan-50 border-2 border-blue-200 rounded-xl">
          {preview.error ? (
            <div className="text-red-600 font-semibold">❌ Error: {preview.error}</div>
          ) : (
            <div className="space-y-3">
              <div>
                <div className="text-sm font-semibold text-gray-600 mb-1">Cleaned Text:</div>
                <div className="bg-white p-3 rounded-lg text-gray-800">{preview.cleaned}</div>
              </div>
              <div>
                <div className="text-sm font-semibold text-gray-600 mb-1">Tokens:</div>
                <div className="bg-white p-3 rounded-lg flex flex-wrap gap-2">
                  {preview.tokens.map((token, i) => (
                    <span key={i} className="px-2 py-1 bg-blue-100 text-blue-700 rounded-md text-sm font-medium">
                      {token}
                    </span>
                  ))}
                </div>
              </div>
            </div>
          )}
        </div>
      )}
    </section>
  )
}
