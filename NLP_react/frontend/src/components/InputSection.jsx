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
    <section>
      <textarea id="nlp-input" rows={6} className="w-full p-2 border rounded" value={text} onChange={(e)=>setText(e.target.value)} placeholder="Type a sentence to analyze" />
      <div className="mt-2 flex gap-2">
        <button className="px-4 py-2 bg-blue-600 text-white rounded" onClick={handleAnalyze} disabled={loading||!text.trim()}>
          {loading ? 'Analyzing...' : 'Analyze'}
        </button>
        <button className="px-4 py-2 bg-gray-200 rounded" onClick={handlePreview} disabled={!text.trim()}>
          Preview transform
        </button>
      </div>

      {preview && (
        <div className="mt-3 p-3 bg-white border rounded">
          {preview.error ? (
            <div className="text-red-600">Error: {preview.error}</div>
          ) : (
            <>
              <div><strong>Cleaned:</strong> {preview.cleaned}</div>
              <div className="mt-1"><strong>Tokens:</strong> {preview.tokens.join(', ')}</div>
            </>
          )}
        </div>
      )}
    </section>
  )
}
