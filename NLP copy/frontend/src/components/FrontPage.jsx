import React, { useEffect, useState } from 'react'
import { artifacts, transform } from '../services/api'

export default function FrontPage(){
  const [art, setArt] = useState(null)
  const [sample, setSample] = useState(null)

  useEffect(()=>{
    let mounted = true
    artifacts().then(a=>{ if(mounted) setArt(a) }).catch(()=>{})
    return ()=>{ mounted=false }
  }, [])

  const trySample = async ()=>{
    try{
      const r = await transform('This product is excellent and I enjoyed using it.')
      setSample(r)
    }catch(e){ setSample({ error: e.message || String(e) }) }
  }

  return (
    <section className="p-4 bg-white border rounded">
      <h2 className="text-xl font-semibold">Front Page</h2>
      <p className="text-sm text-gray-600 mt-1">This app demonstrates preprocessing, training, and prediction for sentiment analysis.</p>
      <div className="mt-3 flex gap-3">
        <button className="px-3 py-1 bg-blue-600 text-white rounded" onClick={trySample}>Try sample transform</button>
      </div>

      {art && (
        <div className="mt-3 text-sm">
          <div>Artifacts:</div>
          <pre className="bg-gray-100 p-2 rounded text-xs">{JSON.stringify(art, null, 2)}</pre>
        </div>
      )}

      {sample && (
        <div className="mt-3 text-sm">
          {sample.error ? <div className="text-red-600">Error: {sample.error}</div> : (
            <div>
              <div><strong>Cleaned:</strong> {sample.cleaned}</div>
              <div className="mt-1"><strong>Tokens:</strong> {sample.tokens.join(', ')}</div>
            </div>
          )}
        </div>
      )}
    </section>
  )
}
