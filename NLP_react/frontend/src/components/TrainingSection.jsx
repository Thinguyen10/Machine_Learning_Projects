import React, { useState } from 'react'
import { train, artifacts } from '../services/api'

export default function TrainingSection({ setArtifacts, setTrainMetrics }){
  const [loading, setLoading] = useState(false)
  const [status, setStatus] = useState(null)

  const handleTrain = async (backend='sklearn')=>{
    setLoading(true)
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
    <section className="p-4 bg-white border rounded">
      <h3 className="font-semibold">Train model</h3>
      <div className="mt-2 flex gap-2">
        <button className="px-3 py-1 bg-green-600 text-white rounded" onClick={()=>handleTrain('sklearn')} disabled={loading}>Train sklearn</button>
        <button className="px-3 py-1 bg-indigo-600 text-white rounded" onClick={()=>handleTrain('keras')} disabled={loading}>Train keras</button>
        <button className="px-3 py-1 border rounded" onClick={checkArtifacts}>Check artifacts</button>
      </div>

      {status && (
        <div className="mt-3">
          {status.error && <div className="text-red-600">Error: {status.error}</div>}
          {status.metrics && <pre className="text-xs bg-gray-100 p-2 rounded">{JSON.stringify(status.metrics, null, 2)}</pre>}
          {status.artifacts && <pre className="text-xs bg-gray-100 p-2 rounded">{JSON.stringify(status.artifacts, null, 2)}</pre>}
        </div>
      )}
    </section>
  )
}
