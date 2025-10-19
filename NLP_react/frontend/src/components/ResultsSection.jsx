import React from 'react'

export default function ResultsSection({ result }){
  if(!result) return null
  if(result.error) return <div className="p-4 bg-red-100 text-red-800 rounded">Error: {result.error}</div>

  const prob = typeof result.probability === 'number' ? result.probability : (parseFloat(result.probability) || 0)
  const percent = Math.round((prob || 0) * 100)
  const isPositive = prob >= 0.5

  return (
    <section className="p-4 bg-white border rounded">
      <h3 className="font-semibold">Prediction</h3>
      <div className="mt-3 grid grid-cols-1 gap-3">
        <div className="flex items-center justify-between">
          <div>Label:</div>
          <div className={`font-bold ${isPositive ? 'text-green-700' : 'text-red-700'}`}>
            {result.label === 1 || result.label === '1' ? 'Positive' : (result.label === 0 || result.label === '0' ? 'Negative' : String(result.label))}
          </div>
        </div>

        <div>
          <div className="flex items-center justify-between text-sm text-gray-600 mb-1">
            <div>Positive probability</div>
            <div className="font-mono">{percent}%</div>
          </div>
          <div className="w-full bg-gray-200 rounded h-3 overflow-hidden">
            <div style={{width: `${percent}%`}} className={`h-full ${isPositive ? 'bg-green-500' : 'bg-red-500'}`}></div>
          </div>
        </div>

        <div className="text-sm text-gray-600">Backend: <em>{result.backend}</em></div>
      </div>
    </section>
  )
}
