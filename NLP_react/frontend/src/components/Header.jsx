import React from 'react'

export default function Header(){
  return (
    <header className="mb-4">
      <h1 className="text-3xl font-bold">NLP Sentiment Analyzer</h1>
      <p className="text-sm text-gray-600">Enter text to see predicted sentiment (positive probability).</p>
      <div className="mt-2 flex gap-2 text-xs text-gray-500">
        <div className="flex items-center gap-1"><span className="w-3 h-3 bg-red-500 inline-block rounded-full"/> Negative</div>
        <div className="flex items-center gap-1"><span className="w-3 h-3 bg-green-500 inline-block rounded-full"/> Positive</div>
      </div>
    </header>
  )
}
