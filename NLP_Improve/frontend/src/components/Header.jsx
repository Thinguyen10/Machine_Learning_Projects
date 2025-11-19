import React from 'react'

export default function Header(){
  return (
    <header className="mb-8 text-center">
      <div className="inline-block">
        <h1 className="text-5xl font-bold bg-gradient-to-r from-purple-600 via-blue-600 to-pink-600 bg-clip-text text-transparent mb-3">
          NLP Sentiment Analyzer
        </h1>
        <div className="h-1 w-full bg-gradient-to-r from-purple-600 via-blue-600 to-pink-600 rounded-full"></div>
      </div>
      <p className="text-lg text-gray-700 mt-4 max-w-2xl mx-auto">
        Advanced neural network-powered sentiment analysis with state-of-the-art hyperparameter optimization
      </p>
      <div className="mt-4 flex gap-4 justify-center text-sm">
        <div className="flex items-center gap-2 px-4 py-2 bg-red-100 text-red-700 rounded-full font-medium">
          <span className="w-3 h-3 bg-red-500 rounded-full animate-pulse"/> Negative
        </div>
        <div className="flex items-center gap-2 px-4 py-2 bg-green-100 text-green-700 rounded-full font-medium">
          <span className="w-3 h-3 bg-green-500 rounded-full animate-pulse"/> Positive
        </div>
      </div>
    </header>
  )
}
