import React from 'react'

export default function InfoSection(){
  return (
    <section className="glass rounded-2xl p-6 hover-lift bg-gradient-to-br from-indigo-50 to-purple-50">
      <div className="flex items-center gap-3 mb-4">
        <div className="w-10 h-10 bg-gradient-to-br from-indigo-500 to-purple-500 rounded-lg flex items-center justify-center text-white text-xl">
          ℹ️
        </div>
        <h4 className="text-xl font-bold text-gray-800">About This App</h4>
      </div>
      <div className="space-y-3 text-gray-700">
        <p className="leading-relaxed">
          This application demonstrates a production-ready sentiment analysis system powered by machine learning.
        </p>
        <div className="bg-white/60 p-4 rounded-xl">
          <div className="text-sm font-semibold text-indigo-600 mb-2">🔧 Technical Stack</div>
          <ul className="text-sm space-y-1">
            <li>• <strong>Backend:</strong> FastAPI (Python)</li>
            <li>• <strong>Frontend:</strong> React + Vite + Tailwind CSS</li>
            <li>• <strong>ML Framework:</strong> TensorFlow & scikit-learn</li>
          </ul>
        </div>
        <div className="text-xs text-gray-500 bg-white/60 p-3 rounded-lg">
          <strong>Note:</strong> Ensure the backend server is running at <code className="bg-gray-200 px-2 py-1 rounded">http://localhost:8000</code>
        </div>
      </div>
    </section>
  )
}
