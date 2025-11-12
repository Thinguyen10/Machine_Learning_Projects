import React, { useState } from 'react'
import Header from './components/Header'
import FrontPage from './components/FrontPage'
import InputSection from './components/InputSection'
import ResultsSection from './components/ResultsSection'
import ExamplesSection from './components/ExamplesSection'
import InfoSection from './components/InfoSection'
import ModelSelector from './components/TrainingSection'
import ImprovementsPage from './components/ImprovementsPage'

function App(){
  const [result, setResult] = useState(null)
  const [text, setText] = useState('')
  const [selectedBackend, setSelectedBackend] = useState('sklearn')
  const [showImprovements, setShowImprovements] = useState(false)

  // If showing improvements page, render it instead
  if (showImprovements) {
    return <ImprovementsPage onBack={() => setShowImprovements(false)} />
  }

  return (
    <div className="min-h-screen py-8">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <Header />
        <main className="space-y-8">
          <FrontPage onLearnMore={() => setShowImprovements(true)} />

          <section className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            <div className="space-y-6">
              <InputSection 
                text={text} 
                setText={setText} 
                onResult={setResult} 
                selectedBackend={selectedBackend}
              />
              <ExamplesSection onPick={setText} />
            </div>

            <div className="space-y-6">
              <ModelSelector 
                selectedBackend={selectedBackend}
                onBackendChange={setSelectedBackend}
              />
              <ResultsSection result={result} />
              <InfoSection />
            </div>
          </section>
        </main>

        {/* Footer */}
        <footer className="mt-16 mb-8 text-center">
          <div className="glass rounded-2xl p-6 inline-block">
            <div className="flex items-center gap-4 text-gray-600">
              <span>Built with</span>
              <span className="px-3 py-1 bg-gradient-to-r from-purple-100 to-pink-100 rounded-full text-sm font-semibold text-purple-700">TensorFlow</span>
              <span className="px-3 py-1 bg-gradient-to-r from-blue-100 to-cyan-100 rounded-full text-sm font-semibold text-blue-700">React</span>
              <span className="px-3 py-1 bg-gradient-to-r from-green-100 to-emerald-100 rounded-full text-sm font-semibold text-green-700">FastAPI</span>
            </div>
          </div>
        </footer>
      </div>
    </div>
  )
}

export default App
