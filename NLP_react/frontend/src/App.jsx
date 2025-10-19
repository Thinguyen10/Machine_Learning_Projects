import React, { useState } from 'react'
import Header from './components/Header'
import FrontPage from './components/FrontPage'
import InputSection from './components/InputSection'
import ResultsSection from './components/ResultsSection'
import ExamplesSection from './components/ExamplesSection'
import InfoSection from './components/InfoSection'
import TrainingSection from './components/TrainingSection'

function App(){
  const [result, setResult] = useState(null)
  const [text, setText] = useState('')
  const [artifacts, setArtifacts] = useState(null)
  const [trainMetrics, setTrainMetrics] = useState(null)

  return (
    <div className="max-w-4xl mx-auto p-6">
      <Header />
      <main className="space-y-6">
        <FrontPage />

        <section className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          <div className="space-y-6">
            <InputSection text={text} setText={setText} onResult={setResult} />
            <ExamplesSection onPick={setText} />
          </div>

          <div className="space-y-6">
            <TrainingSection setArtifacts={setArtifacts} setTrainMetrics={setTrainMetrics} />
            <ResultsSection result={result} />
            <InfoSection />
          </div>
        </section>
      </main>
    </div>
  )
}

export default App
