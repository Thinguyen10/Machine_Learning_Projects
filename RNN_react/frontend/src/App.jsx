import { useState, useEffect } from 'react';
import { Sparkles } from 'lucide-react';
import IntroPage from './components/IntroPage';
import TextGenerationSection from './components/TextGenerationSection';
import LSTMExamplesSection from './components/LSTMExamplesSection';
import LSTMStatusCard from './components/LSTMStatusCard';
import LSTMInfoSection from './components/LSTMInfoSection';
import { checkHealth } from './services/api';

function App() {
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [apiStatus, setApiStatus] = useState('checking');
  const [activeTab, setActiveTab] = useState('sentiment'); // 'sentiment' or 'textgen'
  const [lstmStatus, setLstmStatus] = useState(null);
  const [showIntro, setShowIntro] = useState(true); // Show intro page by default

  useEffect(() => {
    // Check API health on mount
    checkHealth()
      .then(() => setApiStatus('connected'))
      .catch(() => setApiStatus('disconnected'));
  }, []);

  const [selectedSeedText, setSelectedSeedText] = useState('');

  const handleSelectExample = (example) => {
    setSelectedSeedText(example);
  };

  // Handler to navigate from intro to main app
  const handleGetStarted = () => {
    setShowIntro(false);
  };

  // If showing intro, render IntroPage
  if (showIntro) {
    return <IntroPage onGetStarted={handleGetStarted} />;
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 via-indigo-50 to-purple-50">
      {/* API Status Banner */}
      {apiStatus === 'disconnected' && (
        <div className="bg-red-500 text-white px-4 py-2 text-center text-sm">
          ⚠️ Backend API is not responding. Please ensure the FastAPI server is running on port 8000.
        </div>
      )}

      <div className="container mx-auto px-4 py-8 max-w-6xl">
        {/* Header */}
        <div className="text-center mb-8">
          <div className="flex items-center justify-center gap-3 mb-2">
            <Sparkles className="w-10 h-10 text-purple-600" />
            <h1 className="text-4xl font-bold bg-gradient-to-r from-purple-600 to-indigo-600 bg-clip-text text-transparent">
              LSTM Text Generation
            </h1>
          </div>
          <p className="text-gray-600 text-lg">Next-word prediction powered by LSTM neural network</p>
        </div>

        {/* Main Content */}
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6 mt-8">
          {/* Left Column - Text Generation */}
          <div className="lg:col-span-2">
            <TextGenerationSection 
              lstmStatus={lstmStatus}
              selectedSeedText={selectedSeedText}
              onGenerate={(data) => console.log('Generated:', data)}
            />
          </div>

          {/* Right Column - Status and Examples */}
          <div className="space-y-6">
            <LSTMStatusCard onStatusUpdate={setLstmStatus} />
            <LSTMInfoSection />
            <LSTMExamplesSection 
              onSelectExample={handleSelectExample}
              lstmStatus={lstmStatus}
            />
          </div>
        </div>

        {/* Footer */}
        <footer className="mt-16 text-center text-gray-600 text-sm">
          <div className="flex items-center justify-center gap-2 mb-2">
            <Sparkles className="w-4 h-4" />
            <span>LSTM Text Generation</span>
          </div>
          <p>Built with React + FastAPI + TensorFlow/Keras</p>
          <p className="mt-2 text-xs text-gray-500">CST-435 Deep Learning Project</p>
        </footer>
      </div>
    </div>
  );
}

export default App;
