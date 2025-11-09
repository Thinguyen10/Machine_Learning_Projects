import { BookOpen, Brain, Zap, ArrowRight, Network, Layers, TrendingUp } from 'lucide-react';

function IntroPage({ onGetStarted }) {
  return (
    <div className="min-h-screen bg-gradient-to-br from-purple-50 via-white to-indigo-50">
      {/* Hero Section */}
      <div className="max-w-6xl mx-auto px-6 py-16">
        {/* Header */}
        <div className="text-center mb-16">
          <div className="inline-flex items-center gap-2 bg-purple-100 text-purple-700 px-4 py-2 rounded-full text-sm font-medium mb-6">
            <Brain className="w-4 h-4" />
            <span>Deep Learning Project - CST-435</span>
          </div>
          <h1 className="text-5xl md:text-6xl font-bold bg-gradient-to-r from-purple-600 to-indigo-600 bg-clip-text text-transparent mb-6">
            LSTM Text Generation
          </h1>
          <p className="text-xl text-gray-600 max-w-3xl mx-auto leading-relaxed">
            Experience the power of Recurrent Neural Networks trained on F. Scott Fitzgerald's
            <span className="font-semibold text-purple-600"> The Great Gatsby</span> to generate
            authentic early 20th-century American prose.
          </p>
        </div>

        {/* Main Content Grid */}
        <div className="grid md:grid-cols-2 gap-8 mb-12">
          {/* What is LSTM Card */}
          <div className="bg-white rounded-2xl shadow-lg p-8 border border-purple-100 hover:shadow-xl transition-shadow">
            <div className="flex items-start gap-4 mb-4">
              <div className="bg-purple-100 p-3 rounded-lg">
                <Network className="w-6 h-6 text-purple-600" />
              </div>
              <div>
                <h2 className="text-2xl font-bold text-gray-800 mb-2">What is LSTM?</h2>
                <div className="w-16 h-1 bg-gradient-to-r from-purple-600 to-indigo-600 rounded"></div>
              </div>
            </div>
            <p className="text-gray-600 leading-relaxed mb-4">
              <strong className="text-purple-600">Long Short-Term Memory (LSTM)</strong> is a specialized 
              type of Recurrent Neural Network that excels at learning patterns in sequential data like text.
            </p>
            <p className="text-gray-600 leading-relaxed mb-4">
              Unlike traditional neural networks, LSTMs have <strong>memory cells</strong> that can retain 
              information over long sequences, making them perfect for understanding language context and 
              generating coherent text.
            </p>
            <div className="bg-purple-50 rounded-lg p-4 border border-purple-100">
              <p className="text-sm text-purple-800 font-medium">
                💡 Think of LSTM as a writer who remembers the entire story context while writing each new sentence.
              </p>
            </div>
          </div>

          {/* How It Works Card */}
          <div className="bg-white rounded-2xl shadow-lg p-8 border border-indigo-100 hover:shadow-xl transition-shadow">
            <div className="flex items-start gap-4 mb-4">
              <div className="bg-indigo-100 p-3 rounded-lg">
                <Layers className="w-6 h-6 text-indigo-600" />
              </div>
              <div>
                <h2 className="text-2xl font-bold text-gray-800 mb-2">How It Works</h2>
                <div className="w-16 h-1 bg-gradient-to-r from-indigo-600 to-purple-600 rounded"></div>
              </div>
            </div>
            <div className="space-y-4">
              <div className="flex gap-3">
                <div className="flex-shrink-0 w-8 h-8 bg-indigo-100 rounded-full flex items-center justify-center text-indigo-600 font-bold">
                  1
                </div>
                <div>
                  <h3 className="font-semibold text-gray-800 mb-1">Training</h3>
                  <p className="text-sm text-gray-600">
                    The model reads The Great Gatsby sentence by sentence, learning patterns,
                    vocabulary, and F. Scott Fitzgerald's writing style.
                  </p>
                </div>
              </div>
              <div className="flex gap-3">
                <div className="flex-shrink-0 w-8 h-8 bg-indigo-100 rounded-full flex items-center justify-center text-indigo-600 font-bold">
                  2
                </div>
                <div>
                  <h3 className="font-semibold text-gray-800 mb-1">Context Window</h3>
                  <p className="text-sm text-gray-600">
                    Uses a sliding window of 7 words to predict the next word, capturing 
                    grammatical structure and semantic relationships.
                  </p>
                </div>
              </div>
              <div className="flex gap-3">
                <div className="flex-shrink-0 w-8 h-8 bg-indigo-100 rounded-full flex items-center justify-center text-indigo-600 font-bold">
                  3
                </div>
                <div>
                  <h3 className="font-semibold text-gray-800 mb-1">Generation</h3>
                  <p className="text-sm text-gray-600">
                    Given your seed text, the LSTM predicts each next word based on learned patterns, 
                    generating authentic-sounding prose.
                  </p>
                </div>
              </div>
            </div>
          </div>
        </div>

        {/* Model Architecture Section */}
        <div className="bg-gradient-to-br from-purple-600 to-indigo-600 rounded-2xl shadow-xl p-8 text-white mb-12">
          <div className="flex items-center gap-3 mb-6">
            <Brain className="w-8 h-8" />
            <h2 className="text-3xl font-bold">Model Architecture</h2>
          </div>
          <div className="grid md:grid-cols-3 gap-6">
            <div className="bg-white/10 backdrop-blur-sm rounded-xl p-6 border border-white/20">
              <div className="text-4xl font-bold mb-2">1000</div>
              <div className="text-purple-100">Vocabulary Size</div>
              <p className="text-sm text-purple-200 mt-2">Most common words from the novel</p>
            </div>
            <div className="bg-white/10 backdrop-blur-sm rounded-xl p-6 border border-white/20">
              <div className="text-4xl font-bold mb-2">256</div>
              <div className="text-purple-100">LSTM Units</div>
              <p className="text-sm text-purple-200 mt-2">Hidden memory cells for learning</p>
            </div>
            <div className="bg-white/10 backdrop-blur-sm rounded-xl p-6 border border-white/20">
              <div className="text-4xl font-bold mb-2">7</div>
              <div className="text-purple-100">Context Words</div>
              <p className="text-sm text-purple-200 mt-2">Window size for predictions</p>
            </div>
          </div>
        </div>

        {/* Training Stats Section */}
        <div className="grid md:grid-cols-2 gap-6 mb-12">
          <div className="bg-white rounded-xl shadow-md p-6 border border-gray-200">
            <div className="flex items-center gap-3 mb-4">
              <BookOpen className="w-6 h-6 text-purple-600" />
              <h3 className="text-xl font-bold text-gray-800">Training Data</h3>
            </div>
            <ul className="space-y-2 text-gray-600">
              <li className="flex items-center gap-2">
                <span className="w-2 h-2 bg-purple-600 rounded-full"></span>
                <strong>Source:</strong> The Great Gatsby by F. Scott Fitzgerald
              </li>
              <li className="flex items-center gap-2">
                <span className="w-2 h-2 bg-purple-600 rounded-full"></span>
                <strong>Size:</strong> ~299 KB (Project Gutenberg edition)
              </li>
              <li className="flex items-center gap-2">
                <span className="w-2 h-2 bg-purple-600 rounded-full"></span>
                <strong>Examples:</strong> 90,000+ training sequences
              </li>
            </ul>
          </div>
          
          <div className="bg-white rounded-xl shadow-md p-6 border border-gray-200">
            <div className="flex items-center gap-3 mb-4">
              <TrendingUp className="w-6 h-6 text-indigo-600" />
              <h3 className="text-xl font-bold text-gray-800">Performance</h3>
            </div>
            <ul className="space-y-2 text-gray-600">
              <li className="flex items-center gap-2">
                <span className="w-2 h-2 bg-indigo-600 rounded-full"></span>
                <strong>Top-1 Accuracy:</strong> ~17% (1 in 6 correct)
              </li>
              <li className="flex items-center gap-2">
                <span className="w-2 h-2 bg-indigo-600 rounded-full"></span>
                <strong>Top-5 Accuracy:</strong> ~38% (correct in top 5)
              </li>
              <li className="flex items-center gap-2">
                <span className="w-2 h-2 bg-indigo-600 rounded-full"></span>
                <strong>Perplexity:</strong> ~106 (lower is better)
              </li>
            </ul>
          </div>
        </div>

        {/* How to Use Section */}
        <div className="bg-gradient-to-br from-amber-50 to-orange-50 rounded-2xl shadow-md p-8 border border-amber-200 mb-12">
          <div className="flex items-center gap-3 mb-6">
            <Zap className="w-7 h-7 text-amber-600" />
            <h2 className="text-2xl font-bold text-gray-800">How to Use</h2>
          </div>
          <div className="grid md:grid-cols-3 gap-6">
            <div className="flex gap-3">
              <div className="flex-shrink-0 w-10 h-10 bg-amber-500 rounded-full flex items-center justify-center text-white font-bold text-lg">
                1
              </div>
              <div>
                <h4 className="font-semibold text-gray-800 mb-1">Enter Seed Text</h4>
                <p className="text-sm text-gray-600">
                  Start with 3-7 words in Gatsby's style (e.g., "in my younger", "gatsby believed in")
                </p>
              </div>
            </div>
            <div className="flex gap-3">
              <div className="flex-shrink-0 w-10 h-10 bg-amber-500 rounded-full flex items-center justify-center text-white font-bold text-lg">
                2
              </div>
              <div>
                <h4 className="font-semibold text-gray-800 mb-1">Adjust Parameters</h4>
                <p className="text-sm text-gray-600">
                  Choose how many words to generate and set creativity level (temperature)
                </p>
              </div>
            </div>
            <div className="flex gap-3">
              <div className="flex-shrink-0 w-10 h-10 bg-amber-500 rounded-full flex items-center justify-center text-white font-bold text-lg">
                3
              </div>
              <div>
                <h4 className="font-semibold text-gray-800 mb-1">Generate!</h4>
                <p className="text-sm text-gray-600">
                  Watch the AI continue your text in authentic Jane Austen style
                </p>
              </div>
            </div>
          </div>
        </div>

        {/* CTA Button */}
        <div className="text-center">
          <button
            onClick={onGetStarted}
            className="group inline-flex items-center gap-3 bg-gradient-to-r from-purple-600 to-indigo-600 text-white px-8 py-4 rounded-xl font-semibold text-lg shadow-lg hover:shadow-xl transform hover:scale-105 transition-all duration-200"
          >
            <span>Start Generating Text</span>
            <ArrowRight className="w-5 h-5 group-hover:translate-x-1 transition-transform" />
          </button>
          <p className="text-sm text-gray-500 mt-4">
            No setup required • Instant results • Try different creative levels
          </p>
        </div>

        {/* Footer Note */}
        <div className="mt-16 text-center text-sm text-gray-500 border-t border-gray-200 pt-8">
          <p>
            Built with <strong>TensorFlow/Keras</strong> • <strong>React</strong> • <strong>FastAPI</strong>
          </p>
          <p className="mt-2">
            Training corpus: <em>The Great Gatsby</em> by F. Scott Fitzgerald (Project Gutenberg)
          </p>
        </div>
      </div>
    </div>
  );
}

export default IntroPage;
