import { Info, BookOpen, Zap, Brain } from 'lucide-react';

/**
 * LSTMInfoSection Component
 * Explains how LSTM text generation works
 */
function LSTMInfoSection() {
  return (
    <div className="bg-white rounded-xl shadow-lg p-6 border border-purple-100">
      {/* Header */}
      <div className="flex items-center gap-2 mb-4">
        <Info className="w-5 h-5 text-purple-600" />
        <h3 className="text-lg font-bold text-gray-800">How It Works</h3>
      </div>

      {/* Explanation Sections */}
      <div className="space-y-4">
        {/* What is LSTM */}
        <div className="bg-purple-50 p-4 rounded-lg">
          <div className="flex items-start gap-2 mb-2">
            <Brain className="w-5 h-5 text-purple-600 mt-0.5 flex-shrink-0" />
            <div>
              <h4 className="font-semibold text-gray-800 mb-1">What is LSTM?</h4>
              <p className="text-sm text-gray-700">
                LSTM (Long Short-Term Memory) is a type of neural network that learns patterns 
                in sequences. It was trained on text to predict what word comes next.
              </p>
            </div>
          </div>
        </div>

        {/* Seed Text Explanation */}
        <div className="bg-indigo-50 p-4 rounded-lg">
          <div className="flex items-start gap-2 mb-2">
            <BookOpen className="w-5 h-5 text-indigo-600 mt-0.5 flex-shrink-0" />
            <div>
              <h4 className="font-semibold text-gray-800 mb-1">What is Seed Text?</h4>
              <p className="text-sm text-gray-700 mb-2">
                The <span className="font-semibold">seed text</span> is your starting phrase. 
                The model reads these words and continues the sentence.
              </p>
              <div className="bg-white p-2 rounded text-xs">
                <p className="text-gray-600 mb-1">Example:</p>
                <p className="text-indigo-700">
                  Seed: <span className="font-semibold">"once upon a"</span>
                </p>
                <p className="text-gray-700">
                  Generated: "once upon a <span className="font-semibold">time there was</span>"
                </p>
              </div>
            </div>
          </div>
        </div>

        {/* Parameters Explanation */}
        <div className="bg-blue-50 p-4 rounded-lg">
          <div className="flex items-start gap-2 mb-2">
            <Zap className="w-5 h-5 text-blue-600 mt-0.5 flex-shrink-0" />
            <div>
              <h4 className="font-semibold text-gray-800 mb-1">Control Parameters</h4>
              
              <div className="space-y-2 text-sm text-gray-700">
                <div>
                  <span className="font-semibold">Number of Words:</span> How many words to add 
                  after your seed text.
                </div>
                
                <div>
                  <span className="font-semibold">Temperature (Creativity):</span>
                  <ul className="ml-4 mt-1 space-y-1">
                    <li>• <span className="font-mono bg-white px-1 rounded">0.3-0.5</span>: Safe, predictable, coherent</li>
                    <li>• <span className="font-mono bg-white px-1 rounded">0.6-0.8</span>: Balanced (recommended)</li>
                    <li>• <span className="font-mono bg-white px-1 rounded">1.0-2.0</span>: Creative, surprising, risky</li>
                  </ul>
                </div>
              </div>
            </div>
          </div>
        </div>

        {/* Tips */}
        <div className="bg-gray-50 p-3 rounded-lg">
          <p className="text-xs font-semibold text-gray-700 mb-2">💡 Tips for Best Results:</p>
          <ul className="text-xs text-gray-600 space-y-1 ml-4">
            <li>• Use 3-5 words as seed text</li>
            <li>• Start with temperature 0.5-0.7</li>
            <li>• Try different seeds with same settings</li>
            <li>• Lower temperature for formal text</li>
            <li>• Higher temperature for creative stories</li>
          </ul>
        </div>
      </div>
    </div>
  );
}

export default LSTMInfoSection;
