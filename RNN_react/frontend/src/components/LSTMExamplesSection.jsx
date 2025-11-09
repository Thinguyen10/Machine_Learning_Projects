import { useState, useEffect } from 'react';
import { BookOpen, Loader2 } from 'lucide-react';
import { getLSTMExamples } from '../services/api';

/**
 * LSTMExamplesSection Component
 * Displays example seed phrases for text generation
 */
function LSTMExamplesSection({ onSelectExample, lstmStatus }) {
  const [examples, setExamples] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    loadExamples();
  }, []);

  const loadExamples = async () => {
    try {
      const data = await getLSTMExamples();
      setExamples(data);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  if (loading) {
    return (
      <div className="bg-white rounded-xl shadow-lg p-6 border border-purple-100">
        <div className="flex items-center justify-center py-8">
          <Loader2 className="w-6 h-6 animate-spin text-purple-600" />
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="bg-white rounded-xl shadow-lg p-6 border border-purple-100">
        <p className="text-sm text-red-600">Failed to load examples</p>
      </div>
    );
  }

  const categories = [
    { key: 'simple', label: 'Simple Phrases', color: 'blue' },
    { key: 'questions', label: 'Questions', color: 'green' },
    { key: 'actions', label: 'Actions', color: 'orange' },
    { key: 'descriptions', label: 'Descriptions', color: 'purple' },
  ];

  return (
    <div className="bg-white rounded-xl shadow-lg p-6 border border-purple-100">
      {/* Header */}
      <div className="flex items-center gap-2 mb-4">
        <BookOpen className="w-5 h-5 text-purple-600" />
        <h3 className="text-lg font-bold text-gray-800">Example Seeds</h3>
      </div>

      {/* Categories */}
      <div className="space-y-4">
        {categories.map((category) => (
          <div key={category.key}>
            <h4 className="text-xs font-semibold text-gray-600 uppercase mb-2">
              {category.label}
            </h4>
            <div className="space-y-1">
              {examples[category.key]?.map((example, index) => (
                <button
                  key={index}
                  onClick={() => onSelectExample(example)}
                  disabled={!lstmStatus?.model_loaded}
                  className={`w-full text-left px-3 py-2 rounded-lg text-sm transition-all ${
                    lstmStatus?.model_loaded
                      ? `hover:bg-${category.color}-50 hover:border-${category.color}-200 border border-transparent`
                      : 'opacity-50 cursor-not-allowed'
                  }`}
                >
                  <span className="text-gray-700">{example}</span>
                </button>
              ))}
            </div>
          </div>
        ))}
      </div>

      {/* Instructions */}
      <div className="mt-4 pt-4 border-t border-gray-200">
        <p className="text-xs text-gray-500">
          💡 Click any example to use it as seed text
        </p>
      </div>
    </div>
  );
}

export default LSTMExamplesSection;
