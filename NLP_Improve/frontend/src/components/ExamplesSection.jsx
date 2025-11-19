import React from 'react'

const examples = [
  { text: 'I love this product, it works great!', emoji: '😍', color: 'from-green-400 to-emerald-500' },
  { text: 'This is the worst experience I have ever had.', emoji: '😡', color: 'from-red-400 to-orange-500' },
  { text: 'The service was okay, not exceptional but fine.', emoji: '😐', color: 'from-yellow-400 to-amber-500' }
]

export default function ExamplesSection({ onPick }){
  return (
    <section className="glass rounded-2xl p-6 hover-lift">
      <div className="flex items-center gap-3 mb-4">
        <div className="w-10 h-10 bg-gradient-to-br from-yellow-500 to-orange-500 rounded-lg flex items-center justify-center text-white text-xl">
          💡
        </div>
        <h4 className="text-xl font-bold text-gray-800">Try Examples</h4>
      </div>
      <div className="space-y-3">
        {examples.map((ex, i) => (
          <button 
            key={i} 
            className={`w-full p-4 bg-gradient-to-r ${ex.color} text-white rounded-xl text-left hover:shadow-lg transition-all transform hover:scale-105 font-medium`}
            onClick={() => onPick(ex.text)}
          >
            <div className="flex items-center gap-3">
              <span className="text-2xl">{ex.emoji}</span>
              <span className="flex-1 text-sm">{ex.text}</span>
            </div>
          </button>
        ))}
      </div>
    </section>
  )
}
