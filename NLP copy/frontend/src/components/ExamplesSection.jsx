import React from 'react'

const examples = [
  'I love this product, it works great!',
  'This is the worst experience I have ever had.',
  'The service was okay, not exceptional but fine.'
]

export default function ExamplesSection({ onPick }){
  return (
    <section>
      <h4 className="font-medium">Examples</h4>
      <div className="mt-2 flex gap-2">
        {examples.map((ex,i)=> (
          <button key={i} className="px-3 py-1 border rounded text-sm" onClick={()=>onPick(ex)}>{ex}</button>
        ))}
      </div>
    </section>
  )
}
