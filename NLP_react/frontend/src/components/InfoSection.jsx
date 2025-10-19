import React from 'react'

export default function InfoSection(){
  return (
    <section className="text-sm text-gray-700">
      <h4 className="font-medium">About</h4>
      <p className="mt-2">This frontend talks to a FastAPI backend that wraps an existing TF-IDF vectorizer and trained model. Ensure the backend is running at <code>http://localhost:8000</code>.</p>
    </section>
  )
}
