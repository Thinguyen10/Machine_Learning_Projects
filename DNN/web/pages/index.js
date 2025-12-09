import { useState } from 'react';
import Head from 'next/head';
import { useRouter } from 'next/router';

export default function Home() {
  const router = useRouter();
  const [text, setText] = useState('');
  const [sentiment, setSentiment] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [mode, setMode] = useState('hybrid');

  const analyzeSentiment = async () => {
    if (!text.trim()) {
      setError('Please enter some text');
      return;
    }

    setLoading(true);
    setError(null);
    setSentiment(null);

    try {
      // Use local backend server during development, Vercel API in production
      const apiUrl = process.env.NODE_ENV === 'production' 
        ? '/api/predict' 
        : 'http://localhost:8000/api/predict';
      
      const response = await fetch(apiUrl, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text, mode })
      });

      if (!response.ok) {
        throw new Error('Failed to analyze sentiment');
      }

      const data = await response.json();
      setSentiment(data);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  const handleKeyPress = (e) => {
    if (e.key === 'Enter' && e.ctrlKey) {
      analyzeSentiment();
    }
  };

  return (
    <>
      <Head>
        <title>AI Sentiment Analysis | Deep Learning</title>
        <meta name="description" content="Real-time sentiment analysis using DistilBERT transformer" />
        <link rel="icon" href="/favicon.ico" />
      </Head>

      <div className="container">
        <main className="main">
          <h1 className="title">
            🤖 AI Sentiment Analysis
          </h1>
          
          <p className="description">
            🧠 Hybrid AI: <strong>RNN Fast Filter</strong> + <strong>DistilBERT Verifier</strong>
          </p>

          {/* Navigation Buttons */}
          <div style={{ 
            display: 'flex', 
            gap: '10px', 
            marginBottom: '1rem',
            flexWrap: 'wrap',
            justifyContent: 'center'
          }}>
            <button
              onClick={() => router.push('/upload')}
              style={{
                padding: '10px 20px',
                background: 'white',
                color: '#FF512F',
                border: '2px solid white',
                borderRadius: '8px',
                fontSize: '14px',
                fontWeight: '600',
                cursor: 'pointer',
                transition: 'all 0.3s ease'
              }}
            >
              📊 Batch Upload
            </button>
            <button
              onClick={() => router.push('/dashboard')}
              style={{
                padding: '10px 20px',
                background: 'white',
                color: '#FF512F',
                border: '2px solid white',
                borderRadius: '8px',
                fontSize: '14px',
                fontWeight: '600',
                cursor: 'pointer',
                transition: 'all 0.3s ease'
              }}
            >
              📈 Dashboard
            </button>
            <button
              onClick={() => router.push('/learn')}
              style={{
                padding: '10px 20px',
                background: 'white',
                color: '#FF512F',
                border: '2px solid white',
                borderRadius: '8px',
                fontSize: '14px',
                fontWeight: '600',
                cursor: 'pointer',
                transition: 'all 0.3s ease'
              }}
            >
              📚 Learn More
            </button>
          </div>
          
          <div className="mode-selector">
            <button 
              className={`mode-btn ${mode === 'sequential' ? 'active' : ''}`}
              onClick={() => setMode('sequential')}
            >
              🔬 Sequential
            </button>
            <button 
              className={`mode-btn ${mode === 'business-insights' ? 'active' : ''}`}
              onClick={() => setMode('business-insights')}
            >
              💼 Business Insights
            </button>
          </div>

          <div className="card">
            <textarea
              className="textarea"
              value={text}
              onChange={(e) => setText(e.target.value)}
              onKeyPress={handleKeyPress}
              placeholder="Type or paste your text here...\n\nExamples:\n• Product reviews\n• Movie reviews\n• Social media posts\n• Customer feedback"
              rows={8}
            />
            
            <button 
              className={`button ${loading ? 'loading' : ''}`}
              onClick={analyzeSentiment}
              disabled={loading}
            >
              {loading ? '🔄 Analyzing...' : 
               mode === 'business-insights' ? '🚀 Analyze (Get Business Insights)' :
               mode === 'sequential' ? '🚀 Analyze (Sequential)' :
               '🚀 Analyze (Compare)'}
            </button>
          </div>

          {error && (
            <div className="result error">
              <p>❌ {error}</p>
            </div>
          )}

          {sentiment && sentiment.predictions && sentiment.predictions.final_prediction && (
            <div className={`hybrid-result ${sentiment.predictions.final_prediction.label.toLowerCase()}`}>
              <div className="hybrid-header">
                <h2>🎯 Smart Hybrid Result</h2>
                <span className="model-badge">{sentiment.predictions.model_used}</span>
              </div>
              
              <div className="sentiment-display">
                <div className="label-large">
                  {sentiment.predictions.final_prediction.label === 'Positive' ? '😊' : 
                   sentiment.predictions.final_prediction.label === 'Negative' ? '😞' : '😐'} 
                  <span>{sentiment.predictions.final_prediction.label}</span>
                </div>
                <div className="confidence-large">
                  {(sentiment.predictions.final_prediction.confidence * 100).toFixed(2)}%
                </div>
              </div>
              
              <div className="progress-bar-large">
                <div 
                  className="progress-fill"
                  style={{ width: `${sentiment.predictions.final_prediction.confidence * 100}%` }}
                />
              </div>

              <div className="reasoning">
                <p><strong>Decision Process:</strong></p>
                <p>{sentiment.predictions.reason}</p>
                {sentiment.predictions.explanation && (
                  <p className="explanation-text">💡 {sentiment.predictions.explanation}</p>
                )}
                {sentiment.predictions.pipeline_stage && (
                  <p className="pipeline-stage">📍 {sentiment.predictions.pipeline_stage}</p>
                )}
                {sentiment.predictions.agreement && (
                  <p className="agreement-status">
                    {sentiment.predictions.agreement === 'Agreed' ? '✅' : '⚠️'} Models: {sentiment.predictions.agreement}
                  </p>
                )}
              </div>

              <div className="probabilities">
                <div className="prob-item">
                  <span>😞 Negative:</span>
                  <span>{(sentiment.predictions.final_prediction.probabilities.negative * 100).toFixed(1)}%</span>
                </div>
                <div className="prob-item">
                  <span>😊 Positive:</span>
                  <span>{(sentiment.predictions.final_prediction.probabilities.positive * 100).toFixed(1)}%</span>
                </div>
              </div>

              {sentiment.predictions.distilbert_used && (
                <div className="efficiency-note">
                  ⚡ RNN confidence was low - DistilBERT verification used for accuracy
                </div>
              )}
              {!sentiment.predictions.distilbert_used && (
                <div className="efficiency-note">
                  ⚡ RNN confidence was high - Fast prediction without DistilBERT overhead
                </div>
              )}

              <div className="stats-footer">
                <p><strong>Processing:</strong> {sentiment.processing_time}ms</p>
                <p><strong>Text:</strong> {sentiment.text_length} chars</p>
              </div>
            </div>
          )}

          {sentiment && sentiment.predictions && sentiment.predictions.mode === 'business-insights' && (
            <div className="business-insights">
              <div className="insights-header">
                <h2>💼 Business Intelligence Report</h2>
                <div className="insights-meta">
                  <span className="badge">{sentiment.predictions.total_aspects} Aspects Detected</span>
                  <span className="badge">Overall: {sentiment.predictions.overall_sentiment.consensus}</span>
                </div>
              </div>

              <div className="insights-summary">
                <h3>📊 Executive Summary</h3>
                <p>{sentiment.predictions.business_insights.overall_summary}</p>
              </div>

              {Object.keys(sentiment.predictions.aspect_analysis).length > 0 && (
                <div className="aspects-grid">
                  <h3>🎯 Aspect Breakdown</h3>
                  {Object.entries(sentiment.predictions.aspect_analysis).map(([aspect, data]) => (
                    <div key={aspect} className={`aspect-card ${data.dominant_sentiment.toLowerCase()}`}>
                      <div className="aspect-header">
                        <h4>{aspect.toUpperCase()}</h4>
                        <span className={`sentiment-badge ${data.dominant_sentiment.toLowerCase()}`}>
                          {data.dominant_sentiment === 'Positive' ? '😊' : 
                           data.dominant_sentiment === 'Negative' ? '😞' : '😐'} {data.dominant_sentiment}
                        </span>
                      </div>
                      <div className="aspect-stats">
                        <p><strong>Mentions:</strong> {data.mention_count}</p>
                        <p><strong>Confidence:</strong> {(data.confidence * 100).toFixed(1)}%</p>
                      </div>
                      {data.sample_mentions && data.sample_mentions.length > 0 && (
                        <div className="sample-mentions">
                          <p><em>"{data.sample_mentions[0].text}"</em></p>
                        </div>
                      )}
                    </div>
                  ))}
                </div>
              )}

              <div className="insights-sections">
                {sentiment.predictions.business_insights.strengths.length > 0 && (
                  <div className="insight-section strengths">
                    <h3>✅ Strengths (Maintain & Promote)</h3>
                    {sentiment.predictions.business_insights.strengths.map((item, idx) => (
                      <div key={idx} className="insight-item">
                        <strong>{item.aspect}</strong> ({item.mentions} mentions, {item.confidence} confidence)
                        <p className="action">→ {item.action}</p>
                      </div>
                    ))}
                  </div>
                )}

                {sentiment.predictions.business_insights.weaknesses.length > 0 && (
                  <div className="insight-section weaknesses">
                    <h3>⚠️ Weaknesses (Immediate Action Required)</h3>
                    {sentiment.predictions.business_insights.weaknesses.map((item, idx) => (
                      <div key={idx} className="insight-item">
                        <strong>{item.aspect}</strong> ({item.mentions} mentions, {item.confidence} confidence)
                        <p className="action">→ {item.action}</p>
                      </div>
                    ))}
                  </div>
                )}

                {sentiment.predictions.business_insights.priorities.length > 0 && (
                  <div className="insight-section priorities">
                    <h3>🎯 Priority Action Items</h3>
                    {sentiment.predictions.business_insights.priorities.map((item, idx) => (
                      <div key={idx} className="priority-item">
                        <span className="rank">#{item.rank}</span>
                        <div>
                          <strong>{item.aspect}</strong>
                          <span className={`urgency ${item.urgency.toLowerCase()}`}>{item.urgency} Urgency</span>
                          <p>{item.reason}</p>
                        </div>
                      </div>
                    ))}
                  </div>
                )}

                {sentiment.predictions.business_insights.recommendations.length > 0 && (
                  <div className="insight-section recommendations">
                    <h3>💡 Strategic Recommendations</h3>
                    {sentiment.predictions.business_insights.recommendations.map((rec, idx) => (
                      <div key={idx} className="recommendation-item">
                        {rec}
                      </div>
                    ))}
                  </div>
                )}
              </div>
            </div>
          )}

          {sentiment && sentiment.predictions && sentiment.predictions.distilbert && sentiment.predictions.rnn && (
            <div className="results-grid">
              <div className={`model-result ${sentiment.predictions.distilbert.label.toLowerCase()}`}>
                <h3>🤖 DistilBERT</h3>
                <div className="accuracy-badge">94.22% accuracy</div>
                
                <div className="sentiment-info">
                  <div className="label">
                    {sentiment.predictions.distilbert.label === 'Positive' ? '😊' : 
                     sentiment.predictions.distilbert.label === 'Negative' ? '😞' : '😐'} 
                    {sentiment.predictions.distilbert.label}
                  </div>
                  <div className="confidence">
                    {(sentiment.predictions.distilbert.confidence * 100).toFixed(2)}%
                  </div>
                </div>
                
                <div className="progress-bar">
                  <div 
                    className="progress-fill"
                    style={{ width: `${sentiment.predictions.distilbert.confidence * 100}%` }}
                  />
                </div>

                <div className="probabilities">
                  <div className="prob-item">
                    <span>Negative:</span>
                    <span>{(sentiment.predictions.distilbert.probabilities.negative * 100).toFixed(1)}%</span>
                  </div>
                  <div className="prob-item">
                    <span>Positive:</span>
                    <span>{(sentiment.predictions.distilbert.probabilities.positive * 100).toFixed(1)}%</span>
                  </div>
                </div>
              </div>

              <div className={`model-result ${sentiment.predictions.rnn.label.toLowerCase()}`}>
                <h3>🧠 RNN + Attention</h3>
                <div className="accuracy-badge">87.56% accuracy</div>
                
                <div className="sentiment-info">
                  <div className="label">
                    {sentiment.predictions.rnn.label === 'Positive' ? '😊' : 
                     sentiment.predictions.rnn.label === 'Negative' ? '😞' : '😐'} 
                    {sentiment.predictions.rnn.label}
                  </div>
                  <div className="confidence">
                    {(sentiment.predictions.rnn.confidence * 100).toFixed(2)}%
                  </div>
                </div>
                
                <div className="progress-bar">
                  <div 
                    className="progress-fill"
                    style={{ width: `${sentiment.predictions.rnn.confidence * 100}%` }}
                  />
                </div>

                <div className="probabilities">
                  <div className="prob-item">
                    <span>Negative:</span>
                    <span>{(sentiment.predictions.rnn.probabilities.negative * 100).toFixed(1)}%</span>
                  </div>
                  <div className="prob-item">
                    <span>Positive:</span>
                    <span>{(sentiment.predictions.rnn.probabilities.positive * 100).toFixed(1)}%</span>
                  </div>
                </div>
              </div>
            </div>
          )}

          {sentiment && (sentiment.predictions.final_prediction || sentiment.predictions.distilbert) && (
            <div className="stats-footer-main">
              <p><strong>Processing Time:</strong> {sentiment.processing_time}ms</p>
              <p><strong>Three-Class Classification:</strong> Positive / Neutral / Negative</p>
              <p><strong>Neutral Detection:</strong> Confidence &lt; 85% indicates mixed/uncertain sentiment</p>
            </div>
          )}

          <footer className="footer">
            <p>Built with Next.js + Python | Models: RNN (87.56%) vs DistilBERT (94.22%)</p>
          </footer>
        </main>
      </div>

      <style jsx>{`
        .container {
          min-height: 100vh;
          padding: 0 2rem;
          display: flex;
          flex-direction: column;
          justify-content: center;
          align-items: center;
          background: linear-gradient(135deg, #2E3192 0%, #1BFFFF 100%); // Vibrant gradient background
        }

        .main {
          padding: 3rem 0;
          flex: 1;
          display: flex;
          flex-direction: column;
          align-items: center;
          max-width: 800px;
          width: 100%;
        }

        .title {
          margin: 0;
          font-size: 3rem;
          color: white;
          text-align: center;
          text-shadow: 2px 2px 4px rgba(0,0,0,0.2);
        }

        .description {
          text-align: center;
          font-size: 1.2rem;
          color: rgba(255,255,255,0.9);
          margin: 1rem 0 1.5rem;
        }

        .learn-more-btn {
          background: rgba(255,255,255,0.2);
          color: white;
          border: 2px solid white;
          padding: 12px 30px;
          border-radius: 25px;
          font-size: 1rem;
          cursor: pointer;
          transition: all 0.3s;
          margin-bottom: 2rem;
          display: block;
          margin-left: auto;
          margin-right: auto;
        }

        .learn-more-btn:hover {
          background: white;
          color: #667eea;
          transform: translateY(-2px);
          box-shadow: 0 5px 15px rgba(0,0,0,0.2);
        }

        .mode-selector {
          display: flex;
          gap: 1rem;
          margin-bottom: 2rem;
          flex-wrap: wrap;
        }

        .mode-btn {
          flex: 1;
          min-width: 150px;
          padding: 0.8rem 1.5rem;
          font-size: 1rem;
          font-weight: 600;
          color: white;
          background: rgba(255,255,255,0.2);
          border: 2px solid rgba(255,255,255,0.3);
          border-radius: 12px;
          cursor: pointer;
          transition: all 0.3s;
          backdrop-filter: blur(10px);
        }

        .mode-btn:hover {
          background: rgba(255,255,255,0.3);
          transform: translateY(-2px);
        }

        .mode-btn.active {
          background: white;
          color: #667eea;
          border-color: white;
          box-shadow: 0 5px 20px rgba(255,255,255,0.3);
        }

        .card {
          background: white;
          border-radius: 16px;
          padding: 2rem;
          box-shadow: 0 10px 40px rgba(0,0,0,0.2);
          width: 100%;
          margin-bottom: 2rem;
        }

        .textarea {
          width: 100%;
          padding: 1rem;
          font-size: 1rem;
          border: 2px solid #e0e0e0;
          border-radius: 8px;
          resize: vertical;
          font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, sans-serif;
          transition: border-color 0.3s;
        }

        .textarea:focus {
          outline: none;
          border-color: #667eea;
        }

        .button {
          width: 100%;
          margin-top: 1rem;
          padding: 1rem 2rem;
          font-size: 1.1rem;
          font-weight: bold;
          color: white;
          background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
          border: none;
          border-radius: 8px;
          cursor: pointer;
          transition: transform 0.2s, box-shadow 0.2s;
        }

        .button:hover:not(:disabled) {
          transform: translateY(-2px);
          box-shadow: 0 5px 20px rgba(102, 126, 234, 0.4);
        }

        .button:disabled {
          opacity: 0.7;
          cursor: not-allowed;
        }

        .button.loading {
          animation: pulse 1.5s infinite;
        }

        @keyframes pulse {
          0%, 100% { opacity: 1; }
          50% { opacity: 0.7; }
        }

        .result {
          background: white;
          border-radius: 16px;
          padding: 2rem;
          box-shadow: 0 10px 40px rgba(0,0,0,0.2);
          width: 100%;
          animation: slideIn 0.3s ease-out;
        }

        @keyframes slideIn {
          from {
            opacity: 0;
            transform: translateY(20px);
          }
          to {
            opacity: 1;
            transform: translateY(0);
          }
        }

        .result.error {
          background: #fee;
          color: #c00;
        }

        .result.positive {
          border-left: 6px solid #4caf50;
        }

        .result.negative {
          border-left: 6px solid #f44336;
        }

        .result h2 {
          margin-top: 0;
          color: #333;
        }

        .sentiment-info {
          display: flex;
          justify-content: space-between;
          align-items: center;
          margin: 1rem 0;
        }

        .label {
          font-size: 2rem;
          font-weight: bold;
        }

        .result.positive .label {
          color: #4caf50;
        }

        .result.negative .label {
          color: #f44336;
        }

        .confidence {
          font-size: 1.2rem;
          color: #666;
        }

        .progress-bar {
          width: 100%;
          height: 12px;
          background: #e0e0e0;
          border-radius: 6px;
          overflow: hidden;
          margin: 1rem 0;
        }

        .progress-fill {
          height: 100%;
          background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
          transition: width 0.5s ease-out;
        }

        .hybrid-result {
          background: white;
          border-radius: 20px;
          padding: 2.5rem;
          box-shadow: 0 15px 50px rgba(0,0,0,0.25);
          width: 100%;
          animation: slideIn 0.3s ease-out;
          margin-bottom: 1.5rem;
        }

        .hybrid-result.positive {
          border-left: 8px solid #4caf50;
        }

        .hybrid-result.negative {
          border-left: 8px solid #f44336;
        }

        .hybrid-result.neutral {
          border-left: 8px solid #ff9800;
        }

        .hybrid-header {
          display: flex;
          justify-content: space-between;
          align-items: center;
          margin-bottom: 1.5rem;
        }

        .hybrid-header h2 {
          margin: 0;
          color: #333;
        }

        .model-badge {
          background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
          color: white;
          padding: 0.5rem 1rem;
          border-radius: 20px;
          font-size: 0.9rem;
          font-weight: 600;
        }

        .sentiment-display {
          display: flex;
          justify-content: space-between;
          align-items: center;
          margin: 2rem 0;
        }

        .label-large {
          display: flex;
          align-items: center;
          gap: 1rem;
          font-size: 2.5rem;
          font-weight: bold;
        }

        .hybrid-result.positive .label-large {
          color: #4caf50;
        }

        .hybrid-result.negative .label-large {
          color: #f44336;
        }

        .hybrid-result.neutral .label-large {
          color: #ff9800;
        }

        .confidence-large {
          font-size: 3rem;
          font-weight: bold;
          color: #667eea;
        }

        .progress-bar-large {
          width: 100%;
          height: 16px;
          background: #e0e0e0;
          border-radius: 8px;
          overflow: hidden;
          margin: 1.5rem 0;
        }

        .reasoning {
          background: #f5f7fa;
          padding: 1.5rem;
          border-radius: 12px;
          margin: 1.5rem 0;
        }

        .reasoning p {
          margin: 0.5rem 0;
          color: #555;
          line-height: 1.6;
        }

        .reasoning strong {
          color: #333;
        }

        .explanation-text {
          background: linear-gradient(135deg, rgba(102,126,234,0.1) 0%, rgba(118,75,162,0.1) 100%);
          padding: 0.8rem;
          border-radius: 8px;
          margin-top: 0.5rem;
          font-style: italic;
        }

        .pipeline-stage {
          color: #667eea;
          font-weight: 600;
          margin-top: 0.5rem;
        }

        .agreement-status {
          color: #4caf50;
          font-weight: 600;
          margin-top: 0.5rem;
        }

        .probabilities {
          display: flex;
          gap: 1rem;
          margin: 1.5rem 0;
        }

        .prob-item {
          flex: 1;
          background: #f5f7fa;
          padding: 1rem;
          border-radius: 8px;
          display: flex;
          justify-content: space-between;
          font-weight: 600;
          color: #555;
        }

        .efficiency-note {
          background: linear-gradient(135deg, rgba(102,126,234,0.1) 0%, rgba(118,75,162,0.1) 100%);
          padding: 1rem;
          border-radius: 8px;
          text-align: center;
          font-weight: 600;
          color: #667eea;
          margin: 1rem 0;
        }

        .results-grid {
          display: grid;
          grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
          gap: 1.5rem;
          width: 100%;
          margin-bottom: 1.5rem;
        }

        .model-result {
          background: white;
          border-radius: 16px;
          padding: 2rem;
          box-shadow: 0 10px 40px rgba(0,0,0,0.2);
          animation: slideIn 0.3s ease-out;
        }

        .model-result.positive {
          border-left: 6px solid #4caf50;
        }

        .model-result.negative {
          border-left: 6px solid #f44336;
        }

        .model-result.neutral {
          border-left: 6px solid #ff9800;
        }

        .model-result h3 {
          margin: 0 0 1rem 0;
          color: #333;
          font-size: 1.3rem;
        }

        .accuracy-badge {
          background: #667eea;
          color: white;
          padding: 0.3rem 0.8rem;
          border-radius: 12px;
          font-size: 0.85rem;
          font-weight: 600;
          display: inline-block;
          margin-bottom: 1rem;
        }

        .stats-footer {
          border-top: 1px solid #e0e0e0;
          padding-top: 1rem;
          margin-top: 1rem;
          color: #666;
          font-size: 0.9rem;
        }

        .stats-footer p {
          margin: 0.3rem 0;
        }

        .stats-footer-main {
          background: rgba(255,255,255,0.95);
          border-radius: 12px;
          padding: 1.5rem;
          width: 100%;
          text-align: center;
          color: #555;
          font-size: 0.95rem;
          margin-bottom: 1.5rem;
        }

        .stats-footer-main p {
          margin: 0.5rem 0;
        }

        .business-insights {
          background: white;
          border-radius: 20px;
          padding: 2.5rem;
          box-shadow: 0 15px 50px rgba(0,0,0,0.25);
          width: 100%;
          animation: slideIn 0.3s ease-out;
          margin-bottom: 1.5rem;
        }

        .insights-header {
          border-bottom: 3px solid #667eea;
          padding-bottom: 1.5rem;
          margin-bottom: 2rem;
        }

        .insights-header h2 {
          margin: 0 0 1rem 0;
          color: #333;
        }

        .insights-meta {
          display: flex;
          gap: 1rem;
          flex-wrap: wrap;
        }

        .badge {
          background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
          color: white;
          padding: 0.5rem 1rem;
          border-radius: 20px;
          font-size: 0.9rem;
          font-weight: 600;
        }

        .insights-summary {
          background: #f5f7fa;
          padding: 1.5rem;
          border-radius: 12px;
          margin-bottom: 2rem;
          border-left: 4px solid #667eea;
        }

        .insights-summary h3 {
          margin-top: 0;
          color: #333;
        }

        .aspects-grid {
          margin: 2rem 0;
        }

        .aspects-grid h3 {
          color: #333;
          margin-bottom: 1rem;
        }

        .aspect-card {
          background: #f9f9f9;
          border-radius: 12px;
          padding: 1.5rem;
          margin-bottom: 1rem;
          border-left: 5px solid #ccc;
        }

        .aspect-card.positive {
          border-left-color: #4caf50;
          background: linear-gradient(135deg, rgba(76,175,80,0.05) 0%, rgba(76,175,80,0.02) 100%);
        }

        .aspect-card.negative {
          border-left-color: #f44336;
          background: linear-gradient(135deg, rgba(244,67,54,0.05) 0%, rgba(244,67,54,0.02) 100%);
        }

        .aspect-card.neutral {
          border-left-color: #ff9800;
          background: linear-gradient(135deg, rgba(255,152,0,0.05) 0%, rgba(255,152,0,0.02) 100%);
        }

        .aspect-header {
          display: flex;
          justify-content: space-between;
          align-items: center;
          margin-bottom: 1rem;
        }

        .aspect-header h4 {
          margin: 0;
          color: #333;
          font-size: 1.1rem;
        }

        .sentiment-badge {
          padding: 0.4rem 0.8rem;
          border-radius: 20px;
          font-size: 0.85rem;
          font-weight: 600;
        }

        .sentiment-badge.positive {
          background: #4caf50;
          color: white;
        }

        .sentiment-badge.negative {
          background: #f44336;
          color: white;
        }

        .sentiment-badge.neutral {
          background: #ff9800;
          color: white;
        }

        .aspect-stats {
          margin-bottom: 1rem;
        }

        .aspect-stats p {
          margin: 0.3rem 0;
          color: #666;
          font-size: 0.9rem;
        }

        .sample-mentions {
          margin-top: 1rem;
          padding-top: 1rem;
          border-top: 1px solid #e0e0e0;
        }

        .sample-mentions p {
          color: #555;
          font-size: 0.9rem;
          line-height: 1.5;
        }

        .insights-sections {
          display: grid;
          gap: 1.5rem;
          margin-top: 2rem;
        }

        .insight-section {
          background: #fff;
          border-radius: 12px;
          padding: 1.5rem;
          border: 2px solid #e0e0e0;
        }

        .insight-section h3 {
          margin-top: 0;
          margin-bottom: 1rem;
        }

        .insight-section.strengths {
          border-color: #4caf50;
          background: linear-gradient(135deg, rgba(76,175,80,0.03) 0%, rgba(76,175,80,0.01) 100%);
        }

        .insight-section.weaknesses {
          border-color: #f44336;
          background: linear-gradient(135deg, rgba(244,67,54,0.03) 0%, rgba(244,67,54,0.01) 100%);
        }

        .insight-section.priorities {
          border-color: #ff9800;
          background: linear-gradient(135deg, rgba(255,152,0,0.03) 0%, rgba(255,152,0,0.01) 100%);
        }

        .insight-section.recommendations {
          border-color: #667eea;
          background: linear-gradient(135deg, rgba(102,126,234,0.03) 0%, rgba(118,75,162,0.01) 100%);
        }

        .insight-item {
          margin-bottom: 1rem;
          padding: 1rem;
          background: white;
          border-radius: 8px;
        }

        .insight-item strong {
          color: #333;
          font-size: 1rem;
        }

        .insight-item .action {
          margin: 0.5rem 0 0 0;
          color: #667eea;
          font-weight: 600;
        }

        .priority-item {
          display: flex;
          gap: 1rem;
          align-items: flex-start;
          margin-bottom: 1rem;
          padding: 1rem;
          background: white;
          border-radius: 8px;
        }

        .rank {
          background: #ff9800;
          color: white;
          width: 32px;
          height: 32px;
          border-radius: 50%;
          display: flex;
          align-items: center;
          justify-content: center;
          font-weight: bold;
          flex-shrink: 0;
        }

        .urgency {
          display: inline-block;
          margin-left: 0.5rem;
          padding: 0.2rem 0.6rem;
          border-radius: 12px;
          font-size: 0.8rem;
          font-weight: 600;
        }

        .urgency.high {
          background: #f44336;
          color: white;
        }

        .urgency.medium {
          background: #ff9800;
          color: white;
        }

        .recommendation-item {
          padding: 1rem;
          background: white;
          border-radius: 8px;
          margin-bottom: 0.8rem;
          color: #555;
          line-height: 1.6;
        }

        .stats {
          margin-top: 1rem;
          padding-top: 1rem;
          border-top: 1px solid #e0e0e0;
          color: #666;
          font-size: 0.9rem;
        }

        .stats p {
          margin: 0.5rem 0;
        }

        .footer {
          margin-top: 3rem;
          text-align: center;
          color: rgba(255,255,255,0.8);
          font-size: 0.9rem;
        }

        @media (max-width: 600px) {
          .title {
            font-size: 2rem;
          }
          .container {
            padding: 0 1rem;
          }
        }
      `}</style>
    </>
  );
}
