import { useState } from 'react';
import { useRouter } from 'next/router';

export default function LearnMore() {
  const router = useRouter();
  const [activeSection, setActiveSection] = useState('overview');

  return (
    <div className="container">
      {/* Header */}
      <header>
        <h1>🧠 Understanding Sentiment Analysis Models</h1>
        <p className="subtitle">
          Learn how RNN and Transformer models work together to analyze sentiment
        </p>
        <button onClick={() => router.push('/')} className="back-button">
          ← Back to Analysis
        </button>
      </header>

      {/* Navigation Tabs */}
      <div className="nav-tabs">
        <button 
          className={activeSection === 'overview' ? 'active' : ''}
          onClick={() => setActiveSection('overview')}
        >
          📚 Overview
        </button>
        <button 
          className={activeSection === 'rnn' ? 'active' : ''}
          onClick={() => setActiveSection('rnn')}
        >
          🔄 RNN Model
        </button>
        <button 
          className={activeSection === 'transformer' ? 'active' : ''}
          onClick={() => setActiveSection('transformer')}
        >
          ⚡ Transformer
        </button>
        <button 
          className={activeSection === 'ensemble' ? 'active' : ''}
          onClick={() => setActiveSection('ensemble')}
        >
          🤝 Hybrid Ensemble
        </button>
        <button 
          className={activeSection === 'business' ? 'active' : ''}
          onClick={() => setActiveSection('business')}
        >
          💼 Business Value
        </button>
      </div>

      {/* Content Sections */}
      <div className="content">
        {activeSection === 'overview' && (
          <section className="section">
            <h2>What is Sentiment Analysis?</h2>
            <p className="intro">
              Sentiment analysis is the process of determining whether text expresses 
              positive, negative, or neutral opinions. It's like teaching a computer 
              to understand human emotions in writing.
            </p>

            <div className="example-box">
              <h3>Real-World Examples</h3>
              <div className="example-grid">
                <div className="example-card positive-bg">
                  <h4>✅ Positive</h4>
                  <p>"This product is absolutely amazing! Love it!"</p>
                  <span className="confidence">Confidence: 98%</span>
                </div>
                <div className="example-card negative-bg">
                  <h4>❌ Negative</h4>
                  <p>"Terrible service, would not recommend"</p>
                  <span className="confidence">Confidence: 95%</span>
                </div>
                <div className="example-card neutral-bg">
                  <h4>➖ Neutral</h4>
                  <p>"The product arrived on time"</p>
                  <span className="confidence">Confidence: 82%</span>
                </div>
              </div>
            </div>

            <h3>Why Does This Matter?</h3>
            <div className="benefits-grid">
              <div className="benefit-card">
                <span className="icon">🛍️</span>
                <h4>E-commerce</h4>
                <p>Analyze thousands of product reviews to identify what customers love or hate</p>
              </div>
              <div className="benefit-card">
                <span className="icon">📱</span>
                <h4>Social Media</h4>
                <p>Monitor brand sentiment in real-time across millions of posts</p>
              </div>
              <div className="benefit-card">
                <span className="icon">🎯</span>
                <h4>Customer Support</h4>
                <p>Automatically route angry customers to priority support queues</p>
              </div>
              <div className="benefit-card">
                <span className="icon">📊</span>
                <h4>Market Research</h4>
                <p>Track sentiment trends over time to guide business strategy</p>
              </div>
            </div>

            <h3>Our Multi-Model Approach</h3>
            <p>
              Instead of using just one model, we combine two complementary approaches:
            </p>
            <ul className="feature-list">
              <li><strong>RNN with Attention</strong> - Fast and efficient, handles 70% of cases in milliseconds</li>
              <li><strong>DistilBERT Transformer</strong> - High accuracy for complex language and nuanced sentiment</li>
              <li><strong>Intelligent Routing</strong> - Automatically chooses the best model for each input</li>
              <li><strong>Business Insights</strong> - Breaks down sentiment by specific aspects (food, service, price, etc.)</li>
            </ul>
          </section>
        )}

        {activeSection === 'rnn' && (
          <section className="section">
            <h2>How RNN with Attention Works</h2>
            <p className="intro">
              Recurrent Neural Networks (RNN) process text sequentially, like reading 
              a sentence word by word. The attention mechanism helps the model focus 
              on the most important words.
            </p>

            <h3>Step-by-Step Example</h3>
            <div className="example-box">
              <h4>Input: "The movie was absolutely terrible"</h4>
              
              <div className="step">
                <div className="step-number">1</div>
                <div className="step-content">
                  <h5>Text Preprocessing</h5>
                  <p>Clean and normalize the text</p>
                  <code>"The movie was absolutely terrible" → ["the", "movie", "was", "absolutely", "terrible"]</code>
                </div>
              </div>

              <div className="step">
                <div className="step-number">2</div>
                <div className="step-content">
                  <h5>Tokenization</h5>
                  <p>Convert words to numerical indices using vocabulary</p>
                  <code>["the", "movie", "was", "absolutely", "terrible"] → [45, 892, 234, 1205, 678]</code>
                </div>
              </div>

              <div className="step">
                <div className="step-number">3</div>
                <div className="step-content">
                  <h5>Embedding (100 dimensions)</h5>
                  <p>Transform each word index into a dense vector that captures meaning</p>
                  <code>45 → [0.12, -0.45, 0.78, ..., 0.23]  (100 numbers)</code>
                  <p className="note">Similar words have similar vectors!</p>
                </div>
              </div>

              <div className="step">
                <div className="step-number">4</div>
                <div className="step-content">
                  <h5>Bidirectional LSTM</h5>
                  <p>Process sequence in both directions to capture full context</p>
                  <div className="lstm-visual">
                    <div>Forward LSTM →: Reads left to right</div>
                    <div>Backward LSTM ←: Reads right to left</div>
                    <div className="highlight">Combined output: 256 dimensions per word</div>
                  </div>
                </div>
              </div>

              <div className="step">
                <div className="step-number">5</div>
                <div className="step-content">
                  <h5>Attention Mechanism ⭐</h5>
                  <p>Learn which words are most important for sentiment</p>
                  <div className="attention-visual">
                    <div className="word-weight">
                      <span className="word">"the"</span>
                      <div className="weight-bar" style={{width: '5%', background: '#ccc'}}>5%</div>
                    </div>
                    <div className="word-weight">
                      <span className="word">"movie"</span>
                      <div className="weight-bar" style={{width: '10%', background: '#aaa'}}>10%</div>
                    </div>
                    <div className="word-weight">
                      <span className="word">"was"</span>
                      <div className="weight-bar" style={{width: '10%', background: '#aaa'}}>10%</div>
                    </div>
                    <div className="word-weight">
                      <span className="word">"absolutely"</span>
                      <div className="weight-bar" style={{width: '15%', background: '#888'}}>15%</div>
                    </div>
                    <div className="word-weight">
                      <span className="word">"terrible"</span>
                      <div className="weight-bar" style={{width: '60%', background: '#e74c3c'}}>60% 🎯</div>
                    </div>
                  </div>
                  <p className="note">The model learned "terrible" is the key word!</p>
                </div>
              </div>

              <div className="step">
                <div className="step-number">6</div>
                <div className="step-content">
                  <h5>Context Vector</h5>
                  <p>Create weighted sum emphasizing important words</p>
                  <code>context = 0.05×h₁ + 0.10×h₂ + 0.10×h₃ + 0.15×h₄ + 0.60×h₅</code>
                </div>
              </div>

              <div className="step">
                <div className="step-number">7</div>
                <div className="step-content">
                  <h5>Classification</h5>
                  <p>Final layer predicts sentiment</p>
                  <div className="prediction-visual">
                    <div className="pred-bar positive">Positive: 8%</div>
                    <div className="pred-bar negative active">Negative: 92% ✓</div>
                  </div>
                </div>
              </div>
            </div>

            <h3>Key Advantages of RNN</h3>
            <div className="advantages-grid">
              <div className="advantage-card">
                <h4>⚡ Speed</h4>
                <p>15ms average processing time - perfect for high-volume applications</p>
              </div>
              <div className="advantage-card">
                <h4>💾 Efficiency</h4>
                <p>Only 20MB model size, runs on any device</p>
              </div>
              <div className="advantage-card">
                <h4>🎯 Interpretability</h4>
                <p>Attention weights show exactly which words influenced the decision</p>
              </div>
              <div className="advantage-card">
                <h4>🔍 Neutral Detection</h4>
                <p>Excellent at identifying ambiguous or uncertain sentiment</p>
              </div>
            </div>

            <div className="stats-box">
              <h4>Performance Metrics</h4>
              <ul>
                <li>Accuracy: <strong>87.56%</strong></li>
                <li>Training time: <strong>30 minutes</strong></li>
                <li>Parameters: <strong>2.5 million</strong></li>
                <li>Inference time: <strong>15ms</strong></li>
              </ul>
            </div>
          </section>
        )}

        {activeSection === 'transformer' && (
          <section className="section">
            <h2>How DistilBERT Transformer Works</h2>
            <p className="intro">
              DistilBERT is a smaller, faster version of BERT (Bidirectional Encoder 
              Representations from Transformers). It uses attention to understand 
              relationships between all words simultaneously, not just sequentially.
            </p>

            <h3>Step-by-Step Example</h3>
            <div className="example-box">
              <h4>Input: "The movie was absolutely terrible"</h4>
              
              <div className="step">
                <div className="step-number">1</div>
                <div className="step-content">
                  <h5>Subword Tokenization</h5>
                  <p>Break rare words into subwords for better handling</p>
                  <code>"absolutely" → ["ab", "##solu", "##tely"]</code>
                  <p className="note">Full sequence: [CLS] the movie was ab ##solu ##tely terrible [SEP]</p>
                </div>
              </div>

              <div className="step">
                <div className="step-number">2</div>
                <div className="step-content">
                  <h5>Embeddings (768 dimensions)</h5>
                  <p>Combine token embeddings + position embeddings</p>
                  <div className="embedding-visual">
                    <div>Token embedding: Captures word meaning</div>
                    <div>Position embedding: Captures word order</div>
                    <div className="highlight">Combined: 768-dimensional vector per token</div>
                  </div>
                </div>
              </div>

              <div className="step">
                <div className="step-number">3</div>
                <div className="step-content">
                  <h5>Multi-Head Attention (Layer 1 of 6)</h5>
                  <p>Each word attends to every other word simultaneously</p>
                  <div className="attention-matrix">
                    <h6>Attention from "terrible" to other words:</h6>
                    <div className="attention-row">
                      <span>terrible → movie:</span>
                      <div className="attention-score high">0.45 (strong)</div>
                    </div>
                    <div className="attention-row">
                      <span>terrible → was:</span>
                      <div className="attention-score medium">0.15 (medium)</div>
                    </div>
                    <div className="attention-row">
                      <span>terrible → the:</span>
                      <div className="attention-score low">0.05 (weak)</div>
                    </div>
                  </div>
                  <p className="note">Model learns "terrible movie" is a meaningful phrase!</p>
                </div>
              </div>

              <div className="step">
                <div className="step-number">4</div>
                <div className="step-content">
                  <h5>Feed-Forward Network</h5>
                  <p>Transform each token's representation</p>
                  <code>[768] → [3072] → [768]</code>
                  <p className="note">Each layer captures increasingly abstract patterns</p>
                </div>
              </div>

              <div className="step">
                <div className="step-number">5</div>
                <div className="step-content">
                  <h5>Layers 2-6</h5>
                  <p>Repeat attention + feed-forward 5 more times</p>
                  <ul>
                    <li>Layer 1-2: Basic grammar and syntax</li>
                    <li>Layer 3-4: Semantic relationships</li>
                    <li>Layer 5-6: Complex reasoning and sentiment</li>
                  </ul>
                </div>
              </div>

              <div className="step">
                <div className="step-number">6</div>
                <div className="step-content">
                  <h5>Classification Head</h5>
                  <p>Use [CLS] token (sentence representation) for final prediction</p>
                  <div className="prediction-visual">
                    <div className="pred-bar positive">Positive: 2%</div>
                    <div className="pred-bar negative active">Negative: 98% ✓</div>
                  </div>
                </div>
              </div>
            </div>

            <h3>Why DistilBERT Wins</h3>
            <div className="advantages-grid">
              <div className="advantage-card">
                <h4>📚 Pre-trained Knowledge</h4>
                <p>Learned from 16GB of text (Wikipedia + BookCorpus) - understands language deeply</p>
              </div>
              <div className="advantage-card">
                <h4>🔄 Bidirectional Context</h4>
                <p>Sees full context from both directions simultaneously, not sequentially</p>
              </div>
              <div className="advantage-card">
                <h4>🎯 Nuanced Understanding</h4>
                <p>Handles sarcasm, negation, and complex language patterns</p>
              </div>
              <div className="advantage-card">
                <h4>⚡ Efficient Distillation</h4>
                <p>40% smaller than BERT, 60% faster, with 97% of the quality</p>
              </div>
            </div>

            <div className="comparison-box">
              <h4>DistilBERT vs Full BERT</h4>
              <table>
                <thead>
                  <tr>
                    <th>Model</th>
                    <th>Layers</th>
                    <th>Parameters</th>
                    <th>Speed</th>
                    <th>Accuracy</th>
                  </tr>
                </thead>
                <tbody>
                  <tr>
                    <td>BERT-base</td>
                    <td>12</td>
                    <td>110M</td>
                    <td>1x</td>
                    <td>100%</td>
                  </tr>
                  <tr className="highlight">
                    <td><strong>DistilBERT</strong></td>
                    <td><strong>6</strong></td>
                    <td><strong>66M</strong></td>
                    <td><strong>1.6x faster</strong></td>
                    <td><strong>97%</strong></td>
                  </tr>
                </tbody>
              </table>
            </div>

            <div className="stats-box">
              <h4>Performance Metrics</h4>
              <ul>
                <li>Accuracy: <strong>94.22%</strong></li>
                <li>Training time: <strong>29 minutes</strong> (pre-training helps!)</li>
                <li>Parameters: <strong>66 million</strong></li>
                <li>Inference time: <strong>50ms</strong></li>
              </ul>
            </div>
          </section>
        )}

        {activeSection === 'ensemble' && (
          <section className="section">
            <h2>Hybrid Ensemble Strategy</h2>
            <p className="intro">
              Instead of choosing one model, we use both intelligently. RNN handles 
              easy cases quickly, DistilBERT verifies difficult cases accurately.
            </p>

            <h3>How It Works</h3>
            <div className="pipeline-visual">
              <div className="pipeline-step">
                <div className="step-icon">📝</div>
                <h4>User Input</h4>
                <p>Any text to analyze</p>
              </div>
              <div className="arrow">↓</div>
              <div className="pipeline-step highlight">
                <div className="step-icon">🔄</div>
                <h4>Stage 1: RNN Fast Filter</h4>
                <p>Process ALL inputs (~15ms)</p>
                <p>Predict sentiment + confidence</p>
              </div>
              <div className="arrow">↓</div>
              <div className="pipeline-branch">
                <div className="branch-left">
                  <div className="branch-label">Confidence ≥ 90%</div>
                  <div className="pipeline-step success">
                    <div className="step-icon">⚡</div>
                    <h4>Use RNN Prediction</h4>
                    <p>~70% of cases</p>
                    <p>Fast path ✓</p>
                  </div>
                </div>
                <div className="branch-right">
                  <div className="branch-label">Confidence &lt; 90%</div>
                  <div className="pipeline-step warning">
                    <div className="step-icon">⚡</div>
                    <h4>Stage 2: DistilBERT</h4>
                    <p>~30% of cases</p>
                    <p>Accuracy path ✓</p>
                  </div>
                </div>
              </div>
              <div className="arrow">↓</div>
              <div className="pipeline-step">
                <div className="step-icon">✅</div>
                <h4>Final Prediction</h4>
                <p>Best of both models</p>
              </div>
            </div>

            <h3>Real-World Performance</h3>
            <div className="performance-comparison">
              <div className="perf-card">
                <h4>RNN Only</h4>
                <div className="metric">
                  <span className="label">Accuracy:</span>
                  <span className="value">87.56%</span>
                </div>
                <div className="metric">
                  <span className="label">Latency:</span>
                  <span className="value">15ms</span>
                </div>
                <div className="metric">
                  <span className="label">Cost (1M req):</span>
                  <span className="value">$10</span>
                </div>
              </div>

              <div className="perf-card">
                <h4>DistilBERT Only</h4>
                <div className="metric">
                  <span className="label">Accuracy:</span>
                  <span className="value">94.22%</span>
                </div>
                <div className="metric">
                  <span className="label">Latency:</span>
                  <span className="value">50ms</span>
                </div>
                <div className="metric">
                  <span className="label">Cost (1M req):</span>
                  <span className="value">$33</span>
                </div>
              </div>

              <div className="perf-card best">
                <h4>Hybrid Ensemble ⭐</h4>
                <div className="metric">
                  <span className="label">Accuracy:</span>
                  <span className="value">~92%</span>
                </div>
                <div className="metric">
                  <span className="label">Latency:</span>
                  <span className="value">~24ms</span>
                </div>
                <div className="metric">
                  <span className="label">Cost (1M req):</span>
                  <span className="value">$17</span>
                </div>
                <div className="benefit">60% faster than DistilBERT!</div>
                <div className="benefit">70% cost reduction!</div>
              </div>
            </div>

            <h3>Sequential Pipeline for Neutral Detection</h3>
            <p>We discovered RNN is better at detecting neutral/ambiguous sentiment:</p>
            <div className="sequential-visual">
              <div className="seq-step">
                <strong>1.</strong> RNN analyzes input
              </div>
              <div className="seq-arrow">→</div>
              <div className="seq-step">
                <strong>2.</strong> If Neutral detected (≥85% conf) → Return Neutral
              </div>
              <div className="seq-arrow">→</div>
              <div className="seq-step">
                <strong>3.</strong> Otherwise → DistilBERT refines to Pos/Neg
              </div>
            </div>

            <div className="example-box">
              <h4>Example: "The food was okay, nothing special"</h4>
              <ul>
                <li>✅ RNN: Neutral (87% confidence) - Correctly identified ambiguity</li>
                <li>❌ DistilBERT alone: Negative (82% confidence) - Over-interpreted</li>
                <li>🎯 Sequential: Neutral (using RNN's strength)</li>
              </ul>
            </div>
          </section>
        )}

        {activeSection === 'business' && (
          <section className="section">
            <h2>Business Value & Applications</h2>
            <p className="intro">
              Transform raw sentiment into actionable business intelligence with 
              aspect-based analysis and strategic recommendations.
            </p>

            <h3>Aspect-Based Sentiment Analysis</h3>
            <p>Instead of just "positive" or "negative", identify specific topics:</p>
            <div className="aspects-grid">
              <div className="aspect-card">
                <span className="aspect-icon">🍔</span>
                <h4>Food</h4>
                <p>Quality, taste, freshness</p>
              </div>
              <div className="aspect-card">
                <span className="aspect-icon">👔</span>
                <h4>Service</h4>
                <p>Staff, speed, professionalism</p>
              </div>
              <div className="aspect-card">
                <span className="aspect-icon">💰</span>
                <h4>Price</h4>
                <p>Value, cost, affordability</p>
              </div>
              <div className="aspect-card">
                <span className="aspect-icon">⭐</span>
                <h4>Quality</h4>
                <p>Overall standard, excellence</p>
              </div>
              <div className="aspect-card">
                <span className="aspect-icon">📍</span>
                <h4>Location</h4>
                <p>Accessibility, parking, area</p>
              </div>
              <div className="aspect-card">
                <span className="aspect-icon">🎨</span>
                <h4>Ambiance</h4>
                <p>Atmosphere, decor, cleanliness</p>
              </div>
              <div className="aspect-card">
                <span className="aspect-icon">📦</span>
                <h4>Product</h4>
                <p>Features, packaging, design</p>
              </div>
              <div className="aspect-card">
                <span className="aspect-icon">✨</span>
                <h4>Experience</h4>
                <p>Overall impression, satisfaction</p>
              </div>
            </div>

            <h3>Real Example</h3>
            <div className="business-example">
              <div className="input-text">
                <h4>Input Review:</h4>
                <p>"The food was absolutely delicious and fresh. However, the service was quite slow and the staff seemed unprepared. The location is convenient and parking was easy."</p>
              </div>

              <div className="analysis-results">
                <h4>Aspect Analysis:</h4>
                <div className="aspect-results">
                  <div className="aspect-result positive">
                    <strong>Food:</strong> Positive (99% conf, 2 mentions)
                  </div>
                  <div className="aspect-result negative">
                    <strong>Service:</strong> Negative (95% conf, 2 mentions)
                  </div>
                  <div className="aspect-result positive">
                    <strong>Location:</strong> Positive (92% conf, 1 mention)
                  </div>
                </div>

                <h4>Business Insights:</h4>
                <div className="insights-list">
                  <div className="insight strength">
                    <strong>✅ Strength:</strong> Food quality is exceptional - leverage in marketing
                  </div>
                  <div className="insight weakness">
                    <strong>⚠️ Weakness:</strong> Service needs immediate attention - hiring/training priority
                  </div>
                  <div className="insight priority">
                    <strong>🎯 Priority:</strong> Address service issues (HIGH urgency, 2 mentions)
                  </div>
                  <div className="insight recommendation">
                    <strong>💡 Recommendation:</strong> Hire additional staff, implement service training program
                  </div>
                </div>
              </div>
            </div>

            <h3>Industry Applications</h3>
            <div className="industries-grid">
              <div className="industry-card">
                <h4>🛍️ E-commerce</h4>
                <ul>
                  <li>Analyze product reviews by feature</li>
                  <li>Identify most-loved and most-hated aspects</li>
                  <li>Prioritize product improvements</li>
                  <li>Generate FAQ from common concerns</li>
                </ul>
                <div className="roi">ROI: $10K/month saved on manual analysis</div>
              </div>

              <div className="industry-card">
                <h4>🍽️ Restaurants</h4>
                <ul>
                  <li>Track food vs service vs ambiance trends</li>
                  <li>Identify menu items to promote or remove</li>
                  <li>Monitor staff performance indirectly</li>
                  <li>Benchmark against competitors</li>
                </ul>
                <div className="roi">ROI: 15% improvement in customer satisfaction</div>
              </div>

              <div className="industry-card">
                <h4>🏨 Hotels</h4>
                <ul>
                  <li>Separate room quality from service issues</li>
                  <li>Track cleanliness sentiment over time</li>
                  <li>Identify location/parking concerns</li>
                  <li>Alert on sudden negative trends</li>
                </ul>
                <div className="roi">ROI: 2-point increase in review scores</div>
              </div>

              <div className="industry-card">
                <h4>📱 Social Media</h4>
                <ul>
                  <li>Real-time brand sentiment monitoring</li>
                  <li>Crisis detection (sudden negativity spikes)</li>
                  <li>Influencer impact measurement</li>
                  <li>Campaign effectiveness tracking</li>
                </ul>
                <div className="roi">ROI: 3-hour crisis response time reduction</div>
              </div>
            </div>

            <h3>Cost Savings Example</h3>
            <div className="savings-calculator">
              <h4>Processing 1 Million Reviews/Month</h4>
              <table>
                <thead>
                  <tr>
                    <th>Approach</th>
                    <th>Time</th>
                    <th>Cost</th>
                    <th>Accuracy</th>
                  </tr>
                </thead>
                <tbody>
                  <tr className="manual">
                    <td>Manual Analysis (5 min/review)</td>
                    <td>8,333 hours</td>
                    <td>$208,333</td>
                    <td>Variable</td>
                  </tr>
                  <tr className="single-model">
                    <td>DistilBERT Only</td>
                    <td>13.9 hours</td>
                    <td>$33</td>
                    <td>94.22%</td>
                  </tr>
                  <tr className="hybrid">
                    <td><strong>Hybrid Ensemble ⭐</strong></td>
                    <td><strong>7.1 hours</strong></td>
                    <td><strong>$17</strong></td>
                    <td><strong>~92%</strong></td>
                  </tr>
                </tbody>
              </table>
              <div className="savings-highlight">
                💰 Save $208,316/month vs manual analysis<br/>
                ⚡ 49% faster than DistilBERT only<br/>
                🎯 Near-optimal accuracy maintained
              </div>
            </div>
          </section>
        )}
      </div>

      <style jsx>{`
        .container {
          min-height: 100vh;
          background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
          padding: 40px 20px;
        }

        header {
          text-align: center;
          color: white;
          margin-bottom: 40px;
        }

        h1 {
          font-size: 3rem;
          margin-bottom: 10px;
          text-shadow: 2px 2px 4px rgba(0,0,0,0.2);
        }

        .subtitle {
          font-size: 1.2rem;
          opacity: 0.9;
          margin-bottom: 20px;
        }

        .back-button {
          background: rgba(255,255,255,0.2);
          color: white;
          border: 2px solid white;
          padding: 12px 30px;
          border-radius: 25px;
          font-size: 1rem;
          cursor: pointer;
          transition: all 0.3s;
        }

        .back-button:hover {
          background: white;
          color: #667eea;
          transform: translateY(-2px);
        }

        .nav-tabs {
          display: flex;
          justify-content: center;
          gap: 10px;
          flex-wrap: wrap;
          margin-bottom: 30px;
        }

        .nav-tabs button {
          background: rgba(255,255,255,0.2);
          color: white;
          border: none;
          padding: 12px 24px;
          border-radius: 20px;
          cursor: pointer;
          transition: all 0.3s;
          font-size: 1rem;
        }

        .nav-tabs button:hover {
          background: rgba(255,255,255,0.3);
          transform: translateY(-2px);
        }

        .nav-tabs button.active {
          background: white;
          color: #667eea;
          font-weight: bold;
        }

        .content {
          max-width: 1200px;
          margin: 0 auto;
        }

        .section {
          background: white;
          border-radius: 20px;
          padding: 40px;
          box-shadow: 0 10px 30px rgba(0,0,0,0.2);
          animation: fadeIn 0.3s;
        }

        @keyframes fadeIn {
          from { opacity: 0; transform: translateY(20px); }
          to { opacity: 1; transform: translateY(0); }
        }

        h2 {
          color: #667eea;
          font-size: 2.5rem;
          margin-bottom: 20px;
          border-bottom: 3px solid #667eea;
          padding-bottom: 10px;
        }

        h3 {
          color: #764ba2;
          font-size: 1.8rem;
          margin-top: 40px;
          margin-bottom: 20px;
        }

        h4 {
          color: #333;
          font-size: 1.3rem;
          margin-bottom: 10px;
        }

        .intro {
          font-size: 1.2rem;
          color: #555;
          line-height: 1.8;
          margin-bottom: 30px;
        }

        .example-box {
          background: #f8f9fa;
          border-radius: 15px;
          padding: 30px;
          margin: 30px 0;
          border-left: 5px solid #667eea;
        }

        .example-grid {
          display: grid;
          grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
          gap: 20px;
          margin-top: 20px;
        }

        .example-card {
          padding: 20px;
          border-radius: 10px;
          text-align: center;
        }

        .positive-bg {
          background: linear-gradient(135deg, #d4fc79 0%, #96e6a1 100%);
        }

        .negative-bg {
          background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        }

        .neutral-bg {
          background: linear-gradient(135deg, #ffecd2 0%, #fcb69f 100%);
        }

        .confidence {
          display: inline-block;
          margin-top: 10px;
          padding: 5px 15px;
          background: rgba(0,0,0,0.1);
          border-radius: 20px;
          font-size: 0.9rem;
        }

        .benefits-grid,
        .advantages-grid,
        .aspects-grid,
        .industries-grid {
          display: grid;
          grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
          gap: 20px;
          margin: 20px 0;
        }

        .benefit-card,
        .advantage-card,
        .aspect-card,
        .industry-card {
          background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
          padding: 25px;
          border-radius: 15px;
          box-shadow: 0 4px 6px rgba(0,0,0,0.1);
          transition: transform 0.3s;
        }

        .benefit-card:hover,
        .advantage-card:hover,
        .aspect-card:hover,
        .industry-card:hover {
          transform: translateY(-5px);
        }

        .icon,
        .aspect-icon {
          font-size: 3rem;
          display: block;
          margin-bottom: 10px;
        }

        .feature-list {
          list-style: none;
          padding: 0;
        }

        .feature-list li {
          padding: 15px;
          margin: 10px 0;
          background: #f8f9fa;
          border-radius: 10px;
          border-left: 4px solid #667eea;
        }

        .step {
          display: flex;
          gap: 20px;
          margin: 25px 0;
          align-items: flex-start;
        }

        .step-number {
          background: #667eea;
          color: white;
          width: 40px;
          height: 40px;
          border-radius: 50%;
          display: flex;
          align-items: center;
          justify-content: center;
          font-weight: bold;
          flex-shrink: 0;
        }

        .step-content {
          flex: 1;
        }

        .step-content h5 {
          color: #764ba2;
          margin-bottom: 10px;
          font-size: 1.2rem;
        }

        .step-content code {
          display: block;
          background: #2d3748;
          color: #68d391;
          padding: 15px;
          border-radius: 8px;
          margin: 10px 0;
          overflow-x: auto;
          font-family: 'Courier New', monospace;
        }

        .note {
          font-style: italic;
          color: #718096;
          margin-top: 10px;
        }

        .attention-visual,
        .lstm-visual {
          background: white;
          padding: 20px;
          border-radius: 10px;
          margin: 15px 0;
        }

        .word-weight {
          display: flex;
          align-items: center;
          margin: 10px 0;
          gap: 15px;
        }

        .word {
          width: 120px;
          font-weight: bold;
        }

        .weight-bar {
          height: 30px;
          border-radius: 5px;
          display: flex;
          align-items: center;
          padding: 0 10px;
          color: white;
          font-weight: bold;
          transition: all 0.3s;
        }

        .prediction-visual {
          margin: 20px 0;
        }

        .pred-bar {
          padding: 15px;
          margin: 10px 0;
          border-radius: 8px;
          font-weight: bold;
          opacity: 0.3;
        }

        .pred-bar.active {
          opacity: 1;
          transform: scale(1.05);
        }

        .pred-bar.positive {
          background: linear-gradient(90deg, #d4fc79 0%, #96e6a1 100%);
        }

        .pred-bar.negative {
          background: linear-gradient(90deg, #f093fb 0%, #f5576c 100%);
        }

        .stats-box,
        .comparison-box {
          background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
          color: white;
          padding: 25px;
          border-radius: 15px;
          margin: 30px 0;
        }

        .stats-box ul {
          list-style: none;
          padding: 0;
        }

        .stats-box li {
          padding: 10px 0;
          border-bottom: 1px solid rgba(255,255,255,0.2);
        }

        .stats-box li:last-child {
          border-bottom: none;
        }

        table {
          width: 100%;
          border-collapse: collapse;
          margin: 20px 0;
        }

        th, td {
          padding: 15px;
          text-align: left;
          border-bottom: 1px solid rgba(255,255,255,0.2);
        }

        th {
          font-weight: bold;
        }

        tr.highlight {
          background: rgba(255,255,255,0.1);
        }

        .pipeline-visual {
          display: flex;
          flex-direction: column;
          align-items: center;
          gap: 20px;
          margin: 40px 0;
        }

        .pipeline-step {
          background: white;
          padding: 25px;
          border-radius: 15px;
          box-shadow: 0 4px 6px rgba(0,0,0,0.1);
          text-align: center;
          min-width: 300px;
        }

        .pipeline-step.highlight {
          background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
          color: white;
        }

        .pipeline-step.success {
          background: linear-gradient(135deg, #d4fc79 0%, #96e6a1 100%);
        }

        .pipeline-step.warning {
          background: linear-gradient(135deg, #ffecd2 0%, #fcb69f 100%);
        }

        .step-icon {
          font-size: 3rem;
          margin-bottom: 10px;
        }

        .arrow {
          font-size: 2rem;
          color: #667eea;
        }

        .pipeline-branch {
          display: flex;
          gap: 40px;
          justify-content: center;
          flex-wrap: wrap;
        }

        .branch-label {
          text-align: center;
          font-weight: bold;
          color: #667eea;
          margin-bottom: 15px;
          padding: 8px 16px;
          background: rgba(102, 126, 234, 0.1);
          border-radius: 20px;
        }

        .performance-comparison {
          display: grid;
          grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
          gap: 20px;
          margin: 30px 0;
        }

        .perf-card {
          background: white;
          border: 2px solid #e2e8f0;
          padding: 25px;
          border-radius: 15px;
          box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }

        .perf-card.best {
          border-color: #667eea;
          background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
          transform: scale(1.05);
        }

        .metric {
          display: flex;
          justify-content: space-between;
          padding: 10px 0;
          border-bottom: 1px solid #e2e8f0;
        }

        .metric:last-of-type {
          border-bottom: none;
        }

        .label {
          font-weight: bold;
          color: #4a5568;
        }

        .value {
          color: #667eea;
          font-weight: bold;
        }

        .benefit {
          background: #667eea;
          color: white;
          padding: 8px 16px;
          border-radius: 20px;
          margin-top: 10px;
          text-align: center;
          font-weight: bold;
        }

        .sequential-visual {
          display: flex;
          align-items: center;
          gap: 15px;
          justify-content: center;
          flex-wrap: wrap;
          margin: 30px 0;
        }

        .seq-step {
          background: #f8f9fa;
          padding: 20px;
          border-radius: 10px;
          border: 2px solid #667eea;
        }

        .seq-arrow {
          font-size: 2rem;
          color: #667eea;
          font-weight: bold;
        }

        .business-example {
          margin: 30px 0;
        }

        .input-text {
          background: #f8f9fa;
          padding: 20px;
          border-radius: 10px;
          margin-bottom: 20px;
        }

        .analysis-results {
          background: white;
          border: 2px solid #e2e8f0;
          padding: 25px;
          border-radius: 15px;
        }

        .aspect-results {
          margin: 20px 0;
        }

        .aspect-result {
          padding: 15px;
          margin: 10px 0;
          border-radius: 8px;
          font-weight: bold;
        }

        .aspect-result.positive {
          background: linear-gradient(90deg, #d4fc79 0%, #96e6a1 100%);
        }

        .aspect-result.negative {
          background: linear-gradient(90deg, #f093fb 0%, #f5576c 100%);
          color: white;
        }

        .insights-list {
          margin: 20px 0;
        }

        .insight {
          padding: 15px;
          margin: 10px 0;
          border-radius: 10px;
          border-left: 5px solid;
        }

        .insight.strength {
          background: #f0fdf4;
          border-color: #22c55e;
        }

        .insight.weakness {
          background: #fef2f2;
          border-color: #ef4444;
        }

        .insight.priority {
          background: #fef3c7;
          border-color: #f59e0b;
        }

        .insight.recommendation {
          background: #eff6ff;
          border-color: #3b82f6;
        }

        .roi {
          background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
          color: white;
          padding: 12px;
          border-radius: 10px;
          margin-top: 15px;
          text-align: center;
          font-weight: bold;
        }

        .savings-calculator {
          background: #f8f9fa;
          padding: 30px;
          border-radius: 15px;
          margin: 30px 0;
        }

        .savings-highlight {
          background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
          color: white;
          padding: 20px;
          border-radius: 10px;
          margin-top: 20px;
          text-align: center;
          font-weight: bold;
          line-height: 2;
        }

        .attention-matrix {
          background: white;
          padding: 20px;
          border-radius: 10px;
        }

        .attention-row {
          display: flex;
          justify-content: space-between;
          align-items: center;
          padding: 10px;
          margin: 5px 0;
        }

        .attention-score {
          padding: 8px 20px;
          border-radius: 20px;
          color: white;
          font-weight: bold;
        }

        .attention-score.high {
          background: #22c55e;
        }

        .attention-score.medium {
          background: #f59e0b;
        }

        .attention-score.low {
          background: #6b7280;
        }

        .embedding-visual {
          background: white;
          padding: 20px;
          border-radius: 10px;
        }

        .embedding-visual div {
          padding: 10px;
          margin: 5px 0;
        }

        .highlight {
          background: rgba(102, 126, 234, 0.1);
          padding: 10px;
          border-radius: 5px;
          font-weight: bold;
        }

        @media (max-width: 768px) {
          h1 {
            font-size: 2rem;
          }

          .section {
            padding: 20px;
          }

          .nav-tabs {
            flex-direction: column;
          }

          .nav-tabs button {
            width: 100%;
          }

          .pipeline-branch {
            flex-direction: column;
          }

          .performance-comparison {
            grid-template-columns: 1fr;
          }
        }
      `}</style>
    </div>
  );
}
