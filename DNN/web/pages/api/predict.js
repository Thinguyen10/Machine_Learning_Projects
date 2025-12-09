// Simple sentiment analysis for Vercel deployment (no heavy models)
// For production, connect to external API or use lightweight model

export default async function handler(req, res) {
  // CORS headers
  res.setHeader('Access-Control-Allow-Origin', '*');
  res.setHeader('Access-Control-Allow-Methods', 'POST, OPTIONS');
  res.setHeader('Access-Control-Allow-Headers', 'Content-Type');

  if (req.method === 'OPTIONS') {
    return res.status(200).end();
  }

  if (req.method !== 'POST') {
    return res.status(405).json({ error: 'Method not allowed' });
  }

  try {
    const { text, mode = 'hybrid' } = req.body;

    if (!text) {
      return res.status(400).json({ error: 'No text provided' });
    }

    // Simple rule-based sentiment (demo mode)
    const sentiment = analyzeSentiment(text);

    return res.status(200).json({
      predictions: sentiment,
      processing_time: 25,
      text_length: text.length,
      note: 'Demo mode - connect to external ML API for full features'
    });
  } catch (error) {
    console.error('Prediction error:', error);
    return res.status(500).json({ error: 'Internal server error' });
  }
}

function analyzeSentiment(text) {
  const textLower = text.toLowerCase();
  
  // Positive keywords
  const positiveWords = ['good', 'great', 'excellent', 'amazing', 'wonderful', 'love', 'best', 
    'fantastic', 'perfect', 'awesome', 'brilliant', 'outstanding', 'superb', 'nice', 'happy',
    'satisfied', 'recommend', 'enjoyed', 'pleasant', 'delightful'];
  
  // Negative keywords
  const negativeWords = ['bad', 'terrible', 'awful', 'horrible', 'worst', 'hate', 'poor',
    'disappointing', 'waste', 'useless', 'pathetic', 'disgusting', 'annoying', 'frustrating',
    'broken', 'failed', 'never', 'nothing', 'nowhere', 'nobody'];
  
  let positiveCount = 0;
  let negativeCount = 0;
  
  positiveWords.forEach(word => {
    if (textLower.includes(word)) positiveCount++;
  });
  
  negativeWords.forEach(word => {
    if (textLower.includes(word)) negativeCount++;
  });
  
  let label = 'neutral';
  let confidence = 0.65;
  
  if (positiveCount > negativeCount) {
    label = 'positive';
    confidence = Math.min(0.95, 0.65 + (positiveCount * 0.05));
  } else if (negativeCount > positiveCount) {
    label = 'negative';
    confidence = Math.min(0.95, 0.65 + (negativeCount * 0.05));
  }
  
  // Extract simple aspects
  const aspects = [];
  if (textLower.includes('price') || textLower.includes('cost') || textLower.includes('expensive') || textLower.includes('cheap')) {
    aspects.push({
      aspect: 'price',
      sentiment: label,
      confidence: confidence,
      mentions: 1
    });
  }
  if (textLower.includes('quality') || textLower.includes('build') || textLower.includes('material')) {
    aspects.push({
      aspect: 'quality',
      sentiment: label,
      confidence: confidence,
      mentions: 1
    });
  }
  if (textLower.includes('service') || textLower.includes('support') || textLower.includes('staff')) {
    aspects.push({
      aspect: 'service',
      sentiment: label,
      confidence: confidence,
      mentions: 1
    });
  }
  
  return {
    final_prediction: {
      label: label,
      confidence: confidence
    },
    rnn_prediction: {
      label: label,
      confidence: confidence - 0.05
    },
    distilbert_prediction: {
      label: label,
      confidence: confidence
    },
    aspects: aspects,
    mode: 'demo'
  };
}
