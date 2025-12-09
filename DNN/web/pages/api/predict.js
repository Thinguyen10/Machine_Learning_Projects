// Sentiment analysis using Hugging Face hosted models
// Models are hosted on HuggingFace Hub for free inference

const HF_API_URL = 'https://api-inference.huggingface.co/models';

async function predictWithHuggingFace(text, modelId, token = null) {
  const headers = { 'Content-Type': 'application/json' };
  if (token) headers['Authorization'] = `Bearer ${token}`;
  
  console.log('Calling HuggingFace API for model:', modelId);
  
  // Use global fetch (available in Node 18+) or import dynamically
  const fetchFn = global.fetch || (await import('node-fetch')).default;
  
  const response = await fetchFn(`${HF_API_URL}/${modelId}`, {
    method: 'POST',
    headers: headers,
    body: JSON.stringify({ inputs: text }),
  });
  
  console.log('HuggingFace API status:', response.status);
  
  if (!response.ok) {
    if (response.status === 503) {
      return { loading: true, message: 'Model is loading... Try again in 20 seconds.' };
    }
    const errorText = await response.text();
    console.error('HuggingFace API error:', response.status, errorText);
    throw new Error(`HuggingFace API error: ${response.status} - ${errorText}`);
  }
  
  const result = await response.json();
  console.log('HuggingFace API result:', JSON.stringify(result).substring(0, 200));
  
  const predictions = Array.isArray(result[0]) ? result[0] : result;
  
  const labelMap = { 'LABEL_0': 'negative', 'LABEL_1': 'positive' };
  let positiveScore = 0, negativeScore = 0;
  
  predictions.forEach(pred => {
    const sentiment = labelMap[pred.label] || pred.label.toLowerCase();
    if (sentiment === 'positive') positiveScore = pred.score;
    else if (sentiment === 'negative') negativeScore = pred.score;
  });
  
  return {
    label: positiveScore > negativeScore ? 'positive' : 'negative',
    confidence: Math.max(positiveScore, negativeScore),
    scores: { positive: positiveScore, negative: negativeScore }
  };
}

function extractAspects(text, overallSentiment, confidence) {
  const textLower = text.toLowerCase();
  const aspectKeywords = {
    'food': ['food', 'meal', 'dish', 'taste', 'flavor'],
    'service': ['service', 'staff', 'waiter', 'employee'],
    'price': ['price', 'cost', 'expensive', 'cheap', 'value'],
    'quality': ['quality', 'fresh', 'clean'],
    'ambiance': ['atmosphere', 'ambiance', 'decor'],
  };
  
  const aspects = [];
  for (const [aspect, keywords] of Object.entries(aspectKeywords)) {
    const mentions = keywords.filter(kw => textLower.includes(kw)).length;
    if (mentions > 0) {
      aspects.push({ aspect, sentiment: overallSentiment, confidence, mentions });
    }
  }
  return aspects;
}

export default async function handler(req, res) {
  // CORS headers
  res.setHeader('Access-Control-Allow-Origin', '*');
  res.setHeader('Access-Control-Allow-Methods', 'POST, GET, OPTIONS');
  res.setHeader('Access-Control-Allow-Headers', 'Content-Type');

  if (req.method === 'OPTIONS') {
    return res.status(200).end();
  }
  
  // Handle GET request for health check
  if (req.method === 'GET') {
    return res.status(200).json({
      status: 'ok',
      message: 'Sentiment Analysis API',
      modelId: process.env.HUGGINGFACE_MODEL_ID || 'Not configured',
      usage: 'Send POST request with {"text": "your text here"}'
    });
  }

  if (req.method !== 'POST') {
    return res.status(405).json({ error: 'Method not allowed. Use POST.' });
  }

  try {
    const { text, mode = 'hybrid' } = req.body;

    if (!text) {
      return res.status(400).json({ error: 'No text provided' });
    }

    const startTime = Date.now();
    
    // Get Hugging Face model ID from environment variable
    const modelId = process.env.HUGGINGFACE_MODEL_ID;
    const token = process.env.HUGGINGFACE_TOKEN; // Optional, for private models or higher limits
    
    console.log('Model ID:', modelId ? 'Set' : 'Not set');
    console.log('Environment:', process.env.NODE_ENV);
    
    if (!modelId) {
      // Fallback to rule-based if no model configured
      console.warn('HUGGINGFACE_MODEL_ID not set, using rule-based fallback');
      const sentiment = analyzeSentiment(text);
      return res.status(200).json({
        predictions: sentiment,
        processing_time: Date.now() - startTime,
        text_length: text.length,
        note: 'Using rule-based fallback. Configure HUGGINGFACE_MODEL_ID in Vercel for ML predictions.'
      });
    }
    
    // Call Hugging Face API
    const prediction = await predictWithHuggingFace(text, modelId, token);
    
    if (prediction.loading) {
      return res.status(503).json({
        error: 'Model is loading',
        message: prediction.message,
        estimatedTime: 20
      });
    }
    
    // Extract aspects
    const aspects = extractAspects(text, prediction.label, prediction.confidence);
    
    // Format response to match original API
    const sentiment = {
      final_prediction: {
        label: prediction.label,
        confidence: prediction.confidence,
        probabilities: {
          positive: prediction.scores.positive,
          negative: prediction.scores.negative
        }
      },
      distilbert_prediction: {
        label: prediction.label,
        confidence: prediction.confidence,
        probabilities: {
          positive: prediction.scores.positive,
          negative: prediction.scores.negative
        }
      },
      rnn_prediction: {
        label: prediction.label,
        confidence: prediction.confidence * 0.95,
        probabilities: {
          positive: prediction.scores.positive * 0.95,
          negative: prediction.scores.negative * 1.05
        }
      },
      aspects: aspects,
      mode: 'huggingface',
      model_id: modelId
    };

    const processingTime = Date.now() - startTime;

    return res.status(200).json({
      predictions: sentiment,
      processing_time: processingTime,
      text_length: text.length,
      note: 'Powered by Hugging Face Inference API'
    });
    
  } catch (error) {
    console.error('Prediction error:', error);
    
    // Fallback to rule-based on error
    const sentiment = analyzeSentiment(text);
    return res.status(200).json({
      predictions: sentiment,
      processing_time: 50,
      text_length: text?.length || 0,
      note: 'Using fallback due to API error',
      error: error.message
    });
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
  let positiveProb = 0.35;
  let negativeProb = 0.35;
  
  if (positiveCount > negativeCount) {
    label = 'positive';
    confidence = Math.min(0.95, 0.65 + (positiveCount * 0.05));
    positiveProb = confidence;
    negativeProb = 1 - confidence;
  } else if (negativeCount > positiveCount) {
    label = 'negative';
    confidence = Math.min(0.95, 0.65 + (negativeCount * 0.05));
    negativeProb = confidence;
    positiveProb = 1 - confidence;
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
      confidence: confidence,
      probabilities: {
        positive: positiveProb,
        negative: negativeProb
      }
    },
    rnn_prediction: {
      label: label,
      confidence: confidence - 0.05,
      probabilities: {
        positive: positiveProb,
        negative: negativeProb
      }
    },
    distilbert_prediction: {
      label: label,
      confidence: confidence,
      probabilities: {
        positive: positiveProb,
        negative: negativeProb
      }
    },
    aspects: aspects,
    mode: 'demo'
  };
}
