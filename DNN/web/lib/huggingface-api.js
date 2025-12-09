/**
 * Hugging Face Inference API Integration
 * 
 * This module handles predictions using models hosted on Hugging Face.
 * Free tier includes 30,000 requests per month.
 * 
 * Documentation: https://huggingface.co/docs/api-inference/index
 */

const HF_API_URL = 'https://api-inference.huggingface.co/models';

/**
 * Call Hugging Face Inference API for sentiment analysis
 * 
 * @param {string} text - Text to analyze
 * @param {string} modelId - Hugging Face model ID (e.g., "username/model-name")
 * @param {string} token - Optional HF API token (not needed for public models)
 * @returns {Promise<Object>} Prediction results
 */
export async function predictWithHuggingFace(text, modelId, token = null) {
  try {
    const headers = {
      'Content-Type': 'application/json',
    };
    
    // Add token if provided (for private models or higher rate limits)
    if (token) {
      headers['Authorization'] = `Bearer ${token}`;
    }
    
    const response = await fetch(`${HF_API_URL}/${modelId}`, {
      method: 'POST',
      headers: headers,
      body: JSON.stringify({ inputs: text }),
    });
    
    if (!response.ok) {
      const error = await response.text();
      
      // Model is loading (cold start)
      if (response.status === 503) {
        return {
          loading: true,
          message: 'Model is loading... Please try again in 20 seconds.',
          estimatedTime: 20
        };
      }
      
      throw new Error(`HuggingFace API error: ${response.status} - ${error}`);
    }
    
    const result = await response.json();
    
    // Transform HuggingFace output to our format
    return transformHuggingFaceOutput(result);
    
  } catch (error) {
    console.error('HuggingFace API error:', error);
    throw error;
  }
}

/**
 * Transform Hugging Face output to match our app's format
 */
function transformHuggingFaceOutput(hfResult) {
  // HuggingFace returns: [[{label: 'LABEL_1', score: 0.95}, ...]]
  const predictions = Array.isArray(hfResult[0]) ? hfResult[0] : hfResult;
  
  // Find positive and negative scores
  const labelMap = {
    'LABEL_0': 'negative',
    'LABEL_1': 'positive',
    'NEGATIVE': 'negative',
    'POSITIVE': 'positive'
  };
  
  let positiveScore = 0;
  let negativeScore = 0;
  
  predictions.forEach(pred => {
    const sentiment = labelMap[pred.label] || pred.label.toLowerCase();
    if (sentiment === 'positive') {
      positiveScore = pred.score;
    } else if (sentiment === 'negative') {
      negativeScore = pred.score;
    }
  });
  
  // Determine final prediction
  const label = positiveScore > negativeScore ? 'positive' : 'negative';
  const confidence = Math.max(positiveScore, negativeScore);
  
  return {
    label,
    confidence,
    scores: {
      positive: positiveScore,
      negative: negativeScore
    },
    rawOutput: predictions
  };
}

/**
 * Extract aspects from text using simple keyword matching
 * (More sophisticated aspect extraction would require specialized models)
 */
export function extractAspects(text, overallSentiment, confidence) {
  const textLower = text.toLowerCase();
  
  const aspectKeywords = {
    'food': ['food', 'meal', 'dish', 'taste', 'flavor', 'cuisine', 'menu'],
    'service': ['service', 'staff', 'waiter', 'employee', 'server', 'manager'],
    'price': ['price', 'cost', 'expensive', 'cheap', 'value', 'money'],
    'quality': ['quality', 'fresh', 'clean', 'standard', 'condition'],
    'ambiance': ['atmosphere', 'ambiance', 'decor', 'music', 'vibe'],
    'location': ['location', 'place', 'area', 'parking', 'access']
  };
  
  const aspects = [];
  
  for (const [aspect, keywords] of Object.entries(aspectKeywords)) {
    const mentions = keywords.filter(kw => textLower.includes(kw)).length;
    if (mentions > 0) {
      aspects.push({
        aspect,
        sentiment: overallSentiment,
        confidence: confidence * (0.8 + (mentions * 0.1)), // Slightly adjust confidence
        mentions
      });
    }
  }
  
  return aspects;
}

/**
 * Wait for model to load (retry with exponential backoff)
 */
export async function predictWithRetry(text, modelId, token = null, maxRetries = 3) {
  for (let i = 0; i < maxRetries; i++) {
    const result = await predictWithHuggingFace(text, modelId, token);
    
    if (!result.loading) {
      return result;
    }
    
    // Wait before retry (exponential backoff)
    const waitTime = Math.min(20000 * Math.pow(2, i), 60000); // Max 60 seconds
    await new Promise(resolve => setTimeout(resolve, waitTime));
  }
  
  throw new Error('Model loading timeout. Please try again later.');
}

/**
 * Batch prediction with rate limiting
 */
export async function batchPredict(texts, modelId, token = null, delayMs = 1000) {
  const results = [];
  
  for (let i = 0; i < texts.length; i++) {
    try {
      const result = await predictWithRetry(texts[i], modelId, token);
      results.push({
        text: texts[i],
        prediction: result,
        index: i
      });
      
      // Rate limiting: wait between requests
      if (i < texts.length - 1) {
        await new Promise(resolve => setTimeout(resolve, delayMs));
      }
    } catch (error) {
      results.push({
        text: texts[i],
        error: error.message,
        index: i
      });
    }
  }
  
  return results;
}
