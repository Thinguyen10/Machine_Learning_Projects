// Batch upload API using Hugging Face for predictions

const HF_API_URL = 'https://api-inference.huggingface.co/models';

// Simple rule-based sentiment analysis fallback
function analyzeSentimentRuleBased(text) {
  const positive = ['good', 'great', 'excellent', 'amazing', 'wonderful', 'fantastic', 'love', 'best', 'perfect'];
  const negative = ['bad', 'terrible', 'awful', 'horrible', 'worst', 'hate', 'poor', 'disappointing'];
  
  const textLower = text.toLowerCase();
  let positiveCount = positive.filter(word => textLower.includes(word)).length;
  let negativeCount = negative.filter(word => textLower.includes(word)).length;
  
  if (positiveCount > negativeCount) {
    return { label: 'positive', confidence: 0.6 + (positiveCount * 0.05) };
  } else if (negativeCount > positiveCount) {
    return { label: 'negative', confidence: 0.6 + (negativeCount * 0.05) };
  }
  return { label: 'neutral', confidence: 0.5 };
}

async function predictWithHuggingFace(text, modelId, token = null) {
  const headers = { 'Content-Type': 'application/json' };
  if (token) headers['Authorization'] = `Bearer ${token}`;
  
  // Use global fetch (available in Node 18+) or import dynamically  
  const fetchFn = global.fetch || (await import('node-fetch')).default;
  
  const response = await fetchFn(`${HF_API_URL}/${modelId}`, {
    method: 'POST',
    headers: headers,
    body: JSON.stringify({ inputs: text }),
  });
  
  if (!response.ok) {
    if (response.status === 503) return { loading: true };
    throw new Error(`API error: ${response.status}`);
  }
  
  const result = await response.json();
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
    confidence: Math.max(positiveScore, negativeScore)
  };
}

export default async function handler(req, res) {
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
    console.log('[BATCH] Request received');
    const { csv_data, filename, text_columns } = req.body;

    if (!csv_data) {
      console.log('[BATCH] No CSV data in request');
      return res.status(400).json({ error: 'No CSV data provided' });
    }

    console.log('[BATCH] CSV data length:', csv_data.length);
    
    // Parse CSV - handle both comma and quoted fields
    const rows = csv_data.split('\n').filter(row => row.trim());
    console.log('[BATCH] Total rows:', rows.length);
    
    if (rows.length < 2) {
      return res.status(400).json({ error: 'CSV must have at least a header and one data row' });
    }
    
    const headers = rows[0].split(',').map(h => h.trim().replace(/"/g, ''));
    const dataRows = rows.slice(1).filter(row => row.trim());
    
    console.log('[BATCH] Headers:', headers);
    console.log('[BATCH] Data rows:', dataRows.length);
    console.log('[BATCH] Text columns requested:', text_columns);
    
    // Extract text from specified columns
    const textColumnIndex = text_columns?.[0] ? headers.indexOf(text_columns[0]) : 0;
    console.log('[BATCH] Using column index:', textColumnIndex, 'Column name:', headers[textColumnIndex]);
    
    const texts = dataRows.map((row, idx) => {
      // Simple CSV parsing - handles basic quoted fields
      const cols = row.match(/(".*?"|[^",\s]+)(?=\s*,|\s*$)/g) || row.split(',');
      const text = cols[textColumnIndex]?.trim().replace(/^"|"$/g, '') || '';
      if (idx < 3) console.log(`[BATCH] Row ${idx} text:`, text.substring(0, 50));
      return text;
    }).filter(t => t.length > 0);
    
    console.log('[BATCH] Extracted texts:', texts.length);

    const modelId = process.env.HUGGINGFACE_MODEL_ID;
    const token = process.env.HUGGINGFACE_TOKEN;

    console.log('[BATCH] Model ID:', modelId ? 'Set' : 'Not set');
    console.log('[BATCH] Token:', token ? 'Set' : 'Not set');

    if (!modelId) {
      console.log('[BATCH] No model configured, returning early');
      return res.status(200).json({
        message: 'Batch upload received (no model configured)',
        filename: filename || 'upload.csv',
        total_rows: dataRows.length,
        processed: 0,
        processed_rows: 0,
        note: 'Configure HUGGINGFACE_MODEL_ID to enable predictions.',
        success: true
      });
    }

    // Limit batch size to avoid timeout (Vercel has 10s limit for hobby plan)
    const maxBatch = 5;
    const limitedTexts = texts.slice(0, maxBatch);
    
    console.log('[BATCH] Processing', limitedTexts.length, 'texts');
    
    // Process batch with rate limiting
    const results = [];
    for (let i = 0; i < limitedTexts.length; i++) {
      try {
        const result = await predictWithHuggingFace(limitedTexts[i], modelId, token);
        if (!result.loading) {
          results.push({ 
            text: limitedTexts[i], 
            prediction: result, 
            index: i,
            review_id: i + 1 
          });
        }
        // Rate limiting: wait 1 second between requests
        if (i < limitedTexts.length - 1) {
          await new Promise(resolve => setTimeout(resolve, 1000));
        }
      } catch (error) {
        console.error(`Error processing text ${i}:`, error.message);
        // Use rule-based fallback on error
        const fallbackPrediction = analyzeSentimentRuleBased(limitedTexts[i]);
        results.push({ 
          text: limitedTexts[i], 
          prediction: fallbackPrediction,
          index: i,
          review_id: i + 1,
          fallback: true,
          error: error.message 
        });
      }
    }
    
    const successCount = results.filter(r => r.prediction).length;

    console.log('[BATCH] Success count:', successCount);
    console.log('[BATCH] Total results:', results.length);

    // Store results for dashboard (using Vercel KV would be better for production)
    const analysisResults = results.map(r => ({
      text: r.text,
      sentiment: r.prediction?.label || 'error',
      confidence: r.prediction?.confidence || 0,
      timestamp: new Date().toISOString()
    }));

    // Save to global variable that dashboard can read
    // Note: In production, use a real database (Vercel KV, MongoDB, etc.)
    global.lastBatchResults = analysisResults;

    console.log('[BATCH] Returning response with', successCount, 'processed');

    return res.status(200).json({
      message: 'Batch upload completed',
      filename: filename || 'upload.csv',
      total_rows: dataRows.length,
      processed: successCount,
      processed_rows: successCount, // Add for UI compatibility
      previewed: limitedTexts.length,
      results: results,
      stored_count: analysisResults.length,
      note: `Processed first ${limitedTexts.length} rows. Results visible in dashboard.`,
      success: true
    });
    
  } catch (error) {
    console.error('[BATCH] Upload error:', error);
    console.error('[BATCH] Stack:', error.stack);
    return res.status(500).json({ 
      error: 'Internal server error',
      details: error.message,
      stack: process.env.NODE_ENV === 'development' ? error.stack : undefined
    });
  }
}
