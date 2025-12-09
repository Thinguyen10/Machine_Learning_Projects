// Batch upload API using Hugging Face for predictions

const HF_API_URL = 'https://api-inference.huggingface.co/models';

async function predictWithHuggingFace(text, modelId, token = null) {
  const headers = { 'Content-Type': 'application/json' };
  if (token) headers['Authorization'] = `Bearer ${token}`;
  
  const response = await fetch(`${HF_API_URL}/${modelId}`, {
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
    const { csv_data, filename, text_columns } = req.body;

    if (!csv_data) {
      return res.status(400).json({ error: 'No CSV data provided' });
    }

    // Parse CSV
    const rows = csv_data.split('\n').filter(row => row.trim());
    const headers = rows[0].split(',').map(h => h.trim());
    const dataRows = rows.slice(1);
    
    // Extract text from specified columns
    const textColumnIndex = headers.indexOf(text_columns?.[0] || headers[0]);
    const texts = dataRows.map(row => {
      const cols = row.split(',');
      return cols[textColumnIndex]?.trim() || '';
    }).filter(t => t.length > 0);

    const modelId = process.env.HUGGINGFACE_MODEL_ID;
    const token = process.env.HUGGINGFACE_TOKEN;

    if (!modelId) {
      // Fallback without predictions
      return res.status(200).json({
        message: 'Batch upload received (no model configured)',
        filename: filename || 'upload.csv',
        total_rows: dataRows.length,
        processed: 0,
        note: 'Configure HUGGINGFACE_MODEL_ID to enable predictions.',
        success: true
      });
    }

    // Limit batch size to avoid timeout (Vercel has 10s limit for hobby plan)
    const maxBatch = 5;
    const limitedTexts = texts.slice(0, maxBatch);
    
    // Process batch with rate limiting
    const results = [];
    for (let i = 0; i < limitedTexts.length; i++) {
      try {
        const result = await predictWithHuggingFace(limitedTexts[i], modelId, token);
        if (!result.loading) {
          results.push({ text: limitedTexts[i], prediction: result, index: i });
        }
        // Rate limiting: wait 1 second between requests
        if (i < limitedTexts.length - 1) {
          await new Promise(resolve => setTimeout(resolve, 1000));
        }
      } catch (error) {
        results.push({ text: limitedTexts[i], error: error.message, index: i });
      }
    }
    
    const successCount = results.filter(r => !r.error).length;

    return res.status(200).json({
      message: 'Batch upload completed',
      filename: filename || 'upload.csv',
      total_rows: dataRows.length,
      processed: successCount,
      previewed: limitedTexts.length,
      results: results,
      note: `Processed first ${limitedTexts.length} rows. For full batch processing, use dedicated backend.`,
      success: true
    });
    
  } catch (error) {
    console.error('Upload error:', error);
    return res.status(500).json({ 
      error: 'Internal server error',
      details: error.message 
    });
  }
}
