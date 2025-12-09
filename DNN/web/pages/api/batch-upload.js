// Batch upload API using Hugging Face for predictions
import { batchPredict } from '../../lib/huggingface-api';

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
    
    // Process batch with rate limiting (1 request per second)
    const results = await batchPredict(limitedTexts, modelId, token, 1000);
    
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
