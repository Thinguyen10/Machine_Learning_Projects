// Mock batch upload API for Vercel deployment
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

    // Parse CSV to count rows
    const rows = csv_data.split('\n').filter(row => row.trim());
    const rowCount = rows.length - 1; // Minus header

    // Simulate processing
    await new Promise(resolve => setTimeout(resolve, 1000));

    return res.status(200).json({
      message: 'Batch upload completed (demo mode)',
      filename: filename || 'upload.csv',
      total_rows: rowCount,
      processed: rowCount,
      note: 'Demo mode - data not persisted. Deploy with database for full functionality.',
      success: true
    });
  } catch (error) {
    console.error('Upload error:', error);
    return res.status(500).json({ error: 'Internal server error' });
  }
}
