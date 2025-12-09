// Health check endpoint to verify configuration
export default async function handler(req, res) {
  const modelId = process.env.HUGGINGFACE_MODEL_ID;
  const hasToken = !!process.env.HUGGINGFACE_TOKEN;
  
  const status = {
    status: 'ok',
    timestamp: new Date().toISOString(),
    environment: process.env.NODE_ENV,
    huggingface: {
      modelId: modelId || 'NOT_SET',
      hasToken: hasToken,
      configured: !!modelId
    },
    endpoints: {
      predict: '/api/predict',
      dashboard: '/api/dashboard',
      batchUpload: '/api/batch-upload'
    }
  };
  
  if (!modelId) {
    status.warning = 'HUGGINGFACE_MODEL_ID not set. Using rule-based fallback.';
    status.instructions = 'Add HUGGINGFACE_MODEL_ID=Thi144/sentiment-distilbert to Vercel environment variables';
  }
  
  res.status(200).json(status);
}
