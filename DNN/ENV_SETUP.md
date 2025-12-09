# Environment Variables Setup Guide

## For Local Development

1. **Copy the example file**:
   ```bash
   cd web
   cp .env.local.example .env.local
   ```

2. **Upload your model to Hugging Face**:
   ```bash
   # From project root
   pip install huggingface_hub transformers torch
   huggingface-cli login
   python scripts/upload_to_huggingface.py
   ```

3. **Update `.env.local`** with your model ID:
   ```
   HUGGINGFACE_MODEL_ID=your-username/sentiment-distilbert
   ```

4. **Test locally**:
   ```bash
   cd web
   npm run dev
   ```

## For Vercel Deployment

1. **Go to Vercel Dashboard**:
   - Open your project
   - Navigate to **Settings** → **Environment Variables**

2. **Add the following variables**:
   
   | Name | Value | Environment |
   |------|-------|-------------|
   | `HUGGINGFACE_MODEL_ID` | `your-username/sentiment-distilbert` | Production, Preview, Development |
   | `HUGGINGFACE_TOKEN` | (optional) your HF token | Production |

3. **Redeploy**:
   - Go to **Deployments** tab
   - Click the three dots on the latest deployment
   - Select **Redeploy**
   - Or push a new commit to trigger deployment

## Getting Your Hugging Face Model ID

After running `scripts/upload_to_huggingface.py`, you'll get a URL like:
```
https://huggingface.co/your-username/sentiment-distilbert
```

Your model ID is: `your-username/sentiment-distilbert`

## Optional: Hugging Face Token

For public models: **Not required**
For private models or higher rate limits:
1. Go to https://huggingface.co/settings/tokens
2. Create a new token with "Read" permission
3. Add to environment variables

## Rate Limits

**Free tier (no token)**:
- 30,000 requests per month
- Good for demos and light usage

**With token**:
- Higher rate limits
- Access to private models
- Better performance

## Testing Environment Variables

Test if your variables are set correctly:

```bash
cd web
npm run dev
```

Then visit http://localhost:3000 and try a prediction.

Check console logs for:
- ✅ "Using model: your-username/sentiment-distilbert"
- ❌ "HUGGINGFACE_MODEL_ID not set" (means variable missing)

## Troubleshooting

**"Model is loading"**: First request may take 20 seconds (cold start)
**"HUGGINGFACE_MODEL_ID not set"**: Check .env.local file exists and is in web/ directory
**"Unauthorized"**: Check your HUGGINGFACE_TOKEN if using private model
**Rate limit exceeded**: Wait or upgrade Hugging Face plan

## Security Notes

- ✅ `.env.local` is in `.gitignore` (never commit it!)
- ✅ Environment variables are only accessible server-side
- ✅ Tokens are never exposed to the browser
- ✅ Use separate tokens for production and development
