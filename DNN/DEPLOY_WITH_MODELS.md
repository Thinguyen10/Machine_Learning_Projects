# 🚀 Complete Deployment Guide - Full ML Model Support

This guide shows you how to deploy your sentiment analysis app to Vercel with **full ML model functionality** using Hugging Face for free model hosting.

## 📋 Overview

**Problem**: Your ML models (280MB+) are too large for Vercel's limits.

**Solution**: Host models on Hugging Face Hub (free) and call them via API.

**Result**: Your Vercel app performs exactly like the local version! 🎉

---

## 🎯 Quick Start (3 Steps)

### Step 1: Upload Model to Hugging Face

```bash
# Install requirements
pip install huggingface_hub transformers torch

# Login to Hugging Face
huggingface-cli login
# Paste your token from: https://huggingface.co/settings/tokens

# Upload your model
python scripts/upload_to_huggingface.py
```

This will:
- Upload your fine-tuned DistilBERT model
- Give you a model ID like: `your-username/sentiment-distilbert`
- Make it publicly accessible for free

### Step 2: Configure Environment Variables

**In Vercel Dashboard**:

1. Go to your project → **Settings** → **Environment Variables**
2. Add:
   ```
   Name: HUGGINGFACE_MODEL_ID
   Value: your-username/sentiment-distilbert
   Environment: Production, Preview, Development
   ```

### Step 3: Deploy

```bash
git add .
git commit -m "Add Hugging Face integration"
git push origin main
```

Vercel will automatically redeploy with your ML model! ✨

---

## 📚 Detailed Instructions

### Prerequisites

1. **Hugging Face Account** (free)
   - Sign up: https://huggingface.co/join
   - Create API token: https://huggingface.co/settings/tokens

2. **Trained Model**
   - You should have `outputs/transformer/` directory
   - Contains your fine-tuned DistilBERT model

3. **Vercel Account** (free)
   - Connected to your GitHub repository

### Part A: Model Upload

#### 1. Install Hugging Face CLI

```bash
pip install huggingface_hub
```

#### 2. Login to Hugging Face

```bash
huggingface-cli login
```

When prompted, paste your token from: https://huggingface.co/settings/tokens

#### 3. Run Upload Script

```bash
python scripts/upload_to_huggingface.py
```

Follow the prompts:
- Enter your Hugging Face username
- Wait for upload to complete (~2-3 minutes)
- Note your model ID (you'll need this!)

#### 4. Verify Upload

Visit: `https://huggingface.co/your-username/sentiment-distilbert`

You should see your model with all files uploaded.

### Part B: Vercel Configuration

#### 1. Set Environment Variables

**Option A: Via Dashboard (Recommended)**

1. Go to https://vercel.com/dashboard
2. Select your project
3. Go to **Settings** → **Environment Variables**
4. Add variable:
   - **Name**: `HUGGINGFACE_MODEL_ID`
   - **Value**: `your-username/sentiment-distilbert`
   - **Environments**: Check all (Production, Preview, Development)
5. Click **Save**

**Option B: Via Vercel CLI**

```bash
vercel env add HUGGINGFACE_MODEL_ID
# Enter: your-username/sentiment-distilbert
# Select: Production, Preview, Development
```

#### 2. Push Code Updates

```bash
git add .
git commit -m "Integrate Hugging Face API for ML predictions"
git push origin main
```

#### 3. Wait for Deployment

- Vercel automatically deploys on push
- Check progress: https://vercel.com/dashboard
- Deployment takes ~2-3 minutes

### Part C: Testing Your Deployment

#### 1. First Request (Cold Start)

The first prediction may take 15-20 seconds as Hugging Face loads your model.

**You might see**: "Model is loading... Please try again in 20 seconds."

This is normal! Just wait and try again.

#### 2. Subsequent Requests

After the first load, predictions are fast (~1-2 seconds).

#### 3. Test All Features

Visit your Vercel URL and test:

- ✅ **Home Page**: Single text prediction
- ✅ **Upload Page**: CSV batch processing (first 5 rows)
- ✅ **Dashboard**: View analytics
- ✅ **Aspect Analysis**: See detailed breakdowns

---

## 🔧 Advanced Configuration

### Local Development

Create `web/.env.local`:

```bash
cp web/.env.local.example web/.env.local
```

Edit and add:
```
HUGGINGFACE_MODEL_ID=your-username/sentiment-distilbert
```

Test locally:
```bash
cd web
npm run dev
```

### Higher Rate Limits (Optional)

Free tier: 30,000 requests/month

For higher limits, add your Hugging Face token:

**In Vercel**:
```
Name: HUGGINGFACE_TOKEN
Value: hf_xxxxxxxxxxxxx
Environment: Production
```

Get token from: https://huggingface.co/settings/tokens

### Private Models

If you want to keep your model private:

1. Make model private on Hugging Face
2. Add `HUGGINGFACE_TOKEN` to Vercel (required for private models)

---

## 🎯 Performance

### Response Times

| Scenario | Time |
|----------|------|
| Cold start (first request) | 15-20 seconds |
| Warm (subsequent requests) | 1-2 seconds |
| Batch upload (5 rows) | 5-10 seconds |

### Rate Limits

**Free Tier**:
- 30,000 requests/month
- ~1,000 requests/day
- Perfect for demos and portfolios

**With Token**:
- Higher limits
- Better performance
- Access to private models

---

## 📊 Feature Comparison

| Feature | Local Version | Deployed (HuggingFace) |
|---------|--------------|------------------------|
| ML Model | ✅ Local PyTorch | ✅ HuggingFace API |
| Accuracy | ✅ Full | ✅ Full (same model!) |
| Speed | ✅ Very Fast | ✅ Fast (1-2s) |
| Aspect Analysis | ✅ Yes | ✅ Yes |
| Batch Processing | ✅ Unlimited | ⚠️ Limited (5 rows) |
| Database | ✅ SQLite | ❌ Not included |
| Cost | 💻 Free (local) | 🌐 Free (HF tier) |

---

## 🐛 Troubleshooting

### "Model is loading" message

**Cause**: First request triggers model loading (cold start)

**Solution**: Wait 20 seconds and try again

### "HUGGINGFACE_MODEL_ID not set"

**Cause**: Environment variable not configured

**Solution**: 
1. Check Vercel dashboard → Settings → Environment Variables
2. Ensure variable is set for all environments
3. Redeploy if needed

### Predictions seem wrong

**Cause**: Using fallback rule-based analysis

**Solution**:
1. Verify model uploaded successfully to HuggingFace
2. Check model ID is correct in environment variables
3. Check browser console for errors

### Rate limit exceeded

**Cause**: Exceeded 30,000 monthly requests

**Solutions**:
- Wait for next month
- Add `HUGGINGFACE_TOKEN` for higher limits
- Upgrade HuggingFace plan

### Vercel build fails

**Check**:
1. Ensure `web/lib/huggingface-api.js` exists
2. Check Vercel build logs
3. Verify `package.json` is correct

---

## 🎓 For Your Project Submission

### What to Share

1. **Live Demo**: Your Vercel URL with working ML predictions
2. **GitHub Repo**: Full source code
3. **Model Link**: Your HuggingFace model page

### Explaining Your Architecture

**In your report, mention**:

> "The application uses a hybrid cloud architecture:
> - **Frontend & API**: Deployed on Vercel (Next.js)
> - **ML Models**: Hosted on HuggingFace Hub
> - **Inference**: Serverless API calls to HuggingFace
> 
> This architecture provides:
> - ✅ No file size limitations
> - ✅ Same ML accuracy as local version
> - ✅ Scalable and cost-effective (free tier)
> - ✅ Professional deployment pattern"

### Demo Tips

1. **Warm up the model** before live demo (make a test request)
2. **Show aspect analysis** - it's impressive!
3. **Mention cold start** if asked about initial delay
4. **Compare with local version** to show they're identical

---

## 💰 Cost Breakdown

| Service | Plan | Cost | Usage |
|---------|------|------|-------|
| Vercel | Hobby | Free | Hosting & APIs |
| HuggingFace | Free Tier | Free | 30k requests/month |
| GitHub | Free | Free | Source control |
| **Total** | | **$0/month** | 🎉 |

---

## 🚀 Next Steps

### After Successful Deployment

1. ✅ Test all features thoroughly
2. ✅ Add Vercel URL to your README
3. ✅ Take screenshots for your report
4. ✅ Record a demo video (optional but great!)

### Optional Enhancements

- Add caching for faster responses
- Implement database for batch results
- Add more aspect categories
- Create admin dashboard

### Alternative Deployment Options

If you need even more control:

- **Railway**: Deploy full Python backend
- **AWS Lambda**: Containerized functions
- **Google Cloud Run**: Full container support
- **Render**: Free tier with persistent storage

---

## 📞 Support Resources

- **Hugging Face Docs**: https://huggingface.co/docs
- **Vercel Docs**: https://vercel.com/docs
- **This Project's Issues**: https://github.com/your-repo/issues

---

## ✅ Deployment Checklist

Before submitting:

- [ ] Model uploaded to HuggingFace
- [ ] Environment variables set in Vercel
- [ ] App deployed and accessible
- [ ] Test prediction works
- [ ] Test batch upload works
- [ ] Screenshots taken
- [ ] README updated with Vercel URL
- [ ] Model link added to README

---

**🎉 Congratulations!** Your ML-powered app is now live with full functionality!

Your deployment URL: `https://your-project.vercel.app`

Share it with pride! 🚀
