# 🚀 Vercel Deployment Guide

## Overview
This deployment setup allows you to demo your sentiment analysis project on Vercel without uploading heavy model files (280MB+). The deployed version uses a lightweight rule-based sentiment analysis for demonstration purposes.

## Prerequisites
- Vercel account (free): https://vercel.com
- Git repository (already configured)
- Vercel CLI (optional): `npm install -g vercel`

## Deployment Steps

### Option 1: Deploy via Vercel Dashboard (Recommended)

1. **Push your code to GitHub** (if not already pushed):
   ```bash
   cd "/Users/thinguyen/Library/CloudStorage/OneDrive-GrandCanyonUniversity/RECENT CLASSES/SHARED CLASSES/CST-435 JT/DNN"
   git add .
   git commit -m "Prepare for Vercel deployment"
   git push origin main
   ```

2. **Import to Vercel**:
   - Go to https://vercel.com/new
   - Import your GitHub repository: `thinguyen-dev/CST-435`
   - Configure project:
     - **Framework Preset**: Next.js
     - **Root Directory**: `web`
     - **Build Command**: `npm run build`
     - **Output Directory**: `.next`

3. **Deploy**:
   - Click "Deploy"
   - Wait for build to complete (~2-3 minutes)
   - Your site will be live at: `https://your-project.vercel.app`

### Option 2: Deploy via Vercel CLI

1. **Install Vercel CLI**:
   ```bash
   npm install -g vercel
   ```

2. **Login to Vercel**:
   ```bash
   vercel login
   ```

3. **Deploy from project root**:
   ```bash
   cd "/Users/thinguyen/Library/CloudStorage/OneDrive-GrandCanyonUniversity/RECENT CLASSES/SHARED CLASSES/CST-435 JT/DNN"
   vercel
   ```

4. **Follow prompts**:
   - Set up and deploy: Yes
   - Which scope: Your account
   - Link to existing project: No
   - Project name: dnn-sentiment-analysis
   - Directory with code: `./web`
   - Override settings: No

5. **Production deployment**:
   ```bash
   vercel --prod
   ```

## What's Deployed

### ✅ Included:
- ✅ Full Next.js frontend (all pages)
- ✅ Lightweight demo sentiment analysis
- ✅ CSV upload interface
- ✅ Analytics dashboard with demo data
- ✅ Responsive UI and visualizations
- ✅ All educational content

### ❌ Excluded (too large for Vercel):
- ❌ PyTorch RNN model (20MB)
- ❌ DistilBERT transformer (260MB)
- ❌ SQLite database
- ❌ Training data and scripts
- ❌ Virtual environment

## Demo Mode Features

The deployed version runs in **demo mode**:
- Uses rule-based sentiment analysis (keyword matching)
- Shows mock dashboard data
- CSV uploads are processed but not persisted
- Perfect for project demonstration and portfolio

## Upgrading to Full Production

To enable full ML model functionality on Vercel, you have these options:

### Option A: External Model API
Host models on a separate service:
- **Hugging Face Inference API** (free tier available)
- **AWS Lambda** with container support
- **Google Cloud Run**
- **Railway** or **Render**

Update `/web/pages/api/predict.js` to call external API.

### Option B: Vercel Pro + External Storage
- Upgrade to Vercel Pro ($20/month)
- Store models in AWS S3/Cloud Storage
- Download models on cold start
- Use Vercel's 50MB function limit

### Option C: Hybrid Deployment
- Frontend on Vercel (free)
- Backend API on Railway/Render (free tier)
- Update API URLs to point to external backend

## Testing Your Deployment

1. **Home Page**: Test single text sentiment analysis
2. **Upload Page**: Upload a CSV file (any format)
3. **Dashboard**: View analytics and trends
4. **Learn Page**: Check educational content

## Custom Domain (Optional)

1. Go to your Vercel project settings
2. Navigate to "Domains"
3. Add your custom domain
4. Follow DNS configuration instructions

## Environment Variables (if needed)

In Vercel dashboard:
1. Go to Project Settings → Environment Variables
2. Add any API keys or config:
   ```
   NEXT_PUBLIC_API_URL=https://your-backend-api.com
   ```

## Troubleshooting

**Build fails**:
- Check build logs in Vercel dashboard
- Ensure `web/package.json` has all dependencies
- Try: `cd web && npm install && npm run build` locally

**Pages not loading**:
- Check browser console for errors
- Verify API routes at `/api/predict`, `/api/dashboard`

**API errors**:
- Demo mode has limited functionality
- Connect external ML API for full features

## Project Submission

For your class submission:
1. ✅ Share Vercel deployment URL
2. ✅ Include GitHub repository link
3. ✅ Explain demo mode vs full features in README
4. ✅ Models are too large for submission (document this)
5. ✅ Provide screenshots of full local version

## Your Deployment URL
After deployment, your project will be accessible at:
```
https://dnn-sentiment-analysis.vercel.app
(or your custom URL)
```

## Support
- Vercel Docs: https://vercel.com/docs
- Next.js Docs: https://nextjs.org/docs
- Deployment Issues: Check Vercel dashboard logs

---

**Note**: This demo deployment showcases your project architecture and UI without the 280MB+ model files. For grading, emphasize the local version's full ML capabilities while using the Vercel deployment as a live portfolio piece.
