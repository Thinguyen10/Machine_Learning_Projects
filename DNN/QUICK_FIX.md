# 🚀 Quick Deploy Guide - Get Your App Working NOW!

## The Problem
Your Vercel app is running but using fake rule-based predictions instead of your real ML models because the model files (280MB+) are in `.gitignore` and can't be deployed to Vercel.

## The Solution
Host your models on **Hugging Face** (free) and have your Vercel app call them via API.

---

## ⚡ Quick Start (10 Minutes)

### Step 1: Install Hugging Face CLI (1 min)

```bash
pip install huggingface_hub
```

### Step 2: Login to Hugging Face (2 min)

```bash
huggingface-cli login
```

Don't have an account? 
1. Sign up: https://huggingface.co/join
2. Get token: https://huggingface.co/settings/tokens
3. Paste token when prompted

### Step 3: Upload Your Model (3 min)

```bash
python scripts/upload_to_huggingface.py
```

- Enter your Hugging Face username when prompted
- Wait for upload (~2 minutes)
- **Save the model ID** it gives you (e.g., `username/sentiment-distilbert`)

### Step 4: Configure Vercel (2 min)

1. Go to https://vercel.com/dashboard
2. Open your project
3. Go to **Settings** → **Environment Variables**
4. Click **Add New**
5. Enter:
   - **Name**: `HUGGINGFACE_MODEL_ID`
   - **Value**: `your-username/sentiment-distilbert` (from Step 3)
   - **Environment**: Select all (Production, Preview, Development)
6. Click **Save**

### Step 5: Redeploy (2 min)

**Option A: Push to GitHub**
```bash
git add .
git commit -m "Add Hugging Face integration"
git push origin main
```

**Option B: Redeploy from Vercel Dashboard**
1. Go to **Deployments** tab
2. Click ⋯ on latest deployment
3. Select **Redeploy**

### Step 6: Test! (1 min)

1. Visit your Vercel URL
2. Try a prediction (first one takes 20 seconds - model loading)
3. Subsequent predictions are fast (1-2 seconds)
4. ✅ You're now using your real ML model!

---

## 📋 Checklist

- [ ] Installed `huggingface_hub`
- [ ] Logged in with `huggingface-cli login`
- [ ] Ran `python scripts/upload_to_huggingface.py`
- [ ] Got model ID (username/sentiment-distilbert)
- [ ] Added `HUGGINGFACE_MODEL_ID` to Vercel
- [ ] Redeployed app
- [ ] Tested prediction on Vercel URL
- [ ] Confirmed it says "Powered by Hugging Face Inference API"

---

## 🆘 Quick Troubleshooting

**"Model is loading"** after first request?
→ Normal! Wait 20 seconds and try again. Subsequent requests are fast.

**"HUGGINGFACE_MODEL_ID not set"**?
→ Check Vercel environment variables are saved and deployment happened after.

**Upload script fails**?
→ Make sure you ran `huggingface-cli login` first.

**Model not found on Hugging Face**?
→ Check you have `outputs/transformer/` directory with your trained model.

---

## 💡 What This Does

**Before**: Vercel app → Rule-based fake predictions ❌

**After**: Vercel app → Hugging Face API → Your real ML model ✅

**Cost**: $0 (30,000 free requests/month)

**Performance**: Same accuracy as local (94.22%)

---

## 📚 Need More Details?

See `DEPLOY_WITH_MODELS.md` for complete documentation including:
- Advanced configuration
- Rate limits and optimization
- Troubleshooting guide
- Performance benchmarks

---

## ✅ Success!

Once deployed, your app will:
- Use your actual trained DistilBERT model
- Get 94% accuracy predictions
- Extract sentiment aspects
- Process batch uploads
- Work exactly like the local version

**Share your live URL**: `https://your-project.vercel.app` 🎉

---

**Questions?** Check `ENV_SETUP.md` or `DEPLOY_WITH_MODELS.md`
