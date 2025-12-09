# Step-by-Step: Fix Your Vercel Deployment NOW

Follow these exact steps to get your ML models working on Vercel.

## ⏱️ Time Required: 10 minutes

---

## Step 1: Install Hugging Face Tools

Open terminal and run:

```bash
pip install huggingface_hub transformers torch
```

✅ **Check**: You should see "Successfully installed huggingface_hub..."

---

## Step 2: Create Hugging Face Account (if needed)

1. Visit: https://huggingface.co/join
2. Sign up with email
3. Verify your email

✅ **Check**: You can login to https://huggingface.co

---

## Step 3: Get Your Hugging Face Token

1. Visit: https://huggingface.co/settings/tokens
2. Click "New token"
3. Name it: "vercel-sentiment-model"
4. Type: "Read" (or "Write" to be safe)
5. Click "Generate"
6. **Copy the token** (starts with `hf_...`)

✅ **Check**: You have a token that looks like `hf_xxxxxxxxxxxxx`

---

## Step 4: Login to Hugging Face CLI

In terminal:

```bash
huggingface-cli login
```

When prompted:
- Paste your token (from Step 3)
- Press Enter

✅ **Check**: You see "Login successful"

---

## Step 5: Upload Your Model

From your project directory:

```bash
cd "/Users/thinguyen/Documents/GitHub/CST-435/DNN"
python scripts/upload_to_huggingface.py
```

When prompted:
- Enter your Hugging Face username
- Wait 2-3 minutes for upload

✅ **Check**: You see "✅ DistilBERT model uploaded successfully!"

**IMPORTANT**: Copy the model ID from the output. It looks like:
```
HUGGINGFACE_MODEL_ID=your-username/sentiment-distilbert
```

---

## Step 6: Verify Model on Hugging Face

1. Visit: https://huggingface.co/your-username/sentiment-distilbert
   (Replace `your-username` with your actual username)
2. You should see your model files

✅ **Check**: Model page loads and shows files like `config.json`, `pytorch_model.bin`

---

## Step 7: Open Vercel Dashboard

1. Visit: https://vercel.com/dashboard
2. Login if needed
3. Find your project (CST-435 or DNN)
4. Click on it

✅ **Check**: You're on your project's dashboard

---

## Step 8: Add Environment Variable

In Vercel:
1. Click "Settings" (top menu)
2. Click "Environment Variables" (left sidebar)
3. Click "Add New" button
4. Fill in:
   - **Key**: `HUGGINGFACE_MODEL_ID`
   - **Value**: `your-username/sentiment-distilbert` (from Step 5)
   - **Environments**: Check all three boxes ☑️ Production ☑️ Preview ☑️ Development
5. Click "Save"

✅ **Check**: You see the variable in the list

---

## Step 9: Push Your Code Changes

In terminal:

```bash
cd "/Users/thinguyen/Documents/GitHub/CST-435/DNN"
git add .
git commit -m "Add Hugging Face integration for ML models"
git push origin main
```

✅ **Check**: Code pushed successfully, no errors

---

## Step 10: Wait for Deployment

1. Go back to Vercel dashboard
2. Click "Deployments" tab
3. You'll see a new deployment starting
4. Wait for status to change to "Ready" (2-3 minutes)

✅ **Check**: Deployment shows ✅ "Ready"

---

## Step 11: Test Your App!

1. Click on the deployment to get your URL (or use your existing Vercel URL)
2. Visit the URL
3. On the home page, enter a review:
   ```
   This product is absolutely amazing! Great quality and fast shipping.
   ```
4. Click "Analyze Sentiment"
5. **First time only**: You might see "Model is loading... Please try again in 20 seconds"
   - If so, wait 20 seconds and try again
6. You should now see:
   - Sentiment prediction
   - Confidence score
   - Aspect analysis
   - Note: "Powered by Hugging Face Inference API"

✅ **Check**: You get a real ML prediction with aspects analyzed

---

## Step 12: Test Again (Should Be Fast Now)

Try another review immediately:
```
Terrible experience. The service was horrible and it broke after one day.
```

✅ **Check**: Prediction appears in 1-2 seconds (no loading delay)

---

## 🎉 Success Checklist

After completing all steps, verify:

- [ ] Model uploaded to Hugging Face
- [ ] Model visible at huggingface.co/your-username/sentiment-distilbert
- [ ] Environment variable set in Vercel
- [ ] Code pushed to GitHub
- [ ] Vercel deployment successful
- [ ] First prediction works (may take 20 seconds)
- [ ] Subsequent predictions are fast (1-2 seconds)
- [ ] Aspect analysis is shown
- [ ] Message says "Powered by Hugging Face Inference API"

---

## 🐛 Something Wrong?

### "huggingface-cli: command not found"
```bash
pip install --upgrade huggingface_hub
```

### "Model not found" when uploading
- Check you have `outputs/transformer/` folder
- Make sure you trained your model first
- Verify files exist: `ls outputs/transformer/`

### "HUGGINGFACE_MODEL_ID not set" on Vercel
- Go back to Step 8
- Make sure you saved the variable
- Check you selected all three environments
- Try redeploying: Deployments → ⋯ → Redeploy

### "Model is loading" every time
- First request always takes 20 seconds (cold start)
- If ALL requests take 20 seconds, check:
  - Model ID is correct in Vercel
  - Model successfully uploaded to HuggingFace
  - No errors in Vercel function logs

### Predictions seem wrong
- Check browser console (F12) for errors
- Verify model ID matches exactly
- Try the same text locally to compare

---

## 📞 Need Help?

1. Check `DEPLOY_WITH_MODELS.md` for detailed docs
2. Check `QUICK_FIX.md` for quick reference
3. Check Vercel logs: Dashboard → Deployments → View Function Logs
4. Check browser console: Press F12, look for red errors

---

## ✅ You're Done!

Your app now:
- ✅ Uses real ML models (94% accuracy)
- ✅ Deployed on Vercel for free
- ✅ Gets predictions from Hugging Face
- ✅ Works exactly like the local version
- ✅ Can handle thousands of requests per month

**Share your URL**: `https://your-project.vercel.app`

**Next**: Submit your project with the live URL! 🎓
