# 🧠 Sentiment Analysis with Deep Neural Networks

Multi-model ensemble system combining RNN and Transformer models for production-ready sentiment analysis with batch processing and analytics dashboard.

## 🎯 Overview

**Models:**
- RNN with Attention (87.56% accuracy) - Fast processing
- DistilBERT Transformer (94.22% accuracy) - High accuracy
- Hybrid Ensemble (~92% accuracy) - Intelligent routing

**Features:**
- Batch CSV upload (any format - auto-detects columns)
- Real-time analytics dashboard
- Aspect-based sentiment analysis
- SQLite database with trend tracking

**Trained on:** 6,000 IMDB movie reviews + Twitter sentiment data

---

## 🚀 What Was Implemented

### Models & Training

**RNN with Attention (87.56% accuracy)**
- Bidirectional LSTM with 128 hidden units
- Attention mechanism to focus on important words
- 100-dim embeddings, dropout regularization
- Training: 20 epochs, Adam optimizer

**DistilBERT Transformer (94.22% accuracy)**
- 6-layer transformer (66M parameters)
- Pre-trained on Wikipedia, fine-tuned on sentiment
- 768-dim embeddings, 256 token max length
- Training: 4 epochs, AdamW optimizer

**Hybrid Ensemble (~92% accuracy)**
- RNN for fast filtering (70% of cases)
- DistilBERT for uncertain cases (30%)
- 60% faster than DistilBERT alone

### Application Features

**Batch Upload System**
- Flexible CSV format (auto-detects columns)
- Supports: IMDB, Twitter, Amazon, Yelp, custom datasets
- Progress tracking with SQLite database
- Processes hundreds of reviews in seconds

**Analytics Dashboard**
- Real-time sentiment statistics
- Trend visualization by date
- Top aspects breakdown (price, quality, service, etc.)
- Recent upload history

**API Modes**
- `sequential`: Fast neutral detection
- `business-insights`: Aspect-based analysis
- `compare`: Side-by-side model comparison
- `hybrid`: Production mode (recommended)

---

## 🚀 How to Run

### Setup
```bash
# Navigate to project
cd "/path/to/DNN"

# Create virtual environment
python -m venv venv
source venv/bin/activate  # macOS/Linux
# venv\Scripts\activate   # Windows

# Install dependencies
pip install "numpy<2.0" torch transformers scikit-learn pandas matplotlib seaborn tqdm requests
```

### Start Servers
```bash
# Terminal 1: Backend (port 8000)
cd api/predict
source ../../venv/bin/activate
python local_server.py

# Terminal 2: Frontend (port 3000)
cd web
npm install
npm run dev
```

### Access Application
- **Frontend**: http://localhost:3000
- **Upload Page**: http://localhost:3000/upload
- **Dashboard**: http://localhost:3000/dashboard
- **Learn More**: http://localhost:3000/learn

---

## 🌐 Vercel Deployment with Full ML Models

### ✨ NEW: Deploy with Real ML Models!

Your app can now run with **full ML functionality** on Vercel using Hugging Face for model hosting.

**Quick Setup (3 steps)**:

1. **Upload model to Hugging Face**:
   ```bash
   pip install huggingface_hub
   huggingface-cli login
   python scripts/upload_to_huggingface.py
   ```

2. **Set environment variable in Vercel**:
   - Dashboard → Settings → Environment Variables
   - Add: `HUGGINGFACE_MODEL_ID` = `your-username/sentiment-distilbert`

3. **Deploy**:
   ```bash
   git push origin main
   ```

**📚 Complete Guide**: See `DEPLOY_WITH_MODELS.md` for detailed instructions.

### Alternative: Demo Mode Deployment

For a quick demo without ML models:

**Option 1: Deploy via Vercel Dashboard**
1. Push code to GitHub
2. Go to [vercel.com/new](https://vercel.com/new)
3. Import your repository
4. Set Root Directory: `web`
5. Click Deploy

**Option 2: Deploy via CLI**
```bash
# Install Vercel CLI
npm install -g vercel

# Deploy from project root
./deploy.sh
```

### What's Deployed

**With Hugging Face** (Recommended):
- ✅ Full ML model predictions (94% accuracy)
- ✅ Aspect-based sentiment analysis
- ✅ Same performance as local version
- ✅ Free tier: 30,000 requests/month

**Demo Mode** (No setup):
- ✅ Full frontend UI and visualizations
- ✅ Lightweight rule-based sentiment analysis
- ✅ CSV upload interface
- ✅ Analytics dashboard with demo data

---


## 📁 Project Structure

```
DNN/
├── api/predict/
│   ├── route.py              # Prediction logic (5 modes)
│   ├── local_server.py       # HTTP server (batch upload, dashboard)
│   └── database.py           # SQLite utilities (5 tables)
├── web/pages/
│   ├── index.js              # Main prediction UI
│   ├── upload.js             # Batch CSV upload
│   ├── dashboard.js          # Analytics dashboard
│   └── learn.js              # Educational content
├── model_training/
│   ├── model_a/              # Text preprocessing
│   ├── model_b/              # RNN training
│   └── model_c/              # Transformer training
├── outputs/
│   ├── rnn_sentiment_model.pt        # Trained RNN (20MB)
│   ├── transformer/                  # DistilBERT (260MB)
│   ├── sentiment.db                  # SQLite database
│   └── figures/model_comparison.png
└── raw/                      # Training datasets
    ├── IMDB Dataset.csv
    └── Twitter.csv
```

---

## 📊 Performance

| Model | Accuracy | Speed | Memory |
|-------|----------|-------|--------|
| RNN | 87.56% | 15ms | 20MB |
| DistilBERT | 94.22% | 50ms | 260MB |
| Hybrid | ~92% | 24ms | 280MB |

---

## 📚 References

**Academic Papers**
- LSTM: Hochreiter & Schmidhuber (1997)
- Attention: Bahdanau et al. (2014)
- Transformers: Vaswani et al. (2017)
- DistilBERT: Sanh et al. (2019)

**Datasets**
- IMDB Movie Reviews: Maas et al. (2011)
- Twitter Sentiment140: Go et al. (2009)

**Frameworks**
- PyTorch, Hugging Face Transformers, Next.js

---

**CST-435 Deep Neural Networks Project**  
Grand Canyon University | December 2025
