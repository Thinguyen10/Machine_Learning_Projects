# 🧠 Sentiment Analysis with Deep Neural Networks

Production-ready sentiment analysis using transformer models with Streamlit dashboard for batch processing.

## 🎯 Quick Start

**Live Demo:** [Streamlit App](https://your-app.streamlit.app) - Upload CSV and analyze unlimited reviews

**Model:** DistilBERT (94.22% accuracy) - Hosted on HuggingFace: `Thi144/sentiment-distilbert`

**Dataset:** 6,000 IMDB movie reviews

---

## 🤖 ML/Neural Network Algorithms Used

### 1. **DistilBERT Transformer** (Primary Model - 94.22% accuracy)
**Task:** Binary sentiment classification (positive/negative)  
**Algorithm:** 6-layer transformer with multi-head attention (66M parameters)  
**How it works:** Pre-trained on Wikipedia, fine-tuned on 6,000 IMDB reviews using transfer learning  
**Deployment:** Loaded from HuggingFace Hub for predictions

### 2. **Softmax Probability Distribution**
**Task:** Convert model logits to probabilities  
**How it works:** Takes raw scores and normalizes to [0,1] range that sums to 1.0  
**Use:** Powers the confidence scores and probability displays

### 3. **Entropy-Based Neutral Detection**
**Task:** Identify neutral sentiment from binary model  
**Algorithm:** Information entropy calculation on prediction probabilities  
**How it works:** High entropy (uncertainty) = model can't decide = likely neutral  
**Formula:** `entropy = -Σ(p * log₂(p))` where p = probabilities

### 4. **TF-IDF (Term Frequency-Inverse Document Frequency)**
**Task:** Automatic aspect/topic discovery from reviews  
**Algorithm:** Statistical NLP technique to identify important words  
**How it works:** Words frequent in few documents but rare overall = important topics  
**Use:** Replaces manual keyword lists with learned topics

### 5. **Bidirectional LSTM with Attention** (Trained but not deployed)
**Task:** Sequence modeling for sentiment (87.56% accuracy)  
**Algorithm:** Recurrent neural network with attention mechanism  
**Architecture:** 128 hidden units, 100-dim embeddings  
**Note:** Available in `outputs/rnn_sentiment_model.pt`

---

## 📊 Model Performance

### Model A: Preprocessing Pipeline
- Text cleaning, tokenization, embedding preparation
- GloVe embeddings (100-dim)
- Handles IMDB format and custom CSVs

### Model B: RNN with Attention (87.56% accuracy)
- Bidirectional LSTM (128 hidden units)
- Attention mechanism for interpretability
- Training: 20 epochs, Adam optimizer

### Model C: DistilBERT Transformer (94.22% accuracy) ⭐
- 6-layer transformer (66M parameters)
- Fine-tuned on sentiment data
- Training: 4 epochs, AdamW optimizer
- **Best model - deployed in production**

---

## 🚀 Deployment

### Streamlit Cloud (Primary)
**Live App:** Processes unlimited reviews with real-time progress

**Features:**
- Upload any CSV file
- Analyze all 120+ reviews (no limits)
- Interactive charts and dashboard
- Download results with sentiment scores

**Setup:**
1. Go to https://share.streamlit.io
2. Deploy `DNN/streamlit_app.py` from this repo
3. Auto-loads model from HuggingFace

### Vercel (Alternative - API only)

- Serverless API at `/api/predict`
- 10-second timeout (processes ~20 reviews max)
- Environment: `HUGGINGFACE_MODEL_ID`, `HUGGINGFACE_TOKEN`

---

## 💻 Local Development

**Run Streamlit app locally:**

```bash
# Install dependencies
pip install -r requirements.txt

# Run app
streamlit run streamlit_app.py
```

**Or run Next.js version:**

```bash
# Backend (Terminal 1)
cd api/predict
python local_server.py

# Frontend (Terminal 2)
cd web
npm install && npm run dev
```

**Access:** http://localhost:3000

---

## 📁 Project Structure

```
DNN/
├── streamlit_app.py          # Main Streamlit app ⭐
├── model_training/
│   ├── model_a/              # Preprocessing
│   ├── model_b/              # RNN training
│   └── model_c/              # DistilBERT training
├── outputs/
│   ├── rnn_sentiment_model.pt
│   └── transformer/          # DistilBERT (uploaded to HuggingFace)
├── web/                      # Next.js frontend
└── api/predict/              # Python backend
```

---

## 📊 Results

| Model | Accuracy | Algorithm Type | Use Case |
|-------|----------|----------------|----------|
| RNN + Attention | 87.56% | Deep Learning (LSTM) | Trained but not deployed |
| DistilBERT | 94.22% | Deep Learning (Transformer) | **Production deployment** |
| TF-IDF | N/A | Traditional ML | Aspect extraction |
| Entropy | N/A | Statistical ML | Neutral detection |

**Training Dataset:** 6,000 IMDB movie reviews

---

## 🔬 Technical Stack

**Deep Learning:**
- PyTorch (neural network framework)
- Transformers / HuggingFace (pre-trained models)
- Transfer learning (fine-tuning)

**Traditional ML:**
- Scikit-learn (TF-IDF, vectorization)
- Information theory (entropy)

**Frontend:**
- Streamlit (primary dashboard)
- Next.js (alternative API interface)

**Deployment:**
- Streamlit Cloud (free tier)
- Vercel (serverless API)
- **Model Hosting:** HuggingFace Hub

**Frameworks**
- PyTorch, Hugging Face Transformers, Next.js

---

**CST-435 Deep Neural Networks Project**  
Grand Canyon University | December 2025
