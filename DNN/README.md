# 🧠 Sentiment Analysis with Deep Neural Networks

Production-ready 7-class sentiment analysis using DistilBERT with Streamlit dashboard for unlimited batch processing.

## 🎯 Quick Start

**Live Demo:** Deploy to Streamlit Cloud - Upload CSV and analyze unlimited reviews with nuanced sentiment scoring

**Model:** DistilBERT 7-Class (73.7% accuracy) - Hosted on HuggingFace: `Thi144/sentiment-distilbert-7class`

**Scale:** -3 (Very Negative) to +3 (Very Positive) with true neutral detection

**Dataset:** 6,000 IMDB movie reviews

---

## 📝 What Was Implemented

This project implements a complete sentiment analysis pipeline from data preprocessing to production deployment:

**Core ML Implementation:**
- Fine-tuned DistilBERT transformer for 7-class sentiment (-3 to +3 scale)
- Multi-class classification with automatic label conversion from binary IMDB dataset
- Domain adaptation layer for business/professional reviews with sentiment lexicon
- Noun phrase extraction for automatic topic discovery (no manual keywords)
- Softmax probability distribution for confidence scoring

**Production Deployment:**
- Streamlit dashboard with unlimited CSV batch processing
- Model hosted on HuggingFace Hub for cloud access
- Interactive visualizations with Plotly charts
- Real-time progress tracking and downloadable results
- Cross-domain support (movie reviews, business reviews, insurance, etc.)

**Training Infrastructure:**
- PyTorch-based training pipeline with gradient clipping
- AdamW optimizer with learning rate scheduling
- Automatic metric logging (accuracy, precision, recall, F1)
- Confusion matrix generation for performance analysis

**Data Management:**
- Training data archived (507MB → 173MB compressed)
- Model versioning on HuggingFace Hub
- Config files preserved for reproducibility

---

## 🤖 ML/Neural Network Algorithms

### 1. **DistilBERT Transformer 7-Class** (Primary Model - 73.7% accuracy)

**Task:** Multi-class sentiment classification (-3 to +3 scale)  
**Algorithm:** 6-layer transformer with multi-head attention (66M parameters)  
**How it works:** Pre-trained on Wikipedia, fine-tuned on 6,000 IMDB reviews with 7-class labels  
**Deployment:** Loaded from HuggingFace Hub: `Thi144/sentiment-distilbert-7class`  
**Classes:** Very Negative (-3), Negative (-2), Slightly Negative (-1), Neutral (0), Slightly Positive (+1), Positive (+2), Very Positive (+3)

### 2. **Domain Adaptation Layer**

**Task:** Adjust model predictions for business/professional reviews  
**How it works:** Keyword detection for domain-specific sentiment indicators (e.g., "helpful", "frustrating")  
**Use:** Corrects neutral predictions when strong business sentiment signals are present  
**Algorithm:** Rule-based post-processing with sentiment lexicon matching

### 3. **Noun Phrase Extraction**

**Task:** Automatic aspect/topic discovery from reviews  
**Algorithm:** N-gram extraction with stopword filtering  
**How it works:** Extracts meaningful 1-2 word noun phrases (e.g., "customer service", "pricing plan")  
**Use:** Identifies business topics without manual keyword lists

### 4. **Softmax Probability Distribution**

**Task:** Convert model logits to probabilities  
**How it works:** Takes raw scores and normalizes to [0,1] range that sums to 1.0  
**Use:** Powers the confidence scores and probability displays

---

## 📊 Model Performance

**Production Model:** DistilBERT 7-Class Sentiment (73.7% accuracy)

**Class-Specific Performance:**
- Very Negative (-3): 81% precision, 88% recall
- Negative (-2): 83% precision, 77% recall
- Slightly Negative (-1): 54% precision, 58% recall
- Neutral (0): 86% precision, 64% recall
- Slightly Positive (+1): 58% precision, 54% recall
- Positive (+2): 79% precision, 83% recall
- Very Positive (+3): 88% precision, 81% recall

**Key Insights:**
- Best at detecting strong sentiments (Very Negative/Positive)
- Struggles with subtle distinctions (Slightly Negative/Positive)
- Excellent neutral detection (86% precision)

---

## 🚀 Deployment

### Streamlit Cloud

**Live App:** Processes unlimited reviews with real-time progress and nuanced sentiment scoring

**Features:**
- Upload any CSV file with review text
- Analyze unlimited reviews (no size limits)
- 7-class sentiment scale (-3 to +3)
- ML-based aspect extraction
- Interactive charts and dashboard
- Download results with sentiment scores and confidence

**Setup:**
1. Deploy to Streamlit Cloud: https://share.streamlit.io
2. Connect to this repo: `Thinguyen10/CST-435`
3. Set main file path: `DNN/streamlit_app.py`
4. Auto-loads model from HuggingFace Hub

---

## 💻 Local Development

**Run Streamlit app locally:**

```bash
# Install dependencies
pip install -r requirements.txt

# Run app
streamlit run streamlit_app.py
```

Optional: use an isolated virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate   # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run streamlit_app.py
```

Notes:
- Default main file is `streamlit_app.py`; if you have `streamlit_dashboard.py`, replace the filename in the command.
- The app loads the model from the HuggingFace Hub when online; use a token or local model files for offline runs.

**Access:** http://localhost:8501

---

## 📁 Project Structure

```
DNN/
├── streamlit_app.py              # Main Streamlit dashboard ⭐
├── requirements.txt              # Python dependencies
├── data_archive.tar.gz           # Training data (507MB → 173MB compressed)
├── model_training/
│   └── model_c/
│       ├── train_multiclass.py   # 7-class model training script
│       └── upload_7class_model.py
├── outputs/
│   └── transformer_7class/       # Model config & metrics (model files on HuggingFace)
│       ├── config.json
│       ├── vocab.txt
│       ├── tokenizer_config.json
│       ├── metrics_7class.json
│       ├── confusion_matrix_7class.png
│       └── README.md             # Model card
└── report/
    └── final_project.md
```

---

## 🔄 Retraining the Model

To retrain or improve the model with the archived data:

```bash
# 1. Extract training data
tar -xzf data_archive.tar.gz

# 2. Create virtual environment
python -m venv venv_training
source venv_training/bin/activate  # On Windows: venv_training\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run training script
python model_training/model_c/train_multiclass.py

# 5. Upload to HuggingFace (requires token)
python model_training/model_c/upload_7class_model.py
```

**Training Data Contents:**
- IMDB Dataset.csv (63MB): 50,000 movie reviews
- Twitter.csv (228MB): Twitter sentiment data
- Amazon_Health_and_Personal_Care.jsonl (216MB): Product reviews

---

## 📊 Results

**Production Model Performance:**

| Metric | Score |
|--------|-------|
| Overall Accuracy | 73.7% |
| Very Negative/Positive | 81-88% |
| Neutral Detection | 86% precision |
| Training Dataset | 6,000 IMDB reviews |
| Training Time | ~15-20 min (CPU) |

**Algorithm Comparison:**

| Algorithm | Type | Use Case |
|-----------|------|----------|
| DistilBERT 7-Class | Deep Learning (Transformer) | Primary sentiment classification |
| Domain Adaptation | Rule-based ML | Business review sentiment adjustment |
| Noun Phrase Extraction | NLP/ML | Automatic topic discovery |
| Softmax | Neural Network | Probability distribution |

---

## 🔬 Technical Stack

**Deep Learning:**
- PyTorch 2.0+ (neural network framework)
- Transformers/HuggingFace (pre-trained models)
- Transfer learning (fine-tuning DistilBERT)

**Traditional ML:**
- Scikit-learn (TF-IDF, vectorization)
- NumPy/Pandas (data processing)

**Frontend:**
- Streamlit (interactive dashboard)
- Plotly (data visualization)

**Deployment:**
- Streamlit Cloud (free tier, unlimited processing)
- HuggingFace Hub (model hosting)

---

**CST-435 Deep Neural Networks Project**  
**Model:** https://huggingface.co/Thi144/sentiment-distilbert-7class  
Grand Canyon University | December 2025
