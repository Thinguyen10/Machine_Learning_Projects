# Model Files (Not Included in Git)

## 📁 Missing Files (Too Large for GitHub)

The following files are excluded from git due to size limits:

### Trained Models (280MB+):
- `rnn_sentiment_model.pt` (20MB)
- `rnn_checkpoints/best_model.pt` (19MB)
- `transformer/model.safetensors` (255MB)
- `transformer/checkpoint-*/` (1GB+)

### Training Data (500MB+):
- `model_training/data/raw/Twitter.csv` (228MB)
- `model_training/data/raw/IMDB Dataset.csv` (63MB)
- `model_training/data/raw/Amazon_Health_and_Personal_Care.jsonl` (216MB)

## 🔄 How to Get These Files

### Option 1: Train Models Yourself (Recommended)
```bash
# Navigate to project
cd "CST-435 JT/DNN"

# Activate virtual environment
source venv/bin/activate

# Download training data (instructions in model_training/data/raw/README.md)

# Train RNN model
cd model_training/model_b
python train.py

# Train DistilBERT model
cd ../model_c
python train_transformer.py
```

### Option 2: Download Pre-trained Models
**For Graders/Reviewers**: Contact the repository owner for access to pre-trained models.

The trained models can be shared via:
- Google Drive
- Dropbox
- OneDrive
- Direct file transfer

### Option 3: Use Demo Mode (Vercel Deployment)
The deployed version at Vercel uses lightweight rule-based sentiment analysis.

No model files needed for the demo deployment.

## 📊 Expected Model Performance

When you train or obtain the models, you should see:

- **RNN Model**: ~87.56% accuracy
- **DistilBERT Model**: ~94.22% accuracy
- **Hybrid Ensemble**: ~92% accuracy

## 🗂️ File Structure (After Training)

```
outputs/
├── rnn_sentiment_model.pt          # RNN model (20MB)
├── transformer/
│   ├── model.safetensors          # DistilBERT (255MB)
│   ├── config.json
│   ├── tokenizer_config.json
│   └── vocab.txt
├── rnn_checkpoints/
│   └── best_model.pt              # Best RNN checkpoint (19MB)
└── sentiment.db                    # SQLite database (auto-created)
```

## ⚠️ Important Notes

1. **GitHub Limits**: Files over 100MB cannot be pushed to GitHub
2. **Total Size**: All models combined = ~2GB
3. **Git LFS**: Not used to keep repository simple
4. **Submission**: Models excluded from git, documented in README

## 🎓 For Course Submission

**Instructor**: These model files are too large for GitHub submission.

**Alternatives**:
1. ✅ Live demo on Vercel (no models needed)
2. ✅ Source code in GitHub (training scripts included)
3. ✅ Local demo during office hours/presentation
4. ✅ Video recording of full system
5. ✅ Share models via cloud storage if requested

## 📝 Training Time Estimates

- RNN training: ~30 minutes (CPU) / ~10 minutes (GPU)
- DistilBERT fine-tuning: ~2 hours (CPU) / ~30 minutes (GPU)
- Total training time: ~2-3 hours (CPU) / ~40 minutes (GPU)

## 💾 Storage Requirements

- Training data: 500MB
- Trained models: 280MB
- Checkpoints: 1GB (can delete after training)
- Virtual environment: 1GB
- **Total**: ~2.7GB

## 🔗 Related Documentation

- Training guide: `model_training/README.md`
- Deployment guide: `VERCEL_DEPLOYMENT.md`
- Main README: `../README.md`

---

**Note**: This is a standard practice for ML projects. Model files are distributed separately from source code due to size constraints.
