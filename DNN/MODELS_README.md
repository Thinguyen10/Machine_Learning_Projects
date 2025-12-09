# 📦 Trained Models

The trained models are **NOT included** in this repository due to GitHub file size limits.

## Model Files (Total: ~1.8GB)

**RNN Model** (6.2MB) - ✅ Included
- `outputs/rnn_sentiment_model.pt`

**DistilBERT Transformer** (1.7GB) - ❌ Excluded
- `outputs/transformer/pytorch_model.bin` (255MB)
- `outputs/transformer/model.safetensors` (255MB)
- `outputs/transformer/checkpoint-*/` (1.0GB+)

## How to Get the Models

### Option 1: Download Pre-trained (Recommended)
Contact the project author for access to the trained models via cloud storage.

### Option 2: Train from Scratch (~1 hour)
```bash
# Activate environment
source venv/bin/activate

# Train RNN (~30 min)
cd model_training/model_b
python train.py

# Train DistilBERT (~29 min)
cd ../model_c
python train_transformer.py
```

## Configuration Files (Included)
The following config files ARE included and allow model loading:
- `outputs/transformer/config.json`
- `outputs/transformer/tokenizer_config.json`
- `outputs/transformer/vocab.txt`
- `outputs/rnn_checkpoints/tokenizer.pkl`
