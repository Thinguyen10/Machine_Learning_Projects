---
language: en
license: apache-2.0
tags:
- sentiment-analysis
- text-classification
- distilbert
- pytorch
- transformers
datasets:
- imdb
metrics:
- accuracy
- f1
widget:
- text: "This movie was absolutely amazing! Best film I've seen all year!"
  example_title: "Very Positive"
- text: "Pretty good movie, enjoyed it overall."
  example_title: "Slightly Positive"
- text: "It was okay, nothing special but not bad either."
  example_title: "Neutral"
- text: "Not a great movie, pretty disappointing."
  example_title: "Slightly Negative"
- text: "Terrible film, complete waste of time and money!"
  example_title: "Very Negative"
---

# DistilBERT 7-Class Sentiment Analysis Model

A fine-tuned DistilBERT model for nuanced sentiment analysis with 7 sentiment classes on a scale from -3 (Very Negative) to +3 (Very Positive).

## Model Description

This model performs fine-grained sentiment classification, providing more nuanced predictions than traditional binary positive/negative models. It's particularly useful for business applications where understanding the intensity of sentiment matters (e.g., identifying "at-risk" customers vs. extremely dissatisfied ones).

**Architecture:** DistilBERT (distilbert-base-uncased)  
**Parameters:** 66 million  
**Training Data:** 6,000 IMDB movie reviews  
**Accuracy:** 73.7%

## Sentiment Classes

| Class | Scale | Label | Description |
|-------|-------|-------|-------------|
| 0 | -3 | Very Negative | Extremely dissatisfied, angry |
| 1 | -2 | Negative | Clearly unhappy, disappointed |
| 2 | -1 | Slightly Negative | Somewhat disappointed |
| 3 | 0 | Neutral | Balanced, neither positive nor negative |
| 4 | +1 | Slightly Positive | Somewhat satisfied |
| 5 | +2 | Positive | Clearly satisfied, happy |
| 6 | +3 | Very Positive | Extremely satisfied, delighted |

## Usage

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# Load model and tokenizer
model_id = "Thi144/sentiment-distilbert-7class"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForSequenceClassification.from_pretrained(model_id)

# Class mapping
CLASS_LABELS = {
    0: {"scale": -3, "label": "negative", "name": "Very Negative"},
    1: {"scale": -2, "label": "negative", "name": "Negative"},
    2: {"scale": -1, "label": "negative", "name": "Slightly Negative"},
    3: {"scale": 0, "label": "neutral", "name": "Neutral"},
    4: {"scale": 1, "label": "positive", "name": "Slightly Positive"},
    5: {"scale": 2, "label": "positive", "name": "Positive"},
    6: {"scale": 3, "label": "positive", "name": "Very Positive"}
}

# Predict sentiment
def predict_sentiment(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    
    with torch.no_grad():
        outputs = model(**inputs)
        predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
        class_id = predictions.argmax().item()
        confidence = predictions[0][class_id].item()
    
    result = CLASS_LABELS[class_id]
    return {
        "class": class_id,
        "scale": result["scale"],
        "label": result["label"],
        "name": result["name"],
        "confidence": confidence
    }

# Example
result = predict_sentiment("This movie was absolutely amazing!")
print(f"Sentiment: {result['name']} (Scale: {result['scale']}, Confidence: {result['confidence']:.2%})")
```

## Performance Metrics

**Overall Accuracy:** 73.7%

**Class-Specific Performance:**
- **Very Negative (-3):** 81% precision, 88% recall
- **Negative (-2):** 83% precision, 77% recall  
- **Slightly Negative (-1):** 54% precision, 58% recall
- **Neutral (0):** 86% precision, 64% recall
- **Slightly Positive (+1):** 58% precision, 54% recall
- **Positive (+2):** 79% precision, 83% recall
- **Very Positive (+3):** 88% precision, 81% recall

The model performs best at identifying strong sentiments (Very Negative/Positive) and struggles most with subtle distinctions (Slightly Negative/Positive).

## Training Details

- **Base Model:** distilbert-base-uncased
- **Dataset:** 6,000 IMDB reviews (4,800 train, 1,200 test)
- **Label Conversion:** Binary labels converted to 7-class using text intensity analysis
- **Epochs:** 4
- **Batch Size:** 16
- **Optimizer:** AdamW (lr=2e-5)
- **Training Time:** ~15-20 minutes on CPU

## Limitations

- Trained on movie reviews, may not generalize perfectly to other domains
- Slightly Negative/Positive classes have lower accuracy (~54-58%)
- Performance depends on text clarity and length
- May struggle with sarcasm or complex sentiment

## Intended Use

**Primary Use Cases:**
- Customer feedback analysis with nuanced sentiment scoring
- Product review sentiment classification
- Social media monitoring with intensity detection
- Business intelligence dashboards requiring granular sentiment

**Not Recommended For:**
- Safety-critical applications
- Legal decision-making
- Medical diagnosis

## License

Apache 2.0

## Citation

If you use this model, please cite:

```
@model{thi144-sentiment-distilbert-7class,
  author = {Thi144},
  title = {DistilBERT 7-Class Sentiment Analysis},
  year = {2025},
  publisher = {HuggingFace},
  url = {https://huggingface.co/Thi144/sentiment-distilbert-7class}
}
```
