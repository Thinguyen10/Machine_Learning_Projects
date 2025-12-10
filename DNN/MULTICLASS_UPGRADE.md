# 🎯 Upgrading to 7-Class Sentiment Analysis

## Problem with Current Model

Your DistilBERT model was trained on **binary classification** (positive/negative only):
- Outputs: `[negative_score, positive_score]`
- Neutral is **inferred** from uncertainty (entropy-based workaround)
- Not ideal for real business use cases

## Solution: 7-Class Scale (-3 to +3)

### Class Mapping
```
-3: Very Negative     (Extremely dissatisfied)
-2: Negative          (Dissatisfied)
-1: Slightly Negative (Somewhat dissatisfied)
 0: Neutral           (Neither satisfied nor dissatisfied)
+1: Slightly Positive (Somewhat satisfied)
+2: Positive          (Satisfied)
+3: Very Positive     (Extremely satisfied)
```

### Benefits
✅ **Nuanced sentiment** - Captures strength of opinion  
✅ **True neutral** - Model learns what neutral looks like  
✅ **Business insights** - Identify "at-risk" customers (-1) vs truly angry (-3)  
✅ **No dataset needed** - Converts existing binary data automatically  

---

## Training the 7-Class Model

### Step 1: Run Training Script

```bash
cd model_training/model_c
python train_multiclass.py
```

This will:
1. Load IMDB dataset (6,000 reviews)
2. Auto-convert binary labels → 7-class scale using text analysis
3. Train DistilBERT for 4 epochs (~15-20 min on CPU)
4. Save model to `outputs/transformer_7class/`

### Step 2: How Label Conversion Works

The script analyzes text intensity:

**Very Strong Words** → ±3:
- Positive: "amazing", "perfect", "masterpiece"
- Negative: "terrible", "horrible", "worst"

**Strong Words** → ±2:
- Positive: "excellent", "great", "wonderful"  
- Negative: "bad", "awful", "disappointing"

**Moderate Words** → ±1:
- Positive: "good", "nice", "enjoyable"
- Negative: "poor", "lacking", "underwhelming"

**Neutral Words** → 0:
- "average", "okay", "decent", "mixed"

### Step 3: Update Streamlit App

After training, update `streamlit_app.py`:

```python
# Change this line (around line 150):
MODEL_ID = "Thi144/sentiment-distilbert"  # OLD

# To:
MODEL_ID = "/path/to/outputs/transformer_7class"  # NEW - local path
```

Or upload to HuggingFace:
```bash
pip install huggingface_hub
huggingface-cli login
python upload_7class_model.py  # (create this script)
```

---

## Using the 7-Class Model

### In Streamlit App

The model will output 7 probabilities. Map to display:

```python
# Prediction gives class 0-6
prediction_idx = model_output  # 0 to 6
sentiment_scale = prediction_idx - 3  # Convert to -3 to +3

# Map to labels for display
if sentiment_scale <= -2:
    display_label = "negative"
elif sentiment_scale >= 2:
    display_label = "positive"
else:
    display_label = "neutral"

# But also show the scale
display_detail = f"{display_label} ({sentiment_scale:+d}/3)"
```

### Example Outputs

```
Input: "This was the worst movie I've ever seen!"
Output: -3 (Very Negative) → Display: "😞 Negative (-3/3)"

Input: "The website could use improvements but it works"  
Output: 0 (Neutral) → Display: "😐 Neutral (0/3)"

Input: "Absolutely amazing product, highly recommend!"
Output: +3 (Very Positive) → Display: "😊 Positive (+3/3)"
```

---

## Business Applications

### Customer Satisfaction Analysis

```
-3 to -2: Critical issues - immediate intervention needed
-1: At-risk customers - follow up to prevent churn  
 0: Neutral feedback - opportunity for improvement
+1: Satisfied - maintain quality
+2 to +3: Promoters - leverage for testimonials
```

### Dashboard Improvements

Instead of just:
```
Positive: 60%
Neutral: 20%
Negative: 20%
```

Show:
```
Very Positive (+3): 30%
Positive (+2): 30%
Slightly Positive (+1): 10%
Neutral (0): 20%
Slightly Negative (-1): 5%
Negative (-2): 3%
Very Negative (-3): 2%
```

---

## Alternative: Use Existing Multi-Class Datasets

If you prefer real multi-class data instead of conversion:

### Option 1: SST-5 (Stanford Sentiment Treebank)
- 5 classes: Very Negative, Negative, Neutral, Positive, Very Positive
- 11,855 movie reviews
- Download: https://nlp.stanford.edu/sentiment/

### Option 2: Yelp-5 
- 5-star ratings (1-5 stars)
- Map: 1→-2, 2→-1, 3→0, 4→+1, 5→+2
- Large dataset: 8M reviews

### Option 3: Amazon Reviews
- 5-star ratings
- Same mapping as Yelp

---

## Expected Results

### Accuracy by Class
```
-3 (Very Neg):    ~88% (strong signals)
-2 (Neg):         ~82%
-1 (Slightly Neg): ~75% (harder to distinguish)
 0 (Neutral):     ~78% (will improve!)
+1 (Slightly Pos): ~76%
+2 (Pos):         ~83%
+3 (Very Pos):    ~89% (strong signals)

Overall Accuracy: ~82-85%
```

### Why Not 94%?
- More classes = harder task
- But you get much better insights!
- Trade accuracy for nuance

---

## Next Steps

1. ✅ Run `train_multiclass.py` to create 7-class model
2. ✅ Test locally with single reviews
3. ✅ Update Streamlit app to use new model
4. ✅ Optionally upload to HuggingFace
5. ✅ Deploy to Streamlit Cloud

**Pro Tip:** Keep both models:
- Use 7-class for detailed analysis
- Use binary (94% accuracy) for simple positive/negative when you need high accuracy
