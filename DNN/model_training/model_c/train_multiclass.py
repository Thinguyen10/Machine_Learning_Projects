"""
Train DistilBERT for 7-class sentiment analysis (-3 to +3 scale)

This creates a nuanced sentiment classifier:
  -3: Very Negative
  -2: Negative  
  -1: Slightly Negative
   0: Neutral
  +1: Slightly Positive
  +2: Positive
  +3: Very Positive

Converts existing binary datasets to 7-class by analyzing text features.
"""

import torch
import torch.nn as nn
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, AdamW, get_linear_schedule_with_warmup
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import json
import os

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Paths
DATA_PATH = '../../raw/IMDB Dataset.csv'
OUTPUT_DIR = '../../outputs/transformer_7class'
os.makedirs(OUTPUT_DIR, exist_ok=True)

def convert_to_7class(text, binary_label):
    """
    Convert binary label to 7-class scale using text analysis.
    
    Strategy:
    1. Start with binary: 0 (negative) or 1 (positive)
    2. Analyze intensity words to determine strength
    3. Map to -3 to +3 scale
    
    Intensity markers:
    - Very strong: "amazing", "terrible", "worst", "best" → ±3
    - Strong: "great", "bad", "awful", "excellent" → ±2
    - Moderate: "good", "poor", "nice", "disappointing" → ±1
    - Neutral indicators: "okay", "average", "decent" → 0
    """
    text_lower = text.lower()
    
    # Neutral indicators - override binary label
    neutral_keywords = ['average', 'okay', 'decent', 'fine', 'acceptable', 
                       'neither', 'mixed feelings', 'so-so', 'mediocre']
    if any(word in text_lower for word in neutral_keywords):
        return 0  # Neutral (class 3 in 7-class system)
    
    # Base class from binary
    base_class = int(binary_label)  # 0 or 1
    
    # Intensity analysis
    very_strong_pos = ['amazing', 'fantastic', 'incredible', 'outstanding', 
                       'perfect', 'masterpiece', 'brilliant', 'exceptional']
    very_strong_neg = ['terrible', 'horrible', 'awful', 'worst', 'disgusting',
                       'appalling', 'atrocious', 'dreadful']
    
    strong_pos = ['excellent', 'great', 'wonderful', 'superb', 'loved', 
                  'best', 'highly recommend']
    strong_neg = ['bad', 'poor', 'disappointing', 'waste', 'avoid', 'regret']
    
    moderate_pos = ['good', 'nice', 'enjoy', 'pleasant', 'liked', 'solid']
    moderate_neg = ['not good', 'below average', 'lacking', 'underwhelming']
    
    if base_class == 1:  # Positive
        if any(word in text_lower for word in very_strong_pos):
            return 3  # Very Positive
        elif any(word in text_lower for word in strong_pos):
            return 2  # Positive
        else:
            return 1  # Slightly Positive
    else:  # Negative
        if any(word in text_lower for word in very_strong_neg):
            return -3  # Very Negative
        elif any(word in text_lower for word in strong_neg):
            return -2  # Negative
        else:
            return -1  # Slightly Negative

def load_and_convert_data(file_path, sample_size=6000):
    """Load IMDB data and convert to 7-class scale."""
    print("Loading dataset...")
    df = pd.read_csv(file_path)
    
    # Sample if needed
    if len(df) > sample_size:
        df = df.sample(sample_size, random_state=42)
    
    print("Converting to 7-class labels...")
    # Binary: positive=1, negative=0
    binary_labels = (df['sentiment'] == 'positive').astype(int)
    
    # Convert to 7-class scale (-3 to +3)
    df['sentiment_7class'] = [
        convert_to_7class(text, label) 
        for text, label in tqdm(zip(df['review'], binary_labels), total=len(df))
    ]
    
    # Map to 0-6 for PyTorch (model outputs 7 classes)
    df['class_idx'] = df['sentiment_7class'] + 3  # -3→0, -2→1, ..., +3→6
    
    # Show distribution
    print("\n7-Class Distribution:")
    print(df['sentiment_7class'].value_counts().sort_index())
    
    return df

class SentimentDataset(torch.utils.data.Dataset):
    """Dataset for 7-class sentiment."""
    
    def __init__(self, texts, labels, tokenizer, max_length=256):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'label': torch.tensor(label, dtype=torch.long)
        }

def train_epoch(model, dataloader, optimizer, scheduler, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    predictions = []
    actual_labels = []
    
    progress_bar = tqdm(dataloader, desc='Training')
    for batch in progress_bar:
        optimizer.zero_grad()
        
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)
        
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        
        total_loss += loss.item()
        
        preds = torch.argmax(outputs.logits, dim=1)
        predictions.extend(preds.cpu().numpy())
        actual_labels.extend(labels.cpu().numpy())
        
        progress_bar.set_postfix({'loss': loss.item()})
    
    avg_loss = total_loss / len(dataloader)
    accuracy = accuracy_score(actual_labels, predictions)
    
    return avg_loss, accuracy

def evaluate(model, dataloader, device):
    """Evaluate model."""
    model.eval()
    total_loss = 0
    predictions = []
    actual_labels = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc='Evaluating'):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            
            total_loss += loss.item()
            
            preds = torch.argmax(outputs.logits, dim=1)
            predictions.extend(preds.cpu().numpy())
            actual_labels.extend(labels.cpu().numpy())
    
    avg_loss = total_loss / len(dataloader)
    accuracy = accuracy_score(actual_labels, predictions)
    
    return avg_loss, accuracy, predictions, actual_labels

def main():
    # Load and convert data
    df = load_and_convert_data(DATA_PATH)
    
    # Split data
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, 
                                         stratify=df['class_idx'])
    
    print(f"\nTrain size: {len(train_df)}, Test size: {len(test_df)}")
    
    # Initialize tokenizer and model
    print("\nInitializing model...")
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    model = DistilBertForSequenceClassification.from_pretrained(
        'distilbert-base-uncased',
        num_labels=7  # 7 classes: -3 to +3
    )
    model.to(device)
    
    # Create datasets
    train_dataset = SentimentDataset(
        train_df['review'].values,
        train_df['class_idx'].values,
        tokenizer
    )
    test_dataset = SentimentDataset(
        test_df['review'].values,
        test_df['class_idx'].values,
        tokenizer
    )
    
    # Create dataloaders
    batch_size = 16
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size)
    
    # Optimizer and scheduler
    epochs = 4
    optimizer = AdamW(model.parameters(), lr=2e-5)
    total_steps = len(train_loader) * epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0,
        num_training_steps=total_steps
    )
    
    # Training loop
    print("\nStarting training...")
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    
    for epoch in range(epochs):
        print(f"\nEpoch {epoch + 1}/{epochs}")
        
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, scheduler, device)
        val_loss, val_acc, val_preds, val_labels = evaluate(model, test_loader, device)
        
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
    
    # Final evaluation
    print("\nFinal Evaluation:")
    _, final_acc, final_preds, final_labels = evaluate(model, test_loader, device)
    print(f"Test Accuracy: {final_acc:.4f}")
    
    # Convert back to -3 to +3 scale for reporting
    final_preds_scale = np.array(final_preds) - 3
    final_labels_scale = np.array(final_labels) - 3
    
    # Classification report
    class_names = ['-3 (Very Neg)', '-2 (Neg)', '-1 (Slightly Neg)', 
                   '0 (Neutral)', '+1 (Slightly Pos)', '+2 (Pos)', '+3 (Very Pos)']
    print("\nClassification Report:")
    print(classification_report(final_labels_scale, final_preds_scale, 
                                target_names=class_names))
    
    # Confusion matrix
    cm = confusion_matrix(final_labels_scale, final_preds_scale)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title('7-Class Sentiment Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'confusion_matrix_7class.png'))
    print(f"Confusion matrix saved to {OUTPUT_DIR}/confusion_matrix_7class.png")
    
    # Save model
    print("\nSaving model...")
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    
    # Save metrics
    metrics = {
        'test_accuracy': float(final_acc),
        'history': history,
        'class_distribution': df['sentiment_7class'].value_counts().to_dict()
    }
    
    with open(os.path.join(OUTPUT_DIR, 'metrics_7class.json'), 'w') as f:
        json.dump(metrics, f, indent=2)
    
    print(f"\nModel saved to {OUTPUT_DIR}")
    print("\nTo use this model, update MODEL_ID in streamlit_app.py to load from this directory.")

if __name__ == '__main__':
    main()
