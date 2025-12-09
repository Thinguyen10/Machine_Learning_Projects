# Training script for DistilBERT
# Fine-tune pre-trained DistilBERT on sentiment data
#
# Usage:
#   python train_transformer.py
#
# This script:
#   1. Loads data from all 3 datasets
#   2. Creates DistilBERT model and tokenizer
#   3. Fine-tunes using HuggingFace Trainer
#   4. Evaluates and saves model to outputs/transformer/
#
# Fine-tuning takes advantage of pre-trained knowledge
# and adapts it to sentiment analysis task

import torch
import pandas as pd
import json
import os
import sys
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

from transformers import Trainer, TrainingArguments, EarlyStoppingCallback
from distilbert_finetune import create_distilbert_model, SentimentDataset, compute_metrics

# Add model_a to path for preprocessing
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from model_a.cleaning import preprocess_for_sentiment


def load_data(dataset='all', sample_size=None):
    """
    Load sentiment data from files.
    
    Args:
        dataset: Which dataset to use ('imdb', 'amazon', 'twitter', or 'all')
        sample_size: Number of samples per dataset (None = load all)
        
    Returns:
        texts, labels
    """
    DATA_DIR = "../data/raw"
    all_texts = []
    all_labels = []
    
    datasets_to_load = ['imdb', 'amazon', 'twitter'] if dataset == 'all' else [dataset]
    
    for ds_name in datasets_to_load:
        print(f"\nLoading {ds_name.upper()} dataset...")
        
        if ds_name == 'imdb':
            # Load IMDB data
            data_path = os.path.join(DATA_DIR, 'IMDB Dataset.csv')
            try:
                df = pd.read_csv(data_path)
                
                if 'review' in df.columns and 'sentiment' in df.columns:
                    texts = df['review'].astype(str).tolist()
                    labels = (df['sentiment'] == 'positive').astype(int).tolist()
                else:
                    texts = df.iloc[:, 0].astype(str).tolist()
                    labels = df.iloc[:, 1].tolist()
                    if isinstance(labels[0], str):
                        unique_labels = list(set(labels))
                        label_map = {unique_labels[0]: 0, unique_labels[1]: 1}
                        labels = [label_map[l] for l in labels]
                
                if sample_size:
                    texts = texts[:sample_size]
                    labels = labels[:sample_size]
                
                all_texts.extend(texts)
                all_labels.extend(labels)
                print(f"  Loaded {len(texts)} IMDB samples")
                
            except Exception as e:
                print(f"  Error loading IMDB: {e}")
                
        elif ds_name == 'amazon':
            # Load Amazon data
            data_path = os.path.join(DATA_DIR, 'Amazon_Health_and_Personal_Care.jsonl')
            texts = []
            labels = []
            
            try:
                with open(data_path, 'r') as f:
                    for i, line in enumerate(f):
                        if sample_size and i >= sample_size:
                            break
                        
                        data = json.loads(line)
                        if 'reviewText' in data and 'overall' in data:
                            text = data['reviewText']
                            rating = data['overall']
                            
                            # Convert to binary: 1-2 stars = negative, 4-5 stars = positive
                            if rating <= 2:
                                texts.append(text)
                                labels.append(0)
                            elif rating >= 4:
                                texts.append(text)
                                labels.append(1)
                
                all_texts.extend(texts)
                all_labels.extend(labels)
                print(f"  Loaded {len(texts)} Amazon samples")
                
            except Exception as e:
                print(f"  Error loading Amazon: {e}")
                
        elif ds_name == 'twitter':
            # Load Twitter data
            data_path = os.path.join(DATA_DIR, 'Twitter.csv')
            
            try:
                df = pd.read_csv(data_path, encoding='latin-1', header=None)
                
                texts = df.iloc[:, -1].astype(str).tolist()  # Last column is text
                labels = df.iloc[:, 0].tolist()  # First column is sentiment
                
                # Convert 4 to 1 (positive)
                labels = [1 if l == 4 else 0 for l in labels]
                
                if sample_size:
                    texts = texts[:sample_size]
                    labels = labels[:sample_size]
                
                all_texts.extend(texts)
                all_labels.extend(labels)
                print(f"  Loaded {len(texts)} Twitter samples")
                
            except Exception as e:
                print(f"  Error loading Twitter: {e}")
    
    print(f"\nTotal samples loaded: {len(all_texts)}")
    print(f"Label distribution: Negative={all_labels.count(0)}, Positive={all_labels.count(1)}")
    
    return all_texts, all_labels


def plot_confusion_matrix(cm, labels, save_path):
    """Plot and save confusion matrix."""
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=labels, yticklabels=labels)
    plt.title('Confusion Matrix - DistilBERT')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Confusion matrix saved to {save_path}")
    plt.close()


def save_metrics(metrics, filepath):
    """Save evaluation metrics to JSON file."""
    # Convert numpy arrays to lists for JSON serialization
    serializable_metrics = {}
    for key, value in metrics.items():
        if isinstance(value, np.ndarray):
            serializable_metrics[key] = value.tolist()
        elif isinstance(value, (list, tuple)):
            serializable_metrics[key] = [float(v) if isinstance(v, (np.floating, float)) else v for v in value]
        else:
            serializable_metrics[key] = float(value) if isinstance(value, (np.floating, float)) else value
    
    with open(filepath, 'w') as f:
        json.dump(serializable_metrics, f, indent=2)
    
    print(f"Metrics saved to {filepath}")


def main():
    """Main training pipeline for DistilBERT."""
    
    print("="*70)
    print("DISTILBERT FINE-TUNING - MODEL C (TRANSFORMER)")
    print("="*70)
    
    # Hyperparameters
    MAX_LENGTH = 256  # Maximum sequence length for DistilBERT
    BATCH_SIZE = 16  # Smaller batch size due to transformer memory requirements
    N_EPOCHS = 4  # Fewer epochs needed for fine-tuning (transformers learn fast)
    LEARNING_RATE = 2e-5  # Small learning rate typical for fine-tuning
    WARMUP_STEPS = 500  # Gradual learning rate warmup
    WEIGHT_DECAY = 0.01  # L2 regularization
    
    # Paths
    OUTPUT_DIR = "../../outputs/transformer"
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_DIR, '../figures'), exist_ok=True)
    
    # Check GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")
    
    # Load data
    print("\n" + "="*70)
    print("LOADING DATA")
    print("="*70)
    texts, labels = load_data(dataset='all', sample_size=3000)  # 3000 samples per dataset
    
    # Light preprocessing (transformers handle most preprocessing internally)
    print("\nPreprocessing texts...")
    cleaned_texts = [preprocess_for_sentiment(text) for text in tqdm(texts)]
    
    # Split data
    X_train, X_temp, y_train, y_temp = train_test_split(
        cleaned_texts, labels, test_size=0.3, random_state=42, stratify=labels
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
    )
    
    print(f"\nTrain size: {len(X_train)}")
    print(f"Validation size: {len(X_val)}")
    print(f"Test size: {len(X_test)}")
    
    # Create model and tokenizer
    print("\n" + "="*70)
    print("CREATING DISTILBERT MODEL")
    print("="*70)
    model, tokenizer = create_distilbert_model()
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Create datasets
    print("\nTokenizing datasets...")
    train_dataset = SentimentDataset(X_train, y_train, tokenizer, MAX_LENGTH)
    val_dataset = SentimentDataset(X_val, y_val, tokenizer, MAX_LENGTH)
    test_dataset = SentimentDataset(X_test, y_test, tokenizer, MAX_LENGTH)
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=N_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        warmup_steps=WARMUP_STEPS,
        weight_decay=WEIGHT_DECAY,
        learning_rate=LEARNING_RATE,
        logging_dir=os.path.join(OUTPUT_DIR, 'logs'),
        logging_steps=100,
        eval_strategy="epoch",  # Evaluate after each epoch
        save_strategy="epoch",  # Save checkpoint after each epoch
        load_best_model_at_end=True,  # Load best model at the end
        metric_for_best_model="accuracy",  # Use accuracy to determine best model
        greater_is_better=True,
        save_total_limit=2,  # Only keep 2 best checkpoints
        report_to="none",  # Don't report to wandb/tensorboard
        fp16=torch.cuda.is_available(),  # Use mixed precision on GPU
    )
    
    # Create Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
    )
    
    # Train
    print("\n" + "="*70)
    print("TRAINING")
    print("="*70)
    trainer.train()
    
    # Evaluate on test set
    print("\n" + "="*70)
    print("EVALUATING ON TEST SET")
    print("="*70)
    
    # Get predictions
    predictions = trainer.predict(test_dataset)
    pred_labels = np.argmax(predictions.predictions, axis=1)
    
    # Calculate metrics
    test_acc = accuracy_score(y_test, pred_labels)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test, pred_labels, average='weighted'
    )
    cm = confusion_matrix(y_test, pred_labels)
    
    print(f"\nTest Accuracy: {test_acc:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    
    # Save metrics
    metrics = {
        'test_accuracy': test_acc,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'confusion_matrix': cm,
        'hyperparameters': {
            'max_length': MAX_LENGTH,
            'batch_size': BATCH_SIZE,
            'learning_rate': LEARNING_RATE,
            'epochs': N_EPOCHS,
            'warmup_steps': WARMUP_STEPS,
            'weight_decay': WEIGHT_DECAY
        }
    }
    
    save_metrics(metrics, os.path.join(OUTPUT_DIR, '../distilbert_metrics.json'))
    
    # Plot confusion matrix
    plot_confusion_matrix(cm, ['Negative', 'Positive'],
                         os.path.join(OUTPUT_DIR, '../figures/confusion_matrix_distilbert.png'))
    
    # Save final model and tokenizer
    print("\n" + "="*70)
    print("SAVING MODEL")
    print("="*70)
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"Model and tokenizer saved to {OUTPUT_DIR}")
    
    print("\n" + "="*70)
    print("TRAINING COMPLETE!")
    print("="*70)
    print(f"Test accuracy: {test_acc:.4f}")
    print(f"\nModel ready for deployment:")
    print(f"  - Model and tokenizer: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
