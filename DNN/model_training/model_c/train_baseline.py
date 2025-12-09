# Training script for TF-IDF Baseline Model
# Loads data, trains model, evaluates, and saves results
#
# Usage:
#   python train_baseline.py
#
# This script demonstrates:
#   1. Loading sentiment data from CSV/JSONL files
#   2. Training TF-IDF + Logistic Regression baseline
#   3. Evaluating on test set
#   4. Saving confusion matrix and metrics

import pandas as pd
import numpy as np
import json
import os
import sys
from sklearn.model_selection import train_test_split
from baseline_tfidf import TFIDFBaseline, plot_confusion_matrix, save_metrics

# Add parent directory to path for importing model_a modules
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from model_a.cleaning import preprocess_for_sentiment


def load_imdb_data(filepath: str, sample_size: int = None):
    """
    Load IMDB dataset from CSV file.
    
    IMDB dataset contains movie reviews with binary sentiment labels.
    Each review is labeled as either 'positive' or 'negative'.
    This is a classic benchmark dataset for sentiment analysis.
    
    Args:
        filepath: Path to IMDB Dataset.csv
        sample_size: Optional - limit number of samples (useful for quick testing)
        
    Returns:
        texts: List of review texts
        labels: List of binary labels (0=negative, 1=positive)
    """
    print(f"Loading IMDB data from {filepath}...")
    df = pd.read_csv(filepath)
    
    # Assuming columns are 'review' and 'sentiment'
    # Adjust column names based on your actual CSV structure
    if 'review' in df.columns and 'sentiment' in df.columns:
        texts = df['review'].astype(str).tolist()
        # Convert sentiment to binary: positive=1, negative=0
        labels = (df['sentiment'] == 'positive').astype(int).tolist()
    else:
        # Try alternative column names
        text_col = df.columns[0]  # First column is text
        label_col = df.columns[1]  # Second column is label
        texts = df[text_col].astype(str).tolist()
        labels = df[label_col].tolist()
        
        # Convert labels to 0/1 if they're strings
        if isinstance(labels[0], str):
            unique_labels = list(set(labels))
            label_map = {unique_labels[0]: 0, unique_labels[1]: 1}
            labels = [label_map[l] for l in labels]
    
    if sample_size:
        texts = texts[:sample_size]
        labels = labels[:sample_size]
    
    print(f"Loaded {len(texts)} samples")
    print(f"Label distribution: {np.bincount(labels)}")
    
    return texts, labels


def load_amazon_data(filepath: str, sample_size: int = None):
    """
    Load Amazon reviews from JSONL file.
    
    Amazon dataset contains product reviews with star ratings (1-5).
    We convert star ratings to binary sentiment:
        - 1-2 stars → Negative (0)
        - 4-5 stars → Positive (1)
        - 3 stars → Skipped (neutral, harder to classify)
    
    This gives us clearer positive/negative examples for training.
    
    Args:
        filepath: Path to Amazon_Health_and_Personal_Care.jsonl
        sample_size: Optional - limit number of samples
        
    Returns:
        texts: List of review texts
        labels: List of binary labels (0=negative, 1=positive)
    """
    print(f"Loading Amazon data from {filepath}...")
    texts = []
    labels = []
    
    with open(filepath, 'r') as f:
        for i, line in enumerate(f):
            if sample_size and i >= sample_size:
                break
            
            data = json.loads(line)
            # Extract review text and rating
            if 'reviewText' in data and 'overall' in data:
                text = data['reviewText']
                rating = data['overall']  # 1-5 star rating
                
                # Convert 5-star rating to binary sentiment
                # 1-2 stars = negative (0), 4-5 stars = positive (1)
                # Skip neutral (3 stars) - ambiguous sentiment makes training harder
                if rating <= 2:
                    texts.append(text)
                    labels.append(0)
                elif rating >= 4:
                    texts.append(text)
                    labels.append(1)
    
    print(f"Loaded {len(texts)} samples")
    print(f"Label distribution: {np.bincount(labels)}")
    
    return texts, labels


def load_twitter_data(filepath: str, sample_size: int = None):
    """
    Load Twitter/Sentiment140 data from CSV file.
    
    Args:
        filepath: Path to Twitter.csv
        sample_size: Optional - limit number of samples
        
    Returns:
        texts, labels
    """
    print(f"Loading Twitter data from {filepath}...")
    
    # Sentiment140 format: sentiment, id, date, query, user, text
    # Sentiment: 0=negative, 4=positive
    df = pd.read_csv(filepath, encoding='latin-1', header=None)
    
    # Assuming standard Sentiment140 format
    texts = df.iloc[:, -1].astype(str).tolist()  # Last column is text
    labels = df.iloc[:, 0].tolist()  # First column is sentiment
    
    # Convert 4 to 1 (positive)
    labels = [1 if l == 4 else 0 for l in labels]
    
    if sample_size:
        texts = texts[:sample_size]
        labels = labels[:sample_size]
    
    print(f"Loaded {len(texts)} samples")
    print(f"Label distribution: {np.bincount(labels)}")
    
    return texts, labels


def main():
    """
    Main training pipeline for baseline model.
    """
    print("="*70)
    print("TF-IDF BASELINE MODEL TRAINING")
    print("="*70)
    
    # Configuration
    DATA_DIR = "../data/raw"
    OUTPUT_DIR = "../../outputs"
    
    # Choose dataset to use
    dataset = 'imdb'  # Change to 'amazon' or 'twitter' as needed
    
    # Load data based on dataset choice
    if dataset == 'imdb':
        data_path = os.path.join(DATA_DIR, 'IMDB Dataset.csv')
        texts, labels = load_imdb_data(data_path, sample_size=10000)  # Use 10k samples for faster training
        class_names = ['Negative', 'Positive']
    elif dataset == 'amazon':
        data_path = os.path.join(DATA_DIR, 'Amazon_Health_and_Personal_Care.jsonl')
        texts, labels = load_amazon_data(data_path, sample_size=10000)
        class_names = ['Negative', 'Positive']
    elif dataset == 'twitter':
        data_path = os.path.join(DATA_DIR, 'Twitter.csv')
        texts, labels = load_twitter_data(data_path, sample_size=10000)
        class_names = ['Negative', 'Positive']
    else:
        raise ValueError(f"Unknown dataset: {dataset}")
    
    # Preprocess texts using Model A cleaning
    # This removes URLs, HTML tags, normalizes text
    # Preprocessing is crucial - clean data = better results
    print("\nPreprocessing texts...")
    cleaned_texts = [preprocess_for_sentiment(text) for text in texts]
    
    # Split into train and test sets (80/20 split)
    # Train set: Used to learn patterns and weights
    # Test set: Used to evaluate how well model generalizes to unseen data
    # IMPORTANT: We never let the model see test data during training!
    print("\nSplitting data into train/test sets...")
    X_train, X_test, y_train, y_test = train_test_split(
        cleaned_texts, labels, 
        test_size=0.2,  # 20% for testing, 80% for training
        random_state=42,  # Fixed seed for reproducibility
        stratify=labels  # Maintain same class distribution in train/test
                        # If 60% positive overall → 60% positive in both train and test
    )
    
    print(f"Train size: {len(X_train)}")
    print(f"Test size: {len(X_test)}")
    
    # Train baseline model
    print("\n" + "="*70)
    baseline = TFIDFBaseline(max_features=10000, classifier_type='logistic')
    baseline.train(X_train, y_train)
    
    # Evaluate on test set
    print("\n" + "="*70)
    print("EVALUATING ON TEST SET")
    print("="*70)
    metrics = baseline.evaluate(X_test, y_test, label_names=class_names)
    
    # Create output directories
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_DIR, 'figures'), exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_DIR, 'checkpoints'), exist_ok=True)
    
    # Save confusion matrix plot
    # Visual representation helps us understand where the model makes mistakes
    # Example: If many false negatives → model struggles with negative reviews
    cm_path = os.path.join(OUTPUT_DIR, 'figures', f'confusion_matrix_baseline_{dataset}.png')
    plot_confusion_matrix(metrics['confusion_matrix'], class_names, save_path=cm_path)
    
    # Save metrics to JSON
    # Allows us to compare this baseline with RNN and DistilBERT later
    # Metrics include: accuracy, precision, recall, F1-score
    metrics_path = os.path.join(OUTPUT_DIR, f'baseline_metrics_{dataset}.json')
    save_metrics(metrics, metrics_path)
    
    # Save trained model
    # Can load later to make predictions without retraining
    # Useful for deployment or continuing experiments
    model_path = os.path.join(OUTPUT_DIR, 'checkpoints', f'baseline_tfidf_{dataset}.pkl')
    baseline.save_model(model_path)
    
    # Show top features
    # This reveals which words the model learned are most important
    # Helps us understand:
    #   1. Is the model learning sensible patterns?
    #   2. Are there biases in the data?
    #   3. What language patterns indicate sentiment?
    # 
    # Example positive features: "excellent", "amazing", "love", "perfect"
    # Example negative features: "terrible", "worst", "disappointing", "waste"
    print("\n" + "="*70)
    print("TOP PREDICTIVE FEATURES")
    print("="*70)
    top_features = baseline.get_top_features(n=15)
    
    print("\nMost Positive Features:")
    print("(High coefficients → strong indicators of positive sentiment)")
    for word, coef in top_features['positive_features']:
        print(f"  {word:20s} {coef:8.4f}")
    
    print("\nMost Negative Features:")
    print("(Low coefficients → strong indicators of negative sentiment)")
    for word, coef in top_features['negative_features']:
        print(f"  {word:20s} {coef:8.4f}")
    
    print("\n" + "="*70)
    print("TRAINING COMPLETE!")
    print("="*70)
    print(f"Model saved to: {model_path}")
    print(f"Metrics saved to: {metrics_path}")
    print(f"Confusion matrix saved to: {cm_path}")


if __name__ == "__main__":
    main()
