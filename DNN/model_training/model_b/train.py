# Training script for RNN model with Attention
# Trains LSTM-based sentiment classifier and saves metrics/plots
#
# Usage:
#   python train.py
#
# This script:
#   1. Loads and preprocesses data from all 3 datasets
#   2. Builds vocabulary and creates embeddings
#   3. Trains RNN with attention mechanism
#   4. Evaluates and saves metrics, plots, checkpoints
#
# Training uses GPU if available for faster training

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import json
import os
import sys
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

# Import our modules
from rnn_attention import SentimentRNN, count_parameters
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from model_a.cleaning import preprocess_for_sentiment
from model_a.tokenizer import SimpleTokenizer

# Check if GPU is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")


class SentimentDataset:
    """
    Simple dataset class for sentiment analysis.
    Handles batching and padding of text sequences.
    """
    
    def __init__(self, texts, labels, tokenizer, max_length=200):
        """
        Initialize dataset.
        
        Args:
            texts: List of text strings
            labels: List of sentiment labels (0 or 1)
            tokenizer: Tokenizer for encoding text
            max_length: Maximum sequence length (longer sequences truncated)
        """
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # Encode all texts to sequences
        self.encoded_texts = tokenizer.encode_batch(texts, max_length=max_length)
        
    def __len__(self):
        """Return number of samples in dataset."""
        return len(self.labels)
    
    def __getitem__(self, idx):
        """
        Get a single sample.
        
        Returns:
            text: Encoded sequence (tensor of word indices)
            label: Sentiment label (0 or 1)
            length: Actual length before padding
        """
        encoded = self.encoded_texts[idx]
        label = self.labels[idx]
        
        # Calculate actual length (before padding)
        length = len([x for x in encoded if x != 0])  # Count non-padding tokens
        
        return {
            'text': torch.tensor(encoded, dtype=torch.long),
            'label': torch.tensor(label, dtype=torch.long),
            'length': length
        }


def collate_batch(batch):
    """
    Collate function to create batches from individual samples.
    
    Args:
        batch: List of samples from __getitem__
        
    Returns:
        Dictionary with batched tensors
    """
    texts = torch.stack([item['text'] for item in batch])
    labels = torch.stack([item['label'] for item in batch])
    lengths = torch.tensor([item['length'] for item in batch])
    
    return {
        'text': texts,
        'label': labels,
        'length': lengths
    }


def create_dataloader(dataset, batch_size, shuffle=False):
    """
    Create a simple dataloader from dataset.
    
    Args:
        dataset: SentimentDataset instance
        batch_size: Number of samples per batch
        shuffle: Whether to shuffle data
        
    Returns:
        Generator that yields batches
    """
    indices = list(range(len(dataset)))
    if shuffle:
        np.random.shuffle(indices)
    
    for i in range(0, len(indices), batch_size):
        batch_indices = indices[i:i + batch_size]
        batch = [dataset[idx] for idx in batch_indices]
        yield collate_batch(batch)


def train_epoch(model, dataset, batch_size, criterion, optimizer, device):
    """
    Train model for one epoch.
    
    An epoch is one complete pass through the training data.
    For each batch:
        1. Forward pass: Make predictions
        2. Calculate loss: How wrong were we?
        3. Backward pass: Compute gradients
        4. Update weights: Improve the model
    
    Args:
        model: The RNN model
        dataset: Training dataset
        batch_size: Batch size
        criterion: Loss function (CrossEntropyLoss)
        optimizer: Weight update algorithm (Adam)
        device: CPU or GPU
        
    Returns:
        Average loss for the epoch
        Training accuracy
    """
    model.train()  # Set model to training mode (enables dropout)
    epoch_loss = 0
    correct = 0
    total = 0
    
    # Create dataloader
    dataloader = create_dataloader(dataset, batch_size, shuffle=True)
    num_batches = (len(dataset) + batch_size - 1) // batch_size
    
    # Progress bar for visual feedback
    pbar = tqdm(dataloader, total=num_batches, desc='Training')
    
    for batch in pbar:
        # Move data to device (GPU if available)
        texts = batch['text'].to(device)
        labels = batch['label'].to(device)
        lengths = batch['length']
        
        # Zero gradients from previous batch
        # (Gradients accumulate by default in PyTorch)
        optimizer.zero_grad()
        
        # Forward pass: Get predictions
        # Shape: (batch_size, num_classes)
        predictions, _ = model(texts, lengths)
        
        # Calculate loss: How different are predictions from true labels?
        # CrossEntropyLoss combines softmax and negative log likelihood
        loss = criterion(predictions, labels)
        
        # Backward pass: Compute gradients
        # This calculates how to adjust each weight to reduce loss
        loss.backward()
        
        # Clip gradients to prevent exploding gradients
        # (Common issue with RNNs)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        
        # Update weights using gradients
        optimizer.step()
        
        # Track statistics
        epoch_loss += loss.item()
        _, predicted = torch.max(predictions, 1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)
        
        # Update progress bar
        pbar.set_postfix({'loss': loss.item(), 'acc': correct / total})
    
    return epoch_loss / num_batches, correct / total


def evaluate(model, dataset, batch_size, criterion, device):
    """
    Evaluate model on validation/test set.
    
    Args:
        model: The RNN model
        dataset: Evaluation dataset
        batch_size: Batch size
        criterion: Loss function
        device: CPU or GPU
        
    Returns:
        Average loss
        Accuracy
        All predictions and true labels (for metrics)
    """
    model.eval()  # Set to evaluation mode (disables dropout)
    epoch_loss = 0
    all_predictions = []
    all_labels = []
    
    # Create dataloader
    dataloader = create_dataloader(dataset, batch_size, shuffle=False)
    num_batches = (len(dataset) + batch_size - 1) // batch_size
    
    # Disable gradient calculation (saves memory and computation)
    with torch.no_grad():
        for batch in tqdm(dataloader, total=num_batches, desc='Evaluating'):
            texts = batch['text'].to(device)
            labels = batch['label'].to(device)
            lengths = batch['length']
            
            # Forward pass only (no backward pass during evaluation)
            predictions, _ = model(texts, lengths)
            
            # Calculate loss
            loss = criterion(predictions, labels)
            epoch_loss += loss.item()
            
            # Get predicted classes
            _, predicted = torch.max(predictions, 1)
            
            # Store for metrics calculation
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    accuracy = accuracy_score(all_labels, all_predictions)
    
    return epoch_loss / num_batches, accuracy, all_predictions, all_labels


def plot_training_history(train_losses, train_accs, val_losses, val_accs, save_path):
    """
    Plot training and validation metrics over epochs.
    
    This helps us understand:
    - Is the model learning? (loss should decrease)
    - Is it overfitting? (train acc high, val acc low)
    - When to stop training? (val loss stops improving)
    
    Args:
        train_losses: Loss on training set per epoch
        train_accs: Accuracy on training set per epoch
        val_losses: Loss on validation set per epoch
        val_accs: Accuracy on validation set per epoch
        save_path: Where to save the plot
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot loss
    epochs = range(1, len(train_losses) + 1)
    ax1.plot(epochs, train_losses, 'b-', label='Training Loss', linewidth=2)
    ax1.plot(epochs, val_losses, 'r-', label='Validation Loss', linewidth=2)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot accuracy
    ax2.plot(epochs, train_accs, 'b-', label='Training Accuracy', linewidth=2)
    ax2.plot(epochs, val_accs, 'r-', label='Validation Accuracy', linewidth=2)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Training and Validation Accuracy')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Training history saved to {save_path}")
    plt.close()


def plot_confusion_matrix(cm, labels, save_path):
    """Plot and save confusion matrix."""
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=labels, yticklabels=labels)
    plt.title('Confusion Matrix - RNN with Attention')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Confusion matrix saved to {save_path}")
    plt.close()


def save_metrics(metrics, filepath):
    """Save evaluation metrics to JSON file."""
    import json
    
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


def main():
    """Main training pipeline for RNN model."""
    
    print("="*70)
    print("RNN WITH ATTENTION - MODEL B TRAINING")
    print("="*70)
    
    # Hyperparameters
    EMBEDDING_DIM = 100
    HIDDEN_DIM = 128
    N_LAYERS = 2
    DROPOUT = 0.5
    BATCH_SIZE = 32
    N_EPOCHS = 20
    LEARNING_RATE = 0.001
    MAX_VOCAB_SIZE = 10000
    MAX_SEQ_LENGTH = 200
    
    # Paths
    OUTPUT_DIR = "../../outputs"
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_DIR, 'rnn_checkpoints'), exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_DIR, 'figures'), exist_ok=True)
    
    # Load data
    print("\n" + "="*70)
    print("LOADING DATA")
    print("="*70)
    texts, labels = load_data(dataset='all', sample_size=3000)  # 3000 samples per dataset
    
    # Preprocess texts
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
    
    # Build vocabulary
    print("\n" + "="*70)
    print("BUILDING VOCABULARY")
    print("="*70)
    tokenizer = SimpleTokenizer(max_vocab_size=MAX_VOCAB_SIZE, min_freq=2)
    tokenizer.build_vocab(X_train)
    vocab_size = tokenizer.get_vocab_size()
    
    # Create datasets
    train_dataset = SentimentDataset(X_train, y_train, tokenizer, MAX_SEQ_LENGTH)
    val_dataset = SentimentDataset(X_val, y_val, tokenizer, MAX_SEQ_LENGTH)
    test_dataset = SentimentDataset(X_test, y_test, tokenizer, MAX_SEQ_LENGTH)
    
    # Initialize model
    print("\n" + "="*70)
    print("INITIALIZING MODEL")
    print("="*70)
    model = SentimentRNN(
        vocab_size=vocab_size,
        embedding_dim=EMBEDDING_DIM,
        hidden_dim=HIDDEN_DIM,
        output_dim=2,
        n_layers=N_LAYERS,
        bidirectional=True,
        dropout=DROPOUT
    ).to(device)
    
    print(f"Model has {count_parameters(model):,} trainable parameters")
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # Training loop
    print("\n" + "="*70)
    print("TRAINING")
    print("="*70)
    
    train_losses, train_accs = [], []
    val_losses, val_accs = [], []
    best_val_acc = 0
    
    for epoch in range(N_EPOCHS):
        print(f"\nEpoch {epoch+1}/{N_EPOCHS}")
        
        # Train
        train_loss, train_acc = train_epoch(model, train_dataset, BATCH_SIZE, criterion, optimizer, device)
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        
        # Validate
        val_loss, val_acc, _, _ = evaluate(model, val_dataset, BATCH_SIZE, criterion, device)
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        
        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
        print(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'vocab_size': vocab_size,
                'embedding_dim': EMBEDDING_DIM,
                'hidden_dim': HIDDEN_DIM,
                'n_layers': N_LAYERS,
                'dropout': DROPOUT,
            }, os.path.join(OUTPUT_DIR, 'rnn_checkpoints', 'best_model.pt'))
            print(f"✓ New best model saved (val_acc: {val_acc:.4f})")
    
    # Plot training history
    plot_training_history(
        train_losses, train_accs, val_losses, val_accs,
        os.path.join(OUTPUT_DIR, 'figures', 'rnn_training_history.png')
    )
    
    # Evaluate on test set
    print("\n" + "="*70)
    print("EVALUATING ON TEST SET")
    print("="*70)
    
    # Load best model
    checkpoint = torch.load(os.path.join(OUTPUT_DIR, 'rnn_checkpoints', 'best_model.pt'))
    model.load_state_dict(checkpoint['model_state_dict'])
    
    test_loss, test_acc, predictions, true_labels = evaluate(model, test_dataset, BATCH_SIZE, criterion, device)
    
    # Calculate metrics
    precision, recall, f1, _ = precision_recall_fscore_support(
        true_labels, predictions, average='weighted'
    )
    cm = confusion_matrix(true_labels, predictions)
    
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
        'train_losses': train_losses,
        'train_accs': train_accs,
        'val_losses': val_losses,
        'val_accs': val_accs,
        'hyperparameters': {
            'embedding_dim': EMBEDDING_DIM,
            'hidden_dim': HIDDEN_DIM,
            'n_layers': N_LAYERS,
            'dropout': DROPOUT,
            'batch_size': BATCH_SIZE,
            'learning_rate': LEARNING_RATE,
            'epochs': N_EPOCHS
        }
    }
    
    save_metrics(metrics, os.path.join(OUTPUT_DIR, 'rnn_metrics.json'))
    
    # Plot confusion matrix
    plot_confusion_matrix(cm, ['Negative', 'Positive'],
                         os.path.join(OUTPUT_DIR, 'figures', 'confusion_matrix_rnn.png'))
    
    # Save tokenizer for future use
    tokenizer_path = os.path.join(OUTPUT_DIR, 'rnn_checkpoints', 'tokenizer.pkl')
    import pickle
    with open(tokenizer_path, 'wb') as f:
        pickle.dump(tokenizer, f)
    print(f"\nTokenizer saved to {tokenizer_path}")
    
    # Save final model for production use
    final_model_path = os.path.join(OUTPUT_DIR, 'rnn_sentiment_model.pt')
    torch.save({
        'model_state_dict': model.state_dict(),
        'vocab_size': vocab_size,
        'embedding_dim': EMBEDDING_DIM,
        'hidden_dim': HIDDEN_DIM,
        'n_layers': N_LAYERS,
        'dropout': DROPOUT,
        'test_accuracy': test_acc,
        'val_accuracy': best_val_acc,
    }, final_model_path)
    print(f"Final model saved to {final_model_path}")
    
    print("\n" + "="*70)
    print("TRAINING COMPLETE!")
    print("="*70)
    print(f"Best validation accuracy: {best_val_acc:.4f}")
    print(f"Test accuracy: {test_acc:.4f}")
    print(f"\nModel ready for deployment:")
    print(f"  - Model: {final_model_path}")
    print(f"  - Tokenizer: {tokenizer_path}")


if __name__ == "__main__":
    main()
