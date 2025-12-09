# Model C: DistilBERT Fine-tuning
# Transformer-based sentiment analysis using DistilBERT
#
# Usage:
#   model, tokenizer = create_distilbert_model()
#   dataset = SentimentDataset(texts, labels, tokenizer)
#
# This uses pre-trained DistilBERT and fine-tunes it for binary sentiment classification

import torch
from torch.utils.data import Dataset
from transformers import DistilBertForSequenceClassification, DistilBertTokenizer
import numpy as np


class SentimentDataset(Dataset):
    """
    PyTorch Dataset for sentiment analysis with DistilBERT.
    
    Handles tokenization and encoding for transformer models.
    Unlike RNN models that need manual vocabulary building,
    transformers use pre-trained tokenizers.
    """
    
    def __init__(self, texts, labels, tokenizer, max_length=256):
        """
        Initialize dataset.
        
        Args:
            texts: List of text strings
            labels: List of sentiment labels (0 or 1)
            tokenizer: DistilBERT tokenizer
            max_length: Maximum sequence length (default 256 for DistilBERT)
        """
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        """Return number of samples."""
        return len(self.texts)
    
    def __getitem__(self, idx):
        """
        Get a single tokenized sample.
        
        The tokenizer automatically:
        - Converts text to subword tokens
        - Adds special tokens ([CLS], [SEP])
        - Creates attention mask
        - Pads/truncates to max_length
        
        Returns:
            Dictionary with input_ids, attention_mask, and label
        """
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        # Tokenize text
        # return_tensors='pt' returns PyTorch tensors
        # truncation=True ensures sequences don't exceed max_length
        # padding='max_length' pads shorter sequences
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }


def create_distilbert_model(num_labels=2, pretrained_model='distilbert-base-uncased'):
    """
    Create DistilBERT model and tokenizer for sentiment analysis.
    
    Args:
        num_labels: Number of output classes (2 for binary sentiment)
        pretrained_model: Name of pretrained model to use
        
    Returns:
        model: DistilBERT model for sequence classification
        tokenizer: Corresponding tokenizer
    """
    # Load pre-trained tokenizer
    # This tokenizer knows how to split text into subwords
    # that the model understands
    tokenizer = DistilBertTokenizer.from_pretrained(pretrained_model)
    
    # Load pre-trained model with classification head
    # The model has:
    # - DistilBERT base (learned from millions of texts)
    # - Classification layer on top (randomly initialized, needs training)
    model = DistilBertForSequenceClassification.from_pretrained(
        pretrained_model,
        num_labels=num_labels,
        problem_type="single_label_classification"
    )
    
    return model, tokenizer


def compute_metrics(pred):
    """
    Compute evaluation metrics for the Trainer.
    
    Args:
        pred: Predictions object with predictions and label_ids
        
    Returns:
        Dictionary with accuracy and F1 score
    """
    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
    
    # Get predicted labels (argmax of logits)
    predictions = np.argmax(pred.predictions, axis=1)
    labels = pred.label_ids
    
    # Calculate metrics
    accuracy = accuracy_score(labels, predictions)
    f1 = f1_score(labels, predictions, average='weighted')
    precision = precision_score(labels, predictions, average='weighted')
    recall = recall_score(labels, predictions, average='weighted')
    
    return {
        'accuracy': accuracy,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }
