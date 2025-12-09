# Model C: TF-IDF + Logistic Regression Baseline
# Simple but effective baseline for sentiment analysis
#
# Example usage:
#   baseline = TFIDFBaseline()
#   baseline.train(train_texts, train_labels)
#   predictions = baseline.predict(test_texts)
#   accuracy = baseline.evaluate(test_texts, test_labels)
#   # Saves confusion matrix and metrics automatically

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from sklearn.metrics import classification_report
import pickle
import os
from typing import List, Tuple
import matplotlib.pyplot as plt
import seaborn as sns


class TFIDFBaseline:
    """
    Baseline sentiment classifier using TF-IDF features and Logistic Regression.
    This provides a strong baseline to compare against deep learning models.
    """
    
    def __init__(self, max_features: int = 10000, classifier_type: str = 'logistic'):
        """
        Initialize the baseline model.
        
        Args:
            max_features: Maximum number of TF-IDF features to use
            classifier_type: 'logistic' for LogisticRegression or 'svm' for LinearSVC
        """
        self.max_features = max_features
        self.classifier_type = classifier_type
        
        # TF-IDF vectorizer: converts text to numerical features
        # TF-IDF = Term Frequency * Inverse Document Frequency
        # 
        # How TF-IDF works:
        #   - TF (Term Frequency): How often a word appears in a document
        #   - IDF (Inverse Document Frequency): How rare/unique a word is across all documents
        #   - Formula: TF-IDF = (word count in doc / total words in doc) * log(total docs / docs containing word)
        #   - Example: "excellent" appears rarely → high IDF → high TF-IDF score
        #              "the" appears everywhere → low IDF → low TF-IDF score
        # 
        # This gives more weight to important discriminative words, less to common ones
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,  # Keep only top N most important features (reduces dimensionality)
            ngram_range=(1, 2),  # Use unigrams ("good") AND bigrams ("not good")
                                 # Bigrams help capture negations and phrases
            min_df=2,  # Ignore words appearing in fewer than 2 documents (likely typos/rare words)
            max_df=0.9,  # Ignore words in >90% of documents (too common to be useful)
            strip_accents='unicode',  # Remove accents (café → cafe) for consistency
            lowercase=True,  # Convert all to lowercase ("Good" and "good" treated same)
            stop_words='english'  # Remove common words ("the", "is", "and") - usually not sentiment-bearing
        )
        
        # Classifier: Logistic Regression or SVM
        # Both are linear models that work well with high-dimensional sparse TF-IDF features
        if classifier_type == 'logistic':
            # Logistic Regression: Predicts probability of each class
            # Works by learning weights for each feature (word)
            # Equation: P(positive) = 1 / (1 + e^-(w1*x1 + w2*x2 + ... + wn*xn))
            # Advantages: Fast, interpretable, outputs probabilities
            self.classifier = LogisticRegression(
                max_iter=1000,  # Maximum training iterations
                random_state=42,  # For reproducible results
                C=1.0,  # Regularization strength (smaller = more regularization)
                        # Regularization prevents overfitting by penalizing large weights
                solver='liblinear'  # Optimization algorithm - good for small-medium datasets
            )
        elif classifier_type == 'svm':
            # Linear SVM: Finds hyperplane that best separates classes
            # Maximizes margin between positive and negative examples
            # Often achieves slightly better accuracy than Logistic Regression
            # Disadvantage: Doesn't output probabilities (only hard predictions)
            self.classifier = LinearSVC(
                max_iter=1000,  # Maximum training iterations
                random_state=42,  # For reproducible results
                C=1.0  # Regularization parameter (smaller = wider margin, more regularization)
            )
        else:
            raise ValueError("classifier_type must be 'logistic' or 'svm'")
        
        self.is_trained = False
        
    def train(self, texts: List[str], labels: List[int]):
        """
        Train the baseline model on text data.
        
        Args:
            texts: List of text documents
            labels: List of sentiment labels (0=negative, 1=positive, etc.)
        """
        print(f"Training {self.classifier_type} baseline model...")
        print(f"Training samples: {len(texts)}")
        
        # Step 1: Convert texts to TF-IDF features
        # This creates a sparse matrix where:
        #   - Rows = documents (reviews/tweets)
        #   - Columns = features (words/bigrams)
        #   - Values = TF-IDF scores (importance of word in document)
        # Example: "I love this movie" might become:
        #   [0.0, 0.0, 0.82, 0.0, 0.57, ...]  (mostly zeros, sparse)
        #            ^      ^      ^-- "movie" TF-IDF score
        #            |      |-- "love" TF-IDF score  
        #            |-- "this" (stop word, removed)
        print("Extracting TF-IDF features...")
        X_train = self.vectorizer.fit_transform(texts)  # fit_transform learns vocabulary AND transforms
        print(f"Feature matrix shape: {X_train.shape}")  # e.g., (10000, 5000) = 10k docs, 5k features
        
        # Step 2: Train the classifier
        # The classifier learns a weight for each feature (word)
        # Positive weights → word indicates positive sentiment
        # Negative weights → word indicates negative sentiment
        # Example learned weights:
        #   "excellent": +2.5, "terrible": -3.1, "not bad": +0.8
        print(f"Training {self.classifier_type} classifier...")
        self.classifier.fit(X_train, labels)  # Learn the weights from training data
        
        self.is_trained = True
        print("Training complete!")
        
        # Show training accuracy
        train_preds = self.classifier.predict(X_train)
        train_acc = accuracy_score(labels, train_preds)
        print(f"Training accuracy: {train_acc:.4f}")
        
    def predict(self, texts: List[str]) -> np.ndarray:
        """
        Predict sentiment labels for texts.
        
        Args:
            texts: List of text documents
            
        Returns:
            Array of predicted labels
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")
        
        # Transform texts to TF-IDF features using the SAME vocabulary learned during training
        # Note: We use transform() not fit_transform() - we don't learn new vocabulary
        # Example: "great movie" → [0.0, 0.0, 0.75, 0.0, 0.68, ...]
        X = self.vectorizer.transform(texts)
        
        # Predict labels by computing weighted sum of features
        # For each document: score = w1*x1 + w2*x2 + ... + wn*xn
        # If score > threshold → positive, else → negative
        # Example: "great" (0.75 * 2.1) + "movie" (0.68 * 0.3) = 1.78 → POSITIVE
        predictions = self.classifier.predict(X)
        return predictions
    
    def predict_proba(self, texts: List[str]) -> np.ndarray:
        """
        Predict probability distributions for each class.
        Only works with Logistic Regression.
        
        Args:
            texts: List of text documents
            
        Returns:
            Array of probability distributions
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")
        
        if self.classifier_type != 'logistic':
            raise ValueError("predict_proba only available with logistic classifier")
        
        X = self.vectorizer.transform(texts)
        probas = self.classifier.predict_proba(X)
        return probas
    
    def evaluate(self, texts: List[str], labels: List[int], 
                label_names: List[str] = None) -> dict:
        """
        Evaluate model on test data and compute metrics.
        
        Args:
            texts: List of text documents
            labels: True labels
            label_names: Names of classes (e.g., ['negative', 'positive'])
            
        Returns:
            Dictionary of metrics
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before evaluation")
        
        print(f"\nEvaluating on {len(texts)} samples...")
        
        # Get predictions
        predictions = self.predict(texts)
        
        # Calculate metrics
        accuracy = accuracy_score(labels, predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(
            labels, predictions, average='weighted'
        )
        
        # Per-class metrics
        precision_per_class, recall_per_class, f1_per_class, support = \
            precision_recall_fscore_support(labels, predictions, average=None)
        
        # Confusion matrix
        cm = confusion_matrix(labels, predictions)
        
        metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'confusion_matrix': cm,
            'precision_per_class': precision_per_class,
            'recall_per_class': recall_per_class,
            'f1_per_class': f1_per_class,
            'support': support
        }
        
        # Print results
        print(f"\n{'='*60}")
        print("EVALUATION RESULTS")
        print(f"{'='*60}")
        print(f"Accuracy:  {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall:    {recall:.4f}")
        print(f"F1 Score:  {f1:.4f}")
        print(f"\nConfusion Matrix:\n{cm}")
        
        # Detailed classification report
        if label_names is None:
            label_names = [f"Class {i}" for i in range(len(np.unique(labels)))]
        
        print(f"\n{classification_report(labels, predictions, target_names=label_names)}")
        
        return metrics
    
    def save_model(self, filepath: str):
        """
        Save the trained model to disk.
        
        Args:
            filepath: Path to save model (e.g., 'baseline_model.pkl')
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before saving")
        
        model_data = {
            'vectorizer': self.vectorizer,
            'classifier': self.classifier,
            'classifier_type': self.classifier_type,
            'max_features': self.max_features
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """
        Load a trained model from disk.
        
        Args:
            filepath: Path to saved model
        """
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        self.vectorizer = model_data['vectorizer']
        self.classifier = model_data['classifier']
        self.classifier_type = model_data['classifier_type']
        self.max_features = model_data['max_features']
        self.is_trained = True
        
        print(f"Model loaded from {filepath}")
    
    def get_top_features(self, n: int = 20) -> List[Tuple[str, float]]:
        """
        Get the most important features (words) for classification.
        Only works with Logistic Regression.
        
        Args:
            n: Number of top features to return
            
        Returns:
            List of (feature, coefficient) tuples
        """
        if not self.is_trained:
            raise ValueError("Model must be trained first")
        
        if self.classifier_type != 'logistic':
            raise ValueError("Feature importance only available with logistic classifier")
        
        # Get feature names (words/bigrams) and their learned coefficients (weights)
        # Coefficients tell us how much each word contributes to the prediction
        feature_names = self.vectorizer.get_feature_names_out()
        
        # For binary classification (positive vs negative)
        if len(self.classifier.classes_) == 2:
            # Get the learned weight for each feature
            # Large positive weight → strong indicator of positive sentiment
            # Large negative weight → strong indicator of negative sentiment
            # Weight near zero → not useful for prediction
            coefficients = self.classifier.coef_[0]
            
            # Get top positive features (indicate positive sentiment)
            # argsort gives indices sorted by value, [-n:] takes last n (largest), [::-1] reverses
            # Example results: "excellent" (2.8), "amazing" (2.5), "love" (2.3)
            top_positive_idx = np.argsort(coefficients)[-n:][::-1]
            top_positive = [(feature_names[i], coefficients[i]) 
                          for i in top_positive_idx]
            
            # Get top negative features (indicate negative sentiment)
            # [:n] takes first n (most negative)
            # Example results: "terrible" (-3.2), "worst" (-2.9), "awful" (-2.7)
            top_negative_idx = np.argsort(coefficients)[:n]
            top_negative = [(feature_names[i], coefficients[i]) 
                          for i in top_negative_idx]
            
            return {
                'positive_features': top_positive,
                'negative_features': top_negative
            }
        else:
            # Multi-class: return top features per class
            results = {}
            for i, class_label in enumerate(self.classifier.classes_):
                coefficients = self.classifier.coef_[i]
                top_idx = np.argsort(coefficients)[-n:][::-1]
                top_features = [(feature_names[j], coefficients[j]) 
                              for j in top_idx]
                results[f'class_{class_label}'] = top_features
            
            return results


def plot_confusion_matrix(cm: np.ndarray, labels: List[str], 
                         save_path: str = None):
    """
    Plot and save confusion matrix visualization.
    
    Confusion Matrix shows prediction accuracy breakdown:
    
                    Predicted
                 Neg    Pos
    True   Neg   TN     FP     TN = True Negative (correctly predicted negative)
           Pos   FN     TP     TP = True Positive (correctly predicted positive)
                              FP = False Positive (predicted positive, actually negative)
                              FN = False Negative (predicted negative, actually positive)
    
    Diagonal = correct predictions, off-diagonal = errors
    
    Args:
        cm: Confusion matrix (2D array)
        labels: Class labels (e.g., ['Negative', 'Positive'])
        save_path: Path to save figure
    """
    plt.figure(figsize=(8, 6))
    # Create heatmap: darker blue = more predictions
    # annot=True shows numbers in each cell
    # fmt='d' formats as integers
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=labels, yticklabels=labels)
    plt.title('Confusion Matrix - TF-IDF Baseline')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Confusion matrix saved to {save_path}")
    
    plt.show()


def save_metrics(metrics: dict, filepath: str):
    """
    Save evaluation metrics to a JSON file.
    
    Args:
        metrics: Dictionary of metrics
        filepath: Path to save metrics
    """
    import json
    
    # Convert numpy arrays to lists for JSON serialization
    serializable_metrics = {}
    for key, value in metrics.items():
        if isinstance(value, np.ndarray):
            serializable_metrics[key] = value.tolist()
        else:
            serializable_metrics[key] = value
    
    with open(filepath, 'w') as f:
        json.dump(serializable_metrics, f, indent=2)
    
    print(f"Metrics saved to {filepath}")
