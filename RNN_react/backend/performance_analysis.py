"""
performance_analysis.py

Analyze and summarize the performance of the trained LSTM next-word prediction model.

Key metrics and utilities:
- compute_accuracy: Overall next-word prediction accuracy on a test set.
- compute_top_k_accuracy: Top-k accuracy (correct word in top k predictions).
- compute_perplexity: Perplexity metric for language models (lower is better).
- generate_confusion_matrix: For classification, but less meaningful for large vocab; instead we
  provide top-error analysis.
- display_sample_predictions: Show sample inputs, true next word, and model predictions.
- summarize_performance: Generate a summary report of model performance.

Notes:
- For next-word prediction with large vocabularies (thousands of words), a traditional confusion
  matrix is impractical. Instead, we focus on:
    - Accuracy and top-k accuracy.
    - Perplexity (exponential of average cross-entropy loss).
    - Sample predictions to qualitatively assess performance.
- Perplexity: exp(average negative log-likelihood). Lower perplexity means the model is more
  confident and accurate.
"""
from typing import Dict, List, Tuple, Optional
import numpy as np

try:
    from tensorflow.keras.models import Model
except Exception:
    Model = None  # type: ignore


def compute_accuracy(model, X_test, y_test) -> float:
    """
    Compute overall next-word prediction accuracy on a test set.

    Args:
        model: Trained Keras model.
        X_test: Test feature array (num_examples, window_size).
        y_test: Test label array (num_examples,) containing true next-word indices.

    Returns:
        accuracy: Fraction of examples where the predicted word matches the true word.
    """
    # Predict on test set
    predictions = model.predict(X_test, verbose=0)
    predicted_indices = np.argmax(predictions, axis=1)

    # Compute accuracy
    correct = np.sum(predicted_indices == y_test)
    accuracy = correct / len(y_test)

    return float(accuracy)


def compute_top_k_accuracy(model, X_test, y_test, k: int = 5) -> float:
    """
    Compute top-k accuracy: fraction of examples where the true word is in the top k predictions.

    Args:
        model: Trained Keras model.
        X_test: Test feature array.
        y_test: Test label array.
        k: Number of top predictions to consider.

    Returns:
        top_k_accuracy: Fraction of examples where true word is in top k.
    """
    predictions = model.predict(X_test, verbose=0)
    # For each example, get indices of top k predictions
    top_k_preds = np.argsort(predictions, axis=1)[:, -k:]  # shape: (num_examples, k)

    # Check if true label is in top_k for each example
    correct = 0
    for i, true_label in enumerate(y_test):
        if true_label in top_k_preds[i]:
            correct += 1

    top_k_acc = correct / len(y_test)
    return float(top_k_acc)


def compute_perplexity(model, X_test, y_test) -> float:
    """
    Compute perplexity on the test set.

    Perplexity = exp(average cross-entropy loss). Lower perplexity indicates better performance.

    Args:
        model: Trained Keras model.
        X_test: Test feature array.
        y_test: Test label array.

    Returns:
        perplexity: exp(mean negative log-likelihood).

    Notes:
        - Perplexity measures how well the probability distribution predicted by the model matches
          the actual distribution. A perplexity of k means the model is as uncertain as if it had
          to choose uniformly among k possibilities.
    """
    # Evaluate the model to get loss (sparse_categorical_crossentropy)
    loss = model.evaluate(X_test, y_test, verbose=0)
    # loss is a list [total_loss, accuracy] or just total_loss depending on metrics
    if isinstance(loss, list):
        cross_entropy = loss[0]
    else:
        cross_entropy = loss

    perplexity = np.exp(cross_entropy)
    return float(perplexity)


def display_sample_predictions(
    model,
    X_sample,
    y_sample,
    index_word: Dict[int, str],
    num_samples: int = 10,
) -> List[Dict]:
    """
    Display sample predictions: input context, true next word, and predicted next word.

    Args:
        model: Trained Keras model.
        X_sample: Sample input sequences (num_samples, window_size).
        y_sample: True next-word labels (num_samples,).
        index_word: Mapping from index to word.
        num_samples: Number of samples to display.

    Returns:
        samples: List of dicts with keys: 'context', 'true_word', 'predicted_word', 'confidence'.
    """
    predictions = model.predict(X_sample[:num_samples], verbose=0)
    predicted_indices = np.argmax(predictions, axis=1)
    confidences = np.max(predictions, axis=1)

    samples = []
    for i in range(min(num_samples, len(X_sample))):
        context_indices = X_sample[i]
        # Convert context to words (skip padding)
        context_words = [index_word.get(idx, "<PAD>") for idx in context_indices if idx != 0]
        context_str = " ".join(context_words)

        true_word = index_word.get(y_sample[i], "<UNK>")
        pred_word = index_word.get(predicted_indices[i], "<UNK>")
        confidence = float(confidences[i])

        samples.append({
            "context": context_str,
            "true_word": true_word,
            "predicted_word": pred_word,
            "confidence": confidence,
            "correct": (predicted_indices[i] == y_sample[i]),
        })

    return samples


def analyze_top_errors(
    model,
    X_test,
    y_test,
    index_word: Dict[int, str],
    top_n: int = 20,
) -> List[Tuple[str, int, float]]:
    """
    Identify the most frequently mispredicted words (words that appear in y_test but are often
    predicted incorrectly).

    Args:
        model: Trained Keras model.
        X_test: Test feature array.
        y_test: Test label array.
        index_word: Mapping from index to word.
        top_n: Number of top error words to return.

    Returns:
        top_errors: List of tuples (true_word, count, error_rate) sorted by count descending.
    """
    predictions = model.predict(X_test, verbose=0)
    predicted_indices = np.argmax(predictions, axis=1)

    # Count errors per true word
    error_counts: Dict[int, int] = {}
    total_counts: Dict[int, int] = {}

    for i, true_idx in enumerate(y_test):
        total_counts[true_idx] = total_counts.get(true_idx, 0) + 1
        if predicted_indices[i] != true_idx:
            error_counts[true_idx] = error_counts.get(true_idx, 0) + 1

    # Compute error rates and sort
    error_list = []
    for true_idx, err_count in error_counts.items():
        total = total_counts[true_idx]
        error_rate = err_count / total
        word = index_word.get(true_idx, "<UNK>")
        error_list.append((word, err_count, error_rate))

    # Sort by error count descending
    error_list.sort(key=lambda x: x[1], reverse=True)

    return error_list[:top_n]


def summarize_performance(
    model,
    X_test,
    y_test,
    index_word: Dict[int, str],
    model_description: str = "LSTM next-word model",
) -> Dict:
    """
    Generate a comprehensive performance summary report.

    Args:
        model: Trained Keras model.
        X_test: Test feature array.
        y_test: Test label array.
        index_word: Mapping from index to word.
        model_description: Brief description of the model.

    Returns:
        summary: Dictionary containing all performance metrics and sample predictions.
    """
    print(f"\n{'='*80}")
    print(f"Performance Summary: {model_description}")
    print(f"{'='*80}\n")

    # Compute metrics
    accuracy = compute_accuracy(model, X_test, y_test)
    top_5_acc = compute_top_k_accuracy(model, X_test, y_test, k=5)
    top_10_acc = compute_top_k_accuracy(model, X_test, y_test, k=10)
    perplexity = compute_perplexity(model, X_test, y_test)

    print(f"Test set size: {len(y_test)} examples")
    print(f"Overall Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"Top-5 Accuracy: {top_5_acc:.4f} ({top_5_acc*100:.2f}%)")
    print(f"Top-10 Accuracy: {top_10_acc:.4f} ({top_10_acc*100:.2f}%)")
    print(f"Perplexity: {perplexity:.2f}")
    print()

    # Display sample predictions
    print("Sample Predictions (first 10 examples):")
    print("-" * 80)
    samples = display_sample_predictions(model, X_test, y_test, index_word, num_samples=10)
    for i, sample in enumerate(samples, 1):
        status = "✓" if sample["correct"] else "✗"
        print(f"{i}. {status} Context: \"{sample['context']}\"")
        print(f"   True: {sample['true_word']} | Predicted: {sample['predicted_word']} (conf: {sample['confidence']:.3f})")
        print()

    # Analyze top errors
    print("Top 10 Most Frequently Mispredicted Words:")
    print("-" * 80)
    top_errors = analyze_top_errors(model, X_test, y_test, index_word, top_n=10)
    for rank, (word, err_count, err_rate) in enumerate(top_errors, 1):
        print(f"{rank}. \"{word}\" - {err_count} errors (error rate: {err_rate*100:.1f}%)")

    print("\n" + "="*80)
    print("Summary of RNN Functioning:")
    print("="*80)
    print(f"The {model_description} uses an LSTM architecture to predict the next word in a sequence.")
    print(f"Given a context window of words, it outputs a probability distribution over the vocabulary.")
    print(f"The model achieved {accuracy*100:.2f}% accuracy on the test set, meaning it correctly predicts")
    print(f"the next word in {accuracy*100:.2f}% of cases. The top-5 accuracy is {top_5_acc*100:.2f}%, indicating")
    print(f"that the correct word is among the top 5 predictions in {top_5_acc*100:.2f}% of cases.")
    print(f"Perplexity of {perplexity:.2f} suggests the model has moderate uncertainty; lower perplexity")
    print(f"indicates better performance. Overall, the model demonstrates {_performance_label(accuracy)} performance")
    print(f"for next-word prediction on this dataset.")
    print("="*80 + "\n")

    summary = {
        "accuracy": accuracy,
        "top_5_accuracy": top_5_acc,
        "top_10_accuracy": top_10_acc,
        "perplexity": perplexity,
        "test_size": len(y_test),
        "sample_predictions": samples,
        "top_errors": top_errors,
    }

    return summary


def _performance_label(accuracy: float) -> str:
    """Helper to categorize performance based on accuracy."""
    if accuracy >= 0.7:
        return "strong"
    elif accuracy >= 0.5:
        return "moderate"
    elif accuracy >= 0.3:
        return "fair"
    else:
        return "limited"


__all__ = [
    "compute_accuracy",
    "compute_top_k_accuracy",
    "compute_perplexity",
    "display_sample_predictions",
    "analyze_top_errors",
    "summarize_performance",
]
