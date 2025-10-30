"""
sequence_builder.py

Utilities to build sliding-window feature/label pairs for next-word prediction
from integer token sequences.

Key features:
- Set the window size (number of words used as features) via parameter `window_size` (n).
- Build training examples using a sliding window across each sequence:
  e.g. for window_size=n, examples are:
    words[0:n] -> words[n]
    words[1:n+1] -> words[n+1]
    words[2:n+2] -> words[n+2]
    ...
- Optionally filter examples based on `num_words` (vocabulary size). Examples containing
  tokens with index >= num_words will be skipped (useful when tokenizer was created with num_words).
- Simple train/test split helper with optional shuffling.

Note: RNN performance is proportional to the amount of training data; creating many
sliding-window examples increases dataset size and usually improves model performance.
"""
from typing import List, Tuple, Optional
import numpy as np


def build_sliding_window_sequences(
    sequences: List[List[int]],
    window_size: int,
    num_words: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build feature / label pairs using a sliding window across each integer token sequence.

    For each input sequence (a list of ints), this function generates examples where the
    features are `window_size` consecutive tokens and the label is the immediately
    following token.

    Args:
        sequences: List of token sequences (each is a list of integer token ids).
        window_size: Number of words to use as input features (n).
        num_words: Optional vocabulary size cap. If provided, any example containing a token
                   with id >= num_words (or a label >= num_words) will be skipped. This
                   is useful when the tokenizer used `num_words` to limit vocabulary.

    Returns:
        X: numpy array of shape (num_examples, window_size) with dtype int32.
        y: numpy array of shape (num_examples,) with dtype int32 containing the next-word ids.

    Example:
        >>> seqs = [[5,2,8,9,3]]
        >>> X,y = build_sliding_window_sequences(seqs, window_size=2)
        # X -> [[5,2],[2,8],[8,9]]
        # y -> [8,9,3]
    """
    if window_size < 1:
        raise ValueError("window_size must be >= 1")

    X_list: List[List[int]] = []
    y_list: List[int] = []

    for seq_idx, seq in enumerate(sequences):
        if not seq:
            continue
        # iterate over sliding windows within this sequence; stop so label exists
        for i in range(len(seq) - window_size):
            window = seq[i : i + window_size]
            label = seq[i + window_size]

            # If num_words is provided, skip any examples that reference out-of-range tokens
            if num_words is not None:
                # token indices should be non-negative integers; skip if out of range
                out_of_range = any((token is None or token < 0 or token >= num_words) for token in window)
                if out_of_range or label < 0 or label >= num_words:
                    continue

            X_list.append(window)
            y_list.append(label)

    # Convert to numpy arrays; handle empty case
    if X_list:
        X = np.array(X_list, dtype=np.int32)
        y = np.array(y_list, dtype=np.int32)
    else:
        X = np.empty((0, window_size), dtype=np.int32)
        y = np.empty((0,), dtype=np.int32)

    return X, y


def train_test_split(
    X: np.ndarray,
    y: np.ndarray,
    train_frac: float = 0.8,
    shuffle: bool = True,
    seed: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Split features and labels into train and validation sets.

    Args:
        X: Feature array of shape (num_examples, window_size).
        y: Label array of shape (num_examples,).
        train_frac: Fraction of examples to keep for training (rest is validation).
        shuffle: Whether to shuffle the examples before splitting.
        seed: Optional random seed for reproducible shuffling.

    Returns:
        X_train, y_train, X_val, y_val
    """
    if X.shape[0] != y.shape[0]:
        raise ValueError("X and y must have the same number of examples")
    n = X.shape[0]
    if n == 0:
        return X, y, X, y

    indices = np.arange(n)
    if shuffle:
        rng = np.random.RandomState(seed)
        rng.shuffle(indices)

    split_at = int(np.floor(train_frac * n))
    train_idx = indices[:split_at]
    val_idx = indices[split_at:]

    X_train = X[train_idx]
    y_train = y[train_idx]
    X_val = X[val_idx]
    y_val = y[val_idx]

    return X_train, y_train, X_val, y_val


__all__ = ["build_sliding_window_sequences", "train_test_split"]
