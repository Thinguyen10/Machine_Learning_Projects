"""
data_processing.py

Lightweight data-preparation utilities for text inputs.

This module provides functions to:
- remove punctuation and normalize text
- split strings into lists of individual words
- convert words to integer indices using Keras Tokenizer

Each function has inline comments and a short docstring explaining inputs/outputs.
"""
from typing import List, Tuple, Optional
import re

try:
    # Prefer TensorFlow's Keras if available
    # Tokenizer: turns text (or lists of tokens) into sequences of integer indices.
    # - tokenizer.fit_on_texts(texts): learns the word -> index mapping (word counts, ranks)
    # - tokenizer.texts_to_sequences(texts): converts each text (or token list) into a list of ints
    # See docs: https://www.tensorflow.org/api_docs/python/tf/keras/preprocessing/text/Tokenizer
    from tensorflow.keras.preprocessing.text import Tokenizer
    # pad_sequences: utility to make all sequences the same length by padding/truncating
    # - pad_sequences(sequences, maxlen=N, padding='post'|'pre'): returns a numpy array
    # - truncating='post'|'pre': determines where to cut long sequences
    from tensorflow.keras.preprocessing.sequence import pad_sequences
except Exception:
    # If TensorFlow/Keras is unavailable, raise a clear error when tokenizer functions are used
    Tokenizer = None  # type: ignore
    pad_sequences = None  # type: ignore


_PUNCT_RE = re.compile(r"[^\w\s]")  # matches anything that's not a word char or whitespace


def remove_punctuation_and_split(texts: List[str]) -> List[List[str]]:
    """
    Remove punctuation from each string in `texts` and split into lists of words.

    - Lowercases the text.
    - Removes punctuation characters (keeps letters, digits and whitespace).
    - Splits on whitespace to produce token lists.

    Args:
        texts: List of raw strings (sentences/documents).

    Returns:
        A list where each element is a list of tokens (words) for the corresponding input string.

    Example:
        >>> remove_punctuation_and_split(["Hello, world!"])
        [['hello', 'world']]
    """
    cleaned: List[List[str]] = []
    for t in texts:
        if not isinstance(t, str):
            # defensively convert non-strings to string
            t = str(t)
        lower = t.lower()
        # remove punctuation (anything not a word character or whitespace)
        no_punct = _PUNCT_RE.sub("", lower)
        # split into words by whitespace
        tokens = no_punct.split()
        cleaned.append(tokens)
    return cleaned


def texts_to_sequences(
    texts: List[str],
    num_words: Optional[int] = None,
    oov_token: Optional[str] = None,
    max_len: Optional[int] = None,
    padding: str = "post",
) -> Tuple[List[List[int]], object]:
    """
    Convert texts to integer sequences using Keras Tokenizer.

    Steps performed:
    1. Clean texts (remove punctuation, lowercase, split to tokens).
    2. Fit a Keras Tokenizer on the token lists.
    3. Convert token lists to integer sequences.
    4. Optionally pad/truncate sequences to `max_len` using `pad_sequences`.

    Args:
        texts: List of raw strings to process.
        num_words: Maximum number of words to keep (most frequent). If None, keep all.
        oov_token: Token to represent out-of-vocabulary words (e.g. '<OOV>').
        max_len: If provided, pad/truncate sequences to this length (returns numpy array).
        padding: 'pre' or 'post' (used only if `max_len` is provided).

    Returns:
        sequences: List (or numpy array if padded) of integer sequences.
        tokenizer: The fitted Keras Tokenizer instance (contains word_index mapping).

        Notes:
                - This function requires TensorFlow (or standalone Keras) to be installed. If it's not
                    available, a clear ImportError will be raised when calling this function.
                - How Keras Tokenizer works (short):
                        1. tokenizer = Tokenizer(num_words=..., oov_token=...)
                             - Builds an internal mapping `word -> integer` when fit.
                             - `num_words` limits the vocabulary to the top `num_words-1` most frequent words
                                 (the index 0 is usually reserved for padding).
                             - `oov_token` is used to mark words not seen during fit (out-of-vocabulary).
                        2. tokenizer.fit_on_texts(texts)
                             - Updates internal word counts and creates `word_index` mapping.
                        3. tokenizer.texts_to_sequences(texts)
                             - Replaces each token with its integer index according to `word_index`.
                             - Words not in `word_index` map to the `oov_token` index if `oov_token` was set,
                                 otherwise they are skipped.
                - pad_sequences takes a list of integer lists and returns a 2D numpy array shaped
                    (num_samples, maxlen) where sequences shorter than `maxlen` are padded and longer
                    ones are truncated according to the `padding` and `truncating` strategy.
    """
    if Tokenizer is None:
        raise ImportError(
            "Keras Tokenizer not available. Please install TensorFlow (e.g. `pip install tensorflow`) to use texts_to_sequences."
        )

    # 1) clean texts into lists of tokens
    token_lists = remove_punctuation_and_split(texts)

    # 2) create and fit tokenizer on token lists (Tokenizer accepts lists of tokens)
    # Create the tokenizer instance. When `fit_on_texts` is called the tokenizer:
    # - counts word frequencies across the provided token lists
    # - builds `word_index` mapping e.g. {'the': 1, 'a': 2, ...}
    # - if `num_words` is set, only the top `num_words-1` words will be kept for
    #   conversion (index 0 reserved for padding when using pad_sequences).
    tokenizer = Tokenizer(num_words=num_words, oov_token=oov_token)
    # Fit tokenizer to the tokenized texts. This learns the vocabulary and word counts.
    tokenizer.fit_on_texts(token_lists)

    # 3) convert to integer sequences
    # texts_to_sequences maps each token in token_lists to its integer id according to
    # the learned `word_index`. The output is a list of lists of integers. If a token
    # wasn't in the learned vocabulary and `oov_token` was provided, it maps to the
    # oov index; otherwise unseen words are skipped.
    sequences = tokenizer.texts_to_sequences(token_lists)

    # 4) optionally pad sequences
    if max_len is not None:
        if pad_sequences is None:
            raise ImportError(
                "pad_sequences not available. Please install TensorFlow (e.g. `pip install tensorflow`) to use padding features."
            )
        # pad_sequences returns a numpy array of shape (n_samples, max_len). By default
        # we use `truncating='post'` so that tokens beyond `max_len` at the end are dropped.
        sequences = pad_sequences(sequences, maxlen=max_len, padding=padding, truncating="post")

    return sequences, tokenizer


__all__ = [
    "remove_punctuation_and_split",
    "texts_to_sequences",
]
