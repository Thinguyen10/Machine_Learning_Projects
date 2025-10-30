"""
lstm_model.py

Build and train an LSTM next-word prediction model using Keras Sequential API.

This module provides:
- build_lstm_model: constructs the Keras model with Embedding, Masking, LSTM, Dense, Dropout.
- load_glove_embeddings: create an embedding matrix from a GloVe file and a tokenizer's word_index.
- cosine_similarity: compute cosine similarity between two embedding vectors.
- train_model: helper to train the model with ModelCheckpoint and EarlyStopping callbacks.

Assumptions / notes:
- The embedding layer expects input word indices that match the tokenizer used to create
  the sequences (index 0 reserved for padding).
- If `embedding_matrix` is provided to `build_lstm_model`, it will be used to initialize the
  Embedding layer via the `weights` parameter. Set `trainable=False` to keep embeddings fixed.
"""
from typing import Optional, Dict, Tuple
import numpy as np

try:
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Embedding, Masking, LSTM, Dense, Dropout
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
except Exception:
    Sequential = None  # type: ignore
    Embedding = None  # type: ignore
    Masking = None  # type: ignore
    LSTM = None  # type: ignore
    Dense = None  # type: ignore
    Dropout = None  # type: ignore
    Adam = None  # type: ignore
    ModelCheckpoint = None  # type: ignore
    EarlyStopping = None  # type: ignore


def build_lstm_model(
    vocab_size: int,
    embedding_dim: int = 100,
    embedding_matrix: Optional[np.ndarray] = None,
    input_length: Optional[int] = None,
    lstm_units: int = 128,
    dense_units: int = 64,
    dropout_rate: float = 0.5,
    trainable_embeddings: bool = False,
) -> "Sequential":
    """
    Build and compile an LSTM model for next-word prediction.

    Architecture (Keras Sequential):
      - Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=input_length, weights=[embedding_matrix])
        Maps word indices to dense vectors. If `embedding_matrix` is supplied, the layer is
        initialized with pretrained vectors (e.g., GloVe). If `trainable_embeddings` is False,
        embeddings are kept fixed during training.
      - Masking(mask_value=0.0)
        Masks padded positions (index 0) so the LSTM ignores them. Only use this when
        embeddings are not trainable or when zero-padding is meaningful.
      - LSTM(lstm_units, dropout=..., recurrent_dropout=...)
        The core recurrent layer. Since we're using a single LSTM layer for many-to-one
        prediction, return_sequences=False (default).
      - Dense(dense_units, activation='relu')
        A fully-connected layer to increase representational capacity.
      - Dropout(dropout_rate)
        Prevent overfitting.
      - Dense(vocab_size, activation='softmax')
        Output probabilities over the vocabulary for the next-word prediction.

    Args:
        vocab_size: Size of the vocabulary (number of distinct token indices). This is
                    the size of the final softmax layer.
        embedding_dim: Dimensionality of embeddings (100 for GloVe-100).
        embedding_matrix: Optional numpy array of shape (vocab_size, embedding_dim). If
                          provided, used to initialize the Embedding layer.
        input_length: Length of input windows (n). If provided, the Embedding layer will
                      use it to shape its input; otherwise Keras accepts variable lengths.
        lstm_units: Number of LSTM units.
        dense_units: Units in the intermediate Dense layer.
        dropout_rate: Dropout rate applied after Dense.
        trainable_embeddings: Whether to update embedding weights during training.

    Returns:
        A compiled Keras Sequential model ready for training.
    """
    if Sequential is None:
        raise ImportError("TensorFlow / Keras not available. Install tensorflow to use build_lstm_model.")

    model = Sequential()

    # Embedding layer. If you pass an embedding_matrix, Keras expects `weights=[embedding_matrix]`.
    if embedding_matrix is not None:
        # Ensure matrix shape matches vocab_size x embedding_dim
        if embedding_matrix.shape[0] != vocab_size or embedding_matrix.shape[1] != embedding_dim:
            raise ValueError("embedding_matrix shape must be (vocab_size, embedding_dim)")
        model.add(
            Embedding(
                input_dim=vocab_size,
                output_dim=embedding_dim,
                input_length=input_length,
                weights=[embedding_matrix],
                trainable=trainable_embeddings,
                mask_zero=True,  # mask_zero signals that index 0 is reserved for padding
            )
        )
        # When mask_zero=True, Keras handles masking internally; no separate Masking layer needed.
    else:
        # No pretrained embeddings: initialize embeddings randomly and allow training
        model.add(
            Embedding(
                input_dim=vocab_size,
                output_dim=embedding_dim,
                input_length=input_length,
                trainable=trainable_embeddings,
                mask_zero=True,
            )
        )

    # Add the LSTM layer. The `dropout` param applies to inputs; `recurrent_dropout` applies to recurrent state.
    model.add(LSTM(lstm_units, dropout=0.2, recurrent_dropout=0.2))

    # Dense + dropout for extra capacity and regularization
    model.add(Dense(dense_units, activation="relu"))
    model.add(Dropout(dropout_rate))

    # Output layer: probability distribution over the vocabulary
    model.add(Dense(vocab_size, activation="softmax"))

    # Compile the model with Adam optimizer and sparse categorical crossentropy since
    # labels are integer word indices (not one-hot encoded). Use accuracy metric for monitoring.
    model.compile(
        optimizer=Adam(),
        loss="sparse_categorical_crossentropy",
        metrics=["sparse_categorical_accuracy"],
    )

    return model


def load_glove_embeddings(glove_path: str, word_index: Dict[str, int], embedding_dim: int = 100, vocab_size: Optional[int] = None) -> np.ndarray:
    """
    Load GloVe vectors from `glove_path` and create an embedding matrix aligned with `word_index`.

    Args:
        glove_path: Path to the GloVe text file (e.g., glove.6B.100d.txt).
        word_index: Mapping from token (string) to integer index as produced by Keras Tokenizer.
        embedding_dim: Dimensionality of the GloVe vectors (e.g., 100).
        vocab_size: If provided, size of the embedding matrix (number of rows). If None, use
                    max index in word_index + 1.

    Returns:
        embedding_matrix: numpy array of shape (vocab_size, embedding_dim). Row 0 is all zeros
                          (reserved for padding). Words not found in GloVe remain zeros.

    Notes:
        - GloVe format: each line -> word val1 val2 ... valN
        - This function may take some time and memory depending on the GloVe file size.
    """
    # Determine final vocab size
    max_index = max(word_index.values()) if word_index else 0
    if vocab_size is None:
        vocab_size = max_index + 1  # +1 because word indices start at 1; 0 reserved for padding

    # Initialize embedding matrix with zeros. Row 0 corresponds to the padding token.
    embedding_matrix = np.zeros((vocab_size, embedding_dim), dtype=np.float32)

    # Read GloVe file and build a mapping word -> vector
    glove_dict = {}
    with open(glove_path, "r", encoding="utf8") as f:
        for line in f:
            parts = line.rstrip().split(" ")
            word = parts[0]
            vec = np.asarray(parts[1:], dtype=np.float32)
            if vec.size != embedding_dim:
                # skip lines that don't match expected dimension
                continue
            glove_dict[word] = vec

    # Fill embedding matrix using the tokenizer's word_index mapping.
    # Keras Tokenizer indices start at 1 (0 is reserved for padding).
    for word, idx in word_index.items():
        if idx >= vocab_size:
            continue
        vector = glove_dict.get(word)
        if vector is not None:
            embedding_matrix[idx] = vector
        # else leave zeros (unknown words will have zero vector)

    return embedding_matrix


def cosine_similarity(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    """
    Compute cosine similarity between two 1-D vectors.

    Returns a float in [-1, 1] where 1 means identical direction, -1 opposite.
    """
    a = vec_a.astype(np.float32)
    b = vec_b.astype(np.float32)
    denom = (np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)


def train_model(
    model,
    X_train,
    y_train,
    X_val,
    y_val,
    epochs: int = 20,
    batch_size: int = 128,
    checkpoint_path: str = "best_model.h5",
    patience: int = 3,
):
    """
    Train the model with ModelCheckpoint and EarlyStopping callbacks.

    Args:
        model: Compiled Keras model.
        X_train, y_train: Training data (numpy arrays).
        X_val, y_val: Validation data for monitoring.
        epochs: Maximum number of epochs.
        batch_size: Batch size for training.
        checkpoint_path: Filepath to save the best model (by val loss).
        patience: EarlyStopping patience (number of epochs with no improvement).

    Returns:
        history: Keras History object returned by `model.fit`.
    """
    if ModelCheckpoint is None or EarlyStopping is None:
        raise ImportError("TensorFlow / Keras callbacks not available. Install tensorflow to use train_model.")

    # Save the best model according to validation loss
    checkpoint = ModelCheckpoint(checkpoint_path, monitor="val_loss", save_best_only=True, verbose=1)
    # Stop training early if validation loss doesn't improve
    early_stop = EarlyStopping(monitor="val_loss", patience=patience, restore_best_weights=True, verbose=1)

    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[checkpoint, early_stop],
    )

    return history


__all__ = [
    "build_lstm_model",
    "load_glove_embeddings",
    "cosine_similarity",
    "train_model",
]
