"""
train_predict.py

Train the LSTM model with ModelCheckpoint and EarlyStopping, save it, and make predictions
(next-word prediction or sentence completion).

Key functions:
- train_and_save: Train the model using the training helper from lstm_model and save the final model.
- save_training_artifacts: Save model + tokenizer + metadata using pickle for easy reload.
- load_training_artifacts: Load saved model, tokenizer, and metadata from disk.
- load_trained_model: Load a saved Keras model from disk.
- predict_next_word: Given a sequence of word indices, predict the next word.
- generate_sentence: Given a starting sequence (seed text), generate n words by repeatedly predicting the next word.

Notes:
- The model must be compiled before training (handled by lstm_model.build_lstm_model).
- Use ModelCheckpoint and EarlyStopping callbacks during training (handled by lstm_model.train_model).
- For prediction, you need the tokenizer's word_index (to map words -> indices) and an index_word
  mapping (to map predicted indices -> words).
- Use save_training_artifacts to save everything (model + tokenizer + config) in one call to avoid retraining.
"""
from typing import List, Dict, Optional, Tuple, Any
import numpy as np
import pickle
import os
import json

try:
    from tensorflow.keras.models import load_model
    from tensorflow.keras.preprocessing.sequence import pad_sequences
except Exception:
    load_model = None  # type: ignore
    pad_sequences = None  # type: ignore


def train_and_save(
    model,
    X_train,
    y_train,
    X_val,
    y_val,
    epochs: int = 20,
    batch_size: int = 128,
    checkpoint_path: str = "best_model.h5",
    final_model_path: str = "final_model.h5",
    patience: int = 3,
):
    """
    Train the model with ModelCheckpoint and EarlyStopping and save the final trained model.

    This function wraps the train_model function from lstm_model (which uses callbacks) and then
    saves the final model after training completes.

    Args:
        model: Compiled Keras model.
        X_train, y_train: Training feature and label arrays.
        X_val, y_val: Validation feature and label arrays.
        epochs: Maximum number of training epochs.
        batch_size: Training batch size.
        checkpoint_path: Path where the best model (by val_loss) is saved during training.
        final_model_path: Path to save the final model after training ends.
        patience: EarlyStopping patience (epochs with no improvement).

    Returns:
        history: Keras History object from training.

    Notes:
        - ModelCheckpoint saves the best model (by validation loss) to `checkpoint_path`.
        - EarlyStopping stops training early if validation loss doesn't improve for `patience` epochs
          and restores the best weights.
        - After training, the model (with best weights restored) is saved to `final_model_path`.
    """
    # Import train_model from lstm_model to use callbacks
    from lstm_model import train_model

    # Train the model using callbacks (ModelCheckpoint + EarlyStopping)
    history = train_model(
        model=model,
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        epochs=epochs,
        batch_size=batch_size,
        checkpoint_path=checkpoint_path,
        patience=patience,
    )

    # Save the final trained model (with best weights restored by EarlyStopping)
    model.save(final_model_path)
    print(f"Final trained model saved to {final_model_path}")

    return history


def save_training_artifacts(
    model,
    tokenizer,
    window_size: int,
    save_dir: str = "trained_model",
    model_name: str = "lstm_model.h5",
    metadata: Optional[Dict[str, Any]] = None,
):
    """
    Save model, tokenizer, and metadata to disk for easy reloading (avoids retraining).

    This function saves:
    - Keras model (.h5 file)
    - Tokenizer (pickle file)
    - Metadata/config (JSON file with window_size, vocab_size, etc.)

    Args:
        model: Trained Keras model.
        tokenizer: Keras Tokenizer used during training.
        window_size: Window size (n) used for input sequences.
        save_dir: Directory to save all artifacts.
        model_name: Filename for the model (.h5).
        metadata: Optional dict with additional metadata (e.g., training history, hyperparams).

    Returns:
        None
    """
    # Create save directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)

    # Save model
    model_path = os.path.join(save_dir, model_name)
    model.save(model_path)
    print(f"✓ Model saved to {model_path}")

    # Save tokenizer using pickle
    tokenizer_path = os.path.join(save_dir, "tokenizer.pkl")
    with open(tokenizer_path, "wb") as f:
        pickle.dump(tokenizer, f)
    print(f"✓ Tokenizer saved to {tokenizer_path}")

    # Save metadata as JSON
    config = {
        "window_size": window_size,
        "vocab_size": len(tokenizer.word_index) + 1,
        "num_words": tokenizer.num_words,
    }
    if metadata:
        config.update(metadata)

    config_path = os.path.join(save_dir, "config.json")
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    print(f"✓ Config saved to {config_path}")

    print(f"\n✅ All training artifacts saved to '{save_dir}/' directory")
    print(f"   To reload: use load_training_artifacts('{save_dir}')")


def load_training_artifacts(save_dir: str = "trained_model") -> Tuple[Any, Any, Dict]:
    """
    Load saved model, tokenizer, and metadata from disk (avoids retraining).

    Args:
        save_dir: Directory containing saved artifacts.

    Returns:
        model: Loaded Keras model.
        tokenizer: Loaded Keras Tokenizer.
        config: Dictionary with metadata (window_size, vocab_size, etc.).
    """
    if load_model is None:
        raise ImportError("TensorFlow / Keras not available. Install tensorflow to load models.")

    # Load model
    model_files = [f for f in os.listdir(save_dir) if f.endswith(".h5")]
    if not model_files:
        raise FileNotFoundError(f"No .h5 model file found in {save_dir}")
    model_path = os.path.join(save_dir, model_files[0])
    model = load_model(model_path)
    print(f"✓ Model loaded from {model_path}")

    # Load tokenizer
    tokenizer_path = os.path.join(save_dir, "tokenizer.pkl")
    with open(tokenizer_path, "rb") as f:
        tokenizer = pickle.load(f)
    print(f"✓ Tokenizer loaded from {tokenizer_path}")

    # Load config
    config_path = os.path.join(save_dir, "config.json")
    with open(config_path, "r") as f:
        config = json.load(f)
    print(f"✓ Config loaded from {config_path}")

    print(f"\n✅ All artifacts loaded from '{save_dir}/' directory")
    print(f"   Window size: {config.get('window_size')}, Vocab size: {config.get('vocab_size')}")

    return model, tokenizer, config


def load_trained_model(model_path: str):
    """
    Load a saved Keras model from disk.

    Args:
        model_path: Path to the saved .h5 or SavedModel directory.

    Returns:
        model: Loaded Keras model ready for inference.
    """
    if load_model is None:
        raise ImportError("TensorFlow / Keras not available. Install tensorflow to load models.")
    model = load_model(model_path)
    print(f"Model loaded from {model_path}")
    return model


def predict_next_word(
    model,
    seed_sequence: List[int],
    index_word: Dict[int, str],
    top_k: int = 1,
) -> List[Tuple[str, float]]:
    """
    Predict the next word given a seed sequence of word indices.

    Args:
        model: Trained Keras model.
        seed_sequence: List of integer word indices (the input context). Should match the model's
                       expected input length (window_size).
        index_word: Mapping from integer index to word string (inverse of tokenizer.word_index).
        top_k: Number of top predictions to return (sorted by probability descending).

    Returns:
        List of tuples (word, probability) for the top_k most likely next words.

    Notes:
        - If seed_sequence is shorter than the model's input length, it should be padded (or you
          can handle padding here).
        - The model outputs a probability distribution over the vocabulary; we take the top_k.
    """
    # Ensure seed_sequence is a 2D array with shape (1, window_size)
    seed_array = np.array([seed_sequence], dtype=np.int32)

    # Get model prediction (shape: (1, vocab_size))
    predictions = model.predict(seed_array, verbose=0)
    probs = predictions[0]  # shape: (vocab_size,)

    # Get top_k indices sorted by probability (descending)
    top_indices = np.argsort(probs)[::-1][:top_k]

    results = []
    for idx in top_indices:
        word = index_word.get(idx, "<UNK>")
        prob = float(probs[idx])
        results.append((word, prob))

    return results


def generate_sentence(
    model,
    seed_text: str,
    tokenizer,
    window_size: int,
    num_words: int = 10,
    temperature: float = 1.0,
) -> str:
    """
    Generate a sentence by iteratively predicting the next word and appending it to the context.

    Args:
        model: Trained Keras model for next-word prediction.
        seed_text: Starting text (a few words to seed the generation).
        tokenizer: Keras Tokenizer used during training (has word_index and can convert text to sequences).
        window_size: Number of words the model uses as input (n).
        num_words: Number of words to generate.
        temperature: Sampling temperature (1.0 = normal, <1.0 = more confident, >1.0 = more random).
                     Lower temperature makes predictions more deterministic.

    Returns:
        generated_text: The seed text followed by the generated words.

    Notes:
        - The function converts seed_text to token indices, takes the last `window_size` tokens as context,
          predicts the next word, appends it, and repeats.
        - Temperature sampling: instead of taking argmax, sample from the probability distribution scaled
          by temperature. Temperature=1.0 means no change; <1.0 makes high-prob words more likely.
    """
    # Build index_word mapping from tokenizer
    index_word = {idx: word for word, idx in tokenizer.word_index.items()}

    # Convert seed text to token indices
    from data_processing import remove_punctuation_and_split

    # Clean and tokenize seed text using the same preprocessing
    token_lists = remove_punctuation_and_split([seed_text])
    seed_tokens = token_lists[0] if token_lists else []

    # Convert tokens to indices using tokenizer
    seed_indices = []
    for token in seed_tokens:
        idx = tokenizer.word_index.get(token, 0)  # 0 if unknown
        if idx != 0:  # skip padding/unknown for seed
            seed_indices.append(idx)

    # If seed is shorter than window_size, pad with zeros at the start or handle gracefully
    # For simplicity, if seed is too short, we pad; if too long, take the last window_size tokens.
    if len(seed_indices) < window_size:
        # Pad at the beginning with zeros
        seed_indices = [0] * (window_size - len(seed_indices)) + seed_indices
    else:
        # Take the last window_size tokens
        seed_indices = seed_indices[-window_size:]

    generated_tokens = list(seed_indices)

    # Generate num_words tokens
    for _ in range(num_words):
        # Take the last window_size tokens as input
        current_window = generated_tokens[-window_size:]
        current_array = np.array([current_window], dtype=np.int32)

        # Predict next word probabilities
        predictions = model.predict(current_array, verbose=0)
        probs = predictions[0]  # shape: (vocab_size,)

        # Apply temperature scaling
        if temperature != 1.0:
            probs = np.log(probs + 1e-10) / temperature  # avoid log(0)
            probs = np.exp(probs)
            probs = probs / np.sum(probs)

        # Sample from the distribution (or take argmax if temperature is very low)
        next_idx = np.random.choice(len(probs), p=probs)

        # Append predicted token
        generated_tokens.append(next_idx)

    # Convert generated tokens back to words (skip padding tokens for output)
    generated_words = []
    for idx in generated_tokens:
        if idx == 0:
            continue  # skip padding
        word = index_word.get(idx, "<UNK>")
        generated_words.append(word)

    return " ".join(generated_words)


__all__ = [
    "train_and_save",
    "save_training_artifacts",
    "load_training_artifacts",
    "load_trained_model",
    "predict_next_word",
    "generate_sentence",
]
