"""
model.py

Utilities to train a simple NLP classifier on TF-IDF features produced by
`processing.process`. Two trainers are provided:
 - train_sklearn: fast, uses LogisticRegression and works with sparse matrices
 - train_keras: a small dense neural network (converts sparse -> dense)

Usage (basic):
    python model.py --csv data.csv --backend sklearn
    python model.py --csv data.csv --backend keras --epochs 5

The file also exposes save/load helpers for the model and vectorizer.
"""

from typing import Tuple, Optional
import argparse
import joblib
import os

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Defer importing TensorFlow until it's actually needed to avoid import-time
# side-effects (useful when running under process managers like Streamlit).

from processing import process


def train_sklearn(X_train, X_test, y_train, y_test, model_path: Optional[str] = None) -> dict:
    """Train a LogisticRegression on sparse TF-IDF features.

    This supports sparse matrices directly and is fast for text classification.
    Returns a metrics dict and saves the model if model_path is provided.
    """
    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_train, y_train)

    preds = clf.predict(X_test)
    acc = accuracy_score(y_test, preds)
    report = classification_report(y_test, preds, output_dict=True)

    if model_path:
        joblib.dump(clf, model_path)

    metrics = {"accuracy": acc, "report": report}
    # Return metrics and trained model so callers (e.g., UI) can persist or use it
    return metrics, clf


def build_keras_model(input_dim: int) -> object:
    try:
        import tensorflow as tf
    except Exception as e:
        raise ImportError("TensorFlow is required for Keras models but could not be imported: {}".format(e))

    model = tf.keras.Sequential([
        tf.keras.layers.InputLayer(input_shape=(input_dim,)),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(1, activation="sigmoid")
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
                  loss="binary_crossentropy",
                  metrics=["accuracy"])
    return model


def train_keras(X_train, X_test, y_train, y_test, epochs: int = 5, batch_size: int = 32,
                model_path: Optional[str] = None) -> dict:
    """Train a small Keras model on dense arrays.

    Note: this will convert sparse TF-IDF matrices to dense arrays which
    can use a lot of memory for large feature sets. For small-to-medium datasets
    it's fine.
    """
    # Convert to dense if necessary
    if hasattr(X_train, "toarray"):
        X_train_arr = X_train.toarray()
        X_test_arr = X_test.toarray()
    else:
        X_train_arr = np.asarray(X_train)
        X_test_arr = np.asarray(X_test)

    # Convert labels to numeric if they are strings/objects
    # (e.g., 'positive'/'negative' -> 0/1)
    y_train_arr = np.asarray(y_train)
    y_test_arr = np.asarray(y_test)
    
    if y_train_arr.dtype == object or not np.issubdtype(y_train_arr.dtype, np.number):
        from sklearn.preprocessing import LabelEncoder
        le = LabelEncoder()
        y_train_arr = le.fit_transform(y_train_arr)
        y_test_arr = le.transform(y_test_arr)

    model = build_keras_model(X_train_arr.shape[1])

    model.fit(X_train_arr, y_train_arr, validation_split=0.1, epochs=epochs, batch_size=batch_size, verbose=2)
    loss, acc = model.evaluate(X_test_arr, y_test_arr, verbose=0)

    if model_path:
        model.save(model_path)

    metrics = {"loss": float(loss), "accuracy": float(acc)}
    # Return metrics and the trained Keras model as well
    return metrics, model


def save_vectorizer(vectorizer, path: str):
    joblib.dump(vectorizer, path)


def load_vectorizer(path: str):
    return joblib.load(path)


def save_model(model, path: str, backend: str = "sklearn"):
    """Save a trained model.

    - For sklearn, `joblib.dump` is used (path typically ends with .joblib)
    - For keras, `model.save(path)` will write a SavedModel directory
    """
    if backend == "sklearn":
        joblib.dump(model, path)
    elif backend == "keras":
        # Keras models expose a .save() method
        model.save(path)
    else:
        raise ValueError("Unknown backend for save_model: %s" % backend)


def load_model(path: str, backend: str = "sklearn"):
    """Load a model saved with `save_model`.

    Returns the loaded model object.
    """
    if backend == "sklearn":
        return joblib.load(path)
    elif backend == "keras":
        try:
            import tensorflow as tf
        except Exception as e:
            raise ImportError("TensorFlow required to load Keras model: %s" % e)
        return tf.keras.models.load_model(path)
    else:
        raise ValueError("Unknown backend for load_model: %s" % backend)


def predict_text(model, vect, text: str, backend: str = "sklearn"):
    """Predict a single text string and return (label, prob).

    - `vect` should be a fitted vectorizer (e.g., TfidfVectorizer) loaded with `load_vectorizer`.
    - For sklearn: returns model.predict(X)[0] and positive-class probability (if available).
    - For keras: returns 0/1 label and probability (sigmoid output assumed).
    """
    proc_text = text if isinstance(text, str) else str(text)
    X = vect.transform([proc_text])

    if backend == "sklearn":
        # Predict label
        label = model.predict(X)[0]
        prob = None
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(X)[0]
            # Try to pick probability of positive class if label space is {0,1}
            if hasattr(model, 'classes_'):
                classes = list(model.classes_)
                if 1 in classes:
                    pos_idx = classes.index(1)
                else:
                    pos_idx = 1 if len(classes) > 1 else 0
            else:
                pos_idx = 1 if len(probs) > 1 else 0
            prob = float(probs[pos_idx])
        return label, prob

    elif backend == "keras":
        if hasattr(X, 'toarray'):
            X_in = X.toarray()
        else:
            X_in = X
        preds = model.predict(X_in)
        prob = float(preds[0][0])
        label = 1 if prob >= 0.5 else 0
        return label, prob

    else:
        raise ValueError("Unknown backend for predict_text: %s" % backend)


def main():
    parser = argparse.ArgumentParser(description="Train an NLP model on a CSV using TF-IDF preprocessing.")
    parser.add_argument("--csv", default="data.csv", help="CSV file path (expects 'Body' and 'Label')")
    parser.add_argument("--backend", choices=["sklearn", "keras"], default="sklearn")
    parser.add_argument("--model-out", default=None, help="Optional path to save trained model")
    parser.add_argument("--vect-out", default=None, help="Optional path to save fitted vectorizer (joblib)")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    vect, Xtr, Xte, ytr, yte = process(args.csv, text_col="Body", label_col="Label")

    if args.vect_out:
        save_vectorizer(vect, args.vect_out)
        print(f"Saved vectorizer to {args.vect_out}")

    print("Train shape:", Xtr.shape, "Test shape:", Xte.shape)

    if args.backend == "sklearn":
        metrics = train_sklearn(Xtr, Xte, ytr, yte, model_path=args.model_out)
        print("Sklearn results:", metrics["accuracy"])
    else:
        metrics = train_keras(Xtr, Xte, ytr, yte, epochs=args.epochs, batch_size=args.batch_size, model_path=args.model_out)
        print("Keras results:", metrics)


if __name__ == "__main__":
    main()
