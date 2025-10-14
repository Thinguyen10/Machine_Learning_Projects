"""training.py

Thin wrappers that encapsulate model training calls. Move heavy logic out of the
Streamlit UI file so `app.py` stays focused on the interface.
"""
from typing import Any, Dict, Optional

from model import train_sklearn as _train_sklearn, train_keras as _train_keras


def train_sklearn(X_train, X_test, y_train, y_test, model_path: Optional[str] = None) -> Dict[str, Any]:
    """Train a sklearn LogisticRegression model and return metrics dict.

    This wrapper exists so the UI can import a single clear symbol.
    """
    return _train_sklearn(X_train, X_test, y_train, y_test, model_path=model_path)


def train_keras(X_train, X_test, y_train, y_test, epochs: int = 5, batch_size: int = 32,
                model_path: Optional[str] = None) -> Dict[str, Any]:
    """Train the Keras model and return metrics dict.

    This calls into the lazy-loading train_keras defined in `model.py`.
    """
    return _train_keras(X_train, X_test, y_train, y_test, epochs=epochs, batch_size=batch_size, model_path=model_path)
