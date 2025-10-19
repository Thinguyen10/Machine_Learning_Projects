"""training.py

Thin wrappers that encapsulate model training calls. Move heavy logic out of the
Streamlit UI file so `app.py` stays focused on the interface.
"""
from typing import Any, Dict, Optional


def train_sklearn(X_train, X_test, y_train, y_test, model_path: Optional[str] = None) -> Dict[str, Any]:
    """Train a sklearn LogisticRegression model and return metrics dict.

    This wrapper exists so the UI can import a single clear symbol.
    """
    # import trainer lazily to avoid heavy import-time dependencies
    try:
        from backend.model import train_sklearn as _train_sklearn
    except Exception:
        from model import train_sklearn as _train_sklearn

    # _train_sklearn now returns (metrics, model)
    return _train_sklearn(X_train, X_test, y_train, y_test, model_path=model_path)


def train_keras(X_train, X_test, y_train, y_test, epochs: int = 5, batch_size: int = 32,
                model_path: Optional[str] = None) -> Dict[str, Any]:
    """Train the Keras model and return metrics dict.

    This calls into the lazy-loading train_keras defined in `model.py`.
    """
    try:
        from backend.model import train_keras as _train_keras
    except Exception:
        from model import train_keras as _train_keras

    # _train_keras now returns (metrics, model)
    return _train_keras(X_train, X_test, y_train, y_test, epochs=epochs, batch_size=batch_size, model_path=model_path)
