"""tuning.py

Utilities to help tune and compare Keras models for the NLP project. Features:
 - epoch_sweep: train for many epochs and fit a quadratic peak to val-accuracy to find a 'sweet spot'
 - grid_search_keras: simple grid-search loop (lightweight alternative to KerasTuner)
 - helpers to build a default dense model; example build functions for CNN/RNN (illustrative)

This module keeps dependencies optional and provides informative return dicts so
you can integrate it with the existing `backend/model.py` pipeline.
"""
from typing import Callable, Dict, Iterable, Tuple, Any
import numpy as np
import logging
import matplotlib.pyplot as plt
from typing import Optional

try:
    from sklearn.model_selection import ParameterGrid
except Exception:
    # ParameterGrid is small and commonly available with scikit-learn in this repo
    from sklearn.model_selection import ParameterGrid

logger = logging.getLogger(__name__)


def _to_dense(X):
    if hasattr(X, "toarray"):
        return X.toarray()
    return np.asarray(X)


def fit_quadratic_peak(metric_values: Iterable[float]) -> Dict[str, Any]:
    """Fit a quadratic (parabola) to metric_values (y over epochs) and return the vertex.

    We fit y = ax^2 + bx + c. If a < 0 the parabola is concave (has a maximum).
    Vertex at x* = -b/(2a). We return the fitted params and the estimated peak epoch
    and value. Epochs are 1-indexed for user convenience.
    """
    y = np.asarray(list(metric_values), dtype=float)
    x = np.arange(1, len(y) + 1, dtype=float)
    if len(y) < 3:
        return {"fit_ok": False, "reason": "need >= 3 points to fit quadratic", "length": len(y)}

    coeffs = np.polyfit(x, y, 2)  # [a, b, c]
    a, b, c = coeffs
    result = {"a": float(a), "b": float(b), "c": float(c), "fit_ok": True}
    if a >= 0:
        # parabola opens upward -> no concave maximum
        result.update({"has_peak": False, "reason": "parabola concave up (a>=0)"})
        return result

    x_star = -b / (2 * a)
    # clamp to range of epochs
    x_star_clamped = float(np.clip(x_star, x[0], x[-1]))
    peak_val = float(np.polyval(coeffs, x_star_clamped))
    result.update({"has_peak": True, "peak_epoch": x_star_clamped, "peak_val": peak_val})
    return result


def epoch_sweep(build_fn: Callable[[int], Any],
                X_train, y_train,
                X_val, y_val,
                max_epochs: int = 50,
                batch_size: int = 32,
                verbose: int = 1) -> Dict[str, Any]:
    """Train a model (built by build_fn) for `max_epochs` and return history + peak analysis.

    build_fn(input_dim) -> compiled Keras model. X_* may be sparse (TF-IDF) or dense.

    Returns dict with:
      - 'history': list of dicts per epoch (loss, accuracy, val_loss, val_accuracy)
      - 'peak_analysis': output of fit_quadratic_peak on validation accuracy
      - 'best_epoch': int (1-indexed) corresponding to max val_accuracy observed
    """
    try:
        import tensorflow as tf
    except Exception as e:
        raise ImportError("TensorFlow is required for epoch_sweep: %s" % e)

    Xtr = _to_dense(X_train)
    Xv = _to_dense(X_val)
    ytr = np.asarray(y_train)
    yv = np.asarray(y_val)

    input_dim = Xtr.shape[1]
    model = build_fn(input_dim)

    # Train and capture history
    h = model.fit(Xtr, ytr, validation_data=(Xv, yv), epochs=max_epochs, batch_size=batch_size, verbose=verbose)

    hist = h.history
    # convert history to per-epoch dict list
    per_epoch = []
    n_epochs = len(hist.get('loss', []))
    for i in range(n_epochs):
        per_epoch.append({k: float(hist[k][i]) for k in hist.keys()})

    # analyze val_accuracy (support both 'val_accuracy' and 'val_acc' depending on TF version)
    val_acc_key = 'val_accuracy' if 'val_accuracy' in hist else ('val_acc' if 'val_acc' in hist else None)
    if val_acc_key is None:
        # fallback: can't analyze
        peak_analysis = {"fit_ok": False, "reason": "no validation accuracy in history"}
        best_epoch = None
    else:
        val_acc = hist[val_acc_key]
        peak_analysis = fit_quadratic_peak(val_acc)
        # best observed epoch
        best_idx = int(np.argmax(val_acc))
        best_epoch = best_idx + 1

    return {"history": per_epoch, "peak_analysis": peak_analysis, "best_epoch": best_epoch, "model": model}


def plot_learning_curves(history: Any, ax: Optional[plt.Axes] = None, show: bool = False, save_path: Optional[str] = None):
    """Plot training and validation loss/accuracy from a Keras history dict or a list of per-epoch dicts.

    - history can be the `history` dict attribute returned by Keras (with keys like 'loss', 'accuracy', 'val_loss', 'val_accuracy'),
      or a list of dicts per epoch as returned by `epoch_sweep` in the 'history' field.
    """
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(8, 4))
    else:
        fig = ax.figure

    # normalize history format
    if isinstance(history, list):
        # list of per-epoch dicts
        keys = set().union(*(h.keys() for h in history))
        hist = {k: [h.get(k, None) for h in history] for k in keys}
    elif isinstance(history, dict):
        hist = history
    else:
        raise ValueError("Unsupported history format")

    # plot accuracy if available
    acc_key = 'accuracy' if 'accuracy' in hist else ('acc' if 'acc' in hist else None)
    val_acc_key = 'val_accuracy' if 'val_accuracy' in hist else ('val_acc' if 'val_acc' in hist else None)
    loss_key = 'loss' if 'loss' in hist else None
    val_loss_key = 'val_loss' if 'val_loss' in hist else None

    epochs = np.arange(1, len(next(iter(hist.values()))) + 1)

    if acc_key and val_acc_key:
        ax.plot(epochs, hist[acc_key], label='train_acc')
        ax.plot(epochs, hist[val_acc_key], label='val_acc')
    if loss_key and val_loss_key:
        ax2 = ax.twinx()
        ax2.plot(epochs, hist[loss_key], label='train_loss', color='0.5', linestyle='--')
        ax2.plot(epochs, hist[val_loss_key], label='val_loss', color='0.2', linestyle=':')
        ax2.set_ylabel('loss')

    ax.set_xlabel('epoch')
    ax.set_title('Learning curves')
    ax.legend(loc='upper left')

    if save_path:
        fig.savefig(save_path, bbox_inches='tight')
    if show:
        plt.show()
    return fig, ax


def plot_quadratic_fit(val_acc: Iterable[float], peak_analysis: Optional[Dict[str, Any]] = None,
                       ax: Optional[plt.Axes] = None, show: bool = False, save_path: Optional[str] = None):
    """Plot validation accuracy points and an optional quadratic fit described by peak_analysis.

    `peak_analysis` can be the dict returned by `fit_quadratic_peak` (containing 'a','b','c' etc.).
    """
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(6, 4))
    else:
        fig = ax.figure

    y = np.asarray(list(val_acc), dtype=float)
    x = np.arange(1, len(y) + 1)
    ax.plot(x, y, 'o-', label='val_acc')

    if peak_analysis and peak_analysis.get('fit_ok') and 'a' in peak_analysis:
        a, b, c = peak_analysis['a'], peak_analysis['b'], peak_analysis['c']
        xs = np.linspace(x[0], x[-1], 200)
        ys = a * xs ** 2 + b * xs + c
        ax.plot(xs, ys, '-', label='quadratic_fit')
        if peak_analysis.get('has_peak'):
            xstar = peak_analysis['peak_epoch']
            ystar = peak_analysis['peak_val']
            ax.axvline(xstar, color='r', linestyle='--', label=f'peak ~{xstar:.2f}')
            ax.scatter([xstar], [ystar], color='r')

    ax.set_xlabel('epoch')
    ax.set_ylabel('val_acc')
    ax.set_title('Validation accuracy and quadratic fit')
    ax.legend()

    if save_path:
        fig.savefig(save_path, bbox_inches='tight')
    if show:
        plt.show()
    return fig, ax


def grid_search_keras(build_fn_template: Callable[[Dict[str, Any], int], Any],
                      param_grid: Dict[str, Iterable[Any]],
                      X_train, y_train,
                      X_val, y_val,
                      epochs: int = 5,
                      batch_size: int = 32,
                      verbose: int = 0) -> Dict[str, Any]:
    """Lightweight grid search over hyperparameters.

    - build_fn_template(hparams, input_dim) -> compiled Keras model
    - param_grid is a dict accepted by sklearn.model_selection.ParameterGrid

    Returns dict: each param combo -> metrics (val_accuracy, val_loss), plus best params.
    """
    try:
        from sklearn.model_selection import ParameterGrid
    except Exception:
        from sklearn.model_selection import ParameterGrid

    Xtr = _to_dense(X_train)
    Xv = _to_dense(X_val)
    ytr = np.asarray(y_train)
    yv = np.asarray(y_val)

    input_dim = Xtr.shape[1]
    results = []
    for params in ParameterGrid(param_grid):
        logger.info("Testing params: %s", params)
        model = build_fn_template(params, input_dim)
        h = model.fit(Xtr, ytr, validation_data=(Xv, yv), epochs=epochs, batch_size=batch_size, verbose=verbose)
        hist = h.history
        val_acc_key = 'val_accuracy' if 'val_accuracy' in hist else ('val_acc' if 'val_acc' in hist else None)
        val_acc = float(hist[val_acc_key][-1]) if val_acc_key else None
        val_loss = float(hist['val_loss'][-1]) if 'val_loss' in hist else None
        results.append({"params": dict(params), "val_acc": val_acc, "val_loss": val_loss, "history": hist})

    # pick best by val_acc (ignore None)
    valid = [r for r in results if r['val_acc'] is not None]
    if len(valid) == 0:
        best = None
    else:
        best = max(valid, key=lambda r: r['val_acc'])

    return {"results": results, "best": best}


def run_kerastuner_search(hp_builder_fn: Callable, X_train=None, y_train=None, X_val=None, y_val=None,
                          tuner_type: str = 'Hyperband', max_trials: int = 20, executions_per_trial: int = 1,
                          epochs: int = 10, batch_size: int = 32, project_name: str = 'kt_project') -> Dict[str, Any]:
    """Run a KerasTuner search using a hp_builder_fn that accepts a HyperParameters object.

    Example hp_builder_fn signature:
        def hp_builder(hp):
            model = tf.keras.Sequential([...])
            hp.Choice('hidden', [32,64,128])
            return model

    tuner_type: one of 'Hyperband', 'Bayesian', 'Random'. Requires `keras_tuner` to be installed.
    Returns dict with tuner object, best_hyperparameters, and best_model.
    """
    try:
        import keras_tuner as kt
    except Exception as e:
        raise ImportError('keras-tuner is required for run_kerastuner_search: %s' % e)

    if tuner_type.lower() == 'hyperband':
        tuner = kt.Hyperband(hp_builder_fn, objective='val_accuracy', max_epochs=epochs,
                             executions_per_trial=executions_per_trial, directory='kt_dir', project_name=project_name)
    elif tuner_type.lower() == 'bayesian':
        tuner = kt.BayesianOptimization(hp_builder_fn, objective='val_accuracy', max_trials=max_trials,
                                        executions_per_trial=executions_per_trial, directory='kt_dir', project_name=project_name)
    else:
        tuner = kt.RandomSearch(hp_builder_fn, objective='val_accuracy', max_trials=max_trials,
                                executions_per_trial=executions_per_trial, directory='kt_dir', project_name=project_name)

    # Ensure arrays
    Xtr = _to_dense(X_train)
    Xv = _to_dense(X_val) if X_val is not None else None

    tuner.search(Xtr, y_train, validation_data=(Xv, y_val) if Xv is not None else None,
                 epochs=epochs, batch_size=batch_size)

    best_hps = tuner.get_best_hyperparameters(num_trials=1)
    best_models = tuner.get_best_models(num_models=1)
    result = {'tuner': tuner, 'best_hyperparameters': best_hps[0] if best_hps else None,
              'best_model': best_models[0] if best_models else None}
    return result


# -------------------- Example build functions --------------------
def build_default_dense(input_dim: int, hidden: int = 128, dropout: float = 0.5, lr: float = 1e-3):
    """Default dense model compatible with TF-IDF dense input vectors."""
    import tensorflow as tf
    model = tf.keras.Sequential([
        tf.keras.layers.InputLayer(input_shape=(input_dim,)),
        tf.keras.layers.Dense(hidden, activation='relu'),
        tf.keras.layers.Dropout(dropout),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(lr), loss='binary_crossentropy', metrics=['accuracy'])
    return model


def build_fn_from_params(params: Dict[str, Any], input_dim: int):
    """Small adapter: params may contain 'hidden', 'dropout', 'lr'."""
    hidden = int(params.get('hidden', 128))
    dropout = float(params.get('dropout', 0.5))
    lr = float(params.get('lr', 1e-3))
    return build_default_dense(input_dim, hidden=hidden, dropout=dropout, lr=lr)


def example_cnn_for_sequences(vocab_size: int, embed_dim: int = 64, max_len: int = 200):
    """Illustrative: a small text-CNN that expects integer token sequences, not TF-IDF.

    Use this only if you convert the pipeline to token sequences and use an Embedding layer.
    """
    import tensorflow as tf
    model = tf.keras.Sequential([
        tf.keras.layers.InputLayer(input_shape=(max_len,)),
        tf.keras.layers.Embedding(vocab_size, embed_dim, input_length=max_len),
        tf.keras.layers.Conv1D(128, 5, activation='relu'),
        tf.keras.layers.GlobalMaxPool1D(),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model


def example_rnn_for_sequences(vocab_size: int, embed_dim: int = 64, max_len: int = 200):
    """Illustrative: a small LSTM based model for token sequences.

    Like the CNN example, this requires sequence inputs and an embedding.
    """
    import tensorflow as tf
    model = tf.keras.Sequential([
        tf.keras.layers.InputLayer(input_shape=(max_len,)),
        tf.keras.layers.Embedding(vocab_size, embed_dim, input_length=max_len),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64)),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model
