import matplotlib.pyplot as plt
import numpy as np

def plot_training_history(history_dict):
    """
    Plot training and validation loss & accuracy.

    Accepts either:
    - Keras History object (.history)
    - Or dict with keys: 'loss', 'accuracy', 'val_loss', 'val_accuracy'
    """
    if hasattr(history_dict, 'history'):
        raw = history_dict.history
    else:
        raw = history_dict  # assume dict-like

    # Helper to safely extract numeric series from history entries
    def to_numeric_list(x):
        # x may be a list, scalar, numpy array, or nested list
        try:
            arr = np.asarray(x)
            # flatten and convert to float
            flat = arr.ravel().astype(float)
            return flat.tolist()
        except Exception:
            # fallback: try to iterate and cast
            try:
                return [float(v) for v in x]
            except Exception:
                return []

    loss = to_numeric_list(raw.get('loss', []))
    val_loss = to_numeric_list(raw.get('val_loss', []))
    acc = to_numeric_list(raw.get('accuracy', []))
    val_acc = to_numeric_list(raw.get('val_accuracy', []))

    # Determine number of epochs from the longest series
    n_epochs = max(len(loss), len(val_loss), len(acc), len(val_acc))
    epochs = list(range(1, n_epochs + 1)) if n_epochs > 0 else []

    fig, ax = plt.subplots(1, 2, figsize=(12, 4))

    # Plot Loss
    if loss:
        ax[0].plot(epochs[:len(loss)], loss, marker='o', label='Train Loss')
    if val_loss:
        ax[0].plot(epochs[:len(val_loss)], val_loss, marker='o', label='Val Loss')
    ax[0].set_title("Loss over Epochs")
    ax[0].set_xlabel("Epochs")
    ax[0].set_ylabel("Loss")
    if epochs:
        ax[0].set_xticks(epochs)
    ax[0].legend()

    # Plot Accuracy
    if acc:
        ax[1].plot(epochs[:len(acc)], acc, marker='o', label='Train Accuracy')
    if val_acc:
        ax[1].plot(epochs[:len(val_acc)], val_acc, marker='o', label='Val Accuracy')
    ax[1].set_title("Accuracy over Epochs")
    ax[1].set_xlabel("Epochs")
    ax[1].set_ylabel("Accuracy")
    if epochs:
        ax[1].set_xticks(epochs)
    ax[1].legend()

    fig.tight_layout()
    return fig
