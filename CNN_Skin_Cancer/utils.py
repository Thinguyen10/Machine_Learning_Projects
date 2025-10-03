import matplotlib.pyplot as plt

def plot_training_history(history_dict):
    """
    Plot training and validation loss & accuracy.

    Accepts either:
    - Keras History object (.history)
    - Or dict with keys: 'loss', 'accuracy', 'val_loss', 'val_accuracy'
    """
    if hasattr(history_dict, 'history'):
        h = history_dict.history
    else:
        h = history_dict  # assume dict

    fig, ax = plt.subplots(1, 2, figsize=(12, 4))

    # Plot Loss
    ax[0].plot(h['loss'], label='Train Loss')
    ax[0].plot(h['val_loss'], label='Val Loss')
    ax[0].set_title("Loss over Epochs")
    ax[0].set_xlabel("Epochs")
    ax[0].set_ylabel("Loss")
    ax[0].legend()

    # Plot Accuracy
    ax[1].plot(h['accuracy'], label='Train Accuracy')
    ax[1].plot(h['val_accuracy'], label='Val Accuracy')
    ax[1].set_title("Accuracy over Epochs")
    ax[1].set_xlabel("Epochs")
    ax[1].set_ylabel("Accuracy")
    ax[1].legend()

    return fig
