"""
Train the skin cancer CNN model for 70 epochs and save to pickle file.
This script bypasses the Streamlit UI for faster training.
"""

import pickle
import json
import kagglehub
from pathlib import Path
import pandas as pd
from tensorflow.keras.optimizers import Adam

from data_processing import load_skin_cancer_data
from model import create_cnn_model
from utils import plot_training_history
import matplotlib.pyplot as plt

# ----------------------------
# Configuration
# ----------------------------
EPOCHS = 70
BATCH_SIZE = 32
LEARNING_RATE = 0.001
IMG_SIZE = (64, 64)
MODEL_SAVE_PATH = "trained_model_70epochs.pkl"
HISTORY_SAVE_PATH = "training_history_70epochs.json"

print("=" * 60)
print("SKIN CANCER CNN TRAINING - 70 EPOCHS")
print("=" * 60)

# ----------------------------
# Step 1: Download Dataset
# ----------------------------
print("\n[1/6] Downloading dataset from KaggleHub...")
path = kagglehub.dataset_download("jaiahuja/skin-cancer-detection")
DATA_DIR = Path(path)
print(f"✅ Dataset downloaded to: {DATA_DIR}")

# ----------------------------
# Step 2: Find dataset structure
# ----------------------------
print("\n[2/6] Finding dataset structure...")

def find_split_folder(base: Path):
    """Find the root folder that contains 'train' and 'test' subfolders."""
    for child in base.rglob('*'):
        if child.is_dir() and child.name.lower() in ['train', 'test']:
            return child.parent
    return base

DATA_ROOT = find_split_folder(DATA_DIR)
print(f"✅ Dataset root: {DATA_ROOT}")

# ----------------------------
# Step 3: Generate CSV labels
# ----------------------------
print("\n[3/6] Generating CSV labels...")

IMAGE_EXTS = {'.jpg', '.jpeg', '.png'}

def scan_folder(root_dir: Path):
    """Return list of (absolute_path, label) tuples."""
    data = []
    for label_dir in sorted(root_dir.iterdir()):
        if not label_dir.is_dir():
            continue
        label = label_dir.name
        for img_file in sorted(label_dir.rglob('*')):
            if img_file.suffix.lower() in IMAGE_EXTS:
                data.append((str(img_file.resolve()), label))
    return data

csv_data = {}
for split_dir in DATA_ROOT.iterdir():
    if split_dir.is_dir() and split_dir.name.lower() in ['train', 'test']:
        split_name = split_dir.name.lower()
        data = scan_folder(split_dir)
        if not data:
            print(f"⚠️  No images found in {split_dir}")
            continue
        df = pd.DataFrame(data, columns=['filepath', 'label'])
        csv_data[split_name] = df
        print(f"✅ {split_name.capitalize()} CSV: {len(df)} entries")

if 'train' not in csv_data or 'test' not in csv_data:
    raise ValueError("Could not find train and test folders in dataset!")

# ----------------------------
# Step 4: Load dataset
# ----------------------------
print("\n[4/6] Loading dataset into generators...")

train_gen, val_gen, test_gen = load_skin_cancer_data(
    df_train=csv_data['train'],
    df_test=csv_data['test'],
    img_size=IMG_SIZE,
    batch_size=BATCH_SIZE
)

class_labels = list(train_gen.class_indices.keys())
num_classes = len(class_labels)
print(f"✅ Classes detected: {class_labels}")
print(f"✅ Number of classes: {num_classes}")

# ----------------------------
# Step 5: Create and compile model
# ----------------------------
print("\n[5/6] Creating CNN model...")

optimizer = Adam(learning_rate=LEARNING_RATE)
model = create_cnn_model(
    input_shape=(*IMG_SIZE, 3),
    num_classes=num_classes,
    optimizer=optimizer
)

print("✅ Model created successfully!")
model.summary()

# ----------------------------
# Step 6: Train model
# ----------------------------
print(f"\n[6/6] Training model for {EPOCHS} epochs...")
print("This may take a while. Please keep your computer awake.")
print("-" * 60)

history = model.fit(
    train_gen,
    epochs=EPOCHS,
    validation_data=val_gen,
    verbose=1
)

print("\n" + "=" * 60)
print("✅ TRAINING COMPLETE!")
print("=" * 60)

# ----------------------------
# Evaluate on validation set
# ----------------------------
print("\nEvaluating on validation set...")
val_loss, val_accuracy = model.evaluate(val_gen)
print(f"Final Validation Loss: {val_loss:.4f}")
print(f"Final Validation Accuracy: {val_accuracy:.4f}")

# ----------------------------
# Save model to pickle file
# ----------------------------
print(f"\nSaving model to {MODEL_SAVE_PATH}...")
with open(MODEL_SAVE_PATH, 'wb') as f:
    pickle.dump(model, f)
print(f"✅ Model saved to {MODEL_SAVE_PATH}")

# ----------------------------
# Save training history
# ----------------------------
print(f"\nSaving training history to {HISTORY_SAVE_PATH}...")
history_dict = {
    'loss': history.history['loss'],
    'accuracy': history.history['accuracy'],
    'val_loss': history.history['val_loss'],
    'val_accuracy': history.history['val_accuracy']
}

with open(HISTORY_SAVE_PATH, 'w') as f:
    json.dump(history_dict, f, indent=2)
print(f"✅ Training history saved to {HISTORY_SAVE_PATH}")

# ----------------------------
# Plot and save training curves
# ----------------------------
print("\nGenerating training plots...")
fig = plot_training_history(history_dict)
plt.savefig('training_curves_70epochs.png', dpi=150, bbox_inches='tight')
print("✅ Training curves saved to training_curves_70epochs.png")

print("\n" + "=" * 60)
print("ALL DONE! 🎉")
print("=" * 60)
print(f"Model file: {MODEL_SAVE_PATH}")
print(f"History file: {HISTORY_SAVE_PATH}")
print(f"Plots file: training_curves_70epochs.png")
print("\nTo load the model later, use:")
print(f"  import pickle")
print(f"  with open('{MODEL_SAVE_PATH}', 'rb') as f:")
print(f"      model = pickle.load(f)")
print("=" * 60)
