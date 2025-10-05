# app.py - Main Streamlit application
# This file handles the user interface (UI) for training a CNN
# The dataset path is fixed (default) and users only interact with the process.

import streamlit as st
import kagglehub
import numpy as np
import tensorflow as tf
import json
import os
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.preprocessing import image
from pathlib import Path
import csv
import pandas as pd
import random

from data_processing import load_skin_cancer_data
from model import create_cnn_model
from train import train_and_evaluate
from utils import plot_training_history

from streamlit_frontpage import show_front_page

# ----------------------------
# FRONT PAGE
# ----------------------------
show_front_page()


# ----------------------------
# Sidebar: Training Parameters
# ----------------------------
st.sidebar.header("⚙️ Training Configuration")

epochs = st.sidebar.slider("Number of Epochs", min_value=3, max_value=70, value=10, step=2)
batch_size = st.sidebar.selectbox("Batch Size", [16, 32, 64, 128], index=1)
learning_rate = st.sidebar.selectbox("Learning Rate", [0.1, 0.01, 0.001, 0.0001], index=2)
optimizer_choice = st.sidebar.radio("Optimizer", ["Adam", "SGD"], index=0)
image_size_int = st.sidebar.selectbox("Image Size", [32, 64, 128], index=1)

# Convert to tuple for the data loader
img_size = (image_size_int, image_size_int)

# Choose optimizer
if optimizer_choice == "Adam":
    optimizer = Adam(learning_rate=learning_rate)
else:
    optimizer = SGD(learning_rate=learning_rate, momentum=0.9)


# ----------------------------
# STEP 1: Download dataset (default path, auto)
# ----------------------------
st.subheader("Dataset: jaiahuja/skin-cancer-detection")
st.write("The dataset will be downloaded automatically from KaggleHub.")

# Download dataset
path = kagglehub.dataset_download("jaiahuja/skin-cancer-detection")
st.session_state['path'] = path
DATA_DIR = Path(path)
IMAGE_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}

st.success(f"✅ Dataset downloaded to: {DATA_DIR}")


# ----------------------------
# STEP 1.5: Auto-detect dataset root and subfolders
# ----------------------------
def find_split_folder(base: Path):
    """Find the root folder that contains 'train' and 'test' subfolders."""
    for child in base.rglob('*'):
        if child.is_dir() and child.name.lower() in ['train', 'test']:
            return child.parent
    return base  # fallback if structure is flat

DATA_ROOT = find_split_folder(DATA_DIR)
st.info(f"Detected dataset root: {DATA_ROOT}")

# ----------------------------
# STEP 1.6: Generate label CSVs (Train/Test)
# ----------------------------

# def scan_folder(root_dir: Path):
#     """
#     Scan a folder with subfolders per class and return list of (relative_path, label).
#     Paths are relative to DATA_ROOT for compatibility with Keras.
#     """
#     data = []
#     for label_dir in sorted(root_dir.iterdir()):
#         if not label_dir.is_dir():
#             continue
#         label = label_dir.name
#         for img_file in sorted(label_dir.rglob('*')):
#             if img_file.suffix.lower() in IMAGE_EXTS:
#                 # Path relative to dataset root
#                 rel_path = str(img_file.relative_to(DATA_ROOT))
#                 data.append((rel_path, label))
#     return data

def scan_folder(root_dir: Path):
    """Return list of (absolute_path, label) tuples."""
    data = []
    for label_dir in sorted(root_dir.iterdir()):
        if not label_dir.is_dir():
            continue
        label = label_dir.name
        for img_file in sorted(label_dir.rglob('*')):
            if img_file.suffix.lower() in IMAGE_EXTS:
                # absolute path
                abs_path = str(img_file.resolve())
                data.append((abs_path, label))
    return data


def write_csv(data, out_csv: Path):
    """Write the list of (filepath, label) to a CSV."""
    with open(out_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['filepath', 'label'])
        writer.writerows(data)
    st.success(f"✅ Wrote {len(data)} entries → {out_csv}")

# Generate CSVs for all splits
csv_files = []
for split_name in ['Train', 'Test', 'train', 'test']:
    split_dir = DATA_ROOT / split_name
    if split_dir.exists():
        st.write(f"📂 Scanning {split_dir} ...")
        data = scan_folder(split_dir)
        if not data:
            st.warning(f"No images found in {split_dir}")
            continue

        # Save CSV inside dataset root so data_processing.py can find it
        csv_path = Path(path) / f"{split_name.lower()}_labels.csv"
        write_csv(data, csv_path)


        # Quick preview
        df = pd.read_csv(csv_path)
        st.write(df.head())
        csv_files.append(csv_path)

if not csv_files:
    st.error("No CSVs generated. Please verify the folder structure in the downloaded dataset.")

# ----------------------------
# STEP 2: Load dataset
# ----------------------------

if "path" in st.session_state:

    st.subheader("Step 1: Load Training and Validation Data")

    # Button to trigger dataset loading
    if st.button("Load Dataset"):

        # -----------------------------
        # Load data using the CSVs generated in Step 1.6
        # This ensures that all class labels are correctly read, even if the folder structure is nested
        # Returns three generators: training, validation, and test
        # -----------------------------
        train_gen, val_gen, test_gen = load_skin_cancer_data(
            path,             # Path to the dataset root
            img_size=img_size,   # Target size for images (tuple)
            batch_size=batch_size # Batch size for generators
        )
        # Store generators in session state for later use in training and evaluation
        st.session_state.train_gen = train_gen
        st.session_state.val_gen = val_gen
        st.session_state.test_gen = test_gen

        # -----------------------------
        # Preview a batch of training images
        # -----------------------------
        # Get the first batch of images and labels from the training generator
        X_batch, y_batch = next(train_gen)

            # Safety check: ensure batch is not empty
        if len(X_batch) == 0 or len(y_batch) == 0:
            st.error("Training generator is empty. Check CSV paths and image files.")
        else:
            labels_idx = y_batch.argmax(axis=1)

        # Extract class label names from the generator (e.g., ["Melanoma", "Nevus", ...])
        class_labels = list(train_gen.class_indices.keys())
        st.session_state['class_labels'] = class_labels  # store for later steps

        # Display the class labels detected
        st.write(f"Class labels detected: {class_labels}")

        # Convert one-hot encoded labels to integer indices
        labels_idx = y_batch.argmax(axis=1)

        # -----------------------------
        # Display first 5 images with their corresponding labels
        # -----------------------------
        st.write("Sample images with labels:")
        cols = st.columns(5)  # create 5 columns for side-by-side display
        for i, col in enumerate(cols):
            img = X_batch[i]                      # get image
            label_idx = labels_idx[i]             # get label index
            # Map label index to class name; fallback to "Unknown" if out of bounds
            label_name = class_labels[label_idx] if label_idx < len(class_labels) else "Unknown"
            col.image(img, width=100, caption=label_name)  # display image with label

# ----------------------------
# STEP 3: Create CNN Model
# ----------------------------
if "class_labels" in st.session_state:
    st.subheader("Step 2: Create CNN Model")

    if st.button("Create CNN Model"):
        if "train_gen" not in st.session_state:
            st.error("Please load the dataset first!")
        else:
            num_classes = len(st.session_state.train_gen.class_indices)

            if num_classes < 2:
                st.error(
                    f"Detected only {num_classes} class in the dataset.\n"
                    "Please ensure your dataset has at least two class subfolders (e.g. 'benign', 'malignant') or provide a labels CSV."
                )
            else:
                # Apply selected optimizer & create model
                model = create_cnn_model(input_shape=(*img_size, 3),
                                         num_classes=num_classes,
                                         optimizer=optimizer)
                st.session_state['model'] = model
                st.session_state['step3_done'] = True
                st.success("✅ CNN model created successfully! Continue to training.")


# ----------------------------
# STEP 4: Train CNN Model
# ----------------------------
if st.session_state.get("step3_done", False):
    st.subheader("Step 3: Train the CNN Model")

    if st.button("Train CNN"):
        if "train_gen" not in st.session_state or "val_gen" not in st.session_state:
            st.error("Please load the dataset first!")
        else:
            st.info("Training the CNN... please wait ⏳")

            history_dict = {"loss": [], "accuracy": [], "val_loss": [], "val_accuracy": []}
            progress_bar = st.progress(0)
            status_text = st.empty()
            metrics_placeholder = st.empty()

            for epoch in range(epochs):
                status_text.text(f"Training epoch {epoch+1}/{epochs}…")

                hist = st.session_state.model.fit(
                    st.session_state.train_gen,
                    validation_data=st.session_state.val_gen,
                    epochs=1,
                    verbose=0
                )

                history_dict["loss"].append(hist.history["loss"][0])
                history_dict["accuracy"].append(hist.history["accuracy"][0])
                history_dict["val_loss"].append(hist.history["val_loss"][0])
                history_dict["val_accuracy"].append(hist.history["val_accuracy"][0])

                progress_bar.progress((epoch+1)/epochs)

                metrics_placeholder.write({
                    "Epoch": epoch + 1,
                    "Train Loss": round(hist.history['loss'][0], 4),
                    "Train Accuracy": round(hist.history['accuracy'][0], 4),
                    "Val Loss": round(hist.history['val_loss'][0], 4),
                    "Val Accuracy": round(hist.history['val_accuracy'][0], 4),
                })

            st.success("✅ Training complete!")
            st.session_state['history_dict'] = history_dict
            st.session_state['step4_done'] = True

# ----------------------------
# STEP 5: Random Test Image Prediction
# ----------------------------
if st.session_state.get("step4_done", False) and "test_gen" in st.session_state:
    st.subheader("Step 4: Random Test Image Prediction")

    if st.button("Show Random Prediction"):
        import random
        test_gen = st.session_state.test_gen
        model = st.session_state.model

        # Pick a random batch index and image
        batch_index = random.randint(0, len(test_gen) - 1)
        X_batch, y_batch = test_gen[batch_index]

        img_idx = random.randint(0, X_batch.shape[0] - 1)
        img = X_batch[img_idx]
        true_label_idx = y_batch[img_idx].argmax()

        class_labels = list(test_gen.class_indices.keys())
        true_label_name = class_labels[true_label_idx]

        # Predict
        pred_probs = model.predict(img[np.newaxis, ...])
        pred_idx = pred_probs.argmax()
        pred_label = class_labels[pred_idx]
        pred_conf = pred_probs[0][pred_idx] * 100

        st.image(img, width=200,
                 caption=f"True: {true_label_name} | Predicted: {pred_label} ({pred_conf:.1f}%)")
        st.session_state['step5_done'] = True

# ----------------------------
# STEP 6: Display Training History
# ----------------------------
if st.session_state.get("step5_done", False):
    st.subheader("Step 5: Training Performance")

    st.write("Raw history dict:", st.session_state['history_dict'])

    try:
        with open('training_history.json', 'w', encoding='utf-8') as f:
            json.dump(st.session_state['history_dict'], f, ensure_ascii=False, indent=2)
        st.info("Saved training history to training_history.json")
    except Exception as e:
        st.warning(f"Could not save training history to file: {e}")

    st.pyplot(plot_training_history(st.session_state['history_dict']))
