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
from pathlib import Path
import csv
import pandas as pd

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

epochs = st.sidebar.slider("Number of Epochs", min_value=3, max_value=70, value=10, step=5)
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
# STEP 1.5: Auto-generate CSV labels
# ----------------------------
def find_split_folder(base: Path):
    """Find subfolder that contains Train/Test structure."""
    for child in base.rglob('*'):
        if child.is_dir() and child.name.lower() in ['train', 'test']:
            return child.parent
    return base  # fallback if not found

DATA_ROOT = find_split_folder(DATA_DIR)
st.info(f"Detected dataset root: {DATA_ROOT}")

# ----------------------------
# STEP 1.6: Generate label CSVs (Train/Test)
# ----------------------------
def scan_folder(root_dir: Path):
    """Return (relative_path, label) pairs for images under subfolders."""
    data = []
    for label_dir in sorted(root_dir.iterdir()):
        if not label_dir.is_dir():
            continue
        label = label_dir.name
        for img_file in sorted(label_dir.rglob('*')):
            if img_file.suffix.lower() in IMAGE_EXTS:
                rel_path = str(img_file.relative_to(DATA_ROOT.parent))
                data.append((rel_path, label))
    return data

def write_csv(data, out_csv):
    with open(out_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['filepath', 'label'])
        writer.writerows(data)
    st.write(f"✅ Wrote {len(data)} entries → {out_csv}")

csv_files = []
for split_name in ['Train', 'Test', 'train', 'test']:
    split_dir = DATA_ROOT / split_name
    if split_dir.exists():
        st.write(f"📂 Scanning {split_dir} ...")
        data = scan_folder(split_dir)
        if len(data) == 0:
            st.warning(f"No images found in {split_dir}")
            continue
        csv_name = f"{split_name.lower()}_labels.csv"
        write_csv(data, csv_name)
        df = pd.read_csv(csv_name)
        st.write(df.head())
        csv_files.append(csv_name)

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
            # Build CNN model
            num_classes = len(st.session_state.train_gen.class_indices)

            # Safety: refuse to create model if dataset has fewer than 2 classes
            if num_classes < 2:
                st.error(
                    f"Detected only {num_classes} class in the dataset.\n"
                    "Please ensure your dataset has at least two class subfolders (e.g. 'benign', 'malignant') or provide a labels CSV."
                )
            else:
                # set optimizer according to user choice
                optimizer = Adam(learning_rate=learning_rate) if optimizer_choice == "Adam" else SGD(learning_rate=learning_rate, momentum=0.9)

                # apply selected optimizer & create model (model is compiled inside create_cnn_model)
                model = create_cnn_model(input_shape=(*img_size, 3), num_classes=num_classes, optimizer=optimizer)

                st.success("CNN model created successfully! Train CNN Model Now.")
                st.session_state['model'] = model

# ----------------------------
# STEP 4: Train the CNN
# ----------------------------
if "model" in st.session_state:
    st.subheader("Step 3: Train the CNN Model")
    if st.button("Train CNN"):
        if "train_gen" not in st.session_state or "val_gen" not in st.session_state:
            st.error("Please load the dataset first!")
        else:
            st.info("Training the CNN... please wait ⏳")  # blue info box
            
        # Train the CNN and store training history
        history_dict = {"loss": [], "accuracy": [], "val_loss": [], "val_accuracy": []}

        progress_bar = st.progress(0)
        status_text = st.empty()
        metrics_placeholder = st.empty()

        # Train for the specified number of epochs SELCTED
        for epoch in range(epochs):
            status_text.text(f"Training epoch {epoch+1}/{epochs}…")

            # CORE TRAINING STEP
            # - st.session_state.model → your CNN
            # - .fit(...) trains for ONE epoch
            # - train_gen → training data batches
            # - val_gen → vvalidation images (used only to check performance, not update weights).
            # - epochs=1 ensures only one epoch at a time (so we can update UI)
            # - verbose=0 hides console logs (Streamlit handles progress instead)
            hist = st.session_state.model.fit(
                st.session_state.train_gen,
                validation_data=st.session_state.val_gen,
                epochs=1,
                verbose=0
            )
            # hist is a Keras History object
            # hist.history looks like:
            # {
            #   'loss': [0.45],
            #   'accuracy': [0.82],
            #   'val_loss': [0.52],
            #   'val_accuracy': [0.79]
            # }
            # Each list has 1 element because we train for 1 epoch at a time.

            # Save metrics to our custom history_dict
            #val_loss - how bad the predictions are on validation set
            #val_accuracy - how many predictions are correct on validation set
            history_dict["loss"].append(hist.history["loss"][0])
            history_dict["accuracy"].append(hist.history["accuracy"][0])
            history_dict["val_loss"].append(hist.history["val_loss"][0])
            history_dict["val_accuracy"].append(hist.history["val_accuracy"][0])

            # Update progress bar
            progress_bar.progress((epoch+1)/epochs)

            # Update live metrics in the app
            metrics_placeholder.write({
                "Epoch": epoch + 1,
                "Train Loss": round(hist.history['loss'][0], 4),
                "Train Accuracy": round(hist.history['accuracy'][0], 4),
                "Val Loss": round(hist.history['val_loss'][0], 4),
                "Val Accuracy": round(hist.history['val_accuracy'][0], 4),
            })

        st.success("✅ Training complete!")
        st.write("Training Performance:")
        # Debug: show the numeric history to inspect why plots look flat
        st.write("Raw history dict:", history_dict)
        # Persist history in session state for further inspection
        st.session_state['history_dict'] = history_dict

        # Save history to a JSON file for offline inspection
        try:
            with open('training_history.json', 'w', encoding='utf-8') as f:
                json.dump(history_dict, f, ensure_ascii=False, indent=2)
            st.info("Saved training history to training_history.json")
        except Exception as e:
            st.warning(f"Could not save training history to file: {e}")

        # Plot using the persisted history
        st.pyplot(plot_training_history(st.session_state['history_dict']))