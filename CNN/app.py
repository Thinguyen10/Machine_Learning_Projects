# app.py - Main Streamlit application
# This file handles the user interface (UI) for training a CNN
# The dataset path is fixed (default) and users only interact with the process.

import streamlit as st
import kagglehub
import numpy as np
from tensorflow.keras.optimizers import Adam, SGD

from data_processing import load_skin_cancer_data
from model import create_cnn_model
from train import train_and_evaluate
from utils import plot_training_history
from streamlit_frontpage import show_front_page

# ----------------------------
# FRONT PAGE
# ----------------------------
show_front_page()

st.markdown("---")  # Divider line

# ----------------------------
# Sidebar: Training Parameters
# ----------------------------
st.sidebar.header("⚙️ Training Configuration")

epochs = st.sidebar.slider("Number of Epochs", min_value=5, max_value=20, value=10, step=5)
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

# Download dataset from KaggleHub (default, no user input required)
path = kagglehub.dataset_download("jaiahuja/skin-cancer-detection")
# st.success(f"Dataset downloaded to: {path}")
st.session_state['path'] = path  # store path for later steps

# ----------------------------
# STEP 2: Load dataset
# ----------------------------
if "path" in st.session_state:
    st.subheader("Step 1: Load Training and Validation Data")

    if st.button("Load Dataset"):
        # Load and preprocess data (apply image & batch size you set)
        train_gen, val_gen = load_skin_cancer_data(path, img_size=img_size, batch_size=batch_size)
        st.session_state.train_gen = train_gen
        st.session_state.val_gen = val_gen

        # Show first 5 images in the batch
        X_batch, y_batch = next(train_gen)
        st.write("Sample training images:")
        class_labels = list(train_gen.class_indices.keys())  # e.g. ["benign", "malignant"]

        # Convert one-hot or categorical labels to integers
        labels = np.argmax(y_batch[:5], axis=1)

        # Map each label index to its class name
        captions = [class_labels[i] for i in labels]
        st.image(X_batch[:5], caption=captions, width=100)
        st.session_state['class_labels'] = class_labels  # store for later use

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
            
            #set optimizier according to user choice
            optimizer = Adam(learning_rate=learning_rate) if optimizer_choice=="Adam" else SGD(learning_rate=learning_rate, momentum=0.9)
            
            #apply selected optimizer & learning rate
            model = create_cnn_model(input_shape=(*img_size, 3), num_classes=num_classes, optimizer=optimizer)
            model.compile(
                optimizer=optimizer,           # e.g., 'adam' or Adam(lr=0.001)
                loss='categorical_crossentropy',
                metrics=['accuracy']
            )
            
            st.success("CNN model created successfully!")
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
        st.pyplot(plot_training_history(history_dict))