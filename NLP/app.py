# app.py - Main Streamlit application
# The dataset path is fixed (default) and users only interact with the process.

import streamlit as st
import kagglehub
import numpy as np
import json
import os
from pathlib import Path
from model import build_keras_model, save_vectorizer
from training import train_sklearn, train_keras
import pandas as pd
import random

from processing import process, preview_data
from front_page import show_front_page

st.title("Natural Language Processing")

show_front_page()

st.subheader("Step 1: Data Preprocessing")

# Default dataset path (fallback). If you rely only on Kaggle, uploader will
# allow you to provide the CSV when download isn't available.
dataset_csv = None

# Try to download the Kaggle dataset and locate a CSV. If download fails or no
# CSV is present, stop the app and instruct the user to configure Kaggle access.
try:
    dl = kagglehub.dataset_download("mdismielhossenabir/sentiment-analysis")
except Exception as e:
    st.error("Failed to download Kaggle dataset. Ensure kagglehub is configured and you have network access.")
    st.stop()

if dl:
    # Directory returned: search for the first CSV inside
    if os.path.isdir(dl):
        for root, _, files in os.walk(dl):
            for f in files:
                if f.lower().endswith('.csv'):
                    dataset_csv = os.path.join(root, f)
                    break
            if dataset_csv:
                break
    # File returned: CSV or archive
    elif os.path.isfile(dl):
        lower = dl.lower()
        if lower.endswith('.csv'):
            dataset_csv = dl
        elif lower.endswith('.zip'):
            import zipfile, tempfile
            tmpdir = tempfile.mkdtemp()
            try:
                with zipfile.ZipFile(dl) as z:
                    z.extractall(tmpdir)
                for root, _, files in os.walk(tmpdir):
                    for f in files:
                        if f.lower().endswith('.csv'):
                            dataset_csv = os.path.join(root, f)
                            break
                    if dataset_csv:
                        break
            except Exception:
                dataset_csv = None

if not dataset_csv or not os.path.exists(dataset_csv):
    st.error("Could not locate a CSV inside the Kaggle dataset download. Please configure Kaggle access or provide the CSV manually.")
    st.stop()

# Show a preview of data before/after processing
with st.expander("Preview: Original vs. Processed Text"):
    st.write("This preview shows how the preprocessing pipeline transforms raw text:")
    try:
        preview_df, _ = preview_data(dataset_csv, text_col="text", label_col="sentiment", n=3)
        for idx, row in preview_df.iterrows():
            st.markdown(f"**Sample {idx+1}:**")
            st.write(f"**Original:** {row['original'][:200]}...")
            st.write(f"**Cleaned tokens:** {row['tokens']}")
            st.write(f"**Top TF-IDF terms:** {row['top_terms']}")
            if 'sentiment' in row:
                st.write(f"**Label:** {row['sentiment']}")
            st.markdown("---")
    except Exception as e:
        st.warning(f"Could not generate preview: {e}")

# Run full preprocessing pipeline
vect, Xtr, Xte, ytr, yte = process(dataset_csv, text_col="text", label_col="sentiment")
st.write("Processed dataset:")
st.write("  Train shape:", Xtr.shape)
st.write("  Test shape:", Xte.shape)
if ytr is not None:
    st.write("  Train labels:", ytr.shape)
st.session_state.setdefault('vect', None)
st.session_state.setdefault('Xtr', None)
st.session_state.setdefault('Xte', None)
st.session_state.setdefault('ytr', None)
st.session_state.setdefault('yte', None)
st.session_state.setdefault('trained', False)
st.session_state.setdefault('train_metrics', None)

st.session_state['vect'] = vect
st.session_state['Xtr'] = Xtr
st.session_state['Xte'] = Xte
st.session_state['ytr'] = ytr
st.session_state['yte'] = yte

# ============================================================================
# Step 2: Model Training
# ============================================================================
# In this step, users choose a machine learning backend (sklearn or keras) and
# train a sentiment classifier on the preprocessed TF-IDF features.
# 
# - Sklearn backend: Trains a LogisticRegression model directly on sparse TF-IDF
#   matrices (fast, memory-efficient, good baseline).
# 
# - Keras backend: Builds a small feed-forward neural network (Dense layers) and
#   trains it on dense TF-IDF arrays (may use more memory, supports GPU).
#
# After training, accuracy and classification metrics are displayed.
# ============================================================================

st.subheader("Step 2: Model Training")

# Training controls
backend = st.selectbox("Choose backend", ("sklearn", "keras"))

if backend == "sklearn":
    st.write("Sklearn LogisticRegression will be trained on the sparse TF-IDF features.")
    if st.button("Train sklearn model"):
        with st.spinner("Training sklearn model..."):
            try:
                metrics = train_sklearn(st.session_state['Xtr'], st.session_state['Xte'],
                                        st.session_state['ytr'], st.session_state['yte'])
                st.session_state['trained'] = True
                st.session_state['train_metrics'] = metrics
                st.success(f"Sklearn training finished — accuracy: {metrics.get('accuracy')}" )
                st.json(metrics.get('report'))
            except Exception as e:
                st.error(f"Sklearn training failed: {e}")

else:
    epochs = st.number_input("Epochs", min_value=1, value=5)
    batch_size = st.number_input("Batch size", min_value=1, value=32)
    st.write("Keras model: Dense(128)->Dropout(0.5)->Dense(1, sigmoid). Note: sparse TF-IDF is converted to dense.")
    if st.button("Train keras model"):
        with st.spinner("Training Keras model..."):
            try:
                metrics = train_keras(st.session_state['Xtr'], st.session_state['Xte'],
                                      st.session_state['ytr'], st.session_state['yte'],
                                      epochs=int(epochs), batch_size=int(batch_size))
                st.session_state['trained'] = True
                st.session_state['train_metrics'] = metrics
                st.success(f"Keras training finished — accuracy: {metrics.get('accuracy')}" )
                st.write(metrics)
            except Exception as e:
                st.error(f"Keras training failed: {e}")

# Optionally save the fitted vectorizer
if st.session_state.get('vect') is not None:
    if st.button('Save vectorizer to vect.joblib'):
        try:
            save_vectorizer(st.session_state['vect'], 'vect.joblib')
            st.success('Saved vectorizer to vect.joblib')
        except Exception as e:
            st.error(f'Failed to save vectorizer: {e}')

#credit: https://www.kaggle.com/datasets/mdismielhossenabir/sentiment-analysis