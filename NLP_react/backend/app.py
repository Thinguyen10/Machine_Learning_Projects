# app.py - Main Streamlit application
# The dataset path is fixed (default) and users only interact with the process.

import streamlit as st
import kagglehub
import numpy as np
import json
import os
from pathlib import Path
try:
    from backend.model import build_keras_model, save_vectorizer
    from backend.training import train_sklearn, train_keras
    from backend.processing import process, preview_data, preview_text
    from backend.model import load_vectorizer, load_model, predict_text
    from backend.front_page import show_front_page
except Exception:
    # fallback for running this file directly from backend/ folder
    from model import build_keras_model, save_vectorizer
    from training import train_sklearn, train_keras
    from processing import process, preview_data, preview_text
    from model import load_vectorizer, load_model, predict_text
    from front_page import show_front_page

st.title("Natural Language Processing")

show_front_page()

st.subheader("Step 1: Data Preprocessing")
st.write("This step loads the dataset, runs the preprocessing pipeline, and prepares TF-IDF features used for training.")

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

# Store preprocessing artifacts in session state so later steps can use them
st.session_state.setdefault('vect', None)
st.session_state.setdefault('Xtr', None)
st.session_state.setdefault('Xte', None)
st.session_state.setdefault('ytr', None)
st.session_state.setdefault('yte', None)
st.session_state.setdefault('trained', False)
st.session_state.setdefault('train_metrics', None)
st.session_state.setdefault('trained_model', None)
st.session_state.setdefault('step1_done', False)
st.session_state.setdefault('step2_done', False)
st.session_state.setdefault('step3_done', False)

st.session_state['vect'] = vect
st.session_state['Xtr'] = Xtr
st.session_state['Xte'] = Xte
st.session_state['ytr'] = ytr
st.session_state['yte'] = yte

# Mark step1 complete (preprocessing finished)
st.session_state['step1_done'] = True

# If artifacts already exist on disk, allow skipping straight to later steps
if Path('vect.joblib').exists() and (Path('model.joblib').exists() or Path('keras_model').exists()):
    st.session_state['step3_done'] = True

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

if st.session_state.get('step1_done'):
    st.subheader("Step 2: Model Training")
    # Training controls
    backend = st.selectbox("Choose backend", ("sklearn", "keras"))

    if backend == "sklearn":
        st.write("Sklearn: LogisticRegression trained on sparse TF-IDF features. Fast and suitable as a baseline.")
        if st.button("Train sklearn model"):
            with st.spinner("Training sklearn model..."):
                try:
                    metrics, trained_model = train_sklearn(st.session_state['Xtr'], st.session_state['Xte'],
                                                           st.session_state['ytr'], st.session_state['yte'])
                    st.session_state['trained'] = True
                    st.session_state['train_metrics'] = metrics
                    st.session_state['trained_model'] = trained_model
                    st.session_state['step2_done'] = True
                    st.success(f"Sklearn training finished — accuracy: {metrics.get('accuracy')}")
                    # show brief report
                    try:
                        st.json(metrics.get('report'))
                    except Exception:
                        st.write(metrics)
                except Exception as e:
                    st.error(f"Sklearn training failed: {e}")

    else:
        epochs = st.number_input("Epochs", min_value=1, value=5)
        batch_size = st.number_input("Batch size", min_value=1, value=32)
        st.write("Keras model: Dense(128)->Dropout(0.5)->Dense(1, sigmoid). Sparse TF-IDF will be converted to dense arrays.")
        if st.button("Train keras model"):
            with st.spinner("Training Keras model..."):
                try:
                    metrics, trained_model = train_keras(st.session_state['Xtr'], st.session_state['Xte'],
                                                         st.session_state['ytr'], st.session_state['yte'],
                                                         epochs=int(epochs), batch_size=int(batch_size))
                    st.session_state['trained'] = True
                    st.session_state['train_metrics'] = metrics
                    st.session_state['trained_model'] = trained_model
                    st.session_state['step2_done'] = True
                    st.success(f"Keras training finished — accuracy: {metrics.get('accuracy')}")
                    st.write(metrics)
                except Exception as e:
                    st.error(f"Keras training failed: {e}")
else:
    st.info("Complete Step 1 (preprocessing) to enable training")

# Optionally save the fitted vectorizer
st.markdown("---")
if st.session_state.get('step2_done') or st.session_state.get('step3_done'):
    st.subheader("Step 3: Save artifacts")
    st.write("Once a model is trained, save the TF-IDF vectorizer and trained model so you can load them later without retraining.")
    col1, col2 = st.columns(2)
    with col1:
        if st.button('Save vectorizer to vect.joblib'):
            if st.session_state.get('vect') is None:
                st.error('No fitted vectorizer found in session state.')
            else:
                try:
                    save_vectorizer(st.session_state['vect'], 'vect.joblib')
                    st.success('Saved vectorizer to vect.joblib')
                except Exception as e:
                    st.error(f'Failed to save vectorizer: {e}')
    with col2:
        if st.button('Save trained model'):
            if st.session_state.get('trained_model') is None:
                st.error('No trained model available. Train a model first.')
            else:
                try:
                    # decide backend path/name based on selected backend
                    if backend == 'sklearn':
                        from model import save_model
                        save_model(st.session_state['trained_model'], 'model.joblib', backend='sklearn')
                        st.success('Saved sklearn model to model.joblib')
                    else:
                        from model import save_model
                        save_model(st.session_state['trained_model'], 'keras_model', backend='keras')
                        st.success('Saved Keras model to keras_model/')
                except Exception as e:
                    st.error(f'Failed to save model: {e}')
else:
    st.info("Finish training (Step 2) to enable saving artifacts")
    st.stop()


st.markdown("---")

#credit: https://www.kaggle.com/datasets/mdismielhossenabir/sentiment-analysis


# ------------------ Single-sentence demo / prediction ------------------
st.header("Try a sentence")
st.write("Enter a sentence to see how it's cleaned and to get a sentiment prediction from a saved model (if available).")

input_text = st.text_area("Type a sentence to analyze", height=120)
if input_text:
    cleaned, tokens = preview_text(input_text)
    with st.expander("Preview transformation"):
        st.write("Cleaned text:")
        st.write(cleaned)
        st.write("Tokens:")
        st.write(tokens)

    # Attempt to load vectorizer + model from working directory
    vect_path = Path('vect.joblib')
    skl_model_path = Path('model.joblib')
    keras_model_dir = Path('keras_model')

    if vect_path.exists():
        try:
            vect = load_vectorizer(str(vect_path))
        except Exception as e:
            st.error(f"Failed to load vectorizer: {e}")
            vect = None
    else:
        vect = None

    model_obj = None
    model_backend = None
    if skl_model_path.exists():
        try:
            model_obj = load_model(str(skl_model_path), backend='sklearn')
            model_backend = 'sklearn'
        except Exception as e:
            st.warning(f"Failed to load sklearn model: {e}")
    elif keras_model_dir.exists():
        try:
            model_obj = load_model(str(keras_model_dir), backend='keras')
            model_backend = 'keras'
        except Exception as e:
            st.warning(f"Failed to load Keras model: {e}")

    if vect is None or model_obj is None:
        st.info("Saved vectorizer and/or model not found in the working directory. Train and save a model (or place 'vect.joblib' and 'model.joblib' here) to enable prediction.")
    else:
        try:
            label, prob = predict_text(model_obj, vect, cleaned, backend=model_backend)
            st.success(f"Prediction: {label} — probability: {prob}")
        except Exception as e:
            st.error(f"Prediction failed: {e}")