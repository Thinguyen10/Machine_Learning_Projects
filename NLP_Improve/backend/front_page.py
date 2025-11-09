import streamlit as st
import kagglehub
import numpy as np

def show_front_page():
    # --- Front page / Dataset summary -------------------------------------------------
    st.header("Dataset and Project Summary")
    st.markdown(
        """
        **Dataset:** `mdismielhossenabir/sentiment-analysis` (Kaggle)

    This app demonstrates a simple NLP pipeline for binary sentiment
    classification using TF-IDF features and two possible model backends:
    a scikit-learn Logistic Regression (fast, works on sparse matrices) or
    a small Keras neural network (dense, may require more memory).

    The dataset (credited to the original Kaggle author) contains text
    in a `Body` column and corresponding sentiment labels in a `Label`
    column. We use standard text preprocessing (cleaning, tokenization,
    stopword removal, lemmatization) and TF-IDF vectorization before
    training.
    """
    )

    # Download latest version of the dataset from Kaggle (uncomment if needed)
    path = None
    try:
        path = kagglehub.dataset_download("mdismielhossenabir/sentiment-analysis")
    except Exception:
        # If the kagglehub download fails (missing credentials or network), we
        # silently continue — the rest of the app expects a local `data.csv` file.
        path = None

    with st.expander("How the preprocessing pipeline works (from processing.py)"):
        st.write("Key steps performed on the raw text:")
        st.markdown(
            """
            - Load CSV with a `Body` text column and optional `Label`.
            - Clean text: remove emails, URLs, HTML tags, and non-alphabetic characters.
            - Tokenize and normalize: split into tokens, remove short tokens and digits.
            - Remove stopwords and lemmatize tokens (WordNet lemmatizer).
            - Re-join tokens into cleaned documents and fit a TF-IDF vectorizer
            (ngram range (1,2), max features 20000 by default).
            - Split into train/test and transform texts to TF-IDF features.
            """
        )

    with st.expander("Model options and how the model works (from model.py)"):
        st.write("Available training backends and their behavior:")
        st.markdown(
            """
            - Sklearn LogisticRegression:
            - Trains directly on sparse TF-IDF features (fast, memory-efficient).
            - Suitable for quick baselines and smaller compute environments.
            - Keras neural network:
            - A small feed-forward network: Input -> Dense(128, relu) -> Dropout(0.5) -> Dense(1, sigmoid).
            - Uses binary crossentropy loss and Adam optimizer.
            - Converts sparse TF-IDF matrices to dense arrays, which may use
                more memory for large vocabularies.
            """
        )

    st.markdown("---")