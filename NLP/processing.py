"""
processing.py

Simple, well-documented text preprocessing pipeline using NLTK and scikit-learn.

Functions:
 - load_data(csv_file): read CSV into DataFrame
 - clean_text(s): basic regex cleaning
 - tokenize_and_normalize(s): tokenize, remove stopwords, lemmatize
 - prepare_corpus(df): produce X (texts) and y (labels)
 - process(csv_file, ...): full pipeline that returns fitted vectorizer and train/test splits

This file is intentionally simple so you can read and adapt each step.
"""

import re
from typing import Tuple, List, Optional

import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer


# ------------------ Ensure NLTK data is available ------------------
# Attempt to load resources; if not present, download them once.
try:
    _ = word_tokenize("test")
    _ = stopwords.words("english")
except LookupError:
    nltk.download("punkt")
    nltk.download("stopwords")
    nltk.download("wordnet")
    nltk.download("omw-1.4")


# Create single instances to reuse
_STOPWORDS = set(stopwords.words("english"))
_LEMMATIZER = WordNetLemmatizer()


def load_data(csv_file: str) -> pd.DataFrame:
    """Load CSV into a pandas DataFrame. Expects a 'Body' column and optional 'Label'."""
    df = pd.read_csv(csv_file, encoding="utf-8")
    return df


def clean_text(text: str) -> str:
    """Perform lightweight cleaning:
    - remove email addresses
    - remove urls
    - strip html-like tags
    - remove non-alphabetic characters (keeps spaces)
    - lowercase and collapse whitespace
    """
    if text is None:
        return ""
    s = str(text)
    # remove emails
    s = re.sub(r"\S*@\S*\s?", " ", s)
    # remove urls
    s = re.sub(r"https?://\S+|www\.\S+", " ", s)
    # remove html tags
    s = re.sub(r"<[^>]+>", " ", s)
    # keep only letters and spaces
    s = re.sub(r"[^a-zA-Z\s']", " ", s)
    s = s.lower()
    s = re.sub(r"\s+", " ", s).strip()
    return s


def tokenize_and_normalize(text: str,
                           remove_stopwords: bool = True,
                           lemmatize: bool = True,
                           min_token_len: int = 2) -> List[str]:
    """Tokenize text, remove stopwords and short tokens, and lemmatize.

    Returns a list of cleaned tokens.
    """
    if not text:
        return []
    cleaned = clean_text(text)
    toks = word_tokenize(cleaned)
    out = []
    for tk in toks:
        if tk.isdigit():
            continue
        if len(tk) < min_token_len:
            continue
        if remove_stopwords and tk in _STOPWORDS:
            continue
        if lemmatize:
            tk = _LEMMATIZER.lemmatize(tk)
        out.append(tk)
    return out


def prepare_corpus(df: pd.DataFrame, text_col: str = "Body", label_col: Optional[str] = "Label") -> Tuple[pd.Series, Optional[pd.Series]]:
    """Given a DataFrame, produce X (joined tokens) and y (labels if present).

    Steps:
    1. Clean the raw text using regex rules
    2. Tokenize, remove stopwords, lemmatize
    3. Re-join tokens to create a string per document for vectorizers
    """
    texts = df[text_col].astype(str)
    tokens_series = texts.apply(tokenize_and_normalize)
    # join tokens back to space-separated string for TF-IDF vectorizer
    X = tokens_series.apply(lambda toks: " ".join(toks))
    y = df[label_col] if label_col in df.columns else None
    return X, y


def process(csv_file: str,
            text_col: str = "Body",
            label_col: Optional[str] = "Label",
            test_size: float = 0.2,
            random_state: int = 42,
            tfidf_kwargs: Optional[dict] = None) -> Tuple[TfidfVectorizer, object, object, object, object]:
    """Full pipeline:
    - load CSV
    - prepare corpus (clean/tokenize/lemmatize)
    - split train/test
    - fit TF-IDF vectorizer on training data and transform test data

    Returns: (vectorizer, X_train_tfidf, X_test_tfidf, y_train, y_test)
    """
    if tfidf_kwargs is None:
        tfidf_kwargs = {"ngram_range": (1, 2), "max_features": 20000}

    df = load_data(csv_file)

    # Auto-detect text and label columns if the provided names aren't present.
    cols = set(df.columns.astype(str))
    # candidate text columns (check these in order)
    text_candidates = [text_col, text_col.lower(), text_col.title(),
                       'text', 'body', 'review', 'comment', 'content', 'sentence', 'tweet']
    detected_text = None
    for c in text_candidates:
        if c in cols:
            detected_text = c
            break

    if detected_text is None:
        raise KeyError(f"No text column found. Tried candidates: {text_candidates}. CSV columns: {list(df.columns)}")

    # candidate label columns
    detected_label = None
    if label_col is not None:
        label_candidates = [label_col, label_col.lower(), label_col.title(),
                            'label', 'sentiment', 'target', 'rating', 'class']
        for c in label_candidates:
            if c in cols:
                detected_label = c
                break

    # Normalize column names to the expected names used later in pipeline
    if detected_text != text_col:
        df = df.rename(columns={detected_text: text_col})
    if detected_label and detected_label != label_col:
        # only rename if label_col is not None
        df = df.rename(columns={detected_label: label_col})

    X, y = prepare_corpus(df, text_col=text_col, label_col=label_col)

    # train/test split
    if y is None:
        # if no labels, still split X so you can validate downstream
        X_train, X_test = train_test_split(X, test_size=test_size, random_state=random_state)
        y_train = y_test = None
    else:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    # Fit TF-IDF on training texts only
    vectorizer = TfidfVectorizer(**tfidf_kwargs)
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    return vectorizer, X_train_tfidf, X_test_tfidf, y_train, y_test


def preview_data(csv_file: str,
                 text_col: str = "Body",
                 label_col: Optional[str] = "Label",
                 n: int = 5,
                 tfidf_kwargs: Optional[dict] = None):
    """Return a small preview DataFrame showing original text, cleaned text,
    tokens, and top TF-IDF terms for each sampled document.

    Returns (preview_df, fitted_vectorizer)
    """
    if tfidf_kwargs is None:
        tfidf_kwargs = {"ngram_range": (1, 2), "max_features": 20000}

    df = load_data(csv_file)
    # use prepare_corpus to normalize/clean tokens for the whole dataset
    X_all, y_all = prepare_corpus(df, text_col=text_col, label_col=label_col)

    # sample the first n rows (preserve order)
    sample_idx = list(range(min(n, len(X_all))))
    sample_texts = df.iloc[sample_idx][text_col].astype(str)

    # produce tokens and cleaned strings for the sampled texts
    tokens_series = sample_texts.apply(tokenize_and_normalize)
    cleaned_series = tokens_series.apply(lambda toks: " ".join(toks))

    # fit TF-IDF on the whole corpus for a realistic feature set
    vectorizer = TfidfVectorizer(**tfidf_kwargs)
    vectorizer.fit(X_all)

    feat_names = np.array(vectorizer.get_feature_names_out())
    X_sample_tfidf = vectorizer.transform(cleaned_series)

    top_terms = []
    for i in range(X_sample_tfidf.shape[0]):
        row = X_sample_tfidf[i].toarray().ravel()
        if row.sum() == 0:
            top_terms.append([])
            continue
        top_idx = row.argsort()[::-1][:5]
        terms = [feat_names[idx] for idx in top_idx if row[idx] > 0]
        top_terms.append(terms)

    out = pd.DataFrame({
        'original': sample_texts.values,
        'cleaned': cleaned_series.values,
        'tokens': tokens_series.values,
        'top_terms': top_terms
    })
    if label_col in df.columns:
        out[label_col] = df.iloc[sample_idx][label_col].values

    return out, vectorizer


def preview_text(text: str):
    """Return a small preview for a single text: cleaned string and token list.

    This mirrors what the preprocessing pipeline does for dataset rows and is
    intended for quick UI previews (e.g., in Streamlit) so users can see how
    a single sentence will be transformed before vectorization.
    """
    if text is None:
        return "", []
    cleaned = clean_text(text)
    tokens = tokenize_and_normalize(text)
    return cleaned, tokens

#To allow a file to be both importable (providing functions/classes) and runnable (performing a script-like action, e.g., a quick demo, CLI, or tests).
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run simple preprocessing and TF-IDF vectorization on a CSV dataset.")
    parser.add_argument("csv", help="Path to CSV file (expects a 'Body' column)")
    parser.add_argument("--text-col", default="Body", help="Name of the text column")
    parser.add_argument("--label-col", default="Label", help="Name of the label column (optional)")
    parser.add_argument("--out-vect", default=None, help="Optional path to save the fitted vectorizer (joblib)")
    args = parser.parse_args()

    vect, Xtr, Xte, ytr, yte = process(args.csv, text_col=args.text_col, label_col=args.label_col)

    print("Processed dataset:")
    print("  Train shape:", Xtr.shape)
    print("  Test shape:", Xte.shape)
    if ytr is not None:
        print("  Train labels:", ytr.shape)
    if args.out_vect:
        import joblib
        joblib.dump(vect, args.out_vect)
        print("Saved vectorizer to", args.out_vect)

