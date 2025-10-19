"""predict_random.py

Pick a random review from data.csv, vectorize it using the existing
processing pipeline, load a trained model (sklearn or Keras), and print
the review with the predicted label and probability.

Usage:
    python predict_random.py --model model.joblib --vect vect.joblib
    python predict_random.py --model keras_model_dir --vect vect.joblib --backend keras

If no model/vect are provided, the script will attempt to look for
`model.joblib` (sklearn) or a `keras_model` folder and `vect.joblib` in
the current directory.
"""

import argparse
import random
import os
import joblib
import pandas as pd
import numpy as np

from processing import clean_text

def load_vectorizer(path):
    return joblib.load(path)

def load_sklearn_model(path):
    return joblib.load(path)

def load_keras_model(path):
    import tensorflow as tf
    return tf.keras.models.load_model(path)

def predict_with_sklearn(model, vect, text):
    X = vect.transform([text])
    prob = model.predict_proba(X)[0]
    # assume binary: classes_[1] is positive
    if hasattr(model, 'classes_'):
        pos_idx = list(model.classes_).index(1) if 1 in model.classes_ else 0
    else:
        pos_idx = 1
    return model.predict(X)[0], float(prob[pos_idx])

def predict_with_keras(model, vect, text):
    # keras model expects dense array
    X = vect.transform([text])
    if hasattr(X, 'toarray'):
        X = X.toarray()
    preds = model.predict(X)
    # preds are probabilities for positive class
    prob = float(preds[0][0])
    label = 1 if prob >= 0.5 else 0
    return label, prob

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv', default='data.csv')
    parser.add_argument('--model', default=None, help='Path to model (joblib for sklearn or saved_model dir for keras)')
    parser.add_argument('--vect', default='vect.joblib', help='Path to vectorizer joblib')
    parser.add_argument('--backend', choices=['sklearn','keras'], default='sklearn')
    args = parser.parse_args()

    if not os.path.exists(args.csv):
        raise SystemExit(f"CSV not found: {args.csv}")

    df = pd.read_csv(args.csv)
    if 'Body' not in df.columns:
        raise SystemExit("CSV must contain a 'Body' column")

    row = df.sample(1).iloc[0]
    raw = row['Body']
    processed = " ".join([t for t in clean_text(raw).split()])

    # load vectorizer
    if not os.path.exists(args.vect):
        raise SystemExit(f"Vectorizer not found: {args.vect}")
    vect = load_vectorizer(args.vect)

    # load model
    model_path = args.model
    if model_path is None:
        # try defaults
        if os.path.exists('model.joblib'):
            model_path = 'model.joblib'
            args.backend = 'sklearn'
        elif os.path.exists('keras_model'):
            model_path = 'keras_model'
            args.backend = 'keras'
        else:
            raise SystemExit('No model provided and no default model found (model.joblib or keras_model)')

    if args.backend == 'sklearn':
        model = load_sklearn_model(model_path)
        label, prob = predict_with_sklearn(model, vect, processed)
    else:
        model = load_keras_model(model_path)
        label, prob = predict_with_keras(model, vect, processed)

    print('--- Random review prediction ---')
    print('Raw text:')
    print(raw)
    print('\nProcessed (cleaned) text:')
    print(processed)
    print('\nPredicted label:', label)
    print('Predicted probability (positive):', prob)

if __name__ == '__main__':
    main()
