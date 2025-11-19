"""run_tuning.py

Example script showing how to use `backend.tuning` utilities with the project's
preprocessing pipeline. This is a small, runnable demo and not intended for full
experiments (reduce epochs and grid sizes for local testing).
"""
import argparse
import numpy as np

from backend.processing import process
from backend.model import build_keras_model
from backend.tuning import epoch_sweep, grid_search_keras, build_fn_from_params, build_default_dense


def simple_train_val_split(X, y, val_frac=0.1, random_state=42):
    # expects X as sparse/dense matrix and y as array-like
    from sklearn.model_selection import train_test_split
    Xtr, Xv, ytr, yv = train_test_split(X, y, test_size=val_frac, random_state=random_state)
    return Xtr, Xv, ytr, yv


def main(csv_path: str, epochs: int = 10):
    vect, X_train, X_test, y_train, y_test = process(csv_path)

    # further split training set into train/val for tuning
    Xtr, Xv, ytr, yv = simple_train_val_split(X_train, y_train, val_frac=0.1)

    print("Starting epoch sweep (this will train a model for up to", epochs, "epochs)")

    # Use the default dense builder from tuning (same shape as model.build_keras_model)
    def builder(input_dim):
        # use small hidden size for quick iteration
        return build_default_dense(input_dim, hidden=64, dropout=0.4, lr=1e-3)

    sweep = epoch_sweep(builder, Xtr, ytr, Xv, yv, max_epochs=epochs, batch_size=32, verbose=2)
    print("Best observed epoch:", sweep["best_epoch"]) 
    print("Peak analysis:", sweep["peak_analysis"]) 

    # run a tiny grid search (very small grid for demo)
    param_grid = {
        'hidden': [64, 128],
        'dropout': [0.3, 0.5],
        'lr': [1e-3]
    }

    print("Running tiny grid search (this may take a while) ...")
    gs = grid_search_keras(build_fn_from_params, param_grid, Xtr, ytr, Xv, yv, epochs=3, batch_size=32, verbose=0)
    print("Grid search best:", gs['best'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run a short tuning demo')
    parser.add_argument('--csv', default='backend/sentiment_analysis.csv', help='CSV file to load')
    parser.add_argument('--epochs', type=int, default=10, help='Epochs for epoch_sweep')
    args = parser.parse_args()
    main(args.csv, epochs=args.epochs)
