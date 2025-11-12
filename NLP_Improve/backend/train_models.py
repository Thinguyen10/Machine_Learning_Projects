"""
train_models.py

Pre-train both sklearn and keras models with optimized hyperparameters.
This script should be run once to create production-ready models that the
API will load for predictions.

Usage:
    python -m backend.train_models
    
This will create:
    - vect.joblib (TF-IDF vectorizer)
    - model_sklearn.joblib (optimized LogisticRegression)
    - model_keras/ (optimized Keras neural network)
"""

import os
import sys
import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV

# Add backend to path for imports
sys.path.insert(0, os.path.dirname(__file__))

from processing import process
from model import build_keras_model, save_model, save_vectorizer


def train_optimized_sklearn(X_train, X_test, y_train, y_test):
    """Train LogisticRegression with grid search for best hyperparameters."""
    print("\n" + "="*60)
    print("TRAINING SKLEARN MODEL (Optimized LogisticRegression)")
    print("="*60)
    
    # Grid search for best parameters
    param_grid = {
        'C': [0.01, 0.1, 1.0, 10.0],
        'penalty': ['l2'],
        'solver': ['lbfgs', 'liblinear'],
        'max_iter': [1000]
    }
    
    print(f"\nRunning GridSearchCV with {len(param_grid['C']) * len(param_grid['solver'])} combinations...")
    
    clf = LogisticRegression(random_state=42)
    grid_search = GridSearchCV(
        clf, 
        param_grid, 
        cv=5, 
        scoring='accuracy',
        n_jobs=-1,
        verbose=1
    )
    
    grid_search.fit(X_train, y_train)
    
    print(f"\nBest parameters: {grid_search.best_params_}")
    print(f"Best CV score: {grid_search.best_score_:.4f}")
    
    # Get best model
    best_model = grid_search.best_estimator_
    
    # Evaluate on test set
    y_pred = best_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"\nTest Accuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    
    # Prepare metrics
    metrics = {
        "accuracy": float(accuracy),
        "best_params": grid_search.best_params_,
        "cv_score": float(grid_search.best_score_),
        "report": classification_report(y_test, y_pred, output_dict=True)
    }
    
    return best_model, metrics


def train_optimized_keras(X_train, X_test, y_train, y_test, epochs=20):
    """Train Keras model with optimized architecture and hyperparameters."""
    print("\n" + "="*60)
    print("TRAINING KERAS MODEL (Optimized Neural Network)")
    print("="*60)
    
    try:
        import tensorflow as tf
    except ImportError:
        print("ERROR: TensorFlow not installed. Skipping Keras model.")
        return None, None
    
    # Convert sparse to dense
    if hasattr(X_train, "toarray"):
        X_train_dense = X_train.toarray()
        X_test_dense = X_test.toarray()
    else:
        X_train_dense = np.asarray(X_train)
        X_test_dense = np.asarray(X_test)
    
    # Convert labels
    y_train_arr = np.asarray(y_train)
    y_test_arr = np.asarray(y_test)
    
    if y_train_arr.dtype == object or not np.issubdtype(y_train_arr.dtype, np.number):
        from sklearn.preprocessing import LabelEncoder
        le = LabelEncoder()
        y_train_arr = le.fit_transform(y_train_arr)
        y_test_arr = le.transform(y_test_arr)
    
    # Determine number of classes
    num_classes = len(np.unique(y_train_arr))
    
    # Convert to categorical for multi-class
    y_train_cat = tf.keras.utils.to_categorical(y_train_arr, num_classes=num_classes)
    y_test_cat = tf.keras.utils.to_categorical(y_test_arr, num_classes=num_classes)
    
    input_dim = X_train_dense.shape[1]
    
    print(f"\nBuilding optimized model with input_dim={input_dim}, num_classes={num_classes}")
    
    # Optimized architecture for multi-class classification
    model = tf.keras.Sequential([
        tf.keras.layers.InputLayer(input_shape=(input_dim,)),
        tf.keras.layers.Dense(256, activation="relu"),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(64, activation="relu"),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(num_classes, activation="softmax")  # Multi-class output
    ])
    
    # Optimized learning rate
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    
    model.compile(
        optimizer=optimizer,
        loss="categorical_crossentropy",  # Multi-class loss
        metrics=["accuracy"]
    )
    
    print("\nModel architecture:")
    model.summary()
    
    # Early stopping to prevent overfitting
    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=3,
        restore_best_weights=True
    )
    
    # Learning rate reduction
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=2,
        min_lr=1e-6,
        verbose=1
    )
    
    print(f"\nTraining for up to {epochs} epochs with early stopping...")
    
    history = model.fit(
        X_train_dense, 
        y_train_cat,  # Use categorical labels
        validation_split=0.2,
        epochs=epochs,
        batch_size=32,
        callbacks=[early_stop, reduce_lr],
        verbose=2
    )
    
    # Evaluate
    loss, accuracy = model.evaluate(X_test_dense, y_test_cat, verbose=0)  # Use categorical labels
    
    print(f"\nTest Loss: {loss:.4f}")
    print(f"Test Accuracy: {accuracy:.4f}")
    
    # Get predictions for detailed metrics
    y_pred_prob = model.predict(X_test_dense)
    y_pred = np.argmax(y_pred_prob, axis=1)  # Get class with highest probability
    
    print("\nClassification Report:")
    print(classification_report(y_test_arr, y_pred))  # Compare with original labels
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test_arr, y_pred))
    
    metrics = {
        "loss": float(loss),
        "accuracy": float(accuracy),
        "epochs_trained": len(history.history['loss']),
        "final_lr": float(tf.keras.backend.get_value(model.optimizer.learning_rate)),
        "report": classification_report(y_test_arr, y_pred, output_dict=True)
    }
    
    return model, metrics


def main():
    print("="*60)
    print("NLP SENTIMENT ANALYSIS - MODEL TRAINING")
    print("="*60)
    
    # Check for CSV file
    csv_path = 'backend/sentiment_analysis.csv'
    if not os.path.exists(csv_path):
        print(f"ERROR: {csv_path} not found!")
        return
    
    print(f"\nLoading and preprocessing data from {csv_path}...")
    
    # Process data with correct column names
    try:
        vect, X_train, X_test, y_train, y_test = process(
            csv_path, 
            text_col='text',  # Column name from CSV
            label_col='sentiment'  # Column name from CSV
        )
    except Exception as e:
        print(f"ERROR during preprocessing: {e}")
        return
    
    print(f"\nData shapes:")
    print(f"  Training: {X_train.shape}")
    print(f"  Testing: {X_test.shape}")
    print(f"  Vocabulary size: {len(vect.vocabulary_)}")
    
    # Save vectorizer
    vect_path = 'vect.joblib'
    save_vectorizer(vect, vect_path)
    print(f"\n✓ Saved vectorizer to {vect_path}")
    
    # Train sklearn model
    sklearn_model, sklearn_metrics = train_optimized_sklearn(X_train, X_test, y_train, y_test)
    
    if sklearn_model:
        sklearn_path = 'model_sklearn.joblib'
        save_model(sklearn_model, sklearn_path, backend='sklearn')
        print(f"\n✓ Saved sklearn model to {sklearn_path}")
        
        # Save metrics
        metrics_path = 'metrics_sklearn.json'
        import json
        with open(metrics_path, 'w') as f:
            json.dump(sklearn_metrics, f, indent=2)
        print(f"✓ Saved sklearn metrics to {metrics_path}")
    
    # Train keras model
    keras_model, keras_metrics = train_optimized_keras(X_train, X_test, y_train, y_test, epochs=20)
    
    if keras_model:
        keras_path = 'model_keras.keras'  # Add .keras extension
        save_model(keras_model, keras_path, backend='keras')
        print(f"\n✓ Saved keras model to {keras_path}")
        
        # Save metrics
        metrics_path = 'metrics_keras.json'
        import json
        with open(metrics_path, 'w') as f:
            json.dump(keras_metrics, f, indent=2)
        print(f"✓ Saved keras metrics to {metrics_path}")
    
    print("\n" + "="*60)
    print("TRAINING COMPLETE!")
    print("="*60)
    print("\nGenerated files:")
    print("  - vect.joblib (TF-IDF vectorizer)")
    print("  - model_sklearn.joblib (LogisticRegression)")
    print("  - model_keras/ (Keras neural network)")
    print("  - metrics_sklearn.json")
    print("  - metrics_keras.json")
    print("\nYou can now use these models in the API!")


if __name__ == "__main__":
    main()
