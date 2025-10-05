"""
perceptron.py
-------------
Defines and trains a perceptron model.
"""

from sklearn.linear_model import Perceptron

def create_perceptron(max_iter=1000, eta0=1.0, random_state=42):
    """
    Creates a Perceptron model with given hyperparameters.
    """
    return Perceptron(max_iter=max_iter, eta0=eta0, random_state=random_state)

def train_model(model, X_train, y_train):
    """
    Fits the perceptron model on training data.
    """
    model.fit(X_train, y_train)
    return model
