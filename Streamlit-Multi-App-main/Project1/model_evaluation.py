"""
model_evaluation.py
-------------------
Handles evaluation metrics for the perceptron model.
"""

from sklearn.metrics import accuracy_score, classification_report, precision_score, recall_score, f1_score

def evaluate_model(model, X_test, y_test):
    """
    Evaluates model using accuracy and classification report.
    
    Returns:
        acc (float): accuracy score
        report (dict): classification report as dictionary
        y_pred (np.array): predictions
        metrics (dict): additional metrics like precision, recall, f1
    """
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)

    metrics = {
        "precision": precision_score(y_test, y_pred, average="weighted", zero_division=0),
        "recall": recall_score(y_test, y_pred, average="weighted", zero_division=0),
        "f1": f1_score(y_test, y_pred, average="weighted", zero_division=0),
    }

    return acc, report, y_pred, metrics
