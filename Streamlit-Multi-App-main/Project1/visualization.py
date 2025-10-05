"""
visualization.py
----------------------
Handles visualization (confusion matrix, ROC, PR curve, feature importance, etc.)
"""

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_curve
import pandas as pd

def plot_confusion_matrix(y_test, y_pred):
    """Plots confusion matrix using seaborn heatmap."""
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(5,4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix")
    return fig

def plot_roc_curve(model, X_test, y_test):
    """Plots ROC curve for binary classification."""
    try:
        y_score = model.decision_function(X_test)
    except AttributeError:
        y_score = model.predict_proba(X_test)[:, 1]  # fallback

    fpr, tpr, _ = roc_curve(y_test, y_score)
    roc_auc = auc(fpr, tpr)

    fig, ax = plt.subplots()
    ax.plot(fpr, tpr, label=f"ROC Curve (AUC = {roc_auc:.2f})")
    ax.plot([0, 1], [0, 1], "k--")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend()
    ax.set_title("ROC Curve")
    return fig

def plot_precision_recall_curve(model, X_test, y_test):
    """Plots Precision-Recall curve."""
    try:
        y_score = model.decision_function(X_test)
    except AttributeError:
        y_score = model.predict_proba(X_test)[:, 1]  # fallback

    precision, recall, _ = precision_recall_curve(y_test, y_score)
    fig, ax = plt.subplots()
    ax.plot(recall, precision, label="Precision-Recall Curve")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curve")
    return fig

def plot_feature_importance(model, feature_names):
    """Plots feature importance for linear models (coefficients)."""
    if not hasattr(model, "coef_"):
        return None

    coef = model.coef_[0]
    feature_importance = pd.DataFrame({
        "Feature": feature_names,
        "Weight": coef
    }).sort_values("Weight", ascending=False)

    fig, ax = plt.subplots(figsize=(6,4))
    sns.barplot(x="Weight", y="Feature", data=feature_importance, ax=ax)
    ax.set_title("Feature Importance (Weights)")
    return fig
