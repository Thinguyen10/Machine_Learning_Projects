"""
Diabetes Prediction ML System
Core modules for data processing, models, and visualizations.
"""

__version__ = "1.0.0"
__author__ = "CST-425 Project"

from .data_processing import DataGenerator, DataPreprocessor
from .models import (
    NaiveBayesModel,
    RandomForestModel,
    LogisticRegressionModel,
    RiskClusterer,
    DiabetesTypeClassifier
)
from .visualizations import (
    plot_correlation_matrix,
    plot_elbow_curve,
    plot_pca_variance,
    plot_metrics_comparison,
    display_metrics_cards
)

__all__ = [
    'DataGenerator',
    'DataPreprocessor',
    'NaiveBayesModel',
    'RandomForestModel',
    'LogisticRegressionModel',
    'RiskClusterer',
    'DiabetesTypeClassifier',
    'plot_correlation_matrix',
    'plot_elbow_curve',
    'plot_pca_variance',
    'plot_metrics_comparison',
    'display_metrics_cards'
]
