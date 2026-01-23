"""
Machine learning models for diabetes prediction and analysis.
"""
import numpy as np
import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import KMeans
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score
from kneed import KneeLocator


class DiabetesPredictor:
    """Base class for diabetes prediction models."""
    
    def __init__(self):
        self.model = None
        self.model_name = ""
        self.is_trained = False
        
    def train(self, X_train, y_train):
        """Train the model."""
        self.model.fit(X_train, y_train)
        self.is_trained = True
        
    def predict(self, X):
        """Make predictions."""
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        return self.model.predict(X)
    
    def evaluate(self, X_test, y_test):
        """Evaluate model performance."""
        y_pred = self.predict(X_test)
        
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, average='weighted', zero_division=0),
            'recall': recall_score(y_test, y_pred, average='weighted', zero_division=0),
            'f1': f1_score(y_test, y_pred, average='weighted', zero_division=0),
            'report': classification_report(y_test, y_pred, output_dict=True)
        }
        
        return metrics


class NaiveBayesModel(DiabetesPredictor):
    """Naive Bayes classifier for diabetes prediction."""
    
    def __init__(self):
        super().__init__()
        self.model = GaussianNB()
        self.model_name = "Naive Bayes"


class RandomForestModel(DiabetesPredictor):
    """Random Forest classifier with hyperparameter tuning."""
    
    def __init__(self, tune_hyperparameters=False):
        super().__init__()
        self.model = RandomForestClassifier(random_state=42)
        self.model_name = "Random Forest"
        self.tune_hyperparameters = tune_hyperparameters
        self.best_params = None
        
    def train(self, X_train, y_train):
        """Train with optional hyperparameter tuning."""
        if self.tune_hyperparameters:
            param_grid = {
                'n_estimators': [5, 10, 18, 20, 25],
                'max_depth': [None, 1, 2, 5],
                'min_samples_split': [2, 3, 5, 10],
                'min_samples_leaf': [1, 5, 10, 13, 15, 20],
            }
            
            grid_search = GridSearchCV(
                estimator=self.model,
                param_grid=param_grid,
                cv=5,
                n_jobs=-1,
                verbose=0
            )
            grid_search.fit(X_train, y_train)
            
            self.best_params = grid_search.best_params_
            self.model = RandomForestClassifier(**self.best_params, random_state=42)
        
        self.model.fit(X_train, y_train)
        self.is_trained = True


class LogisticRegressionModel(DiabetesPredictor):
    """Logistic Regression classifier."""
    
    def __init__(self):
        super().__init__()
        self.model = LogisticRegression(max_iter=1000, random_state=42)
        self.model_name = "Logistic Regression"
        self.scaler = StandardScaler()
        
    def train(self, X_train, y_train):
        """Train with feature scaling."""
        X_train_scaled = self.scaler.fit_transform(X_train)
        self.model.fit(X_train_scaled, y_train)
        self.is_trained = True
        
    def predict(self, X):
        """Make predictions with scaling."""
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)


class RiskClusterer:
    """K-Means clustering for risk assessment of non-diabetic patients."""
    
    def __init__(self):
        self.model = None
        self.optimal_k = None
        self.high_risk_cluster = None
        self.centroids = None
        
    def find_optimal_k(self, X, k_range=range(1, 11)):
        """Find optimal number of clusters using elbow method."""
        sse = []
        
        for k in k_range:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init='auto')
            kmeans.fit(X)
            sse.append(kmeans.inertia_)
        
        # Detect the elbow point
        kneedle = KneeLocator(
            list(k_range),
            sse,
            curve="convex",
            direction="decreasing"
        )
        
        self.optimal_k = kneedle.elbow if kneedle.elbow is not None else 3
        
        return self.optimal_k, sse
    
    def fit(self, X):
        """Fit K-Means clustering."""
        if self.optimal_k is None:
            self.find_optimal_k(X)
        
        self.model = KMeans(n_clusters=self.optimal_k, random_state=42, n_init='auto')
        clusters = self.model.fit_predict(X)
        
        # Store centroids
        self.centroids = pd.DataFrame(
            self.model.cluster_centers_,
            columns=X.columns if hasattr(X, 'columns') else [f'Feature_{i}' for i in range(X.shape[1])]
        )
        
        # Determine high-risk cluster (highest average feature values)
        self.high_risk_cluster = self.centroids.mean(axis=1).idxmax()
        
        return clusters
    
    def predict_risk(self, X):
        """Predict risk level for patients."""
        clusters = self.model.predict(X)
        risk_labels = ['High Risk' if c == self.high_risk_cluster else 'Low Risk' for c in clusters]
        return risk_labels


class DiabetesTypeClassifier:
    """PCA + SVM for diabetes type classification."""
    
    def __init__(self, variance_threshold=0.90):
        self.variance_threshold = variance_threshold
        self.pca = None
        self.svm = None
        self.scaler = StandardScaler()
        self.n_components = None
        self.cluster_to_type = None
        
    def apply_pca(self, X):
        """Apply PCA for dimensionality reduction."""
        # Standardize
        X_scaled = self.scaler.fit_transform(X)
        
        # Initial PCA to get all components
        pca_full = PCA()
        pca_full.fit(X_scaled)
        
        # Calculate cumulative variance
        cumulative_variance = np.cumsum(pca_full.explained_variance_ratio_)
        
        # Find number of components for threshold
        self.n_components = np.argmax(cumulative_variance >= self.variance_threshold) + 1
        
        # Apply PCA with optimal components
        self.pca = PCA(n_components=self.n_components)
        X_pca = self.pca.fit_transform(X_scaled)
        
        return X_pca, cumulative_variance
    
    def create_pseudo_labels(self, X_pca):
        """Create pseudo labels using K-Means (Type 1 vs Type 2)."""
        kmeans = KMeans(n_clusters=2, random_state=42, n_init='auto')
        y_simulated = kmeans.fit_predict(X_pca)
        
        # Analyze cluster centers
        centers = kmeans.cluster_centers_
        
        # Lower PC1 = Type 1 (younger/lower BMI)
        if centers[0][0] < centers[1][0]:
            self.cluster_to_type = {0: "Type 1 Diabetes", 1: "Type 2 Diabetes"}
        else:
            self.cluster_to_type = {1: "Type 1 Diabetes", 0: "Type 2 Diabetes"}
        
        return y_simulated, centers
    
    def train(self, X):
        """Train the classifier."""
        # Apply PCA
        X_pca, _ = self.apply_pca(X)
        
        # Create pseudo labels
        y_simulated, centers = self.create_pseudo_labels(X_pca)
        
        # Train SVM
        self.svm = SVC(kernel='linear', random_state=42)
        self.svm.fit(X_pca, y_simulated)
        
        return centers
    
    def predict(self, X):
        """Predict diabetes type."""
        # Transform with existing scaler and PCA
        X_scaled = self.scaler.transform(X)
        X_pca = self.pca.transform(X_scaled)
        
        # Predict cluster
        cluster = self.svm.predict(X_pca)
        
        # Map to type
        diabetes_type = [self.cluster_to_type[c] for c in cluster]
        
        return diabetes_type
