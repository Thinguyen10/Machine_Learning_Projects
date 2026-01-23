"""
Data generation and preprocessing module for diabetes prediction.
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split


class DataGenerator:
    """Generate synthetic diabetes dataset."""
    
    def __init__(self, n_samples=1000, random_state=42):
        self.n_samples = n_samples
        self.random_state = random_state
        np.random.seed(random_state)
    
    def generate_data(self):
        """Generate synthetic diabetes patient data."""
        data = []
        
        for _ in range(self.n_samples):
            if np.random.rand() < 0.5:
                # Non-Diabetic (class 0)
                age = np.random.randint(18, 80)
                gender = np.random.choice(["Male", "Female"])
                bmi = round(np.random.normal(27, 5), 1)
                glucose = round(np.random.normal(110, 30), 1)
                bp = round(np.random.normal(80, 10), 1)
                insulin = round(np.random.normal(100, 40), 1)
                activity_level = np.random.choice(["Low", "Medium", "High"])
                stress_level = np.random.randint(1, 11)
                family_history = np.random.choice(["Yes", "No"], p=[0.6, 0.4])
                fatigue = np.random.choice([0, 1], p=[0.7, 0.3])
                urination = np.random.choice([0, 1], p=[0.8, 0.2])
                thirst = np.random.choice([0, 1], p=[0.75, 0.25])
                diabetes = 0
            else:
                # Diabetic (class 1)
                age = np.random.randint(18, 80)
                gender = np.random.choice(["Male", "Female"])
                bmi = round(np.random.normal(30, 5), 1)
                glucose = round(np.random.normal(180, 50), 1)
                bp = round(np.random.normal(90, 15), 1)
                insulin = round(np.random.normal(150, 40), 1)
                activity_level = np.random.choice(["Low", "Medium", "High"])
                stress_level = np.random.randint(1, 11)
                family_history = np.random.choice(["Yes", "No"], p=[0.6, 0.4])
                fatigue = np.random.choice([0, 1], p=[0.4, 0.6])
                urination = np.random.choice([0, 1], p=[0.3, 0.7])
                thirst = np.random.choice([0, 1], p=[0.25, 0.75])
                diabetes = 1

            data.append([age, gender, bmi, glucose, bp, insulin, activity_level, 
                        stress_level, family_history, fatigue, urination, thirst, diabetes])

        df = pd.DataFrame(data, columns=[
            "Age", "Gender", "BMI", "Glucose", "Blood Pressure", "Insulin",
            "Physical Activity Level", "Stress Level", "Family History",
            "Fatigue", "Frequent Urination", "Thirst", "Diabetes Status"
        ])
        
        return df


class DataPreprocessor:
    """Clean and prepare data for machine learning."""
    
    def __init__(self):
        self.label_encoders = {}
        self.scaler = StandardScaler()
        self.top_features = None
        
    def clean_data(self, df):
        """Clean the dataset."""
        # Remove blank values
        df.replace(r'^\s*$', np.nan, regex=True, inplace=True)
        
        # Remove missing values if any
        df.dropna(inplace=True)
        
        # Normalize column names
        df.columns = df.columns.str.strip().str.lower()
        
        return df
    
    def encode_features(self, df):
        """Encode categorical features."""
        df_encoded = df.copy()
        
        categorical_cols = ['gender', 'physical activity level', 'family history']
        
        for col in categorical_cols:
            if col in df_encoded.columns:
                le = LabelEncoder()
                df_encoded[col] = le.fit_transform(df_encoded[col])
                self.label_encoders[col] = le
        
        return df_encoded
    
    def get_top_features(self, df, target_col='diabetes status', threshold=0.1):
        """Get features with correlation above threshold."""
        corr_matrix = df.corr(numeric_only=True)
        
        if target_col in corr_matrix.columns:
            correlation_with_target = corr_matrix[target_col].sort_values(ascending=False)
            top_features = correlation_with_target[correlation_with_target > threshold].index.tolist()
            
            # Remove target from the list
            if target_col in top_features:
                top_features.remove(target_col)
            
            self.top_features = top_features
            return top_features
        
        return []
    
    def prepare_data(self, df, target_col='diabetes status', test_size=0.2, random_state=42):
        """Prepare data for training."""
        # Clean and encode
        df = self.clean_data(df)
        df = self.encode_features(df)
        
        # Get top features
        top_features = self.get_top_features(df, target_col)
        
        # Prepare X and y
        X = df[top_features]
        y = df[target_col]
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        
        return X_train, X_test, y_train, y_test, top_features
    
    def scale_data(self, X_train, X_test):
        """Scale features using StandardScaler."""
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        return X_train_scaled, X_test_scaled


def get_diabetic_data(df):
    """Filter diabetic patients only."""
    return df[df['diabetes status'] == 1].copy()


def get_non_diabetic_data(df):
    """Filter non-diabetic patients only."""
    return df[df['diabetes status'] == 0].copy()
