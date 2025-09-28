"""
train_test_split.py
-------------------
Handles splitting dataset into training and test sets,
and optionally scales features.
"""

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder

def prepare_features_target(df, target_column='target'):
    """
    Separates features (X) and target (y) from dataframe.
    
    Args:
        df (pd.DataFrame): input dataframe
        target_column (str): name of target column
    
    Returns:
        X (pd.DataFrame): features
        y (pd.Series): target variable
    """
    X = df.drop(columns=[target_column])
    y = df[target_column]
    le = LabelEncoder()
    df[target_column] = le.fit_transform(df[target_column])   #HELP WITH PREDICTION

    return X, y


def split_data(X, y, test_size=0.2, random_state=42, scale=True):
    """
    Splits features and target into train and test sets,
    and optionally scales features using StandardScaler.
    
    Args:
        X (pd.DataFrame or np.array): features
        y (pd.Series or np.array): target
        test_size (float): proportion of test set
        random_state (int): for reproducibility
        scale (bool): whether to standardize features
    
    Returns:
        X_train_scaled, X_test_scaled, y_train, y_test
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    if scale:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)    #USEFUL FOR ACCURACY
        X_test = scaler.transform(X_test)
    
    return X_train, X_test, y_train, y_test

