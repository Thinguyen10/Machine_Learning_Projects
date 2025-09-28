"""
data_cleaner.py
---------------
Handles missing values, normalization, or other cleaning steps.
"""

import pandas as pd


def clean_data(df):
    
    df = df.dropna()  # drop rows with missing values
    df = pd.get_dummies(df, drop_first=True)  # drops first category to avoid multicollinearity
    
    return df


