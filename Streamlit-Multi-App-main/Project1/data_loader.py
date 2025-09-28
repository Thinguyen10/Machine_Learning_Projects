""" data_loader.py -------------- 
Loads a dataset (here we use sklearn’s built-in dataset for simplicity). 
"""

from sklearn.datasets import load_breast_cancer 
import pandas as pd

def load_data(): 
    data = load_breast_cancer() 
    df = pd.DataFrame(data.data,columns=data.feature_names) 
    df['target'] = data.target 
    return df