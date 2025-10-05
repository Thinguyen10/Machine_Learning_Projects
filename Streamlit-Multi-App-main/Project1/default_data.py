import pandas as pd

def load_default_frog_dataset():
    """
    Load the default frog dataset from UCI ML Repository.

    Returns
    -------
    df : pandas.DataFrame or None
    """
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00275/Frogs_MFCCs.csv"
    try:
        df = pd.read_csv(url)
        return df
    except Exception as e:
        print(f"Error loading default frog dataset: {e}")
        return None
