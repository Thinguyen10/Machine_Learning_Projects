import pandas as pd
import random
from typing import List

class PlayerSelector:
    def __init__(self, file_input):
        """
        file_input: can be a path (str) or a file-like object (Streamlit upload)
        """
        if isinstance(file_input, pd.DataFrame):
            self.df = file_input
        else:
            self.df = pd.read_csv(file_input)  # works for path or file-like object

    def select_random_players(self, n=100, seasons=[]):
        df_filtered = self.df[self.df['season'].isin(seasons)]
        if len(df_filtered) < n:
            raise ValueError("Not enough players in the filtered dataset.")
        return df_filtered.sample(n=n, random_state=42)


