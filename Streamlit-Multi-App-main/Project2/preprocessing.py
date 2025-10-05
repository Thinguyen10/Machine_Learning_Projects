# preprocess.py
import pandas as pd

def preprocess_data(df):
    # Drop unnamed index columns (if exist)
    df = df.loc[:, ~df.columns.str.contains("^Unnamed")]

    # Convert draft_year to numeric
    df["draft_year"] = pd.to_numeric(df.get("draft_year", pd.Series([0]*len(df))), errors="coerce")

    # Extract season start year
    df["season_start_year"] = df.get("season", pd.Series(["0"]*len(df))).str[:4]
    df["season_start_year"] = pd.to_numeric(df["season_start_year"], errors="coerce").fillna(0).astype(int)

    # Compute experience (years since draft)
    df["experience"] = df["season_start_year"] - df["draft_year"]
    df["experience"] = df["experience"].fillna(0).astype(int)

    return df
