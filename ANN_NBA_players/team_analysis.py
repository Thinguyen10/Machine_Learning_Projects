import pandas as pd
import matplotlib.pyplot as plt

def analyze_team(team_df: pd.DataFrame):
    """
    Analyze and print contributions of each player in the team.
    Returns a DataFrame with key features.
    """
    features_to_show = ["player_name", "position", "scoring_type", 
                        "fg_pct", "three_pt_pct", "assists", "rebounds", "defense", "experience"]
    
    analysis_df = team_df[features_to_show].copy()
    
    # Calculate some contribution metrics
    analysis_df["offense_score"] = (analysis_df["fg_pct"] + analysis_df["three_pt_pct"]) / 2
    analysis_df["total_impact"] = analysis_df["offense_score"] + analysis_df["assists"] + analysis_df["rebounds"] + analysis_df["defense"]
    
    print("\n--- Team Analysis ---")
    print(analysis_df.sort_values(by="total_impact", ascending=False))
    
    return analysis_df
