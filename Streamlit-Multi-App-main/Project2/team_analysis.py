import pandas as pd
import matplotlib.pyplot as plt

# ------------------------------
# Assign positions to team
# ------------------------------
def assign_positions(team_df: pd.DataFrame):
    """
    Assign PG, SG, SF, PF, C to the 5 players.
    Right now it's sequential. Can later be improved by heuristics.
    """
    positions = ["PG", "SG", "SF", "PF", "C"]
    team_df = team_df.copy()
    team_df["assigned_position"] = positions[:len(team_df)]
    return team_df

# ------------------------------
# Analyze team performance
# ------------------------------
def analyze_team(team_df: pd.DataFrame):
    # Assign positions first
    team_df = assign_positions(team_df)

    # Build normalized analysis DataFrame
    analysis_df = pd.DataFrame()
    analysis_df["player_name"] = team_df.get("player_name", pd.Series(["Unknown"] * len(team_df)))
    analysis_df["position"] = team_df.get("assigned_position", "Unknown")
    analysis_df["fg_pct"] = team_df.get("fg_pct", team_df.get("ts_pct", 0))
    analysis_df["three_pt_pct"] = team_df.get("three_pt_pct", 0)
    analysis_df["assists"] = team_df.get("assists", team_df.get("ast", 0))
    analysis_df["rebounds"] = team_df.get("rebounds", team_df.get("reb", 0))
    analysis_df["defense"] = team_df.get("defense", team_df.get("net_rating", 0))
    analysis_df["experience"] = team_df.get("experience", 0)

    # Contribution metrics
    analysis_df["offense_score"] = (analysis_df["fg_pct"] + analysis_df["three_pt_pct"]) / 2
    analysis_df["total_impact"] = (
        analysis_df["offense_score"] +
        analysis_df["assists"] +
        analysis_df["rebounds"] +
        analysis_df["defense"]
    )

    return analysis_df
