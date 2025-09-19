import matplotlib.pyplot as plt
import seaborn as sns

def plot_team_skills(team_analysis_df):
    """
    Create bar charts of each player’s contribution metrics.
    """
    metrics = ["offense_score", "assists", "rebounds", "defense", "experience"]
    
    team_analysis_df = team_analysis_df.set_index("player_name")
    team_analysis_df[metrics].plot(kind="bar", figsize=(12,6))
    plt.title("Player Skill Contributions in Selected Team")
    plt.ylabel("Score / Metric (normalized)")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

def plot_position_distribution(team_analysis_df):
    """
    Show a pie chart of positions in the team to see balance.
    """
    position_counts = team_analysis_df["position"].value_counts()
    plt.figure(figsize=(6,6))
    plt.pie(position_counts, labels=position_counts.index, autopct="%1.1f%%", startangle=90, colors=sns.color_palette("pastel"))
    plt.title("Team Position Distribution")
    plt.show()
