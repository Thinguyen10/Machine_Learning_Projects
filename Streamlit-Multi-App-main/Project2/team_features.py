import numpy as np

# -----------------------------
# Position encoding
# -----------------------------
POSITION_MAP = {"PG": 0, "SG": 1, "SF": 2, "PF": 3, "C": 4}
NUM_POSITIONS = len(POSITION_MAP)

# -----------------------------
# Scoring type encoding (optional)
# -----------------------------
SCORING_MAP = {"Shooter": 0, "Slasher": 1, "All-Around": 2}
NUM_SCORING = len(SCORING_MAP)

# -----------------------------
# Encode single player into feature vector
# -----------------------------
def encode_player(player):
    features = []

    # --- Position one-hot ---
    pos_vec = [0] * NUM_POSITIONS
    pos = player.get("position", None)
    if pos in POSITION_MAP:
        pos_vec[POSITION_MAP[pos]] = 1
    # If missing, all zeros
    features.extend(pos_vec)

    # --- Scoring type one-hot ---
    scoring_vec = [0] * NUM_SCORING
    scoring_type = player.get("scoring_type", None)
    if scoring_type in SCORING_MAP:
        scoring_vec[SCORING_MAP[scoring_type]] = 1
    features.extend(scoring_vec)

    # --- Numeric features ---
    # Use available stats, set 0 if missing
    fg_pct = player.get("fg_pct", player.get("ts_pct", 0))
    three_pt_pct = player.get("three_pt_pct", 0)
    assists = player.get("assists", player.get("ast", 0))
    rebounds = player.get("rebounds", player.get("reb", 0))
    defense = player.get("defense", player.get("net_rating", 0))
    experience = player.get("experience", 0)

    features.extend([
        fg_pct,
        three_pt_pct,
        assists,
        rebounds,
        defense,
        experience
    ])

    return np.array(features, dtype=float)

# -----------------------------
# Convert a team (DataFrame) to a single vector
# -----------------------------
def create_team_vector(team_df):
    """
    Input: DataFrame of 5 players
    Output: concatenated feature vector
    """
    vectors = [encode_player(team_df.iloc[i]) for i in range(len(team_df))]
    return np.concatenate(vectors)
