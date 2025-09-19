import streamlit as st
import pandas as pd
from Select100_random import PlayerSelector
from MLP import train_mlp, select_optimal_team
from team_analysis import analyze_team
from visualization import plot_team_skills, plot_position_distribution

st.set_page_config(page_title="Optimal Basketball Team Selector", layout="wide")
st.title("🏀 Optimal Basketball Team Selector with Deep MLP")

# -----------------------------
# Step 1: Upload player dataset
# -----------------------------
uploaded_file = st.file_uploader("Upload your CSV file with player data", type=["csv"])

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
        st.success(f"Loaded dataset with {len(df)} players!")

        
        # -----------------------------
        # Robust experience calculation
        # -----------------------------
        # fill in any non-numeric field as NaN, then replace with 0
        df["draft_year"] = pd.to_numeric(df.get("draft_year", pd.Series([0]*len(df))), errors="coerce")

        # Extract season start year as integer (e.g., "2004-05" -> 2004)
        df["season_start_year"] = df.get("season", pd.Series(["0"]*len(df))).str[:4]
        df["season_start_year"] = pd.to_numeric(df["season_start_year"], errors="coerce").fillna(0).astype(int)

        # Compute experience (years since draft), fill NaN with 0
        df["experience"] = df["season_start_year"] - df["draft_year"]
        df["experience"] = df["experience"].fillna(0).astype(int)
        
        st.subheader("Dataset Preview")
        st.dataframe(df.head())

        # -----------------------------
        # Step 1b: Fix / map columns for MLP
        # -----------------------------
        # Map columns expected by team_features.py
        # position and scoring_type may be missing; they are handled in team_features
        if "experience" not in df.columns:
            # derive experience from draft_year and season
            if "draft_year" in df.columns and "season" in df.columns:
                df["experience"] = df["season"].str[:4].astype(int) - df["draft_year"]
            else:
                df["experience"] = 0  # fallback


        # -----------------------------
        # Step 2: Select 100-player pool
        # -----------------------------
        seasons_window = st.multiselect(
            "Select seasons to filter",
            options=df["season"].unique(),
            default=["2004-05", "2005-06", "2006-07", "2007-08", "2008-09"]
        )

        if len(seasons_window) == 0:
            st.warning("Please select at least one season.")
            st.stop()

        # Pass the DataFrame directly to PlayerSelector
        selector = PlayerSelector(df)
        try:
            player_pool = selector.select_random_players(n=100, seasons=seasons_window)
            st.success(f"Selected 100-player pool from seasons: {seasons_window}")
        except ValueError as e:
            st.error(str(e))
            st.stop()

        # -----------------------------
        # Step 3: Train MLP
        # -----------------------------
        st.subheader("MLP Training Parameters")
        epochs = st.slider("Number of training epochs", min_value=100, max_value=1000, value=500, step=50)
        num_samples = st.slider("Number of training samples", min_value=500, max_value=5000, value=2000, step=500)

        st.info("Training deep MLP... This may take a while depending on your dataset and epochs.")
        model = train_mlp(player_pool, epochs=epochs, num_samples=num_samples)
        st.success("MLP trained successfully!")

        # -----------------------------
        # Step 4: Predict optimal team
        # -----------------------------
        num_trials = st.slider("Number of team trials", min_value=1000, max_value=10000, value=5000, step=500)
        st.info("Predicting optimal 5-player team...")
        optimal_team = select_optimal_team(model, player_pool, num_trials=num_trials)

        st.subheader("Optimal 5-Player Team")
        st.dataframe(optimal_team)

        # -----------------------------
        # Step 5: Analyze and visualize team
        # -----------------------------
        st.subheader("Team Analysis")
        analysis_df = analyze_team(optimal_team)
        st.dataframe(analysis_df)

        st.subheader("Player Skill Contributions")
        plot_team_skills(analysis_df)

        st.subheader("Team Position Distribution")
        plot_position_distribution(analysis_df)

        st.success("Team selection and analysis complete!")

    except pd.errors.EmptyDataError:
        st.error("Uploaded file is empty or invalid. Please upload a valid CSV file.")
else:
    st.info("Please upload a CSV file to start.")
