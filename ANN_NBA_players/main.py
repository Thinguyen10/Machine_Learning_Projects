import streamlit as st
import pandas as pd
from Select100_random import PlayerSelector
from MLP import train_mlp, select_optimal_team
from team_analysis import analyze_team
from visualization import plot_team_skills, plot_position_distribution

st.set_page_config(page_title="Optimal Basketball Team Selector", layout="wide")
st.title("🏀 Optimal Basketball Team Selector with Deep MLP")

# -----------------------------
# Step 1: Upload dataset
# -----------------------------
st.subheader("Step 1: Upload Your Basketball Dataset ")
uploaded_file = st.file_uploader("CSV File", type=["csv"])

if uploaded_file:
    try:
        df = pd.read_csv(uploaded_file)
        st.success(f"Loaded dataset with {len(df)} players!")

        # Robust experience calculation
        df["draft_year"] = pd.to_numeric(df.get("draft_year", pd.Series([0]*len(df))), errors="coerce")
        df["season_start_year"] = df.get("season", pd.Series(["0"]*len(df))).str[:4]
        df["season_start_year"] = pd.to_numeric(df["season_start_year"], errors="coerce").fillna(0).astype(int)
        df["experience"] = (df["season_start_year"] - df["draft_year"]).fillna(0).astype(int)

        st.subheader("Dataset Preview")
        st.dataframe(df.head())

        # -----------------------------
        # Step 2: Select seasons
        # -----------------------------
        st.subheader("Step 2: Select Seasons to Filter — Within a 5 year Window")
        seasons_window = st.multiselect(
            "Select seasons to include:",
            options=df["season"].unique(),
            default=["2004-05", "2005-06", "2006-07", "2007-08", "2008-09"]
        )

        if seasons_window:
            selector = PlayerSelector(df)

            if st.button("Generate 100-Player Pool"):
                try:
                    player_pool = selector.select_random_players(n=100, seasons=seasons_window)
                    st.session_state["player_pool"] = player_pool  # store for later steps
                    st.success(f"Selected 100-player pool from seasons: {seasons_window}")
                    st.dataframe(player_pool.head())
                except ValueError as e:
                    st.error(str(e))

        # -----------------------------
        # Step 3: Train MLP
        # -----------------------------
        if "player_pool" in st.session_state:
            st.subheader("Step 3: Train MLP Model")

            epochs = st.slider("Number of training epochs", 100, 1000, 500, step=50)
            num_samples = st.slider("Number of training samples", 500, 5000, 2000, step=500)

            if st.button("Train Model"):
                st.info("Training deep MLP... please wait ⏳")
                model = train_mlp(st.session_state["player_pool"], epochs=epochs, num_samples=num_samples)
                st.session_state["mlp_model"] = model
                st.success("MLP trained successfully!")

        # -----------------------------
        # Step 4: Predict optimal team
        # -----------------------------
        if "mlp_model" in st.session_state:
            st.subheader("Step 4: Predict Optimal Team")

            num_trials = st.slider("Number of team trials", 1000, 10000, 5000, step=500)
            if st.button("Predict Optimal Team"):
                st.info("Selecting optimal team...")
                optimal_team = select_optimal_team(st.session_state["mlp_model"], st.session_state["player_pool"], num_trials=num_trials)
                st.session_state["optimal_team"] = optimal_team

                st.subheader("Optimal 5-Player Team")
                st.dataframe(optimal_team)

        # -----------------------------
        # Step 5: Analyze and visualize team
        # -----------------------------
        if "optimal_team" in st.session_state:
            st.subheader("Step 5: Team Analysis & Visualization")

            analysis_df = analyze_team(st.session_state["optimal_team"])
            st.dataframe(analysis_df)

            st.subheader("📈 Player Skill Contributions")
            plot_team_skills(analysis_df)

            st.subheader("📊 Team Position Distribution")
            plot_position_distribution(analysis_df)

            st.success("✅ Team selection and analysis complete!")

    except pd.errors.EmptyDataError:
        st.error("Uploaded file is empty or invalid. Please upload a valid CSV file.")

else:
    st.info("Please upload a CSV file to start.")
