import streamlit as st
import pandas as pd
from preprocessing import preprocess_data
from Select100_random import PlayerSelector
from MLP import train_mlp, select_optimal_team
from team_analysis import analyze_team


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
        df = preprocess_data(df)
        st.success(f"Loaded dataset with {len(df)} players!")

        
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
                    st.success(f"Preview 100-player pool from seasons: {seasons_window}")
                    st.dataframe(player_pool.head())
                except ValueError as e:
                    st.error(str(e))

        # -----------------------------
        # Step 3: Train MLP
        # -----------------------------
        if "player_pool" in st.session_state:
            st.subheader("Step 3: Train MLP Model")

            epochs = st.slider("Number of training epochs", 100, 1000, 500, step=50)
            num_samples = st.slider("Number of training samples — *how many teams generated for training*", 500, 5000, 2000, step=500)

            if st.button("Train Model"):
                st.info("Training deep MLP... please wait ⏳") # blue info box
                model = train_mlp(st.session_state["player_pool"], epochs=epochs, num_samples=num_samples)
                st.session_state["mlp_model"] = model
                st.success("MLP trained successfully!") # green success box

        # -----------------------------
        # Step 4: Predict optimal team
        # -----------------------------
        if "mlp_model" in st.session_state:
            st.subheader("Step 4: Predict Optimal Team")

            num_trials = st.slider("Number of team trials — *how many optimal teams tested*", 1000, 10000, 5000, step=500)
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
            st.subheader("Step 5: Team Position Distribution & Skill Analysis")
            
            if st.button("Reveal Positions"):
                st.session_state["analysis_df"]  = analyze_team(st.session_state["optimal_team"])
                st.dataframe(st.session_state["analysis_df"])

        if "analysis_df" in st.session_state:
            if st.button("Player Skill Analysis"):
                st.write("Sorted by **Total Impact** (overall contribution) descending:")
                st.dataframe(st.session_state["analysis_df"].sort_values(by="total_impact", ascending=False))

            st.success("Team selection and analysis complete!")

    except pd.errors.EmptyDataError:
        st.error("Uploaded file is empty or invalid. Please upload a valid CSV file.")

else:
    st.info("Please upload a CSV file to start.")
