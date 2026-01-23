import streamlit as st
import pandas as pd
from preprocessing import preprocess_data
from Select100_random import PlayerSelector
from MLP import train_mlp, select_optimal_team
from team_analysis import analyze_team

#initial sidebar state collapsed - no sidebar
st.set_page_config(page_title="Optimal Basketball Team Selector", layout="wide", initial_sidebar_state="collapsed")
st.title("🏀 Optimal Basketball Team Selector with Deep MLP")

# Introduction
st.markdown("""
### Welcome to the NBA Team Optimizer!

This application uses a **Deep Multi-Layer Perceptron (MLP)** neural network to predict and select the optimal 5-player basketball team 
based on historical NBA player statistics.

**Dataset:** NBA Player Statistics (1996-2023)  
The dataset contains comprehensive player performance metrics including points, rebounds, assists, net rating, 
usage percentage, true shooting percentage, and more across 27 NBA seasons.

**What This Model Does:**
- Analyzes player performance metrics from your selected 5 consecutive seasons
- Generates and evaluates thousands of possible team combinations
- Uses deep learning to predict team synergy and overall performance
- Identifies the optimal 5-player lineup that maximizes predicted team success

---
""")

# -----------------------------
# Step 1: Load pre-existing dataset
# -----------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("all_seasons.csv")
    df = preprocess_data(df)
    return df

try:
    df = load_data()
    st.success(f"Dataset loaded with {len(df)} players across {len(df['season'].unique())} seasons!")
    
    st.subheader("Dataset Preview")
    st.dataframe(df.head())

    # -----------------------------
    # Step 2: Select 5 consecutive seasons
    # -----------------------------
    st.subheader("Step 1: Select 5 Consecutive Seasons")
    
    # Get all available seasons sorted
    all_seasons = sorted([s for s in df["season"].unique() if isinstance(s, str) and '-' in s])
    
    # Create a slider to select the starting season
    if len(all_seasons) >= 5:
        start_idx = st.slider(
            "Choose the starting season (5 consecutive seasons will be selected):",
            min_value=0,
            max_value=len(all_seasons) - 5,
            value=8,  # Default starts at 2004-05 (index 8)
            format=""
        )
        
        seasons_window = all_seasons[start_idx:start_idx + 5]
        st.info(f"Selected seasons: **{', '.join(seasons_window)}**")
    else:
        st.error("Not enough seasons in the dataset")
        seasons_window = []

    if seasons_window:
        selector = PlayerSelector(df)

        # -----------------------------
        # Step 2: Generate Player Pool
        # -----------------------------
        st.subheader("Step 2: Generate 100-Player Pool")
        if st.button("Generate 100-Player Pool"):
            try:
                player_pool = selector.select_random_players(n=100, seasons=seasons_window)
                st.session_state["player_pool"] = player_pool  # store for later steps
                st.success(f"Generated 100-player pool from seasons: {', '.join(seasons_window)}")
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

    # -----------------------------
    # Dataset Credits & Technical Details
    # -----------------------------
    st.markdown("---")
    st.markdown("""
    ### 📊 Dataset Credits
    **Source:** NBA Player Statistics Dataset  
    **Coverage:** 1996-97 to 2022-23 seasons  
    **Metrics:** Player demographics, physical attributes, performance statistics, and advanced analytics
    
    This dataset aggregates player statistics from official NBA sources and includes advanced metrics 
    such as net rating, usage percentage, offensive/defensive rebound percentages, and true shooting percentage.
    """)
    
    col1, col2 = st.columns([1, 4])
    with col1:
        if st.button("📚 How the Model Works"):
            st.switch_page("pages/1_📚_Model_Architecture.py")

except Exception as e:
    st.error(f"Error loading dataset: {str(e)}")
