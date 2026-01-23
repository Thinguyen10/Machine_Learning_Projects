import streamlit as st
import pandas as pd
import os
from preprocessing import preprocess_data
from Select100_random import PlayerSelector
from MLP import train_mlp, select_optimal_team
from team_analysis import analyze_team

# Configure page settings - wide layout, no sidebar
st.set_page_config(page_title="Optimal Basketball Team Selector", layout="wide", initial_sidebar_state="collapsed")

# CSS to completely hide the sidebar (removes the blank space on the left)
st.markdown("""
<style>
    /* Hide the sidebar completely */
    [data-testid="stSidebar"] {display: none;}
    /* Hide the sidebar collapse button */
    [data-testid="collapsedControl"] {display: none;}
    /* Make main content use full width */
    .main .block-container {max-width: 100%; padding-left: 5rem; padding-right: 5rem;}
</style>
""", unsafe_allow_html=True)

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
# Load pre-existing dataset (cached for performance)
# -----------------------------
@st.cache_data
def load_data():
    """Load and preprocess the NBA player statistics CSV file."""
    # Try multiple path strategies for both local and Streamlit Cloud deployment
    possible_paths = [
        "all_seasons.csv",  # Same directory (works on Streamlit Cloud)
        os.path.join(os.path.dirname(__file__), "all_seasons.csv"),  # Relative to script
        os.path.join(os.getcwd(), "all_seasons.csv"),  # Current working directory
    ]
    
    # Try each path until one works
    for csv_path in possible_paths:
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            df = preprocess_data(df)
            return df
    
    # If no path works, raise informative error
    raise FileNotFoundError(
        f"Could not find 'all_seasons.csv'. Tried paths:\n" + 
        "\n".join(f"  - {p}" for p in possible_paths) +
        f"\n\nCurrent directory: {os.getcwd()}\n" +
        f"Files in current directory: {os.listdir(os.getcwd())}"
    )

try:
    # Load data automatically when page loads
    df = load_data()
    st.success(f"Dataset loaded with {len(df)} players across {len(df['season'].unique())} seasons!")
    
    st.subheader("Dataset Preview")
    st.dataframe(df.head())

    # -----------------------------
    # STEP 1: Select 5 consecutive seasons
    # -----------------------------
    st.subheader("Step 1: Select 5 Consecutive Seasons")
    
    # Get all available seasons from dataset, filter out invalid entries, and sort
    all_seasons = sorted([s for s in df["season"].unique() if isinstance(s, str) and '-' in s])
    
    # Create a slider to select the starting season (will auto-select 5 consecutive seasons)
    if len(all_seasons) >= 5:
        start_idx = st.slider(
            "Choose the starting season (5 consecutive seasons will be selected):",
            min_value=0,
            max_value=len(all_seasons) - 5,
            value=8,  # Default starts at 2004-05 (index 8)
            format=""
        )
        
        # Get the 5 consecutive seasons starting from the selected index
        seasons_window = all_seasons[start_idx:start_idx + 5]
        st.info(f"Selected seasons: **{', '.join(seasons_window)}**")
        
        # Store seasons in session state to track completion of Step 1
        st.session_state["seasons_selected"] = True
        st.session_state["seasons_window"] = seasons_window
    else:
        st.error("Not enough seasons in the dataset")
        st.session_state["seasons_selected"] = False
        seasons_window = []

    # -----------------------------
    # STEP 2: Generate 100-Player Pool
    # Only show this step if Step 1 is complete (seasons selected)
    # -----------------------------
    if st.session_state.get("seasons_selected", False):
        selector = PlayerSelector(df)
        
        st.subheader("Step 2: Generate 100-Player Pool")
        st.write("Click the button below to randomly select 100 players from your chosen seasons.")
        
        if st.button("Generate 100-Player Pool"):
            try:
                # Randomly select 100 players from the selected seasons
                player_pool = selector.select_random_players(n=100, seasons=st.session_state["seasons_window"])
                
                # Store player pool in session state for later steps
                st.session_state["player_pool"] = player_pool
                st.session_state["pool_generated"] = True  # Mark Step 2 as complete
                
                st.success(f"Generated 100-player pool from seasons: {', '.join(st.session_state['seasons_window'])}")
                st.dataframe(player_pool.head())
            except ValueError as e:
                st.error(str(e))

    # -----------------------------
    # STEP 3: Train MLP Model
    # Only show this step if Step 2 is complete (player pool generated)
    # -----------------------------
    if st.session_state.get("pool_generated", False):
        st.subheader("Step 3: Train MLP Model")
        st.write("Configure and train the neural network on randomly generated team combinations.")
        
        # Sliders to configure training hyperparameters
        epochs = st.slider("Number of training epochs", 100, 1000, 500, step=50)
        num_samples = st.slider("Number of training samples — *how many teams generated for training*", 500, 5000, 2000, step=500)

        if st.button("Train Model"):
            st.info("Training deep MLP... please wait ⏳")
            
            # Train the MLP model on the player pool
            model = train_mlp(st.session_state["player_pool"], epochs=epochs, num_samples=num_samples)
            
            # Store trained model in session state for prediction
            st.session_state["mlp_model"] = model
            st.session_state["model_trained"] = True  # Mark Step 3 as complete
            
            st.success("MLP trained successfully!")

    # -----------------------------
    # STEP 4: Predict Optimal Team
    # Only show this step if Step 3 is complete (model trained)
    # -----------------------------
    if st.session_state.get("model_trained", False):
        st.subheader("Step 4: Predict Optimal Team")
        st.write("Use the trained model to find the best 5-player combination from your pool.")
        
        # Slider to configure how many team combinations to test
        num_trials = st.slider("Number of team trials — *how many optimal teams tested*", 1000, 10000, 5000, step=500)
        
        if st.button("Predict Optimal Team"):
            st.info("Selecting optimal team...")
            
            # Use the trained model to find the optimal team
            optimal_team = select_optimal_team(st.session_state["mlp_model"], st.session_state["player_pool"], num_trials=num_trials)
            
            # Store optimal team in session state for analysis
            st.session_state["optimal_team"] = optimal_team
            st.session_state["team_predicted"] = True  # Mark Step 4 as complete

            st.subheader("Optimal 5-Player Team")
            st.dataframe(optimal_team)

    # -----------------------------
    # STEP 5: Analyze Team
    # Only show this step if Step 4 is complete (team predicted)
    # -----------------------------
    if st.session_state.get("team_predicted", False):
        st.subheader("Step 5: Team Position Distribution & Skill Analysis")
        st.write("Analyze the predicted team's composition and individual player contributions.")
        
        # Button to reveal position analysis
        if st.button("Reveal Positions"):
            # Analyze team positions and skills
            st.session_state["analysis_df"] = analyze_team(st.session_state["optimal_team"])
            st.session_state["positions_revealed"] = True  # Mark positions as revealed
            st.dataframe(st.session_state["analysis_df"])

    # Show detailed player skill analysis if positions have been revealed
    if st.session_state.get("positions_revealed", False):
        if st.button("Player Skill Analysis"):
            st.write("Sorted by **Total Impact** (overall contribution) descending:")
            # Sort by total impact to see which players contribute most
            st.dataframe(st.session_state["analysis_df"].sort_values(by="total_impact", ascending=False))
            st.session_state["analysis_complete"] = True  # Mark all analysis as complete

        # Show completion message if full analysis is done
        if st.session_state.get("analysis_complete", False):
            st.success("Team selection and analysis complete!")

    # -----------------------------
    # Dataset Credits & Technical Details
    # Always show at the bottom for reference
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
