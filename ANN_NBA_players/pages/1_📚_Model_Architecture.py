import streamlit as st

#initial sidebar state collapsed - no sidebar
st.set_page_config(page_title="Model Architecture", page_icon="📚", layout="wide", initial_sidebar_state="collapsed")

st.title("📚 How the Model Works")
st.markdown("---")

st.markdown("""
### 🧮 Technical Overview: Deep MLP Architecture

Our system uses a sophisticated deep learning approach to identify optimal basketball team compositions 
from historical player performance data.

#### Model Architecture
Our Deep Multi-Layer Perceptron consists of:
- **Input Layer:** Player feature vectors (normalized statistics)
  - Takes aggregated team statistics as input
  - Features are normalized to ensure balanced learning
- **Hidden Layers:** Multiple dense layers with ReLU activation
  - Non-linear transformations capture complex player interactions
  - Dropout layers prevent overfitting
- **Output Layer:** Team performance prediction score
  - Single neuron with linear activation
  - Outputs predicted team success metric

#### Training Process
1. **Data Generation:** Creates thousands of random 5-player team combinations
   - Randomly samples 5 players from the 100-player pool
   - Ensures diversity in training samples
   
2. **Feature Engineering:** Aggregates individual player stats into team-level metrics
   - Sums: Total points, rebounds, assists
   - Averages: Net rating, shooting percentages
   - Weighted combinations of advanced metrics
   
3. **Target Variable:** Computes composite team score based on weighted performance metrics
   - Combines offensive and defensive capabilities
   - Accounts for synergy through statistical interactions
   
4. **Optimization:** Uses backpropagation with Adam optimizer to minimize prediction error
   - Loss function: Mean Squared Error (MSE)
   - Learning rate: Adaptive with Adam optimizer
   - Early stopping to prevent overfitting

#### Optimization Strategy
Once trained, the model is used to find the best team:

- **Candidate Generation:** Creates N random team combinations from the 100-player pool
  - Typically evaluates 5,000-10,000 candidate teams
  - Each candidate is a unique 5-player combination
  
- **Batch Prediction:** Evaluates all candidates using the trained MLP
  - Efficient batch processing for fast evaluation
  - Predicts performance score for each team
  
- **Selection:** Returns the team with the highest predicted performance score
  - Argmax operation identifies optimal team
  - Returns player details for the winning combination

#### Key Metrics Considered
- **Offensive Efficiency**
  - Points per game (pts)
  - True shooting percentage (ts_pct) - accounts for 2-pt, 3-pt, and free throws
  - Usage percentage (usg_pct) - measures offensive involvement
  
- **Defensive Capability**
  - Total rebounds (reb)
  - Offensive rebound percentage (oreb_pct)
  - Defensive rebound percentage (dreb_pct)
  
- **Playmaking Ability**
  - Assists per game (ast)
  - Assist percentage (ast_pct) - measures passing efficiency
  
- **Advanced Analytics**
  - Net rating (net_rating) - team's point differential per 100 possessions
  - Player Efficiency Rating (derived from multiple stats)
  
- **Team Balance**
  - Position diversity
  - Complementary skill sets
  - Height and physical attributes

#### Why Deep Learning?
Traditional approaches might simply sum player statistics, but this misses crucial synergies:
- A great shooter needs a great passer
- Defensive specialists enable offensive stars
- Role players complement superstars

The MLP learns these **non-linear relationships** between individual player statistics and overall team success, 
capturing synergies that simple additive models would miss. The deep architecture allows the model to learn 
hierarchical patterns: lower layers detect basic stat combinations, while deeper layers identify complex 
team chemistry patterns.

#### Model Performance Considerations
- **Training Size:** More training samples generally improve predictions
- **Epochs:** More training iterations help the model converge
- **Candidate Trials:** More optimization trials increase chance of finding the true optimum
- **Overfitting Prevention:** Dropout and validation monitoring ensure generalization

""")

st.markdown("---")
st.info("💡 **Tip:** Experiment with different hyperparameters to see how they affect the selected team!")

# Add a button to go back to main page
if st.button("← Back to Team Selector"):
    st.switch_page("main.py")
