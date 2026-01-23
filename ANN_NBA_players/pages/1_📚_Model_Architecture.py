import streamlit as st

# Configure page settings - wide layout, no sidebar
st.set_page_config(page_title="Model Architecture", page_icon="📚", layout="wide", initial_sidebar_state="collapsed")

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

st.title("📚 How the Model Works")
st.markdown("---")

st.markdown("""
### 🧮 Deep Multi-Layer Perceptron (MLP) Explained

This project uses an **Artificial Neural Network (ANN)** called a **Deep Multi-Layer Perceptron** to predict which combination 
of 5 basketball players will perform best together. Think of it as a smart calculator that learns patterns from thousands 
of team combinations to find the ultimate starting lineup!

---

### 🏗️ Model Architecture (The Brain Structure)

Imagine the neural network as a series of connected layers, like floors in a building. Each layer processes information 
and passes it to the next level:

#### **Input Layer** → The Starting Point
- Takes in the **combined stats of 5 players** (like points, assists, rebounds, shooting %)
- All player data is concatenated (joined together) into one long feature vector
- Example: Player 1's stats + Player 2's stats + ... + Player 5's stats = Team Vector
- Features are **normalized** (scaled to similar ranges) so no single stat dominates

#### **Hidden Layer 1 (128 neurons, ReLU activation)** 
- **What it does:** Learns broad patterns and interactions between player features
- **Why 128 neurons?** More neurons = more capacity to detect different patterns
- **ReLU (Rectified Linear Unit):** An activation function that adds non-linearity
  - Without ReLU, the network would only learn straight-line (linear) relationships
  - ReLU allows it to learn complex curves and interactions (e.g., "great passer + great shooter = amazing combo")

#### **Hidden Layer 2 (64 neurons, ReLU activation)**
- **What it does:** Refines the patterns found in Layer 1, focuses on higher-level team dynamics
- **Fewer neurons:** Gradually narrows down from 128 → 64 to filter out noise and keep strong signals
- **Learns synergies:** Detects how certain player types complement each other

#### **Hidden Layer 3 (32 neurons, ReLU activation)**
- **What it does:** Final refinement stage, identifies the strongest predictive patterns
- **32 neurons:** Even more selective, keeping only the most important team chemistry indicators
- **Deep = Hierarchical:** Each layer builds on previous layers to learn increasingly sophisticated patterns

#### **Output Layer (1 neuron, Linear activation)**
- **What it does:** Produces a single number—the **predicted team score**
- **Higher score = Better team:** The model's prediction of how well this 5-player lineup will perform
- **Linear activation:** No transformation, just outputs the raw predicted value

---

### 📚 Training Process (How the Model Learns)

The model doesn't magically know which teams are good—it has to **learn** by seeing many examples. 
Here's how training works:

#### **Step 1: Generate Training Data**
- The system creates **thousands of random 5-player teams** from your 100-player pool
- Each random team becomes one training example
- More training samples = more diverse examples for the model to learn from

#### **Step 2: Calculate "Ground Truth" Scores**
- For each random team, we calculate a **proxy score** (a simplified measure of team strength)
- **Proxy Score Formula:** `Points + Assists + Rebounds + Net Rating + True Shooting %`
- This isn't perfect, but it gives the model a target to aim for
- These scores act as the "answer key" during training

#### **Step 3: Make Predictions**
- The neural network looks at each team's stats and tries to predict its score
- At first, predictions are completely random (the network hasn't learned anything yet!)

#### **Step 4: Calculate Loss (Error)**
- **Loss Function:** Mean Squared Error (MSE)
- Measures the difference between the model's prediction and the actual proxy score
- Example: Model predicts 150, actual score is 200 → Loss = (150-200)² = 2500
- **Goal:** Make this loss as small as possible

#### **Step 5: Update Weights (Learn from Mistakes)**
- The model uses **backpropagation** to figure out which weights (internal numbers) caused the error
- **Adam Optimizer:** A smart algorithm that adjusts the weights to reduce future errors
- It's like a teacher correcting homework—each correction makes the model slightly better

#### **Step 6: Repeat (Epochs)**
- **One Epoch = One pass through all training samples**
- With each epoch, the model gets better at predicting team scores
- More epochs = more learning opportunities (but too many can cause overfitting!)

---

### 🎯 Finding the Optimal Team (Team Selection)

After training, the model has learned what makes a good team. Now we use it to find the **best** team:

1. **Generate Candidate Teams:** Create thousands of random 5-player combinations (e.g., 5,000-10,000)
2. **Score Each Team:** Run each candidate through the trained MLP to get a predicted score
3. **Pick the Winner:** Select the team with the **highest predicted score** as your optimal lineup!

**Why so many trials?** With 100 players, there are millions of possible 5-player combinations. Testing more combinations 
increases the chance of finding the true best team!

---

### 🏀 Basketball Positions Explained

Each player is assigned a position that defines their role. These are **one-hot encoded** (converted to binary numbers) 
so the model can understand them:

- **PG – Point Guard** 🎯
  - The "floor general" who runs the offense
  - Focuses on assists, ball handling, and setting up plays
  
- **SG – Shooting Guard** 🏹
  - Primary scoring threat, especially from long range
  - Strong shooter and secondary playmaker
  
- **SF – Small Forward** ⚔️
  - Versatile player who can shoot, drive, and defend
  - Balanced between inside and outside play
  
- **PF – Power Forward** 💪
  - Physical player who dominates near the basket
  - Strong rebounder and interior scorer
  
- **C – Center** 🏔️
  - Tallest player, anchors defense and rebounding
  - Protects the paint, blocks shots, scores close to the rim

A balanced team typically has all 5 positions covered to maximize versatility!

---

### 📊 Key Metrics the Model Considers

- **Offensive Efficiency**
  - Points per game (pts) — scoring ability
  - True shooting percentage (ts_pct) — scoring efficiency (accounts for 2pt, 3pt, free throws)
  - Usage percentage (usg_pct) — how much of the offense runs through this player
  
- **Defensive Capability**
  - Rebounds (reb) — securing possession after missed shots
  - Offensive/Defensive rebound % — rebounding efficiency
  
- **Playmaking Ability**
  - Assists (ast) — setting up teammates for scores
  - Assist percentage (ast_pct) — how often passes lead to baskets
  
- **Advanced Analytics**
  - Net rating (net_rating) — team's point differential per 100 possessions when player is on court
  - Experience — years in the league (maturity and skill development)

---

### 🤔 Why Use Deep Learning Instead of Simple Math?

**Traditional Approach:** Just add up player stats → Team Score  
**Problem:** This misses crucial team chemistry!

**Deep Learning Approach:** Learn complex relationships between players  
**Benefits:**
- Understands that a great shooter needs a great passer to be effective
- Recognizes that defensive specialists enable offensive stars to take more risks
- Detects position balance (can't have 5 centers!)
- Learns non-linear synergies that simple addition can't capture

The **deep** architecture (multiple layers) allows the model to learn **hierarchical patterns**:
- **Lower layers:** Detect basic stat combinations
- **Deeper layers:** Identify sophisticated team chemistry patterns

---

### ⚙️ Hyperparameters You Can Tune

- **Training Samples (500-5000):** How many random teams to generate for training
  - More samples = more diverse learning examples = better predictions
  
- **Epochs (100-1000):** How many times to iterate through all training samples
  - More epochs = more learning time = better convergence (but risk of overfitting)
  
- **Optimization Trials (1000-10000):** How many candidate teams to test after training
  - More trials = higher chance of finding the true optimal team

---

### 📂 Dataset Information

**Source:** [NBA Player Statistics Dataset on Kaggle](https://www.kaggle.com/datasets/justinas/nba-players-data)  
**Coverage:** 1996-97 to 2022-23 NBA seasons  
**Players:** 12,000+ player-season records  
**Metrics:** Demographics, physical attributes, performance statistics, and advanced analytics

This dataset aggregates official NBA statistics and includes:
- Basic stats: Points, rebounds, assists, games played
- Shooting efficiency: True shooting %, usage %, field goal %
- Advanced metrics: Net rating, offensive/defensive rebound %, assist %
- Player info: Height, weight, college, country, draft information

---

### 💡 Tips for Best Results

- **Start with default settings** to understand the baseline
- **Increase training samples** if predictions seem random
- **Increase epochs** if the model hasn't converged (loss still decreasing)
- **Increase trials** to find better optimal teams
- **Experiment with different season ranges** to see how player eras affect optimal lineups!

""")

st.markdown("---")

# Add a button to go back to main page
if st.button("← Back to Team Selector"):
    st.switch_page("main.py")
