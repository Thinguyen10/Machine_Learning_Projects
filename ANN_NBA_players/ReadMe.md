# 🏀 Optimal Basketball Team Selector with Deep MLP

This project uses **Artificial Neural Networks (ANN)** and **Streamlit** to build an interactive tool that selects an optimal 5-player basketball team based on player stats.

---

## Code Structure & Organization

- **`preprocess.py`**
  - Handles dataset cleaning and transformation.
  - Drops unwanted columns like `Unnamed`.
  - Converts categorical features (e.g., draft year, season) into numeric form.
  - Computes derived features like `experience`.

- **`team_features.py`**
  - Defines how individual players are encoded into numeric feature vectors.
  - Position and scoring type are **one-hot encoded**.
  - Creates team vectors by concatenating 5 player vectors.
  - Ensures consistent input format for the MLP.

- **`MLP.py`**
  - Defines the **Deep Multi-Layer Perceptron (MLP)** model.
  - Handles model training, loss calculation, and optimizer setup.
  - Selects the optimal team by simulating thousands of random lineups.

- **`team_analysis.py`**
  - Provides reusable functions to analyze an optimal team.
  - Calculates stats like offense score, total impact, and experience.
  - Assigns each player a position (PG, SG, SF, PF, C).
  - Keeps analysis logic independent of UI (no Streamlit inside).

- **`visualization.py`**
  - Generates charts for player skill contributions and position distribution.
  - Used by the Streamlit app to visually explain team choices.

- **`main.py`**
  - Streamlit app for user interaction.
  - Handles file upload, parameter selection, training, team selection, and analysis.
  - Uses `st.session_state` to enforce step-by-step flow.

---

## The Deep MLP Model

The core of this project is a **Deep Multi-Layer Perceptron (MLP)** implemented in PyTorch.  
It predicts a *team score* based on the combined stats of 5 players.

### Architecture
- **Input layer** → takes a team feature vector (all 5 players concatenated).  
- **Hidden Layer 1 (128 neurons, ReLU)** → learns broad interactions between player features.  
- **Hidden Layer 2 (64 neurons, ReLU)** → refines higher-level patterns.  
- **Hidden Layer 3 (32 neurons, ReLU)** → narrows down to strongest signals.  
- **Output Layer (1 neuron)** → predicts a single numeric *team score*.  

### Training
- **Dataset generation**:  
  - Random 5-player teams are sampled from the pool.  
  - Each team gets a **proxy score** = `PTS + AST + REB + NET + TS%` (sum of key stats).  
  - These pairs (team vector, proxy score) form the training dataset.  

- **Loss function**:  
  - **Mean Squared Error (MSE)** between the predicted score and proxy score.  
  - Guides the model to minimize the gap between its predictions and stat-based “ground truth.”  

- **Optimizer**:  
  - **Adam** optimizer adapts learning rate for efficient weight updates.  

- **Epochs**:  
  - One epoch = training over all sampled teams once.  
  - More epochs → more chances to adjust weights and reduce error.  

### Why ReLU?
- **ReLU (Rectified Linear Unit)** introduces non-linearity.  
- Ensures the model can learn complex patterns, not just straight-line (linear) relationships.  
- Also reduces vanishing gradient problems common in deep nets.  

### Team Selection
1. After training, the model evaluates thousands of random trial teams.  
2. Each trial team is scored by the MLP.  
3. The team with the **highest predicted score** is chosen as the **optimal lineup**.  

---

## Features & Data Handling
- **Position & scoring one-hot encoding**  
Each player in basketball is assigned a **position** that defines their role on the court. These are encoded numerically for the MLP:
- **PG – Point Guard**
  - Primary ball handler and playmaker.
  - Focused on assists, controlling the offense, and perimeter shooting.
- **SG – Shooting Guard**
  - Scoring-focused player, especially from mid-range or 3-point shots.
  - Often assists in defense and secondary playmaking.
- **SF – Small Forward**
  - Versatile scorer and defender.
  - Balances outside shooting, driving to the basket, and rebounding.
- **PF – Power Forward**
  - Strong inside player, mixes scoring and rebounding.
  - Often plays physically near the paint (post).
- **C – Center**
  - Tallest players, focus on rebounds, shot-blocking, and interior defense.
  - Anchors the team in the paint and contributes high-efficiency scoring close to the basket.
In the code, these positions are **one-hot encoded** so the model can understand each player’s role without assuming any ordinal relationship between positions.
- **Dropped `Unnamed` column**  
  - Removes CSV auto-generated index column issues.  
- **`encode_player` + `create_team_vector`**  
  - Standardized way to convert player/team into numeric arrays.  
- **Mapped optimal team to positions**  
  - Each player explicitly assigned a role in the final team.

---

##  UI Flow / Streamlit Improvements
- **Step 1: Upload CSV** → Cleaned and preprocessed automatically.  
- **Step 2: Choose seasons** → Define the pool of players.  
- **Step 3: Train Model**
- **Step 4: Reveal Optimal Team** → Shows 5 best players predicted by MLP.  
- **Step 5: Analyze Team**
  - *Reveal Positions & Stats* → Shows player positions + stats.  
  - *Player Skill Analysis* → Ranks players by **Total Impact** and visualizes contributions.  

---

## Key Concepts
- **Training Samples** → Number of random teams generated to train the model.  
- **Trials** → Number of random teams evaluated after training to find the best one.  
- **Total Impact** → A derived metric combining offense, assists, rebounds, defense.  
  - Helps identify which player contributes most overall.  

---

## How to Run
```bash
 - suggest: python -m venv .venv
            source .venv/bin/activate
pip install -r requirements.txt
streamlit run main.py
