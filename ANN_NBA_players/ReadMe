# 🏀 ANN NBA Team Optimizer

This project builds and analyzes NBA teams using an Artificial Neural Network (ANN) with PyTorch + Streamlit.  
Upload your player dataset, train the model, and reveal your optimal 5-player lineup with analysis.

---

## Code Structure & Organization

- **`preprocess.py`**  
  - Cleans raw CSV data.  
  - Drops auto-generated `Unnamed` index column.  
  - Converts categorical features to numeric form.  

- **`team_features.py`**  
  - Defines **position** and **scoring type** one-hot encodings.  
  - `encode_player()` → translates individual player stats into numeric features.  
  - `create_team_vector()` → concatenates 5 players into one team vector.  

- **`mlp_model.py`**  
  - Defines the `DeepMLP` class (multi-layer perceptron).  
  - Input: encoded team vector.  
  - Hidden layers with ReLU activations (128 → 64 → 32).  
  - Output: predicted **team score**.  
  - Includes training loop with MSE loss + Adam optimizer.  

- **`team_analysis.py`**  
  - `analyze_team()` → computes per-player and team metrics (offense score, total impact, etc.).  
  - `print_team_analysis()` → console/terminal output (keeps logic separate from UI).  

- **`main.py`**  
  - Streamlit app for user interaction.  
  - Step-by-step workflow: Upload → Preprocess → Train Model → Reveal Team → Analyze.  
  - UI-only logic: buttons, headers, session state.  

---

##  Features & Data Handling

- **Position & scoring encodings**  
  - Positions: `PG, SG, SF, PF, C`  
  - Scoring style: `Shooter, Slasher, All-Around`  
  - Encoded numerically so the ANN can process them.  

- **Optimal team position assignment**  
  - Selected team is mapped explicitly to 5 roles.  
  - Ensures lineup balance.  

- **Proxy team score for training**  
  - Combines key stats (PTS + AST + REB + NET + TS%).  
  - Used as supervised label for ANN training (since real outcomes aren’t available).  

---

## UI Flow / Streamlit

1. Upload CSV with player stats.  
2. Preprocess data (auto).  
3. **Click “Train Model”** → trains ANN on random 5-player teams.  
4. Reveal the predicted **optimal team**.  
5. Analyze lineup:  
   - **Reveal Positions & Stats** → shows assigned roles + raw stats.  
   - **Player Skill Analysis** → sorts by *Total Impact* and highlights contribution.  

---

## Machine Learning & Training

- **Training samples**  
  - Thousands of random 5-player teams are generated.  
  - Each team → feature vector + proxy score.  
  - Provides supervised training dataset.  

- **Trials for optimal team**  
  - After training, thousands of trial teams are evaluated.  
  - Model predicts scores, and the highest-scoring team is chosen.  

- **Epochs & Loss**  
  - **Epoch** = one full pass through training samples.  
  - **Loss** = Mean Squared Error (MSE) between predicted score and proxy score.  
  - ANN learns to approximate the scoring function, refining weights each epoch.  

---

##  Example Output

- **Optimal Team Prediction** → best 5-player lineup with highest predicted score.  
- **Analysis** → sorted table of contributions (`Total Impact` = offense + assists + rebounds + defense).  

---

## 🔹 Why This Structure?

✅ Separation of concerns → modular, testable, clean.  
✅ ANN model isolated from UI → can be tested or swapped easily.  
✅ Reusable preprocessing & analysis functions.  
✅ Step-based UI → logical user flow without clutter.  

---
