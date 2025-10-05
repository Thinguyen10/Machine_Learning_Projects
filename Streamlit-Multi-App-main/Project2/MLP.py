import torch  # PyTorch library for deep learning: tensor operations, auto-differentiation
import torch.nn as nn  # Provides modules to build neural network layers
import torch.optim as optim  # Optimizers like Adam for training
from team_features import create_team_vector
import numpy as np
import pandas as pd

# ------------------------------
# Define the Deep MLP
# ------------------------------
class DeepMLP(nn.Module):
    """
    Deep multi-layer perceptron (MLP) to predict a team's score based on player features.
    """
    def __init__(self, input_size):
        super(DeepMLP, self).__init__()
        # Fully connected layers
        self.fc1 = nn.Linear(input_size, 128)  # Input layer -> hidden layer 1
        self.fc2 = nn.Linear(128, 64)          # hidden layer 1 -> hidden layer 2
        self.fc3 = nn.Linear(64, 32)           # hidden layer 2 -> hidden layer 3
        self.output = nn.Linear(32, 1)         # hidden layer 3 -> output layer (team score)
        
        # ReLU activation: introduces non-linearity to help the network model complex patterns
        self.relu = nn.ReLU()

    def forward(self, x):
        """
        Forward pass: propagates input through the network to produce output.
        """
        x = self.relu(self.fc1(x))  # Apply ReLU after first hidden layer
        x = self.relu(self.fc2(x))  # Apply ReLU after second hidden layer
        x = self.relu(self.fc3(x))  # Apply ReLU after third hidden layer
        x = self.output(x)           # Output layer (no activation since it's regression)
        return x

# ------------------------------
# Calculate a simple team score from stats
# ------------------------------
def calculate_team_score(team_df):
    """
    Sum of key numeric features as a proxy for team quality.
    This is our "target" for the MLP to learn.
    """
    score = 0
    for _, player in team_df.iterrows():
        pts = player.get("pts", 0)
        ast = player.get("ast", 0)
        reb = player.get("reb", 0)
        net = player.get("net_rating", 0)
        ts = player.get("ts_pct", 0)
        score += pts + ast + reb + net + ts
    return score

# ------------------------------
# Training function
# ------------------------------
def train_mlp(player_pool, epochs=500, num_samples=2000, lr=0.001):
    """
    Train the DeepMLP to predict team scores based on player features.

    Parameters:
    - player_pool: DataFrame with player stats
    - epochs: number of training iterations
    - num_samples: number of random 5-player teams to sample
    - lr: learning rate for optimizer
    """
    # Replace missing numeric values with 0 to avoid errors in torch tensors
    player_pool = player_pool.fillna(0)

    # Determine input size dynamically from 5-player team vector
    input_size = len(create_team_vector(player_pool.iloc[:5]))
    model = DeepMLP(input_size)

    # Mean Squared Error (MSE) loss: suitable for regression tasks
    criterion = nn.MSELoss()
    
    # Adam optimizer: adjusts weights efficiently using gradients
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Generate training dataset
    X, y = [], []
    all_indices = list(range(len(player_pool)))
    for _ in range(num_samples):
        team_indices = np.random.choice(all_indices, 5, replace=False)
        team_df = player_pool.iloc[team_indices]
        team_vector = create_team_vector(team_df)
        score = calculate_team_score(team_df)
        X.append(team_vector)
        y.append(score)
    
    # Convert to torch tensors for training
    X = torch.tensor(np.array(X), dtype=torch.float32)
    y = torch.tensor(np.array(y), dtype=torch.float32).unsqueeze(1)

    # Training loop
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()       # Reset gradients before backpropagation
        outputs = model(X)          # Forward pass
        loss = criterion(outputs, y) # Compute error
        loss.backward()             # Backpropagation: compute gradients
        optimizer.step()            # Update weights
        if (epoch + 1) % 50 == 0:
            print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")

    return model

# ------------------------------
# Predict optimal team
# ------------------------------
def select_optimal_team(model, player_pool, num_trials=5000):
    """
    Sample random teams and use the trained MLP to select the team with highest predicted score.
    """
    best_score = -np.inf
    best_team = None
    all_indices = list(range(len(player_pool)))

    for _ in range(num_trials):
        team_indices = np.random.choice(all_indices, 5, replace=False)
        team_df = player_pool.iloc[team_indices].fillna(0)
        team_vector = torch.tensor(create_team_vector(team_df), dtype=torch.float32).unsqueeze(0)
        pred_score = model(team_vector).item()
        if pred_score > best_score:
            best_score = pred_score
            best_team = team_df

    print(f"Predicted optimal team score: {best_score:.2f}")
    return best_team
