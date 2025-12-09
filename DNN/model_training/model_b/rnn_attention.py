# Model B: RNN with Attention mechanism
# LSTM/GRU-based sentiment classifier
#
# Example usage:
#   # Initialize the model
#   model = SentimentRNN(vocab_size=10000, embedding_dim=100, hidden_dim=128)
#   
#   # Make predictions on a batch of reviews
#   input_sequences = torch.tensor([[145, 892, 23, 0], [67, 234, 891, 12]])  # 2 reviews
#   predictions, attention_weights = model(input_sequences)
#   # predictions: [[2.1, -1.3], [−0.5, 1.8]] (logits for negative/positive)
#   # attention_weights: [[0.1, 0.7, 0.2, 0], [0.3, 0.4, 0.25, 0.05]] (word importance)
#   
#   # Get probabilities and predicted classes
#   probs, preds, _ = model.predict(input_sequences)
#   # probs: [[0.95, 0.05], [0.15, 0.85]] (probabilities)
#   # preds: [0, 1] (0=negative, 1=positive)
#
# Architecture:
#   1. Embedding layer: Converts word indices to dense vectors
#   2. LSTM layer: Processes sequence, captures context
#   3. Attention layer: Focuses on important words
#   4. Dense layer: Maps to sentiment classes

import torch
import torch.nn as nn
import torch.nn.functional as F


class AttentionLayer(nn.Module):
    """
    Attention mechanism that learns to focus on important parts of the input.
    
    How it works:
    - LSTM produces hidden states for each word: [h1, h2, h3, ..., hn]
    - Attention learns importance scores: [0.1, 0.6, 0.05, ..., 0.25]
    - Creates weighted combination: context = 0.1*h1 + 0.6*h2 + 0.05*h3 + ...
    
    This lets the model focus on sentiment-bearing words like "excellent" 
    and ignore less important words like "the" or "it".
    
    Example: "The movie was absolutely excellent and entertaining"
                ^    ^     ^    ^         ^           ^
               0.02 0.05  0.08 0.15      0.50        0.20  <- attention weights
    The model learns to pay most attention to "excellent"
    """
    
    def __init__(self, hidden_dim: int):
        """
        Initialize attention layer.
        
        Args:
            hidden_dim: Dimension of LSTM hidden states
        """
        super(AttentionLayer, self).__init__()
        
        # Learn attention weights through a small neural network
        # This maps hidden states to attention scores
        self.attention = nn.Linear(hidden_dim, 1)
        
    def forward(self, lstm_output):
        """
        Compute attention weights and create context vector.
        
        Args:
            lstm_output: LSTM outputs for all time steps, shape (batch, seq_len, hidden_dim)
                        Example: (32, 50, 128) = 32 reviews, 50 words each, 128-dim hidden states
        
        Returns:
            context: Weighted sum of hidden states, shape (batch, hidden_dim)
            attention_weights: Importance scores for each word, shape (batch, seq_len)
        """
        # Step 1: Compute attention scores for each word
        # For each word's hidden state, predict how important it is
        # Shape: (batch, seq_len, 1)
        attention_scores = self.attention(lstm_output)
        
        # Step 2: Convert scores to probabilities using softmax
        # This ensures weights sum to 1.0 for each sequence
        # Example: raw scores [2.1, 0.5, 3.8] → weights [0.23, 0.04, 0.73]
        # Shape: (batch, seq_len, 1)
        attention_weights = F.softmax(attention_scores, dim=1)
        
        # Step 3: Create context vector as weighted sum
        # Multiply each hidden state by its attention weight and sum
        # This emphasizes important words, de-emphasizes unimportant ones
        # Shape: (batch, hidden_dim)
        context = torch.sum(attention_weights * lstm_output, dim=1)
        
        # Return context vector and attention weights (weights useful for visualization)
        return context, attention_weights.squeeze(-1)


class SentimentRNN(nn.Module):
    """
    RNN-based sentiment classifier with attention mechanism.
    
    Model flow:
        Input words → Embedding → LSTM → Attention → Dense → Softmax → Sentiment
        
    Example:
        "great movie" → [145, 892] → [[0.2, 0.5, ...], [0.8, 0.1, ...]] 
        → LSTM processing → Attention focuses on "great" 
        → [0.95, 0.05] → Positive (95% confidence)
    """
    
    def __init__(self, vocab_size: int, embedding_dim: int = 100, 
                 hidden_dim: int = 128, output_dim: int = 2, 
                 n_layers: int = 2, bidirectional: bool = True,
                 dropout: float = 0.5, pretrained_embeddings=None):
        """
        Initialize RNN sentiment classifier.
        
        Args:
            vocab_size: Number of words in vocabulary (e.g., 10000)
            embedding_dim: Dimension of word embeddings (e.g., 100, 300)
            hidden_dim: Dimension of LSTM hidden states (e.g., 128, 256)
            output_dim: Number of classes (2 for binary sentiment)
            n_layers: Number of stacked LSTM layers (more = more complex patterns)
            bidirectional: If True, LSTM reads text forwards AND backwards
                          Forward: "I love it" → captures "love" affects "it"
                          Backward: "ti evol I" → captures "it" is affected by "love"
            dropout: Probability of dropping neurons during training (prevents overfitting)
            pretrained_embeddings: Optional pre-trained word vectors (GloVe, FastText)
        """
        super(SentimentRNN, self).__init__()
        
        # Embedding layer: Maps word indices to dense vectors
        # Example: word 145 ("great") → [0.2, -0.5, 0.8, ..., 0.3] (100-dim vector)
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        
        # If we have pre-trained embeddings (GloVe/FastText), use them
        if pretrained_embeddings is not None:
            self.embedding.weight.data.copy_(torch.from_numpy(pretrained_embeddings))
            # Can optionally freeze embeddings: self.embedding.weight.requires_grad = False
        
        # LSTM layer: Processes sequence and captures context
        # Bidirectional LSTM reads text both ways for better context understanding
        self.lstm = nn.LSTM(
            embedding_dim,
            hidden_dim,
            num_layers=n_layers,
            bidirectional=bidirectional,
            dropout=dropout if n_layers > 1 else 0,  # Dropout between LSTM layers
            batch_first=True  # Input shape: (batch, seq_len, features)
        )
        
        # Attention layer: Learns which words are most important
        # If bidirectional, hidden_dim is doubled (forward + backward states)
        attention_dim = hidden_dim * 2 if bidirectional else hidden_dim
        self.attention = AttentionLayer(attention_dim)
        
        # Dropout for regularization (randomly zeros out neurons during training)
        # Prevents model from relying too heavily on specific features
        self.dropout = nn.Dropout(dropout)
        
        # Final classification layer: Maps context vector to sentiment scores
        # Example: [0.5, -0.2, 0.8, ...] (context) → [2.1, -1.3] → [0.95, 0.05] (probabilities)
        self.fc = nn.Linear(attention_dim, output_dim)
        
    def forward(self, text, text_lengths=None):
        """
        Forward pass through the network.
        
        Args:
            text: Input sequences, shape (batch, seq_len)
                 Example: [[145, 892, 23, 0, 0], [67, 234, 891, 12, 45]]
            text_lengths: Actual lengths before padding (optional, for efficiency)
                         Example: [3, 5] (first review is 3 words, second is 5)
        
        Returns:
            predictions: Class probabilities, shape (batch, output_dim)
                        Example: [[0.95, 0.05], [0.12, 0.88]] (2 reviews, 2 classes)
            attention_weights: Importance of each word, shape (batch, seq_len)
        """
        # Step 1: Embed words
        # Convert word indices to dense vectors
        # Shape: (batch, seq_len) → (batch, seq_len, embedding_dim)
        # Example: (32, 50) → (32, 50, 100)
        embedded = self.dropout(self.embedding(text))
        
        # Step 2: Pack padded sequences (optional, for efficiency)
        # This tells LSTM to ignore padding tokens
        if text_lengths is not None:
            embedded = nn.utils.rnn.pack_padded_sequence(
                embedded, text_lengths.cpu(), batch_first=True, enforce_sorted=False
            )
        
        # Step 3: Process through LSTM
        # LSTM reads the sequence and produces hidden states for each word
        # Shape: (batch, seq_len, hidden_dim * num_directions)
        # Example: (32, 50, 256) if bidirectional with hidden_dim=128
        lstm_output, (hidden, cell) = self.lstm(embedded)
        
        # Unpack if we packed earlier
        if text_lengths is not None:
            lstm_output, _ = nn.utils.rnn.pad_packed_sequence(lstm_output, batch_first=True)
        
        # Step 4: Apply attention
        # Attention learns which words are most important for sentiment
        # context contains the weighted combination of all hidden states
        # Shape: (batch, hidden_dim * num_directions)
        context, attention_weights = self.attention(lstm_output)
        
        # Step 5: Apply dropout for regularization
        context = self.dropout(context)
        
        # Step 6: Final classification
        # Map context vector to class scores
        # Shape: (batch, output_dim)
        # Example: (32, 2) for 32 reviews, 2 classes (positive/negative)
        predictions = self.fc(context)
        
        return predictions, attention_weights
    
    def predict(self, text, text_lengths=None):
        """
        Make predictions with probability scores.
        
        Args:
            text: Input sequences
            text_lengths: Actual lengths before padding
            
        Returns:
            probabilities: Softmax probabilities for each class
            predictions: Predicted class labels (0 or 1)
        """
        logits, attention_weights = self.forward(text, text_lengths)
        
        # Convert logits to probabilities using softmax
        # Example: [2.1, -1.3] → [0.95, 0.05]
        probabilities = F.softmax(logits, dim=1)
        
        # Get predicted class (highest probability)
        # Example: [0.95, 0.05] → 0 (negative)
        predictions = torch.argmax(probabilities, dim=1)
        
        return probabilities, predictions, attention_weights


def count_parameters(model):
    """
    Count trainable parameters in the model.
    Useful for understanding model complexity.
    
    Args:
        model: PyTorch model
        
    Returns:
        Number of trainable parameters
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
