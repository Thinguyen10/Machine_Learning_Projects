# model_pytorch.py
# This module defines the CNN architecture for skin cancer detection using PyTorch

import torch
import torch.nn as nn
import torch.nn.functional as F

class SkinCancerCNN(nn.Module):
    """
    Build a Convolutional Neural Network (CNN) using PyTorch.
    
    Arguments:
    - num_classes: int, number of classes in the dataset (default: 9)
    
    Architecture:
    - 3 Convolutional blocks with MaxPooling
    - Fully connected layers
    - Softmax output for classification
    """
    
    def __init__(self, num_classes=9):
        super(SkinCancerCNN, self).__init__()
        
        # 1st Convolutional Layer
        # Detects low-level features like edges and corners
        self.conv1 = nn.Conv2d(
            in_channels=3,         # RGB images
            out_channels=32,       # Number of feature maps
            kernel_size=3,         # 3x3 convolution window
            padding=1              # Keep output size same as input
        )
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)  # Max pooling reduces spatial dimensions
        
        # 2nd Convolutional Layer
        # Detects more complex features
        self.conv2 = nn.Conv2d(
            in_channels=32,
            out_channels=64,       # More feature maps to capture complex patterns
            kernel_size=3,
            padding=1
        )
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # 3rd Convolutional Layer
        # Detects even higher-level patterns
        self.conv3 = nn.Conv2d(
            in_channels=64,
            out_channels=128,      # Further increase in feature maps
            kernel_size=3,
            padding=1
        )
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Calculate flattened size: 64x64 image -> after 3 pooling layers -> 8x8
        # 8 * 8 * 128 = 8192
        self.fc1 = nn.Linear(8 * 8 * 128, 128)  # Fully connected layer
        self.fc2 = nn.Linear(128, num_classes)   # Output layer
        
        # Dropout for regularization
        self.dropout = nn.Dropout(0.5)
        
    def forward(self, x):
        """
        Forward pass through the network.
        
        Arguments:
        - x: input tensor of shape (batch_size, 3, 64, 64)
        
        Returns:
        - output: tensor of shape (batch_size, num_classes)
        """
        # Conv block 1: 64x64x3 -> 64x64x32 -> 32x32x32
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        
        # Conv block 2: 32x32x32 -> 32x32x64 -> 16x16x64
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        
        # Conv block 3: 16x16x64 -> 16x16x128 -> 8x8x128
        x = F.relu(self.conv3(x))
        x = self.pool3(x)
        
        # Flatten: 8x8x128 -> 8192
        x = x.view(x.size(0), -1)
        
        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x  # Raw logits (use CrossEntropyLoss which applies softmax)

def create_model(num_classes=9, learning_rate=0.001):
    """
    Create and return the CNN model with optimizer and loss function.
    
    Arguments:
    - num_classes: number of output classes
    - learning_rate: learning rate for Adam optimizer
    
    Returns:
    - model: PyTorch model
    - optimizer: Adam optimizer
    - criterion: CrossEntropyLoss
    """
    model = SkinCancerCNN(num_classes=num_classes)
    
    # Move model to GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    # Optimizer (Adam)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    # Loss function (CrossEntropyLoss combines LogSoftmax and NLLLoss)
    criterion = nn.CrossEntropyLoss()
    
    print(f"Model initialized on device: {device}")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    return model, optimizer, criterion, device
