# model.py
# This module defines the CNN architecture for skin cancer detection

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

def create_cnn_model(input_shape, num_classes, optimizer):
    """
    Build a Convolutional Neural Network (CNN).
    
    Arguments:
    - input_shape: tuple, shape of input images (height, width, channels)
    - num_classes: int, number of classes in the dataset
    
    Returns:
    - model: compiled Keras CNN model
    """
    
    model = Sequential()  # Initialize a sequential model
    
    # 1st Convolutional Layer
    # Detects low-level features like edges and corners
    model.add(Conv2D(
        filters=32,           # Number of feature maps
        kernel_size=(3,3),    # 3x3 convolution window
        padding='same',       # Keep output size same as input
        activation='relu',    # ReLU activation introduces non-linearity
        input_shape=input_shape  # Only for first layer
    ))
    model.add(MaxPooling2D(pool_size=(2,2)))  # Max pooling reduces spatial dimensions

    # 2nd Convolutional Layer
    # Detects more complex features
    model.add(Conv2D(
        filters=64,          # More feature maps to capture complex patterns
        kernel_size=(3,3),
        padding='same',
        activation='relu'
    ))
    model.add(MaxPooling2D(pool_size=(2,2)))

    # 3rd Convolutional Layer
    # Detects even higher-level patterns
    model.add(Conv2D(
        filters=128,         # Further increase in feature maps
        kernel_size=(3,3),
        padding='same',
        activation='relu'
    ))
    model.add(MaxPooling2D(pool_size=(2,2)))

    # Flatten the 3D feature maps to 1D vector for fully connected layer
    model.add(Flatten())

    # Fully connected layer with ReLU activation
    model.add(Dense(128, activation='relu'))

    # Output layer with Softmax activation for multi-class probabilities (e.g., benign vs malignant)
    model.add(Dense(num_classes, activation='softmax'))

     # ------------------------
    # Compile the model AFTER adding all layers
    # ------------------------
    model.compile(
        optimizer=optimizer,
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model
