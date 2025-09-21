# import libraries — TensorFlow and Keras
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import mnist
import numpy as np

# Load and preprocess the MNIST dataset 
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Normalize pixel values to range [0, 1] - to train model faster
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

# Flatten the images for ANN (flatten 28x28 pixels to 784 features) - since ANN requires 1D input, not 2D images
x_train_flat = x_train.reshape(-1, 28 * 28)
x_test_flat = x_test.reshape(-1, 28 * 28)

print(f"Training data shape: {x_train_flat.shape}")
print(f"Test data shape: {x_test_flat.shape}")

# Create the ANN model
model = models.Sequential([
    # Input layer - take in all 784 input features
        #using Rectified Linear Unit (ReLU) activation function
    layers.Dense(512, activation='relu', input_shape=(784,)),
    layers.Dropout(0.2),  # randomly drop 20% of the neurons to prevent overfitting
    
    # Hidden layer 1
        #using 256 neurons and ReLU activation function, with dropout of 20%
    layers.Dense(256, activation='relu'),
    layers.Dropout(0.2),
    
    # Hidden layer 2
        #using 128 neurons and ReLU activation function, with dropout of 10%
        #decresing # of neurons to help model learn more complex patterns
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.1),
    
    # Output layer - 10 neurons for digits 0-9 since MNIST has 10 classes
        '''
        each neuron represents a class (ex:neuron 0 for digit '0'.)
        after training, the neuron with highest value indicates predicted class 
        
         using Softmax activation function to get probabilities for each class 
        Softmax ensures all output values sum to 1, making them interpretable as probabilities 
        Ex: Raw outputs from 10 neurons: [2.0, 1.0, 0.1, ...]
            Softmax outputs: [0.6, 0.2, 0.05, ...]  # sums to 1
            you don't use Relu here because it doesn't output probabilities'''
    layers.Dense(10, activation='softmax')
])

# Display model architecture
model.summary()

# Compile the model

 ''' 
Using Adam optimizer - popular and efficient for training deep learning models
loss='sparse_categorical_crossentropy' → Suitable for multi-class classification with integer labels.
metrics=['accuracy'] → We’ll monitor classification accuracy.
 '''

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Train the model
print("Training the ANN model...")
history = model.fit(
    x_train_flat, y_train,
    epochs=10, #the model will go through the entire training dataset 10 times
    batch_size=128, #128 samples per gradient update
    validation_split=0.1, #10% of training data used for validation
    verbose=1
)

# Evaluate the model
test_loss, test_acc = model.evaluate(x_test_flat, y_test, verbose=0)
print(f'Test accuracy: {test_acc:.4f}')

# Save the model
model.save('mnist_ann_model.h5')
print("ANN model saved as 'mnist_ann_model.h5'")

# Print comparison info
print(f"\n{'='*50}")
print("ANN Model Summary:")
print(f"Architecture: Fully Connected Neural Network")
print(f"Input: 784 features (28x28 flattened)")
print(f"Hidden Layers: 512 -> 256 -> 128 neurons")
print(f"Output: 10 classes (digits 0-9)")
print(f"Total Parameters: {model.count_params():,}")
print(f"Test Accuracy: {test_acc:.4f}")
print(f"{'='*50}")