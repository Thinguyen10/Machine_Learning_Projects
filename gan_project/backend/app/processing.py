"""
Data Processing Module
Handles data loading, preprocessing, and preparation for GAN training
"""

import numpy as np
from tensorflow import keras


class DataProcessor:
    """Handle data loading and preprocessing for GAN training"""
    
    def __init__(self, dataset_name='mnist'):
        """
        Initialize the data processor
        
        Args:
            dataset_name: Name of the dataset to use (default: 'mnist')
        """
        self.dataset_name = dataset_name
        self.x_train = None
        self.y_train = None
        self.x_test = None
        self.y_test = None
        self.img_shape = None
        
    def load_data(self):
        """
        Load dataset from keras.datasets
        
        Returns:
            Tuple of training and test data
        """
        if self.dataset_name == 'mnist':
            from tensorflow.keras.datasets import mnist
            (self.x_train, self.y_train), (self.x_test, self.y_test) = mnist.load_data()
        elif self.dataset_name == 'fashion_mnist':
            from tensorflow.keras.datasets import fashion_mnist
            (self.x_train, self.y_train), (self.x_test, self.y_test) = fashion_mnist.load_data()
        elif self.dataset_name == 'cifar10':
            from tensorflow.keras.datasets import cifar10
            (self.x_train, self.y_train), (self.x_test, self.y_test) = cifar10.load_data()
        else:
            raise ValueError(f"Dataset {self.dataset_name} not supported")
        
        # Store image shape
        self.img_shape = self.x_train.shape[1:]
        print(f"Loaded {self.dataset_name} dataset: {self.x_train.shape[0]} training samples")
        print(f"Image shape: {self.img_shape}")
        
        return (self.x_train, self.y_train), (self.x_test, self.y_test)
    
    def preprocess_data(self, normalize=True, reshape=False):
        """
        Preprocess the loaded data
        
        Args:
            normalize: Whether to normalize pixel values
            reshape: Whether to reshape the data
            
        Returns:
            Preprocessed data
        """
        if self.x_train is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        
        # Normalize pixel values to [-1, 1] for tanh activation
        if normalize:
            self.x_train = (self.x_train.astype('float32') - 127.5) / 127.5
            self.x_test = (self.x_test.astype('float32') - 127.5) / 127.5
            print("Normalized data to [-1, 1]")
        
        # Add channel dimension if grayscale (for Conv2D compatibility)
        if reshape and len(self.x_train.shape) == 3:
            self.x_train = np.expand_dims(self.x_train, axis=-1)
            self.x_test = np.expand_dims(self.x_test, axis=-1)
            self.img_shape = self.x_train.shape[1:]
            print(f"Reshaped to: {self.img_shape}")
        
        return self.x_train, self.x_test
    
    def get_random_samples(self, n_samples):
        """
        Get random samples from the training data
        
        Args:
            n_samples: Number of samples to return
            
        Returns:
            Random batch of real images
        """
        if self.x_train is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        
        # Get random indices
        idx = np.random.randint(0, self.x_train.shape[0], n_samples)
        
        # Return random samples
        return self.x_train[idx]
    
    def generate_noise(self, n_samples, noise_dim):
        """
        Generate random noise as input for the generator
        
        Args:
            n_samples: Number of noise samples
            noise_dim: Dimension of noise vector
            
        Returns:
            Random noise array
        """
        # Generate random noise from normal distribution
        return np.random.normal(0, 1, (n_samples, noise_dim))
