"""
Model Module
Contains Generator, Discriminator, and GAN architecture definitions
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import (
    Dense, Flatten, Reshape, LeakyReLU, 
    BatchNormalization, Dropout, Input
)
from tensorflow.keras.optimizers import Adam


class Generator:
    """Generator network for creating fake images"""
    
    def __init__(self, noise_dim=100, img_shape=(28, 28, 1)):
        """
        Initialize the Generator
        
        Args:
            noise_dim: Dimension of input noise vector
            img_shape: Shape of output images
        """
        self.noise_dim = noise_dim
        self.img_shape = img_shape
        self.model = None
        
    def build(self):
        """
        Build the Generator network
        
        Returns:
            Compiled generator model
        """
        # Initialize the neural network
        model = Sequential(name='Generator')
        
        # Add input layer (Dense layer that takes noise as input)
        model.add(Dense(256, input_dim=self.noise_dim))
        
        # Activate the layer with LeakyReLU
        model.add(LeakyReLU(alpha=0.2))
        
        # Apply batch normalization
        model.add(BatchNormalization(momentum=0.8))
        
        # Add a second layer
        model.add(Dense(512))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        
        # Add a third layer
        model.add(Dense(1024))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        
        # Add output layer
        # Output: flattened image reshaped to img_shape
        img_size = np.prod(self.img_shape)  # Total number of pixels
        model.add(Dense(img_size, activation='tanh'))
        model.add(Reshape(self.img_shape))
        
        self.model = model
        return model
    
    def compile(self, optimizer='adam', loss='binary_crossentropy'):
        """
        Compile the Generator network
        
        Args:
            optimizer: Optimizer to use
            loss: Loss function
        """
        if isinstance(optimizer, str):
            optimizer = Adam(learning_rate=0.0002, beta_1=0.5)
        
        self.model.compile(optimizer=optimizer, loss=loss)
    
    def summary(self):
        """Print model summary"""
        if self.model:
            self.model.summary()


class Discriminator:
    """Discriminator network for classifying real vs fake images"""
    
    def __init__(self, img_shape=(28, 28, 1)):
        """
        Initialize the Discriminator
        
        Args:
            img_shape: Shape of input images
        """
        self.img_shape = img_shape
        self.model = None
        
    def build(self):
        """
        Build the Discriminator network
        
        Returns:
            Compiled discriminator model
        """
        # Initialize the neural network
        model = Sequential(name='Discriminator')
        
        # Add input layer (flatten the image first)
        model.add(Flatten(input_shape=self.img_shape))
        model.add(Dense(512))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.3))
        
        # Add a second layer
        model.add(Dense(256))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.3))
        
        # Add a third layer
        model.add(Dense(128))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.3))
        
        # Add output layer
        # Output: single value (probability of being real)
        model.add(Dense(1, activation='sigmoid'))
        
        self.model = model
        return model
    
    def compile(self, optimizer='adam', loss='binary_crossentropy', metrics=['accuracy']):
        """
        Compile the Discriminator network
        
        Args:
            optimizer: Optimizer to use
            loss: Loss function
            metrics: List of metrics to track
        """
        if isinstance(optimizer, str):
            optimizer = Adam(learning_rate=0.0002, beta_1=0.5)
        
        self.model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
    
    def summary(self):
        """Print model summary"""
        if self.model:
            self.model.summary()


class GAN:
    """Combined GAN model (Generator + Discriminator)"""
    
    def __init__(self, generator, discriminator):
        """
        Initialize the GAN by stacking generator and discriminator
        
        Args:
            generator: Generator model instance
            discriminator: Discriminator model instance
        """
        self.generator = generator
        self.discriminator = discriminator
        self.model = None
        
    def build(self):
        """
        Build the GAN by stacking the generator and discriminator
        
        Returns:
            Combined GAN model
        """
        # CRITICAL: Must set trainable BEFORE adding to any new model
        # Setting on the model AND each individual layer
        self.discriminator.model.trainable = False
        for layer in self.discriminator.model.layers:
            layer.trainable = False
        
        # Use Functional API instead of Sequential to ensure proper freezing
        from tensorflow.keras.models import Model
        from tensorflow.keras.layers import Input
        
        # Create input for noise
        noise_input = Input(shape=(self.generator.model.input_shape[1],), name='gan_input')
        
        # Generate fake image
        fake_image = self.generator.model(noise_input)
        
        # Discriminator evaluates fake image (frozen!)
        validity = self.discriminator.model(fake_image)
        
        # Build the combined model
        model = Model(noise_input, validity, name='GAN')
        
        self.model = model
        return model
    
    def compile(self, optimizer='adam', loss='binary_crossentropy'):
        """
        Compile the GAN
        
        Args:
            optimizer: Optimizer to use
            loss: Loss function
        """
        if isinstance(optimizer, str):
            optimizer = Adam(learning_rate=0.0002, beta_1=0.5)
        
        self.model.compile(optimizer=optimizer, loss=loss)
        
        # Verify discriminator is frozen
        d_trainable = self.discriminator.model.trainable
        d_layers_trainable = [layer.trainable for layer in self.discriminator.model.layers]
        d_trainable_count = sum(d_layers_trainable)
        
        print(f"\n{'='*60}")
        print(f"GAN COMPILATION VERIFICATION:")
        print(f"Discriminator model.trainable: {d_trainable}")
        print(f"Discriminator layers trainable: {d_trainable_count}/{len(d_layers_trainable)}")
        print(f"GAN total trainable params: {self.model.count_params():,}")
        
        # Count trainable params in GAN
        trainable_params = sum([tf.size(w).numpy() for w in self.model.trainable_weights])
        non_trainable_params = sum([tf.size(w).numpy() for w in self.model.non_trainable_weights])
        print(f"GAN trainable weights: {trainable_params:,}")
        print(f"GAN non-trainable weights: {non_trainable_params:,}")
        print(f"{'='*60}\n")
    
    def summary(self):
        """Print model summary"""
        if self.model:
            self.model.summary()
