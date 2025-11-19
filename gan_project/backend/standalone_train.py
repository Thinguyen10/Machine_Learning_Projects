"""
Standalone Training Script
Trains GAN without web interface (command line only)
Use this for headless training or testing
"""

import numpy as np
from tqdm import tqdm

# Import custom modules from app directory
from app.processing import DataProcessor
from app.model import Generator, Discriminator, GAN
from app.analysis import GANAnalyzer
from app.visual import GANVisualizer


class GANTrainer:
    """Main trainer class for the GAN"""
    
    def __init__(self, noise_dim=100, img_shape=(28, 28, 1)):
        """
        Initialize the GAN trainer
        
        Args:
            noise_dim: Dimension of noise vector
            img_shape: Shape of images
        """
        self.noise_dim = noise_dim
        self.img_shape = img_shape
        
        # Initialize components
        self.data_processor = DataProcessor()
        self.generator = Generator(noise_dim, img_shape)
        self.discriminator = Discriminator(img_shape)
        self.gan = None
        self.analyzer = GANAnalyzer()
        self.visualizer = GANVisualizer(img_shape)
        
    def build_and_compile_models(self):
        """Build and compile all models"""
        print("Building Generator...")
        self.generator.build()
        self.generator.compile()
        self.generator.summary()
        
        print("\nBuilding Discriminator...")
        self.discriminator.build()
        self.discriminator.compile()
        self.discriminator.summary()
        
        print("\nBuilding GAN...")
        self.gan = GAN(self.generator, self.discriminator)
        self.gan.build()
        self.gan.compile()
        self.gan.summary()
        
    def train(self, epochs=10000, batch_size=128, sample_interval=1000):
        """
        Train the GAN
        
        Args:
            epochs: Number of training epochs
            batch_size: Batch size for training
            sample_interval: Interval for saving generated images
        """
        print("Starting GAN training...")
        
        # Load and preprocess data
        print("\nLoading data...")
        self.data_processor.load_data()
        self.data_processor.preprocess_data(normalize=True, reshape=True)
        
        # Labels for real and fake images
        real_labels = np.ones((batch_size, 1))
        fake_labels = np.zeros((batch_size, 1))
        
        # Training loop
        for epoch in tqdm(range(epochs), desc="Training"):
            
            # ---------------------
            #  Train Discriminator
            # ---------------------
            
            # Get batch of real images
            real_images = self.data_processor.get_random_samples(batch_size)
            
            # Generate fake images
            noise = self.data_processor.generate_noise(batch_size, self.noise_dim)
            fake_images = self.generator.model.predict(noise, verbose=0)
            
            # Train discriminator on real images (labels = 1)
            d_loss_real = self.discriminator.model.train_on_batch(real_images, real_labels)
            
            # Train discriminator on fake images (labels = 0)
            d_loss_fake = self.discriminator.model.train_on_batch(fake_images, fake_labels)
            
            # Average discriminator loss
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
            
            # ---------------------
            #  Train Generator
            # ---------------------
            
            # Generate noise
            noise = self.data_processor.generate_noise(batch_size, self.noise_dim)
            
            # Train generator via GAN (labels = 1, to fool discriminator)
            g_loss = self.gan.model.train_on_batch(noise, real_labels)
            
            # ---------------------
            #  Track Progress
            # ---------------------
            
            # Update metrics
            self.analyzer.update_metrics(d_loss[0], d_loss[1], g_loss, epoch)
            
            # Print progress
            if epoch % 100 == 0:
                print(f"\nEpoch {epoch}/{epochs}")
                print(f"D Loss: {d_loss[0]:.4f} | D Acc: {100*d_loss[1]:.2f}%")
                print(f"G Loss: {g_loss:.4f}")
            
            # Save sample images at intervals and milestones
            # Save at epoch milestones: 1, 30, 100, 400, etc.
            if epoch % sample_interval == 0 or epoch in [1, 30, 100, 400]:
                self.save_samples(epoch)
        
        print("\nTraining complete!")
    
    def generate_samples(self, n_samples=16):
        """
        Generate sample images using the trained generator
        
        Args:
            n_samples: Number of samples to generate
            
        Returns:
            Array of generated images
        """
        noise = self.data_processor.generate_noise(n_samples, self.noise_dim)
        generated_images = self.generator.model.predict(noise, verbose=0)
        return generated_images
    
    def save_samples(self, epoch, n_samples=16):
        """
        Generate and save sample images
        
        Args:
            epoch: Current epoch number
            n_samples: Number of samples to generate
        """
        import os
        os.makedirs('generated_images', exist_ok=True)
        
        generated_images = self.generate_samples(n_samples)
        self.visualizer.plot_generated_images(
            generated_images, 
            epoch=epoch, 
            save_path=f'generated_images/epoch_{epoch}.png'
        )
    
    def evaluate(self):
        """Evaluate the trained GAN"""
        print("\nEvaluating GAN...")
        
        # Generate samples
        print("Generating samples...")
        generated_images = self.generate_samples(16)
        
        # Visualize results
        self.visualizer.plot_generated_images(generated_images, save_path='final_results.png')
        
        # Plot training history
        print("Creating training history plot...")
        self.visualizer.plot_training_history(self.analyzer.history, save_path='training_history.png')
        
        # Get summary report
        report = self.analyzer.get_summary_report()
        print(report)
        
        print("\nEvaluation complete!")
    
    def save_models(self, path='./saved_models'):
        """
        Save trained models
        
        Args:
            path: Directory to save models
        """
        import os
        os.makedirs(path, exist_ok=True)
        
        self.generator.model.save(f'{path}/generator.h5')
        self.discriminator.model.save(f'{path}/discriminator.h5')
        self.gan.model.save(f'{path}/gan.h5')
        
        print(f"Models saved to {path}/")
    
    def load_models(self, path='./saved_models'):
        """
        Load pre-trained models
        
        Args:
            path: Directory containing saved models
        """
        from tensorflow.keras.models import load_model
        
        self.generator.model = load_model(f'{path}/generator.h5')
        self.discriminator.model = load_model(f'{path}/discriminator.h5')
        self.gan.model = load_model(f'{path}/gan.h5')
        
        print(f"Models loaded from {path}/")


def main():
    """Main function to run the GAN application"""
    
    print("=" * 60)
    print("GAN-based Fake Image Generator")
    print("=" * 60)
    
    # Configuration
    NOISE_DIM = 100
    IMG_SHAPE = (28, 28, 1)  # MNIST image shape
    EPOCHS = 400  # Train for at least 400 epochs as required
    BATCH_SIZE = 128
    SAMPLE_INTERVAL = 100  # Save images more frequently to capture milestones
    
    # Initialize trainer
    trainer = GANTrainer(noise_dim=NOISE_DIM, img_shape=IMG_SHAPE)
    
    # Build and compile models
    trainer.build_and_compile_models()
    
    # Train the GAN
    trainer.train(epochs=EPOCHS, batch_size=BATCH_SIZE, sample_interval=SAMPLE_INTERVAL)
    
    # Evaluate results
    trainer.evaluate()
    
    # Save models
    trainer.save_models()
    
    print("\nTraining complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
