"""
Visualization Module
Contains functions for visualizing training progress and generated images
"""

import matplotlib.pyplot as plt
import numpy as np


class GANVisualizer:
    """Visualize GAN training and results"""
    
    def __init__(self, img_shape=(28, 28, 1)):
        """
        Initialize the visualizer
        
        Args:
            img_shape: Shape of images to visualize
        """
        self.img_shape = img_shape
        
    def plot_generated_images(self, images, epoch=None, n_rows=4, n_cols=4, save_path=None):
        """
        Plot a grid of generated images
        
        Args:
            images: Array of generated images
            epoch: Current epoch number (for title)
            n_rows: Number of rows in the grid
            n_cols: Number of columns in the grid
            save_path: Path to save the figure (optional)
        """
        # Rescale images from [-1, 1] to [0, 1]
        images = 0.5 * images + 0.5
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(10, 10))
        
        idx = 0
        for i in range(n_rows):
            for j in range(n_cols):
                if idx < len(images):
                    # Handle grayscale vs color images
                    if images[idx].shape[-1] == 1:
                        axes[i, j].imshow(images[idx, :, :, 0], cmap='gray')
                    else:
                        axes[i, j].imshow(images[idx])
                    
                    axes[i, j].axis('off')
                    idx += 1
        
        if epoch is not None:
            fig.suptitle(f'Generated Images - Epoch {epoch}', fontsize=16)
        else:
            fig.suptitle('Generated Images', fontsize=16)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
            print(f"Saved image grid to {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    def plot_training_history(self, history, save_path=None):
        """
        Plot training history (losses and accuracy)
        
        Args:
            history: Dictionary containing training metrics
            save_path: Path to save the figure (optional)
        """
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        
        epochs = history['epochs']
        
        # Plot losses
        axes[0].plot(epochs, history['discriminator_loss'], label='Discriminator Loss', alpha=0.7)
        axes[0].plot(epochs, history['generator_loss'], label='Generator Loss', alpha=0.7)
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].set_title('Training Losses')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Plot discriminator accuracy
        axes[1].plot(epochs, np.array(history['discriminator_accuracy']) * 100, 
                    label='Discriminator Accuracy', color='green', alpha=0.7)
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Accuracy (%)')
        axes[1].set_title('Discriminator Accuracy')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
            print(f"Saved training history to {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    def compare_real_vs_fake(self, real_images, fake_images, n_samples=5):
        """
        Create side-by-side comparison of real and fake images
        
        Args:
            real_images: Array of real images
            fake_images: Array of fake images
            n_samples: Number of samples to compare
        """
        # TODO: Implement comparison visualization
        # - Create 2 rows (real and fake)
        # - Display samples side by side
        pass
    
    def visualize_latent_space(self, generator, n_samples=15):
        """
        Visualize the latent space by interpolating between points
        
        Args:
            generator: Generator model
            n_samples: Number of interpolation steps
        """
        # TODO: Implement latent space visualization
        # - Generate random points in latent space
        # - Interpolate between them
        # - Generate and display images
        pass
    
    def plot_discriminator_predictions(self, discriminator, real_images, fake_images):
        """
        Plot histogram of discriminator predictions on real vs fake images
        
        Args:
            discriminator: Discriminator model
            real_images: Batch of real images
            fake_images: Batch of fake images
        """
        # TODO: Implement prediction distribution plotting
        # - Get predictions for real images
        # - Get predictions for fake images
        # - Plot histograms
        pass
    
    def create_gif(self, image_folder, output_path='training.gif', duration=100):
        """
        Create GIF from saved training images
        
        Args:
            image_folder: Folder containing saved images
            output_path: Path for output GIF
            duration: Duration per frame in milliseconds
        """
        # TODO: Implement GIF creation from training snapshots
        pass
    
    def save_image_grid(self, images, save_path, n_rows=4, n_cols=4):
        """
        Save a grid of images to file
        
        Args:
            images: Array of images
            save_path: Path to save the image
            n_rows: Number of rows
            n_cols: Number of columns
        """
        # TODO: Implement image saving
        pass
