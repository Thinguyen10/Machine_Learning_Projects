"""
Quick test script to verify GAN components are working
Run this with fewer epochs to test the setup
"""

from main import GANTrainer

def test_gan():
    """Test GAN with a small number of epochs"""
    
    print("Testing GAN setup...")
    print("-" * 60)
    
    # Configuration for quick test
    NOISE_DIM = 100
    IMG_SHAPE = (28, 28, 1)  # MNIST
    EPOCHS = 100  # Small number for testing
    BATCH_SIZE = 32
    SAMPLE_INTERVAL = 50
    
    # Initialize trainer
    trainer = GANTrainer(noise_dim=NOISE_DIM, img_shape=IMG_SHAPE)
    
    # Build models
    print("\n1. Building models...")
    trainer.build_and_compile_models()
    
    # Test data loading
    print("\n2. Testing data loading...")
    trainer.data_processor.load_data()
    trainer.data_processor.preprocess_data(normalize=True, reshape=True)
    print(f"Data shape: {trainer.data_processor.x_train.shape}")
    
    # Test noise generation
    print("\n3. Testing noise generation...")
    noise = trainer.data_processor.generate_noise(5, NOISE_DIM)
    print(f"Noise shape: {noise.shape}")
    
    # Test generator
    print("\n4. Testing generator...")
    fake_images = trainer.generator.model.predict(noise, verbose=0)
    print(f"Generated images shape: {fake_images.shape}")
    
    # Test discriminator
    print("\n5. Testing discriminator...")
    predictions = trainer.discriminator.model.predict(fake_images, verbose=0)
    print(f"Discriminator predictions shape: {predictions.shape}")
    
    # Quick training test
    print("\n6. Running quick training test...")
    print(f"Training for {EPOCHS} epochs...")
    trainer.train(epochs=EPOCHS, batch_size=BATCH_SIZE, sample_interval=SAMPLE_INTERVAL)
    
    # Generate final samples
    print("\n7. Generating final samples...")
    samples = trainer.generate_samples(16)
    trainer.visualizer.plot_generated_images(samples, save_path='test_output.png')
    
    print("\n" + "=" * 60)
    print("TEST COMPLETE!")
    print("Check 'generated_images/' folder for sample outputs")
    print("Check 'test_output.png' for final test results")
    print("=" * 60)

if __name__ == "__main__":
    test_gan()
