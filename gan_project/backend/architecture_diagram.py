"""
Architecture Diagram Generator
Creates a simple visualization of the GAN architecture
"""

def print_architecture():
    """Print ASCII diagram of GAN architecture"""
    
    diagram = """
╔══════════════════════════════════════════════════════════════════════════╗
║                    GAN ARCHITECTURE DIAGRAM                              ║
╚══════════════════════════════════════════════════════════════════════════╝

┌─────────────────────────────────────────────────────────────────────────┐
│                         GENERATOR NETWORK                               │
└─────────────────────────────────────────────────────────────────────────┘

    Random Noise (100)
          │
          ▼
    ┌─────────┐
    │ Dense   │ ──► 256 neurons
    │ (256)   │
    └─────────┘
          │
          ▼
    LeakyReLU (α=0.2)
          │
          ▼
    Batch Normalization
          │
          ▼
    ┌─────────┐
    │ Dense   │ ──► 512 neurons
    │ (512)   │
    └─────────┘
          │
          ▼
    LeakyReLU (α=0.2)
          │
          ▼
    Batch Normalization
          │
          ▼
    ┌─────────┐
    │ Dense   │ ──► 1024 neurons
    │ (1024)  │
    └─────────┘
          │
          ▼
    LeakyReLU (α=0.2)
          │
          ▼
    Batch Normalization
          │
          ▼
    ┌─────────┐
    │ Dense   │ ──► 784 neurons (28x28)
    │ (784)   │
    └─────────┘
          │
          ▼
    Tanh Activation
          │
          ▼
    Reshape (28, 28, 1)
          │
          ▼
    FAKE IMAGE (28x28)


┌─────────────────────────────────────────────────────────────────────────┐
│                      DISCRIMINATOR NETWORK                              │
└─────────────────────────────────────────────────────────────────────────┘

    Real or Fake Image (28x28)
          │
          ▼
    Flatten (784)
          │
          ▼
    ┌─────────┐
    │ Dense   │ ──► 512 neurons
    │ (512)   │
    └─────────┘
          │
          ▼
    LeakyReLU (α=0.2)
          │
          ▼
    Batch Normalization
          │
          ▼
    Dropout (0.3)
          │
          ▼
    ┌─────────┐
    │ Dense   │ ──► 256 neurons
    │ (256)   │
    └─────────┘
          │
          ▼
    LeakyReLU (α=0.2)
          │
          ▼
    Batch Normalization
          │
          ▼
    Dropout (0.3)
          │
          ▼
    ┌─────────┐
    │ Dense   │ ──► 128 neurons
    │ (128)   │
    └─────────┘
          │
          ▼
    LeakyReLU (α=0.2)
          │
          ▼
    Batch Normalization
          │
          ▼
    Dropout (0.3)
          │
          ▼
    ┌─────────┐
    │ Dense   │ ──► 1 neuron
    │   (1)   │
    └─────────┘
          │
          ▼
    Sigmoid Activation
          │
          ▼
    Probability [0-1]
    (0=Fake, 1=Real)


┌─────────────────────────────────────────────────────────────────────────┐
│                      TRAINING PROCESS                                   │
└─────────────────────────────────────────────────────────────────────────┘

Step 1: Train Discriminator
    
    Real Images ──┐
                  ├──► Discriminator ──► Loss (should predict 1)
    Fake Images ──┘                       + 
    (from Generator)                      Loss (should predict 0)
                                          = 
                                          Discriminator Loss


Step 2: Train Generator
    
    Random Noise ──► Generator ──► Fake Images ──► Discriminator ──► Loss
                                                   (frozen)       (want it to predict 1)
                                                                      │
                                                                      ▼
                                                              Generator Loss


Step 3: Repeat for many epochs until equilibrium!


┌─────────────────────────────────────────────────────────────────────────┐
│                      DATA FLOW                                          │
└─────────────────────────────────────────────────────────────────────────┘

Training Data (MNIST):
    60,000 images of handwritten digits (28x28)
    Normalized to [-1, 1]

Noise:
    Random samples from Normal Distribution (μ=0, σ=1)
    Shape: (batch_size, 100)

Generator Output:
    Fake images (28x28)
    Values in range [-1, 1] (due to tanh)

Discriminator Output:
    Probability [0-1] that image is real


┌─────────────────────────────────────────────────────────────────────────┐
│                      SUCCESS METRICS                                    │
└─────────────────────────────────────────────────────────────────────────┘

Good Training:
    ✓ Discriminator Accuracy: 50-80%
    ✓ Discriminator Loss: 0.5-0.7
    ✓ Generator Loss: Decreasing then stable
    ✓ Generated images look like digits

Poor Training (Mode Collapse):
    ✗ All generated images look the same
    ✗ Discriminator accuracy > 95%
    ✗ Generator loss increasing

Poor Training (Generator Failure):
    ✗ Generated images are noise
    ✗ Discriminator accuracy near 100%
    ✗ Generator loss not decreasing

"""
    
    print(diagram)

if __name__ == "__main__":
    print_architecture()
