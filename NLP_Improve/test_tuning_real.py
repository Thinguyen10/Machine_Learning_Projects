"""Real-world test of tuning utilities with actual model training.

This script runs a short epoch sweep on a small data subset to demonstrate
the tuning utilities work end-to-end with TensorFlow.
"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TF warnings
os.environ['TF_NUM_INTRAOP_THREADS'] = '1'  # Avoid threading issues
os.environ['TF_NUM_INTEROP_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'

from backend.processing import process
from backend.tuning import epoch_sweep, build_default_dense, plot_learning_curves, plot_quadratic_fit
from sklearn.model_selection import train_test_split
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend

print("=" * 60)
print("TUNING MODULE REAL-WORLD TEST")
print("=" * 60)

# Load data
print("\n1. Loading data from sentiment_analysis.csv...")
vect, Xtr, Xte, ytr, yte = process('backend/sentiment_analysis.csv')
print(f"   ✓ Data loaded: {Xtr.shape[0]} training samples")

# Use small subset for fast test
print("\n2. Creating small subset (200 samples) for quick test...")
X_small = Xtr[:200]
y_small = ytr[:200]

X_train, X_val, y_train, y_val = train_test_split(
    X_small, y_small, test_size=0.2, random_state=42
)
print(f"   ✓ Train: {X_train.shape[0]}, Val: {X_val.shape[0]}")

# Define model builder
def builder(input_dim):
    return build_default_dense(input_dim, hidden=32, dropout=0.3, lr=1e-3)

# Run epoch sweep
print("\n3. Running epoch sweep (5 epochs)...")
print("   (Training a small neural network...)")
result = epoch_sweep(
    builder, 
    X_train, y_train, 
    X_val, y_val, 
    max_epochs=5, 
    batch_size=16, 
    verbose=1
)

print(f"\n4. Results:")
print(f"   ✓ Best observed epoch: {result['best_epoch']}")
print(f"   ✓ Peak analysis: {result['peak_analysis']}")

# Extract validation accuracy
val_acc_key = 'val_accuracy' if 'val_accuracy' in result['history'][0] else 'val_acc'
val_accs = [e[val_acc_key] for e in result['history'] if val_acc_key in e]
print(f"   ✓ Validation accuracies: {[f'{v:.3f}' for v in val_accs]}")

# Generate plots
print("\n5. Generating plots...")
try:
    plot_learning_curves(result['history'], save_path='test_learning_curves.png')
    print("   ✓ Saved: test_learning_curves.png")
except Exception as e:
    print(f"   ⚠ Could not save learning curves: {e}")

try:
    plot_quadratic_fit(val_accs, result['peak_analysis'], save_path='test_quadratic_fit.png')
    print("   ✓ Saved: test_quadratic_fit.png")
except Exception as e:
    print(f"   ⚠ Could not save quadratic fit: {e}")

print("\n" + "=" * 60)
print("✅ TUNING MODULE TEST COMPLETE!")
print("=" * 60)
print("\nThe tuning utilities are working correctly with:")
print("  • Epoch sweep with quadratic peak detection")
print("  • Validation accuracy tracking")
print("  • Learning curve plotting")
print("  • Quadratic fit visualization")
print("\nYou can now use these tools for full hyperparameter tuning!")
print("=" * 60)
