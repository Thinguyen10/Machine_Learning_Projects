"""Quick demo of the tuning module with real data (small scale)."""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from backend.processing import process
from backend.tuning import fit_quadratic_peak, plot_quadratic_fit
import numpy as np

print("=" * 60)
print("TUNING MODULE DEMO")
print("=" * 60)

# Simulate a realistic validation accuracy curve
print("\n📊 Simulating training for 20 epochs...")
epochs = np.arange(1, 21)
# Model improves until epoch 12, then starts overfitting
true_peak = 12
val_acc = -0.005 * (epochs - true_peak) ** 2 + 0.87 + np.random.normal(0, 0.01, len(epochs))

print("   Validation accuracies:")
for e, acc in zip(epochs, val_acc):
    bar = "█" * int(acc * 50)
    print(f"   Epoch {e:2d}: {bar} {acc:.3f}")

# Apply quadratic peak detection
print("\n🔍 Applying quadratic peak detection...")
result = fit_quadratic_peak(val_acc)

print(f"\n✅ Results:")
print(f"   Best observed epoch: {int(np.argmax(val_acc)) + 1}")
print(f"   Peak analysis:")
print(f"     - Fitted peak epoch: {result['peak_epoch']:.2f}")
print(f"     - Predicted peak accuracy: {result['peak_val']:.3f}")
print(f"     - Quadratic coefficients: a={result['a']:.6f}, b={result['b']:.6f}, c={result['c']:.3f}")

# Generate visualization
print("\n📈 Generating visualization...")
plot_quadratic_fit(val_acc, result, save_path='demo_epoch_optimization.png')
print("   ✓ Saved: demo_epoch_optimization.png")

print("\n" + "=" * 60)
print("💡 INSIGHTS")
print("=" * 60)
print(f"• Training beyond epoch {int(result['peak_epoch'])} leads to overfitting")
print(f"• Optimal stopping point: epoch {int(result['peak_epoch'])}")
print(f"• Expected accuracy at peak: {result['peak_val']:.1%}")
print(f"• This saves {20 - int(result['peak_epoch'])} unnecessary epochs")
print("\n✅ Demo complete! Check demo_epoch_optimization.png for visualization.")
print("=" * 60)
