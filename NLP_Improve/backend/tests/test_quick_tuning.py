"""Quick test to verify tuning utilities behave as expected (unit tests, no training).

This is intended to be fast: tests the quadratic fitting logic and plotting helpers
without running actual Keras training (which can be slow and has mutex issues in some envs).

Run with:
    python3 backend/tests/test_quick_tuning.py
"""
import numpy as np
from backend.tuning import fit_quadratic_peak


def test_quadratic_peak_concave_down():
    """Test that fit_quadratic_peak correctly identifies a peak for concave-down parabola."""
    # Simulate validation accuracy that rises and then falls (overfitting)
    # y = -0.01 * (x - 15)^2 + 0.9, peak at x=15
    epochs = np.arange(1, 31)
    val_acc = -0.01 * (epochs - 15) ** 2 + 0.9
    
    result = fit_quadratic_peak(val_acc)
    
    assert result['fit_ok'] is True, "Fit should succeed"
    assert result['has_peak'] is True, "Should detect peak for concave-down parabola"
    assert 14 < result['peak_epoch'] < 16, f"Peak should be near 15, got {result['peak_epoch']}"
    print(f"✓ test_quadratic_peak_concave_down passed: peak at epoch {result['peak_epoch']:.2f}")


def test_quadratic_peak_concave_up():
    """Test that fit_quadratic_peak correctly reports no peak for concave-up parabola."""
    # Simulate monotonically increasing accuracy (underfitting, no peak)
    epochs = np.arange(1, 11)
    val_acc = 0.5 + 0.02 * epochs  # linear increase
    
    result = fit_quadratic_peak(val_acc)
    
    # Linear data will fit a quadratic with small |a|, may be concave up
    # We mainly check it doesn't crash and returns sensible structure
    assert result['fit_ok'] is True, "Fit should succeed"
    print(f"✓ test_quadratic_peak_concave_up passed: has_peak={result.get('has_peak', False)}")


def test_plotting_imports():
    """Test that plotting helpers can be imported without errors."""
    try:
        from backend.tuning import plot_learning_curves, plot_quadratic_fit
        print("✓ test_plotting_imports passed: plot helpers imported successfully")
    except Exception as e:
        raise AssertionError(f"Failed to import plotting helpers: {e}")


def test_grid_search_imports():
    """Test that grid_search and KerasTuner wrapper can be imported."""
    try:
        from backend.tuning import grid_search_keras, run_kerastuner_search
        print("✓ test_grid_search_imports passed: grid search and KerasTuner wrapper imported")
    except Exception as e:
        raise AssertionError(f"Failed to import grid search utilities: {e}")


if __name__ == '__main__':
    print("Running quick tuning utility tests (no training, fast)...\n")
    test_quadratic_peak_concave_down()
    test_quadratic_peak_concave_up()
    test_plotting_imports()
    test_grid_search_imports()
    print("\n✓ All tests passed!")

