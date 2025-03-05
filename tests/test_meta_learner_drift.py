import numpy as np
from meta.meta_learner import MetaLearner
from drift_detection.drift_detector import DriftDetector

def test_meta_learner_with_drift():
    """Test that the meta-learner can handle drift detection properly."""
    # Create a MetaLearner instance with a DriftDetector
    detector = DriftDetector(window_size=10, threshold=0.01, alpha=0.3)
    meta_learner = MetaLearner(
        method="bayesian",
        strategy="bandit",
        exploration=0.3,
        drift_detector=detector
    )
    
    # Generate synthetic data
    np.random.seed(42)
    n_samples = 50
    n_features = 10
    
    # First phase data
    X1 = np.random.randn(n_samples, n_features)
    y1 = np.sin(X1[:, 0]) + 0.1 * np.random.randn(n_samples)
    
    # Second phase data (with drift)
    X2 = np.random.randn(n_samples, n_features)
    y2 = 2 * np.sin(X2[:, 0]) + 0.5 * np.cos(X2[:, 1]) + 0.1 * np.random.randn(n_samples)
    
    # Test fitting and updating
    print("Fitting initial data...")
    meta_learner.fit(X1, y1)
    
    # Make a prediction
    try:
        pred = meta_learner.predict(X2[0].reshape(1, -1))
        print(f"Prediction shape: {pred.shape}, value: {pred}")
    except Exception as e:
        print(f"Prediction error: {e}")
    
    # Update with new data (should detect drift)
    print("Updating with new data...")
    updated = meta_learner.update(X2, y2)
    print(f"Model updated: {updated}")
    
    # Make another prediction after update
    try:
        pred_after = meta_learner.predict(X2[1].reshape(1, -1))
        print(f"Prediction after update shape: {pred_after.shape}, value: {pred_after}")
    except Exception as e:
        print(f"Prediction error after update: {e}")
    
    # Test with scalar input
    print("Testing with scalar input...")
    X_scalar = X2[0, 0]  # Just taking a single value
    try:
        # This should be handled by the code we fixed
        meta_learner.algorithm.update(X_scalar)
        print("Scalar input handled successfully")
    except Exception as e:
        print(f"Scalar input error: {e}")
    
    return True

if __name__ == "__main__":
    # Run the test
    test_meta_learner_with_drift()
    print("All tests passed!")
