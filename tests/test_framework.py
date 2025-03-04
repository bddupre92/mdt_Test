"""Test framework integration with drift detection."""

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import os
import logging

from framework.integration import FrameworkIntegrator
from framework.config import FrameworkConfig
from drift_detection.detector import DriftDetector

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def generate_synthetic_data(n_samples=1000, n_features=10, n_informative=5):
    """Generate synthetic classification data."""
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=n_informative,
        n_redundant=2,
        random_state=42
    )
    feature_names = [f'feature_{i}' for i in range(X.shape[1])]
    return pd.DataFrame(X, columns=feature_names), pd.Series(y)

def simulate_drift(X, magnitude=1.0):
    """Simulate concept drift by shifting feature values."""
    drift_X = X.copy()
    # Add random shift to each feature
    for col in drift_X.columns:
        drift_X[col] += np.random.normal(0, magnitude, size=len(drift_X))
    return drift_X

def test_drift_detection():
    """Test drift detection component."""
    logger.info("Testing drift detection...")
    
    # Generate initial data
    X, y = generate_synthetic_data(n_samples=500)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4)
    
    # Initialize drift detector
    detector = DriftDetector(
        window_size=50,
        drift_threshold=1.8,
        significance_level=0.01,
        feature_names=X.columns.tolist()
    )
    
    # Set initial reference
    detector.set_reference_window(X_train.values)
    
    # Process normal data
    normal_drifts = []
    for i in range(0, len(X_test), 10):
        batch = X_test.iloc[i:i+10].values
        drift, severity, info = detector.detect_drift(
            curr_data=batch,
            ref_data=detector.reference_window
        )
        normal_drifts.append(drift)
    
    logger.info(f"Normal data drift detections: {sum(normal_drifts)}")
    
    # Process data with artificial drift
    X_drift = simulate_drift(X_test, magnitude=2.0)
    drift_detections = []
    for i in range(0, len(X_drift), 10):
        batch = X_drift.iloc[i:i+10].values
        drift, severity, info = detector.detect_drift(
            curr_data=batch,
            ref_data=detector.reference_window
        )
        drift_detections.append(drift)
    
    logger.info(f"Drift data detections: {sum(drift_detections)}")
    return sum(normal_drifts), sum(drift_detections)

def test_optimizer_tracking():
    """Test optimizer state tracking."""
    logger.info("Testing optimizer state tracking...")
    
    # Create output directory
    output_dir = "test_results/plots"
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate data
    X, y = generate_synthetic_data()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    
    # Initialize framework
    config = FrameworkConfig()
    integrator = FrameworkIntegrator(
        window_size=config.drift.window_size,
        drift_threshold=config.drift.drift_threshold,
        significance_level=config.drift.significance_level
    )
    integrator.feature_names = X.columns.tolist()
    
    # Initialize model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Simulate optimization steps
    results = []
    for step in range(10):
        # Make predictions
        predictions = model.predict(X_test)
        probabilities = model.predict_proba(X_test)[:, 1]
        
        # Simulate optimizer state
        optimizer_state = {
            'parameters': {
                'n_estimators': model.n_estimators,
                'max_depth': model.max_depth if model.max_depth else -1
            },
            'fitness': model.score(X_test, y_test),
            'generation': step,
            'landscape_metrics': {
                'ruggedness': np.random.random(),
                'modality': np.random.randint(1, 5)
            },
            'gradient': np.random.randn(len(X.columns))
        }
        
        # Integrate step
        result = integrator.integrate_optimization_step(
            optimizer_state=optimizer_state,
            features=X_test.values,
            predictions=predictions,
            true_values=y_test.values
        )
        results.append(result)
        
        logger.info(f"Step {step}:")
        logger.info(f"  Drift detected: {result['drift_detected']}")
        logger.info(f"  Drift severity: {result['drift_severity']:.3f}")
        logger.info(f"  Current fitness: {result['optimization_metrics']['current_fitness']:.3f}")
        
        # Simulate some drift every 3 steps
        if step > 0 and step % 3 == 0:
            X_test = simulate_drift(X_test, magnitude=0.5)
    
    # Generate visualizations
    logger.info("Generating visualizations...")
    integrator.generate_visualizations(output_dir)
    
    return results

def main():
    """Run all framework tests."""
    # Test drift detection
    normal_drifts, drift_drifts = test_drift_detection()
    logger.info("\nDrift Detection Results:")
    logger.info(f"Normal data drift detections: {normal_drifts}")
    logger.info(f"Artificial drift detections: {drift_drifts}")
    
    # Test optimizer tracking
    results = test_optimizer_tracking()
    logger.info("\nOptimizer Tracking Results:")
    logger.info(f"Steps completed: {len(results)}")
    logger.info(f"Final fitness: {results[-1]['optimization_metrics']['current_fitness']:.3f}")
    
    logger.info("\nTest complete! Check test_results/plots directory for visualizations.")

if __name__ == "__main__":
    main()
