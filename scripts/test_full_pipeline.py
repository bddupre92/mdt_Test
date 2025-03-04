"""
test_full_pipeline.py
-------------------
Test full pipeline with synthetic data and drift detection
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from typing import Dict, Any, List, Tuple
from tqdm import tqdm
import os

from drift_detection.detector import DriftDetector
from models.model_factory import ModelFactory
from meta.meta_learner import MetaLearner
from optimizers.optimizer_factory import create_optimizers

def generate_synthetic_data(n_samples: int = 1000) -> Tuple[pd.DataFrame, np.ndarray]:
    """Generate synthetic data with concept drift"""
    # Generate base features with more pronounced initial differences
    np.random.seed(42)
    data = {
        'temperature': np.random.normal(37, 0.2, n_samples),  # Further reduced variance
        'pressure': np.random.normal(120, 3, n_samples),      # Further reduced variance
        'stress_level': np.random.normal(5, 0.3, n_samples),  # Further reduced variance
        'sleep_hours': np.random.normal(7, 0.3, n_samples),   # Further reduced variance
        'screen_time': np.random.normal(4, 0.3, n_samples)    # Further reduced variance
    }
    
    # Add concept drift in second half
    mid_point = n_samples // 2
    
    # First drift point (more pronounced changes)
    drift_point1 = n_samples // 4
    # Second drift point (change feature relationships)
    drift_point2 = mid_point
    # Third drift point (extreme changes)
    drift_point3 = 3 * n_samples // 4
    
    # Create extremely abrupt changes at drift points
    # First drift: massive stress increase and temperature change
    data['stress_level'][drift_point1:] += 10.0   # Extreme stress increase at first drift
    data['temperature'][drift_point1:] += 3.0     # Significant temperature increase
    
    # Second drift: extreme sleep decrease and pressure change
    data['sleep_hours'][drift_point2:] -= 5.0     # Extreme sleep decrease at second drift
    data['pressure'][drift_point2:] += 30.0       # Major pressure increase
    
    # Third drift: extreme screen time increase and all other changes
    data['screen_time'][drift_point3:] += 15.0    # Extreme screen time increase at third drift
    
    # Add stronger feature correlations that change dramatically at drift points
    # Before first drift: baseline correlations
    for i in range(drift_point1):
        noise = np.random.normal(0, 0.1)  # Very low noise
        data['pressure'][i] = 120 + 10 * data['temperature'][i] + noise
    
    # Between first and second drift: completely different correlations
    for i in range(drift_point1, drift_point2):
        noise = np.random.normal(0, 0.1)  # Very low noise
        data['sleep_hours'][i] = 15 - 2.0 * data['stress_level'][i] + noise
    
    # Between second and third drift: another set of correlations
    for i in range(drift_point2, drift_point3):
        noise = np.random.normal(0, 0.1)  # Very low noise
        data['stress_level'][i] = 2 + 3.0 * data['screen_time'][i] + noise
    
    # After third drift: extreme randomness and new correlations
    for i in range(drift_point3, n_samples):
        noise = np.random.normal(0, 0.1)  # Very low noise
        data['temperature'][i] = 40 + 0.5 * data['screen_time'][i] + noise
    
    # Generate target with dramatically changing relationship
    X = np.column_stack([
        data['temperature'],
        data['pressure'],
        data['stress_level'],
        data['sleep_hours'],
        data['screen_time']
    ])
    
    # First quarter: baseline weights with strong temperature influence
    y1 = (0.8 * data['temperature'][:drift_point1] +
          0.2 * data['pressure'][:drift_point1])  # Only temperature and pressure matter
    
    # Second quarter: complete shift to stress dominance
    y2 = (1.0 * data['stress_level'][drift_point1:drift_point2])  # Only stress matters
    
    # Third quarter: complete shift to sleep dominance
    y3 = (1.0 * data['sleep_hours'][drift_point2:drift_point3])   # Only sleep matters
    
    # Fourth quarter: complete shift to screen time dominance
    y4 = (1.0 * data['screen_time'][drift_point3:])               # Only screen time matters
    
    y = np.concatenate([y1, y2, y3, y4])
    y = (y > np.median(y)).astype(int)  # Convert to binary classification
    
    return pd.DataFrame(data), y

def plot_performance(drift_detector: DriftDetector,
                    feature_names: List[str],
                    save_path: str = 'pipeline_performance.png'):
    """Plot drift detection results"""
    # Create figure
    plt.figure(figsize=(12, 10))
    
    # Plot 1: Drift Scores and Severity
    plt.subplot(3, 1, 1)
    plt.plot(drift_detector.severity_history, label='Drift Severity', color='blue')
    plt.plot(drift_detector.mean_shift_history, label='Mean Shift', color='green', alpha=0.5)
    plt.axhline(y=drift_detector.drift_threshold, color='r', linestyle='--',
               label='Drift Threshold')
    plt.title('Drift Detection Metrics Over Time')
    plt.ylabel('Score')
    plt.legend()
    
    # Plot 2: Feature Drifts
    plt.subplot(3, 1, 2)
    for i, name in enumerate(feature_names):
        drifts = [1 if name in d else 0 for d in drift_detector.drifting_features_history]
        if any(drifts):  # Only plot if feature had any drifts
            plt.plot(drifts, label=name, alpha=0.7)
    plt.title('Feature Drift Occurrences')
    plt.ylabel('Drift Detected')
    plt.legend()
    
    # Plot 3: Statistical Measures
    plt.subplot(3, 1, 3)
    plt.plot(drift_detector.ks_stat_history, label='KS Statistic', color='purple')
    plt.plot(drift_detector.p_value_history, label='p-value', color='orange', alpha=0.5)
    plt.axhline(y=0.05, color='r', linestyle='--', label='Significance Level')
    plt.title('Statistical Measures Over Time')
    plt.ylabel('Value')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def main():
    print("Starting full pipeline test...\n")
    
    # Create progress bar for overall progress
    stages = ['Data Generation', 'Meta-optimization', 'Model Training', 'Drift Detection']
    overall_progress = tqdm(stages, desc='Overall Progress', position=0)
    
    # 1. Generate synthetic data
    overall_progress.set_description('Generating Data')
    X_full, y_full = generate_synthetic_data()
    print(f"\nData shape: {X_full.shape}")
    print("Feature statistics:")
    print(pd.concat([X_full, pd.Series(y_full, name='target')], axis=1).describe())
    overall_progress.update(1)
    
    # 2. Initialize components
    print("\nInitializing components...")
    feature_names = list(X_full.columns)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_full)
    
    # Split into train/test
    train_size = int(0.8 * len(X_scaled))
    X_train = X_scaled[:train_size]
    y_train = y_full[:train_size]
    X_test = X_scaled[train_size:]
    y_test = y_full[train_size:]
    
    # 3. Run meta-optimization
    overall_progress.set_description('Running Meta-optimization')
    print("\nRunning meta-optimization...")
    meta_learner = MetaLearner()
    bounds = [(0, 1), (0, 1)]  # Parameters will be scaled in the objective function
    optimizers = create_optimizers(dim=2, bounds=bounds)
    meta_learner.set_algorithms([optimizers['DE (Adaptive)']])  # Use adaptive DE optimizer
    
    # Add progress callback to meta-learner
    with tqdm(total=100, desc='Optimization Progress', position=1, leave=False) as pbar:
        def progress_callback(iteration, best_score):
            pbar.update(1)
            pbar.set_postfix({'best_score': f'{best_score:.4f}'})
        
        meta_learner.optimize(
            X_train, y_train,
            progress_callback=progress_callback,
            feature_names=feature_names,
            context={'task_type': 'classification'}  # Specify classification task
        )
    
    overall_progress.update(1)
    
    # Get best configuration
    config = meta_learner.get_best_configuration()
    config['task_type'] = 'classification'  # Ensure task type is set
    print("\nBest model configuration:")
    print(config)
    
    # 4. Train model
    overall_progress.set_description('Training Model')
    print("\nTraining model with best configuration...")
    factory = ModelFactory()
    model = factory.create_model(config)
    history = model.fit(X_train, y_train, feature_names=feature_names)
    overall_progress.update(1)
    
    # 5. Evaluate performance
    print("\nEvaluating performance...")
    train_acc = model.score(X_train, y_train)
    test_acc = model.score(X_test, y_test)
    print(f"Train accuracy: {train_acc:.4f}")
    print(f"Test accuracy: {test_acc:.4f}")
    
    # 6. Test drift detection
    overall_progress.set_description('Running Drift Detection')
    print("\nTesting drift detection...")
    detector = DriftDetector(window_size=50, feature_names=feature_names)
    
    # Get prediction probabilities
    train_probs = model.predict_proba(X_train)[:, 1]
    test_probs = model.predict_proba(X_test)[:, 1]
    
    # Set reference window from training data
    detector.set_reference_window(train_probs[-50:], X_train[-50:])
    
    # Process test data with progress bar
    drifts_detected = 0
    with tqdm(total=len(X_test), desc='Processing Samples', position=1, leave=False) as pbar:
        for i in range(len(X_test)):
            drift, severity, info = detector.add_sample(
                test_probs[i],
                features=X_test[i],
                prediction_proba=model.predict_proba(X_test[i].reshape(1, -1))[0]
            )
            if drift:
                drifts_detected += 1
                print(f"\nDrift detected at sample {i}:")
                print(f"Severity: {severity:.3f}")
                print("Drifting features:", info['drifting_features'])
                print(f"Confidence: {info['confidence']:.3f}")
            pbar.update(1)
            pbar.set_postfix({'drifts': drifts_detected})
    
    overall_progress.update(1)
    print(f"\nTotal drifts detected: {drifts_detected}")
    
    # 7. Plot results
    print("\nPlotting results...")
    os.makedirs('plots', exist_ok=True)
    plot_path = os.path.join('plots', 'pipeline_performance.png')
    plot_performance(detector, feature_names, save_path=plot_path)
    print(f"Performance plot saved as '{plot_path}'")
    
    overall_progress.close()

if __name__ == "__main__":
    main()
