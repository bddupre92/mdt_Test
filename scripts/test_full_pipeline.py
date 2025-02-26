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
    # Generate base features
    np.random.seed(42)
    data = {
        'temperature': np.random.normal(22, 5, n_samples),
        'pressure': np.random.normal(1015, 5, n_samples),
        'stress_level': np.random.uniform(0, 12, n_samples),
        'sleep_hours': np.random.normal(7, 1, n_samples),
        'screen_time': np.random.normal(6, 2, n_samples)
    }
    
    # Add concept drift in second half
    mid_point = n_samples // 2
    data['temperature'][mid_point:] += 2  # Temperature increases
    data['stress_level'][mid_point:] *= 1.2  # Stress has more impact
    data['sleep_hours'][mid_point:] -= 1  # Sleep decreases
    
    # Generate target with changing relationship
    X = np.column_stack([
        data['temperature'],
        data['pressure'],
        data['stress_level'],
        data['sleep_hours'],
        data['screen_time']
    ])
    
    # First half: more weight on temperature and pressure
    y1 = (0.3 * data['temperature'][:mid_point] +
          0.3 * data['pressure'][:mid_point] +
          0.2 * data['stress_level'][:mid_point] +
          0.1 * data['sleep_hours'][:mid_point] +
          0.1 * data['screen_time'][:mid_point])
    
    # Second half: more weight on stress and sleep
    y2 = (0.1 * data['temperature'][mid_point:] +
          0.1 * data['pressure'][mid_point:] +
          0.4 * data['stress_level'][mid_point:] +
          0.3 * data['sleep_hours'][mid_point:] +
          0.1 * data['screen_time'][mid_point:])
    
    y = np.concatenate([y1, y2])
    y = (y > np.median(y)).astype(int)  # Convert to binary classification
    
    return pd.DataFrame(data), y

def plot_performance(drift_detector: DriftDetector,
                    feature_names: List[str],
                    save_path: str = 'pipeline_performance.png'):
    """Plot drift detection results"""
    # Create figure
    plt.figure(figsize=(12, 10))
    
    # Plot 1: Drift Scores
    plt.subplot(3, 1, 1)
    plt.plot(drift_detector.drift_scores, label='Drift Score')
    plt.axhline(y=drift_detector.drift_threshold, color='r', linestyle='--',
               label='Drift Threshold')
    plt.title('Drift Scores Over Time')
    plt.ylabel('Score')
    plt.legend()
    
    # Plot 2: Feature Drift Severities
    plt.subplot(3, 1, 2)
    for name in feature_names:
        if name in drift_detector.feature_drift_scores:
            scores = drift_detector.feature_drift_scores[name]
            plt.plot(scores, label=name)
    plt.title('Feature Drift Severities')
    plt.ylabel('Severity')
    plt.legend()
    
    # Plot 3: Confidence Scores
    plt.subplot(3, 1, 3)
    plt.plot(drift_detector.confidence_scores, label='Confidence')
    plt.axhline(y=drift_detector.confidence_threshold, color='r', linestyle='--',
               label='Confidence Threshold')
    plt.title('Model Confidence Over Time')
    plt.ylabel('Confidence')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"Saving plot to: {os.path.abspath(save_path)}")
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
            feature_names=feature_names
        )
    
    overall_progress.update(1)
    
    # Get best configuration
    config = meta_learner.get_best_configuration()
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
