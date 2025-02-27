"""
test_framework.py
--------------
Comprehensive framework testing with synthetic data
"""

from typing import List, Tuple, Dict
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from datetime import datetime, timedelta
import os
from tqdm.auto import tqdm, trange
import matplotlib.pyplot as plt
import seaborn as sns

from meta.meta_learner import MetaLearner
from drift_detection.detector import DriftDetector
from evaluation.framework_evaluator import FrameworkEvaluator
from optimizers.optimizer_factory import create_optimizers

def generate_temporal_data(n_samples: int = 1000, n_features: int = 5,
                         drift_points: List[int] = None) -> Tuple[pd.DataFrame, np.ndarray]:
    """Generate synthetic temporal data with concept drift"""
    if drift_points is None:
        drift_points = [n_samples // 3, 2 * n_samples // 3]
        
    print("Generating temporal patterns...")
    with tqdm(total=4, desc="Data Generation") as pbar:
        # Generate timestamps
        base_time = datetime.now()
        timestamps = [base_time + timedelta(hours=i) for i in range(n_samples)]
        pbar.update(1)
        
        # Generate features with temporal patterns
        data = []
        for i in trange(n_samples, desc="Generating features", leave=False):
            # Base patterns
            temperature = 20 + 5 * np.sin(2 * np.pi * i / 24)  # Daily cycle
            pressure = 1015 + 5 * np.random.randn()
            stress = 5 + 3 * np.random.randn()
            sleep = 7 + np.random.randn()
            screen_time = 6 + 2 * np.random.randn()
            
            # Add more pronounced drift at specified points
            for drift_point in drift_points:
                if i >= drift_point:
                    temperature += 5  # More significant temperature change
                    stress *= 1.5  # More stress amplification
                    sleep -= 2  # More significant sleep reduction
                    screen_time += 3  # More screen time increase
                    pressure += 10  # Add pressure change
                    
            data.append([temperature, pressure, stress, sleep, screen_time])
        pbar.update(1)
        
        # Convert to DataFrame
        df = pd.DataFrame(data, columns=['temperature', 'pressure', 'stress_level', 
                                       'sleep_hours', 'screen_time'])
        df['timestamp'] = timestamps
        pbar.update(1)
        
        # Generate target with more pronounced drift effects
        y = np.zeros(n_samples)
        for i in trange(n_samples, desc="Generating labels", leave=False):
            base_prob = 0.3  # Base probability
            
            # Increase impact of features on migraine probability
            prob = base_prob
            prob += 0.2 * (data[i][2] > 8)  # High stress
            prob += 0.2 * (data[i][3] < 6)  # Low sleep
            prob += 0.15 * (data[i][4] > 8)  # High screen time
            prob += 0.15 * (data[i][0] > 25)  # High temperature
            
            # Add drift effects
            for drift_point in drift_points:
                if i >= drift_point:
                    prob += 0.1  # Increase base probability
            
            y[i] = np.random.binomial(1, min(prob, 0.95))  # Cap at 95% probability
        pbar.update(1)
        
    return df, y

def plot_drift_analysis(data: pd.DataFrame, drift_points: List[int], 
                       drift_detections: List[int], save_path: str):
    """Plot comprehensive drift analysis"""
    fig, axes = plt.subplots(3, 2, figsize=(15, 12))
    fig.suptitle('Drift Analysis', fontsize=16)
    
    # Plot feature evolution
    for i, feature in enumerate(data.columns[:5]):
        ax = axes[i // 2, i % 2] if i < 5 else axes[2, 0]
        data[feature].plot(ax=ax, alpha=0.7)
        ax.set_title(f'{feature} Evolution')
        
        # Add drift points
        for drift_point in drift_points:
            ax.axvline(x=drift_point, color='r', linestyle='--', alpha=0.3)
        
        # Add detected drifts
        for detect_point in drift_detections:
            ax.axvline(x=detect_point, color='g', linestyle=':', alpha=0.3)
    
    # Plot drift severity
    ax = axes[2, 1]
    if hasattr(data, 'drift_severity'):
        data.drift_severity.plot(ax=ax)
        ax.set_title('Drift Severity')
        
        # Add reference lines
        for drift_point in drift_points:
            ax.axvline(x=drift_point, color='r', linestyle='--', alpha=0.3)
        for detect_point in drift_detections:
            ax.axvline(x=detect_point, color='g', linestyle=':', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def main():
    print("Starting comprehensive framework test...")
    
    # Create plots directory if it doesn't exist
    os.makedirs('plots', exist_ok=True)
    
    # Initialize components
    with tqdm(total=4, desc="Initialization") as pbar:
        evaluator = FrameworkEvaluator()
        pbar.update(1)
        
        meta_learner = MetaLearner()
        pbar.update(1)
        
        detector = DriftDetector()
        pbar.update(1)
        
        # Set optimizers
        optimizers = create_optimizers(dim=5)
        meta_learner.set_algorithms([optimizers['DE (Adaptive)']])
        pbar.update(1)
    
    # Generate temporal data with drift
    drift_points = [300, 600]  # Explicit drift points
    data, y = generate_temporal_data(n_samples=1000, drift_points=drift_points)
    feature_names = ['temperature', 'pressure', 'stress_level', 'sleep_hours', 'screen_time']
    
    # Print data statistics
    print("\nData Statistics:")
    print(data[feature_names].describe())
    print(f"\nTarget distribution: {np.mean(y):.2%} positive cases")
    
    # Split into initial training and streaming data
    X_init, X_stream, y_init, y_stream = train_test_split(
        data[feature_names].values, y, test_size=0.7, shuffle=False
    )
    
    # Initial training
    print("\nPerforming initial training...")
    with tqdm(total=3, desc="Initial Setup") as pbar:
        meta_learner.optimize(X_init, y_init, feature_names=feature_names)
        pbar.update(1)
        
        initial_pred = meta_learner.predict(X_init)
        metrics = evaluator.evaluate_prediction_performance(y_init, initial_pred)
        print("Initial performance:", metrics)
        pbar.update(1)
        
        # Track initial feature importance
        importance = meta_learner.get_feature_importance()
        evaluator.track_feature_importance(feature_names, importance)
        pbar.update(1)
    
    # Set up drift detection
    detector.set_reference_window(X_init)
    
    # Simulate streaming predictions
    print("\nSimulating streaming predictions...")
    window_size = 50
    n_windows = len(X_stream) // window_size
    drift_detections = []
    drift_severities = []
    
    with tqdm(total=n_windows, desc="Processing Windows") as window_pbar:
        for i in range(0, len(X_stream), window_size):
            # Get current window
            X_window = X_stream[i:i+window_size]
            y_window = y_stream[i:i+window_size]
            
            # Make predictions
            y_pred = meta_learner.predict(X_window)
            metrics = evaluator.evaluate_prediction_performance(y_window, y_pred)
            
            # Process samples with inner progress bar
            with tqdm(total=len(X_window), desc=f"Window {i//window_size + 1}/{n_windows}", 
                     leave=False) as sample_pbar:
                for j in range(len(X_window)):
                    drift_detected, severity, info = detector.detect_drift()
                    drift_severities.append(severity)
                    
                    if drift_detected:
                        drift_detections.append(i + j)
                        info['severity'] = severity
                        evaluator.track_drift_event(info)
                        tqdm.write(f"\nDrift detected at sample {i+j}")
                        tqdm.write(f"Drift info: {info}")
                        
                        # Retrain on recent data if drift detected
                        recent_X = X_stream[max(0, i-100):i+j]
                        recent_y = y_stream[max(0, i-100):i+j]
                        meta_learner.optimize(recent_X, recent_y, feature_names=feature_names)
                        
                        # Track updated feature importance
                        importance = meta_learner.get_feature_importance()
                        evaluator.track_feature_importance(feature_names, importance)
                    
                    # Add sample to detector
                    detector.add_sample(X_window[j])
                    sample_pbar.update(1)
                    
                    # Update metrics in progress bar
                    sample_pbar.set_postfix({
                        'accuracy': f"{metrics['accuracy']:.3f}",
                        'drift_severity': f"{severity:.3f}"
                    })
            
            window_pbar.update(1)
            # Update window progress bar with overall metrics
            window_pbar.set_postfix({
                'avg_accuracy': f"{np.mean([m['accuracy'] for m in evaluator.metrics_history]):.3f}",
                'drifts': len(evaluator.drift_history)
            })
    
    # Add drift severity to data for plotting
    data['drift_severity'] = pd.Series(drift_severities + [0] * (len(data) - len(drift_severities)))
    
    # Generate final evaluation
    print("\nGenerating final evaluation...")
    with tqdm(total=3, desc="Final Evaluation") as pbar:
        evaluator.plot_framework_performance(save_path='plots/framework_performance.png')
        pbar.update(1)
        
        plot_drift_analysis(data, drift_points, drift_detections, 
                          save_path='plots/drift_analysis.png')
        pbar.update(1)
        
        report = evaluator.generate_performance_report()
        pbar.update(1)
    
    print("\nFinal Performance Report:")
    print("-------------------------")
    print(f"Current Performance: {report['current_performance']}")
    print("\nMetric Trends:")
    for metric, trend in report['metric_trends'].items():
        print(f"{metric}: {trend}")
    print("\nDrift Summary:")
    print(f"Total Drifts: {report['drift_summary']['total_drifts']}")
    print(f"Average Severity: {report['drift_summary']['avg_severity']:.3f}")
    print("Most Drifted Features:", report['drift_summary']['most_drifted_features'])
    
    # Print drift detection analysis
    print("\nDrift Detection Analysis:")
    print(f"True Drift Points: {drift_points}")
    print(f"Detected Drift Points: {drift_detections}")
    
    if drift_detections:
        detection_delays = []
        for true_point in drift_points:
            closest_detection = min(drift_detections, 
                                 key=lambda x: abs(x - true_point))
            delay = closest_detection - true_point
            detection_delays.append(delay)
        print(f"Average Detection Delay: {np.mean(detection_delays):.1f} samples")
    else:
        print("No drifts detected")

if __name__ == "__main__":
    main()
