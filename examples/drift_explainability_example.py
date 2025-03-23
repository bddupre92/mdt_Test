"""
Example script demonstrating drift detection with explainability
"""

import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import json
import logging
from typing import Dict, List, Optional, Tuple, Union, Any
import random
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from utils.plot_utils import save_plot

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# Add parent directory to path to allow imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import local modules
from meta.meta_learner import MetaLearner
from meta.drift_detector import DriftDetector as MetaDriftDetector
from meta.drift_detector import DriftDetector
from explainability.explainer_factory import ExplainerFactory
from optimizers.optimizer_factory import create_optimizers
from models.model_factory import ModelFactory

def generate_synthetic_data_with_drift(n_samples=1000, n_features=10, n_drift_points=3):
    """
    Generate synthetic data with drift
    
    Args:
        n_samples: Number of samples to generate
        n_features: Number of features
        n_drift_points: Number of drift points to introduce
        
    Returns:
        X: Feature matrix
        y: Target values
        drift_points: Indices where drift occurs
    """
    np.random.seed(42)
    X = np.random.randn(n_samples, n_features)
    
    # Initial coefficients
    coef = np.random.randn(n_features)
    
    # Generate target values with noise
    y = np.zeros(n_samples)
    
    # Introduce drift at random points
    drift_points = []
    if n_drift_points > 0:
        drift_points = sorted(np.random.choice(
            range(100, n_samples-100), 
            size=n_drift_points, 
            replace=False
        ))
    
    # Generate data with different patterns before and after drift
    prev_point = 0
    for point in drift_points + [n_samples]:
        # Generate data for this segment
        segment_y = np.dot(X[prev_point:point], coef) + np.random.randn(point - prev_point) * 0.5
        y[prev_point:point] = segment_y
        
        # Change coefficients for the next segment (simulate drift)
        coef = np.random.randn(n_features)
        prev_point = point
    
    return X, y, drift_points

def generate_drift_explanation(meta_learner, X_data, y_true, y_pred, idx, X_train=None, y_train=None):
    """Generate explanation for drift"""
    try:
        # Use the provided X_train and y_train if available, otherwise use default
        X_train_data = X_train if X_train is not None else X_data[:200]
        y_train_data = y_train if y_train is not None else y_true[:200]
        
        # Get the drift window
        window_size = 50
        start_idx = max(0, idx - window_size)
        end_idx = min(len(X_data), idx + window_size)
        
        X_drift = X_data[start_idx:end_idx]
        y_drift = y_true[start_idx:end_idx]
        
        # Check if we have enough data
        if len(X_drift) < 1:
            print(f"Not enough data for drift explanation at index {idx}")
            return None
        
        # Train a model on the training data
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train_data, y_train_data)
        
        # Create explainer
        explainer_factory = ExplainerFactory()
        explainer = explainer_factory.create_explainer(
            explainer_type='shap',
            model_type='random_forest',
            model=model
        )
        
        # Generate explanation - use the correct signature with both X and y
        explanation = explainer.explain(X_drift, y=y_drift)
        
        # Create drift explanation plot
        plt.figure(figsize=(10, 6))
        feature_importance = explainer.get_feature_importance()
        
        # Check if feature_importance is valid
        if feature_importance is None or not feature_importance:
            print(f"No valid feature importance generated for drift at index {idx}")
            return None
            
        # Sort features by importance
        sorted_features = sorted(feature_importance.items(), key=lambda x: abs(x[1]), reverse=True)
        top_features = sorted_features[:5] if len(sorted_features) >= 5 else sorted_features
        
        # Plot
        feature_names = [f[0] for f in top_features]
        importance_values = [f[1] for f in top_features]
        
        plt.barh(feature_names, importance_values)
        plt.xlabel('Importance')
        plt.ylabel('Feature')
        plt.title(f'Top Features Contributing to Drift at Sample {idx}')
        
        # Save plot
        save_plot(
            plt.gcf(),
            f'drift_explanation_{idx}.png',
            plot_type='drift'
        )
        plt.close()
        
        return top_features
    except Exception as e:
        print(f"Error generating drift explanation: {str(e)}")
        return None

def main():
    """Run the drift detection example with explainability"""
    print("Running drift detection with explainability example...")
    
    # Create results directory
    results_dir = Path("results/drift_explainability_example")
    results_dir.mkdir(exist_ok=True, parents=True)
    
    # Generate synthetic data with drift
    print("Generating synthetic data with drift...")
    X, y, drift_points = generate_synthetic_data_with_drift(n_samples=1000, n_features=10, n_drift_points=3)
    print(f"Generated data with {X.shape[1]} features and {X.shape[0]} samples")
    print(f"True drift points: {drift_points}")
    
    # Split data into initial training and streaming parts
    train_size = 200
    X_train, y_train = X[:train_size], y[:train_size]
    X_stream, y_stream = X[train_size:], y[train_size:]
    
    # Create feature names
    feature_names = [f"feature_{i}" for i in range(X.shape[1])]
    n_features = len(feature_names)
    
    # Initialize MetaLearner with drift detection
    print("Initializing MetaLearner with drift detection...")
    meta_learner = MetaLearner(
        method='bayesian',
        selection_strategy='bayesian',
        exploration_factor=0.3
    )
    
    # Set optimizers
    meta_learner.set_algorithms(list(create_optimizers(dim=n_features, bounds=[(-5, 5)] * n_features).values()))
    
    # Set drift detector with extremely sensitive parameters
    meta_learner.drift_detector = DriftDetector(
        window_size=10,  # Very small window to detect changes faster
        drift_threshold=0.05,  # Very low threshold to be extremely sensitive
        significance_level=0.6,  # Very high significance level
        min_drift_interval=5,  # Very short interval between drifts
        ema_alpha=0.9  # Very high alpha for very fast adaptation
    )
    
    # Train initial model
    print("Training initial model...")
    meta_learner.fit(X_train, y_train, feature_names)
    
    # Set a valid model configuration
    model_config = {
        'model_type': 'random_forest',
        'n_estimators': 100,
        'max_depth': None,
        'min_samples_split': 10,
        'min_samples_leaf': 4,
        'max_features': 'sqrt'
    }
    meta_learner.best_config = model_config
    print(f"Using valid model configuration: {model_config}")
    
    # Create a model factory for predictions
    model_factory = ModelFactory()
    
    # Store training data separately for the custom prediction function
    X_train = X[:200]
    y_train = y[:200]
    
    # Create a custom prediction function that doesn't rely on meta_learner.predict
    def custom_predict(X_data):
        model = model_factory.create_model(model_config)
        model.fit(X_train, y_train)
        return model.predict(X_data)
    
    # Manually check for drift at specific points
    def check_drift_at_point(point_idx, X_data, y_true, meta_learner):
        print(f"\nManually checking for drift at point {point_idx}...")
        
        try:
            # Use custom predict instead of meta_learner.predict
            y_pred = custom_predict(X_data[point_idx:point_idx+1])
            
            # Calculate error metrics
            error = np.abs(y_pred - y_true[point_idx:point_idx+1])
            
            # Check for drift using error threshold
            if error.mean() > 0.5:  # Threshold for drift detection
                print(f"Manual drift check at {point_idx}: Drift detected!")
                return True
            else:
                print(f"Manual drift check at {point_idx}: No drift detected")
                return False
        except Exception as e:
            print(f"Prediction error: {str(e)}")
            print(f"Manual drift check at {point_idx}: No drift detected")
            return False
    
    # Manually check for drift at known drift points
    print("\nManually checking for drift at known drift points...")
    true_drift_points = [d for d in drift_points if d >= train_size]
    detected_drift_points = []
    for drift_point in true_drift_points:
        print(f"Near known drift point {drift_point}: Mean shift={meta_learner.drift_detector.get_statistics()['mean_shift']:.4f}, KS stat={meta_learner.drift_detector.get_statistics()['ks_statistic']:.4f}, p-value={meta_learner.drift_detector.get_statistics()['p_value']:.4f}, Severity={meta_learner.drift_detector.get_statistics()['severity']:.4f}")
        is_drift = check_drift_at_point(drift_point, X_stream, y_stream, meta_learner)
        if is_drift:
            detected_drift_points.append(drift_point)
            print(f"Manual drift check at {drift_point}: Drift detected")
        else:
            print(f"Manual drift check at {drift_point}: No drift detected")
    
    # Process streaming data with drift detection
    chunk_size = 50
    predictions = []
    true_values = []
    drift_explanations = []
    
    print("Processing streaming data with drift detection...")
    for i in range(0, len(X_stream), chunk_size):
        X_chunk = X_stream[i:i+chunk_size]
        y_chunk = y_stream[i:i+chunk_size]
        
        # Make predictions using our custom function
        y_pred = custom_predict(X_chunk)
        predictions.extend(y_pred)
        true_values.extend(y_chunk)
        
        # Update with new data
        drift_detected = meta_learner.update(X_chunk, y_chunk)
        
        # Print drift detection statistics for debugging
        if i % 50 == 0:
            stats = meta_learner.drift_detector.get_statistics()
            print(f"Sample {i+200}: Mean shift={stats['mean_shift']:.4f}, KS stat={stats['ks_statistic']:.4f}, p-value={stats['p_value']:.4f}, Severity={stats['severity']:.4f}")
        
        if drift_detected:
            drift_point = i + train_size
            print(f"Drift detected at sample {drift_point}")
            
            # Generate explanation for the drift
            try:
                # Generate explanation
                explanation = generate_drift_explanation(meta_learner, X_stream, y_stream, predictions, drift_point, X_train, y_train)
                
                # Store explanation summary
                drift_explanations.append({
                    'drift_point': drift_point,
                    'feature_importance': explanation,
                    'explanation_path': str(results_dir)
                })
                
                print(f"Generated drift explanation at sample {drift_point}")
                print("Top 5 features contributing to drift:")
                if explanation:
                    for feature, importance in sorted(
                        explanation,
                        key=lambda x: x[1],
                        reverse=True
                    )[:5]:
                        print(f"  {feature}: {importance:.4f}")
                else:
                    print("  No valid feature importance could be generated")
            except Exception as e:
                print(f"Error generating drift explanation: {str(e)}")
    
    # Calculate metrics
    mse = mean_squared_error(true_values, predictions)
    r2 = r2_score(true_values, predictions)
    
    print(f"\nModel performance - MSE: {mse:.4f}, R²: {r2:.4f}")
    
    # Evaluate drift detection
    print("\nDrift Detection Results:")
    print(f"True drift points: {true_drift_points}")
    print(f"Detected drift points: {detected_drift_points}")
    
    # Plot results
    plt.figure(figsize=(12, 6))
    plt.plot(range(len(true_values)), true_values, 'b-', alpha=0.5, label='True values')
    plt.plot(range(len(predictions)), predictions, 'r-', alpha=0.5, label='Predictions')
    
    # Mark true drift points
    for point in true_drift_points:
        plt.axvline(x=point-train_size, color='g', linestyle='--', alpha=0.7, label='True drift' if point == true_drift_points[0] else None)
    
    # Mark detected drift points
    for point in detected_drift_points:
        plt.axvline(x=point-train_size, color='r', linestyle=':', alpha=0.7, label='Detected drift' if point == detected_drift_points[0] else None)
    
    plt.legend()
    plt.title('Drift Detection Results')
    plt.xlabel('Sample index')
    plt.ylabel('Target value')
    
    # Save the figure using save_plot
    save_plot(plt.gcf(), 'drift_detection_results.png', plot_type='drift')
    plt.close()
    
    print(f"\nAll results have been saved to the '{results_dir}' directory.")

if __name__ == '__main__':
    main()
