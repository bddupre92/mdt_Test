import os
import numpy as np
import pandas as pd
import logging
import json
import matplotlib.pyplot as plt
from typing import Dict, List, Any, Tuple, Optional, Union
import argparse
from pathlib import Path
import time
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_squared_error
from scipy.stats import ks_2samp, ttest_ind

from utils.json_utils import save_json
from utils.plotting import save_plot, setup_plot_style

def generate_synthetic_data_with_drift(n_samples=1000, n_features=10, drift_points=None, noise_level=0.1):
    """
    Generate synthetic data with concept drift at specified points.
    
    Parameters:
    -----------
    n_samples : int, optional
        Number of samples to generate
    n_features : int, optional
        Number of features to generate
    drift_points : List[int], optional
        Points where drift should occur. If None, random points are generated.
    noise_level : float, optional
        Level of noise to add to the data
        
    Returns:
    --------
    Tuple[np.ndarray, np.ndarray, List[int]]
        X (features), y (target), and drift points
    """
    # Generate random drift points if not specified
    if drift_points is None:
        drift_points = sorted(np.random.choice(
            range(100, n_samples - 100), 
            size=3, 
            replace=False
        ))
    
    logging.info(f"Generated synthetic data with drift points at: {drift_points}")
    
    # Generate features (same distribution throughout)
    X = np.random.randn(n_samples, n_features)
    
    # Generate target with concept drift
    y = np.zeros(n_samples)
    
    # Define different target functions for each segment
    def target_function_1(x):
        return 2 * x[:, 0] - 1 * x[:, 1] + 0.5 * x[:, 2]
    
    def target_function_2(x):
        return -1 * x[:, 0] + 2 * x[:, 1] + 0.3 * x[:, 3]
    
    def target_function_3(x):
        return 0.5 * x[:, 0] + 0.5 * x[:, 1] - 2 * x[:, 2] + x[:, 4]
    
    def target_function_4(x):
        return 3 * x[:, 0] - 0.5 * x[:, 2] + 0.7 * x[:, 3] - x[:, 4]
    
    # Apply different functions to different segments
    functions = [target_function_1, target_function_2, target_function_3, target_function_4]
    
    # Determine segment boundaries
    segment_boundaries = [0] + drift_points + [n_samples]
    
    # Generate data for each segment
    for i in range(len(segment_boundaries) - 1):
        start = segment_boundaries[i]
        end = segment_boundaries[i + 1]
        
        # Apply the function for this segment
        function = functions[i % len(functions)]
        y[start:end] = function(X[start:end]) + noise_level * np.random.randn(end - start)
    
    # Add timestamps as feature 0
    timestamps = np.arange(n_samples)
    X_with_time = np.column_stack((timestamps, X))
    
    return X_with_time, y, drift_points

def check_drift_at_point(meta_learner, X_before, y_before, X_after, y_after, window_size=10, drift_threshold=0.01, significance_level=0.9):
    """
    Check for drift at a specific point using data before and after the point.
    
    Parameters:
    -----------
    meta_learner : object
        Model that supports predict() method
    X_before : np.ndarray
        Features before the potential drift point
    y_before : np.ndarray
        Targets before the potential drift point
    X_after : np.ndarray
        Features after the potential drift point
    y_after : np.ndarray
        Targets after the potential drift point
    window_size : int, optional
        Size of the window to use for drift detection
    drift_threshold : float, optional
        Threshold for drift detection
    significance_level : float, optional
        Significance level for hypothesis testing
        
    Returns:
    --------
    Tuple[bool, Dict[str, Any]]
        Whether drift was detected and detailed results
    """
    results = {
        'drift_detected': False,
        'p_value': None,
        'statistic': None,
        'prediction_error_before': None,
        'prediction_error_after': None,
        'error_change': None,
        'metrics': {}
    }
    
    # Make predictions
    try:
        y_pred_before = meta_learner.predict(X_before)
        y_pred_after = meta_learner.predict(X_after)
    except Exception as e:
        logging.error(f"Prediction error in drift check: {str(e)}")
        # Skip the meta_learner fallback and go straight to a simple model
        # This avoids any potential issues with invalid parameters
        simple_model = RandomForestRegressor(
            n_estimators=10,
            max_depth=5,
            min_samples_split=2,
            random_state=42
        )
        
        try:
            simple_model.fit(X_before, y_before)
            y_pred_before = simple_model.predict(X_before)
            y_pred_after = simple_model.predict(X_after)
        except Exception as inner_e:
            logging.error(f"Simple model fallback failed: {str(inner_e)}")
            return False, results
    
    # Calculate prediction errors
    error_before = mean_squared_error(y_before, y_pred_before)
    error_after = mean_squared_error(y_after, y_pred_after)
    
    # Calculate error change
    error_change = error_after / (error_before + 1e-10) - 1  # Avoid division by zero
    
    # Update results
    results['prediction_error_before'] = float(error_before)
    results['prediction_error_after'] = float(error_after)
    results['error_change'] = float(error_change)
    
    # Error-based drift detection
    drift_detected_by_error = error_change > drift_threshold
    results['drift_detected_by_error'] = drift_detected_by_error
    
    # Statistical test based drift detection
    residuals_before = y_before - y_pred_before
    residuals_after = y_after - y_pred_after
    
    # KS test on residuals
    ks_statistic, ks_p_value = ks_2samp(residuals_before, residuals_after)
    results['ks_statistic'] = float(ks_statistic)
    results['ks_p_value'] = float(ks_p_value)
    
    drift_detected_by_ks = ks_p_value < (1 - significance_level)
    results['drift_detected_by_ks'] = drift_detected_by_ks
    
    # T-test on residuals
    t_statistic, t_p_value = ttest_ind(residuals_before, residuals_after, equal_var=False)
    results['t_statistic'] = float(t_statistic)
    results['t_p_value'] = float(t_p_value)
    
    drift_detected_by_t = t_p_value < (1 - significance_level)
    results['drift_detected_by_t'] = drift_detected_by_t
    
    # Final decision based on all metrics
    results['drift_detected'] = drift_detected_by_error or drift_detected_by_ks or drift_detected_by_t
    
    return results['drift_detected'], results

def detect_drift_in_stream(X, y, meta_learner=None, window_size=50, drift_threshold=0.01, significance_level=0.9, min_drift_interval=30):
    """
    Detect drift in a data stream.
    
    Parameters:
    -----------
    X : np.ndarray
        Features
    y : np.ndarray
        Targets
    meta_learner : object, optional
        Model that supports predict() method
    window_size : int, optional
        Size of the window to use for drift detection
    drift_threshold : float, optional
        Threshold for drift detection
    significance_level : float, optional
        Significance level for hypothesis testing
    min_drift_interval : int, optional
        Minimum interval between drift points
        
    Returns:
    --------
    Tuple[List[int], List[Dict[str, Any]]]
        Detected drift points and detailed results
    """
    logging.info("Starting drift detection in stream...")
    
    # Create default model if none provided
    if meta_learner is None:
        meta_learner = RandomForestRegressor(
            n_estimators=50,
            max_depth=5,
            random_state=42
        )
    
    n_samples = len(y)
    detected_drift_points = []
    drift_results = []
    
    # Initialize model with initial window
    initial_window = min(window_size * 2, n_samples // 10)
    X_initial = X[:initial_window]
    y_initial = y[:initial_window]
    
    try:
        meta_learner.fit(X_initial, y_initial)
    except Exception as e:
        logging.error(f"Error fitting initial model: {str(e)}")
        return [], []
    
    # Scan through the stream
    last_drift_point = 0
    
    for i in range(initial_window + window_size, n_samples - window_size, window_size):
        # Skip if too close to last drift point
        if i - last_drift_point < min_drift_interval:
            continue
        
        # Get windows before and after current point
        X_before = X[i - window_size:i]
        y_before = y[i - window_size:i]
        X_after = X[i:i + window_size]
        y_after = y[i:i + window_size]
        
        # Check for drift
        drift_detected, results = check_drift_at_point(
            meta_learner, 
            X_before, 
            y_before, 
            X_after, 
            y_after, 
            window_size=window_size, 
            drift_threshold=drift_threshold, 
            significance_level=significance_level
        )
        
        # Record results with position information
        results['position'] = i
        results['timestamp'] = X[i, 0] if X.shape[1] > 1 else i
        drift_results.append(results)
        
        if drift_detected:
            logging.info(f"Drift detected at position {i}")
            detected_drift_points.append(i)
            last_drift_point = i
            
            # Retrain model with recent data
            try:
                meta_learner.fit(X[i - window_size:i + window_size], y[i - window_size:i + window_size])
            except Exception as e:
                logging.error(f"Error retraining model after drift: {str(e)}")
    
    return detected_drift_points, drift_results

def run_drift_detection(args):
    """
    Run drift detection on time series data.
    
    Parameters:
    -----------
    args : argparse.Namespace
        Command-line arguments
        
    Returns:
    --------
    Dict[str, Any]
        Results of drift detection
    """
    logging.info("Running drift detection...")
    
    # Parse arguments
    window_size = getattr(args, 'drift_window', 50)
    drift_threshold = getattr(args, 'drift_threshold', 0.5)
    significance_level = getattr(args, 'significance_level', 0.05)
    min_drift_interval = getattr(args, 'min_drift_interval', 30)
    visualize = getattr(args, 'visualize', False)
    data_path = getattr(args, 'data_path', None)
    is_synthetic = not bool(data_path)
    
    logging.info(f"Parameters: window_size={window_size}, drift_threshold={drift_threshold}, "
                 f"significance_level={significance_level}, min_drift_interval={min_drift_interval}")
    
    # Create results directory
    results_dir = Path('results/drift')
    results_dir.mkdir(exist_ok=True, parents=True)
    
    # Load or generate data
    if data_path:
        try:
            # Load data from file
            data = pd.read_csv(data_path)
            
            # Assume the first column is timestamp and the last column is target
            X = data.iloc[:, :-1].values
            y = data.iloc[:, -1].values
            
            logging.info(f"Loaded {len(y)} samples from {data_path}")
            
            # No ground truth drift points
            true_drift_points = []
            
        except Exception as e:
            logging.error(f"Error loading data: {str(e)}")
            logging.info("Falling back to synthetic data")
            is_synthetic = True
    
    if is_synthetic or not data_path:
        # Generate synthetic data with drift
        n_samples = getattr(args, 'n_samples', 1000)
        n_features = getattr(args, 'n_features', 10)
        
        X, y, true_drift_points = generate_synthetic_data_with_drift(
            n_samples=n_samples, 
            n_features=n_features,
            noise_level=0.1
        )
        
        logging.info(f"Generated {n_samples} synthetic samples with drift points at {true_drift_points}")
    
    # Create model
    meta_learner = RandomForestRegressor(
        n_estimators=50,
        max_depth=5,
        random_state=42
    )
    
    # Detect drift
    detected_drift_points, drift_results = detect_drift_in_stream(
        X, 
        y, 
        meta_learner=meta_learner,
        window_size=window_size,
        drift_threshold=drift_threshold,
        significance_level=significance_level,
        min_drift_interval=min_drift_interval
    )
    
    logging.info(f"Detected {len(detected_drift_points)} drift points at: {detected_drift_points}")
    
    # Prepare results
    results = {
        'metadata': {
            'data_source': data_path if data_path else 'synthetic',
            'n_samples': len(y),
            'n_features': X.shape[1] - 1 if X.shape[1] > 1 else X.shape[1],
            'window_size': window_size,
            'drift_threshold': drift_threshold,
            'significance_level': significance_level,
            'min_drift_interval': min_drift_interval
        },
        'true_drift_points': true_drift_points if is_synthetic else [],
        'detected_drift_points': detected_drift_points,
        'drift_results': drift_results
    }
    
    # Calculate performance metrics if ground truth is available
    if is_synthetic and true_drift_points:
        # Calculate detection performance
        true_positives = 0
        false_positives = 0
        false_negatives = 0
        
        # Check each detected point
        for point in detected_drift_points:
            # Check if it's close to any true drift point
            is_true_positive = any(abs(point - true_point) < window_size for true_point in true_drift_points)
            
            if is_true_positive:
                true_positives += 1
            else:
                false_positives += 1
        
        # Check missed drift points
        for point in true_drift_points:
            is_detected = any(abs(point - detected_point) < window_size for detected_point in detected_drift_points)
            
            if not is_detected:
                false_negatives += 1
        
        # Calculate metrics
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
        f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        results['performance'] = {
            'true_positives': true_positives,
            'false_positives': false_positives,
            'false_negatives': false_negatives,
            'precision': precision,
            'recall': recall,
            'f1_score': f1_score
        }
    
    # Visualize results if requested
    if visualize:
        try:
            setup_plot_style()
            
            # Plot data with drift points
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
            
            # Plot the target variable
            ax1.plot(y, alpha=0.7, label='Target')
            ax1.set_title('Target Variable with Drift Points')
            ax1.set_ylabel('Target Value')
            
            # Add true drift points if available
            if is_synthetic and true_drift_points:
                for point in true_drift_points:
                    ax1.axvline(x=point, color='green', linestyle='--', alpha=0.7)
                    ax1.text(point, ax1.get_ylim()[1] * 0.9, 'True', rotation=90, color='green')
            
            # Add detected drift points
            for point in detected_drift_points:
                ax1.axvline(x=point, color='red', linestyle='-', alpha=0.7)
                ax1.text(point, ax1.get_ylim()[1] * 0.8, 'Detected', rotation=90, color='red')
            
            # Add legend
            ax1.legend()
            
            # Plot drift metrics
            if drift_results:
                positions = [result['position'] for result in drift_results]
                error_changes = [result.get('error_change', 0) for result in drift_results]
                
                ax2.plot(positions, error_changes, marker='o', markersize=4, alpha=0.7, label='Error Change')
                ax2.axhline(y=drift_threshold, color='red', linestyle='--', alpha=0.7, label=f'Threshold ({drift_threshold})')
                ax2.set_xlabel('Position')
                ax2.set_ylabel('Error Change')
                ax2.set_title('Drift Detection Metrics')
                ax2.legend()
            
            plt.tight_layout()
            
            # Save plot
            save_path = save_plot(fig, 'drift_detection', plot_type='drift')
            logging.info(f"Drift detection visualization saved to {save_path}")
            
            # Create performance visualization if ground truth is available
            if is_synthetic and true_drift_points and 'performance' in results:
                fig, ax = plt.subplots(figsize=(8, 6))
                
                metrics = ['precision', 'recall', 'f1_score']
                values = [results['performance'][metric] for metric in metrics]
                
                ax.bar(metrics, values)
                ax.set_ylim(0, 1)
                ax.set_ylabel('Score')
                ax.set_title('Drift Detection Performance')
                
                # Add value labels
                for i, v in enumerate(values):
                    ax.text(i, v + 0.05, f"{v:.2f}", ha='center')
                
                # Save plot
                save_path = save_plot(fig, 'drift_detection_performance', plot_type='drift')
                logging.info(f"Drift detection performance visualization saved to {save_path}")
                
        except Exception as e:
            logging.error(f"Error creating visualization: {str(e)}")
    
    # Save results
    timestamp = time.strftime('%Y%m%d_%H%M%S')
    results_file = results_dir / f'drift_detection_results_{timestamp}.json'
    
    save_json(results, results_file)
    logging.info(f"Drift detection results saved to {results_file}")
    
    # Print summary
    print("\nDrift Detection Summary:")
    print("=======================")
    print(f"Data source: {'Synthetic' if is_synthetic else data_path}")
    print(f"Number of samples: {len(y)}")
    print(f"Detected {len(detected_drift_points)} drift points")
    
    if is_synthetic and true_drift_points:
        print("\nPerformance Metrics:")
        print(f"  Precision: {results['performance']['precision']:.2f}")
        print(f"  Recall: {results['performance']['recall']:.2f}")
        print(f"  F1 Score: {results['performance']['f1_score']:.2f}")
    
    print(f"\nResults saved to {results_file}")
    
    return results
