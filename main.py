import os
import sys
import time
import json
import logging
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Any, Optional, Union
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import pickle
import traceback
from pathlib import Path

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import local modules
from meta.meta_learner import MetaLearner
from meta.meta_optimizer import MetaOptimizer
from models.model_factory import ModelFactory
from explainability.explainer_factory import ExplainerFactory
from optimizers.optimizer_factory import (
    DifferentialEvolutionOptimizer, 
    EvolutionStrategyOptimizer,
    AntColonyOptimizer,
    GreyWolfOptimizer,
    create_optimizers
)
from visualization.drift_analysis import DriftAnalyzer
from visualization.optimizer_analysis import OptimizerAnalyzer
from evaluation.framework_evaluator import FrameworkEvaluator
from optimizers import (
    DifferentialEvolutionOptimizer,
    EvolutionStrategyOptimizer,
    AntColonyOptimizer,
    GreyWolfOptimizer,
    create_optimizers
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

"""
Comprehensive benchmark and comparison of optimization algorithms
including meta-optimization for novel algorithm creation.
"""

import seaborn as sns
import matplotlib
matplotlib.use('TkAgg')  # Try 'MacOSX' or 'Qt5Agg' if this doesn't work
from concurrent.futures import ProcessPoolExecutor
from functools import partial

# Import benchmark functions
from benchmarking.test_functions import TEST_FUNCTIONS, create_test_suite

# Create benchmark functions dictionary
benchmark_functions = TEST_FUNCTIONS

# Import optimizers
from optimizers.optimizer_factory import (
    DifferentialEvolutionOptimizer, 
    EvolutionStrategyOptimizer,
    AntColonyOptimizer,
    GreyWolfOptimizer,
    create_optimizers
)

# Import meta-optimizer
from meta.meta_optimizer import MetaOptimizer

# Import drift detection
from drift_detection.drift_detector import DriftDetector

# Import visualization tools
from visualization.optimizer_analysis import OptimizerAnalyzer
from visualization.live_visualization import LiveOptimizationMonitor

# Import explainability framework
from explainability.explainer_factory import ExplainerFactory
from explainability.base_explainer import BaseExplainer

# Import utility functions
from utils.plot_utils import save_plot

# Setup logging and directories
def setup_environment():
    """Configure logging and create necessary directories"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('optimization.log'),
            logging.StreamHandler()
        ]
    )
    
    # Create directories
    directories = ['results', 'results/plots', 'results/data', 'results/explainability']
    for directory in directories:
        Path(directory).mkdir(exist_ok=True, parents=True)

# Create optimizers for benchmarking
def create_optimizers(dim: int, bounds: List[Tuple[float, float]], verbose: bool = True) -> Dict[str, Any]:
    """Create all optimizer instances for benchmarking"""
    return {
        'ACO': AntColonyOptimizer(dim=dim, bounds=bounds, name="ACO", verbose=verbose),
        'GWO': GreyWolfOptimizer(dim=dim, bounds=bounds, name="GWO", verbose=verbose),
        'DE': DifferentialEvolutionOptimizer(dim=dim, bounds=bounds, name="DE", verbose=verbose),
        'ES': EvolutionStrategyOptimizer(dim=dim, bounds=bounds, name="ES", verbose=verbose),
        'DE (Adaptive)': DifferentialEvolutionOptimizer(dim=dim, bounds=bounds, adaptive=True, name="DE (Adaptive)", verbose=verbose),
        'ES (Adaptive)': EvolutionStrategyOptimizer(dim=dim, bounds=bounds, adaptive=True, name="ES (Adaptive)", verbose=verbose),
        'Meta-Optimizer': MetaOptimizer(
            dim=dim, 
            bounds=bounds, 
            optimizers={
                'DE': DifferentialEvolutionOptimizer(dim=dim, bounds=bounds),
                'ES': EvolutionStrategyOptimizer(dim=dim, bounds=bounds)
            },
            verbose=verbose
        )
    }

# Run optimization
def run_optimization(
    dim: int = 30, 
    max_evals: int = 10000, 
    n_runs: int = 5, 
    live_viz: bool = False, 
    save_plots: bool = True
) -> Tuple[Dict[str, Any], pd.DataFrame]:
    """
    Run optimization process with multiple optimizers and test functions.
    
    Args:
        dim: Dimensionality of the optimization problem
        max_evals: Maximum number of function evaluations
        n_runs: Number of independent runs per optimizer
        live_viz: Whether to enable live visualization
        save_plots: Whether to save plots
        
    Returns:
        Tuple of (results dictionary, results dataframe)
    """
    logging.info("Starting optimization process")
    
    # Create results directories
    results_dir = Path('results')
    data_dir = results_dir / 'data'
    plots_dir = results_dir / 'performance'
    
    for directory in [results_dir, data_dir, plots_dir]:
        directory.mkdir(exist_ok=True, parents=True)
    
    # Define test functions
    test_suite = create_test_suite()
    selected_functions = {
        'unimodal': ['sphere', 'rosenbrock'],
        'multimodal': ['rastrigin', 'ackley'],
    }
    
    # Prepare benchmark functions
    benchmark_functions = {}
    for category, functions in selected_functions.items():
        for func_name in functions:
            func_creator = TEST_FUNCTIONS[func_name]
            bounds = [(-5, 5)] * dim  # Default bounds
            benchmark_functions[func_name] = func_creator(dim, bounds)
    
    # Create optimizers
    bounds = [(-5, 5)] * dim
    optimizers = create_optimizers(dim, bounds)
    
    # Create meta-optimizer
    meta_opt = MetaOptimizer(
        dim=dim,
        bounds=bounds,
        optimizers=optimizers,
        history_file=str(data_dir / 'meta_history.json'),
        selection_file=str(data_dir / 'meta_selection.json')
    )
    
    # Enable live visualization if requested
    if live_viz:
        meta_opt.enable_live_visualization(str(plots_dir) if save_plots else None)
    
    # Add meta-optimizer to the comparison
    all_optimizers = optimizers.copy()
    all_optimizers['Meta-Optimizer'] = meta_opt
    
    # Create analyzer and run comparison
    analyzer = OptimizerAnalyzer(all_optimizers)
    results = {}
    all_results_data = []
    
    # Run optimization for each test function
    for func_name, func in benchmark_functions.items():
        logging.info(f"Optimizing {func_name} function")
        
        function_results = analyzer.run_comparison(
            {func_name: func},
            n_runs=n_runs,
            record_convergence=True,
            max_evals=max_evals
        )
        
        # Store results
        results[func_name] = function_results[func_name]
        
        # Collect data for summary dataframe
        for opt_name, opt_results in function_results[func_name].items():
            for run, result in enumerate(opt_results):
                all_results_data.append({
                    'function': func_name,
                    'dimension': dim,
                    'optimizer': opt_name,
                    'run': run,
                    'best_score': result.best_score,
                    'execution_time': result.execution_time,
                    'convergence_length': len(result.convergence_curve)
                })
        
        # Generate plots for this function
        if save_plots:
            # Plot convergence
            fig = analyzer.plot_convergence_comparison()
            if fig:  # Only save if a figure was returned
                save_plot(fig, f"{func_name}_convergence.png", plot_type='performance')
                plt.close(fig)
            
            # Plot parameter adaptation and diversity for each optimizer
            for opt_name in all_optimizers:
                try:
                    fig = analyzer.plot_parameter_adaptation(opt_name, func_name)
                    if fig:  # Only save if a figure was returned
                        save_plot(fig, f"{func_name}_{opt_name}_parameters.png", plot_type='performance')
                        plt.close(fig)
                except Exception as e:
                    logging.warning(f"Could not plot parameter adaptation for {opt_name}: {str(e)}")
                
                try:
                    fig = analyzer.plot_diversity_analysis(opt_name, func_name)
                    if fig:  # Only save if a figure was returned
                        save_plot(fig, f"{func_name}_{opt_name}_diversity.png", plot_type='performance')
                        plt.close(fig)
                except Exception as e:
                    logging.warning(f"Could not plot diversity analysis for {opt_name}: {str(e)}")
    
    # Create overall performance heatmap
    if save_plots:
        try:
            analyzer.plot_performance_heatmap()
        except Exception as e:
            logging.warning(f"Could not plot performance heatmap: {str(e)}")
        
        # Create boxplot of performance across functions
        try:
            plt.figure(figsize=(14, 10))
            sns.boxplot(data=pd.DataFrame(all_results_data), x='optimizer', y='best_score', hue='function')
            plt.yscale('log')
            plt.title('Performance Comparison Across Functions')
            plt.xticks(rotation=45)
            plt.tight_layout()
            fig = plt.gcf()
            save_plot(fig, "performance_boxplot.png", plot_type='performance')
            plt.close(fig)
        except Exception as e:
            logging.warning(f"Could not plot performance boxplot: {str(e)}")
    
    # Create and save summary dataframe
    results_df = pd.DataFrame(all_results_data)
    results_df.to_csv(data_dir / 'optimization_results.csv', index=False)
    
    # Create summary statistics
    summary_stats = results_df.groupby(['function', 'optimizer']).agg({
        'best_score': ['mean', 'std', 'min'],
        'execution_time': ['mean', 'std']
    }).reset_index()
    
    summary_stats.to_csv(data_dir / 'optimization_summary.csv')
    
    # Prepare final results
    final_results = {
        'best_score': results_df['best_score'].min(),
        'best_optimizer': results_df.loc[results_df['best_score'].idxmin(), 'optimizer'],
        'best_function': results_df.loc[results_df['best_score'].idxmin(), 'function'],
        'summary_stats': summary_stats,
        'optimizers': list(all_optimizers.keys())
    }
    
    return final_results, results_df

# Run benchmark comparison
def run_benchmark_comparison(n_runs: int = 30, max_evals: int = 10000, live_viz: bool = False, save_plots: bool = True):
    """Run comprehensive benchmark comparison of all optimizers"""
    logging.info("Starting benchmark comparison")
    
    # Get test functions
    test_suite = create_test_suite()
    selected_functions = {
        'unimodal': ['sphere', 'rosenbrock'],
        'multimodal': ['rastrigin', 'ackley', 'griewank'],
        'hybrid': ['levy']
    }
    
    # Prepare benchmark functions
    benchmark_functions = {}
    for category, functions in selected_functions.items():
        for func_name in functions:
            for dim in [10, 30]:
                key = f"{func_name}_{dim}D"
                func_creator = TEST_FUNCTIONS[func_name]
                bounds = [(-5, 5)] * dim  # Default bounds
                benchmark_functions[key] = func_creator(dim, bounds)
    
    # Run comparison for each function
    results = {}
    all_results_data = []
    
    for func_name, func in benchmark_functions.items():
        logging.info(f"Benchmarking function: {func_name}")
        
        # Create optimizers for this dimension
        dim = func.dim
        bounds = func.bounds
        optimizers = create_optimizers(dim, bounds)
        
        # Create meta-optimizer
        meta_opt = MetaOptimizer(
            dim=dim,
            bounds=bounds,
            optimizers=optimizers,
            history_file=f'results/data/meta_history_{func_name}.json'
        )
        
        # Enable live visualization if requested
        if live_viz:
            save_path = 'results/plots' if save_plots else None
            meta_opt.enable_live_visualization(save_path)
            
        # Add meta-optimizer to the comparison
        all_optimizers = optimizers.copy()
        all_optimizers['Meta-Optimizer'] = meta_opt
        
        # Create analyzer and run comparison
        analyzer = OptimizerAnalyzer(all_optimizers)
        function_results = analyzer.run_comparison(
            {func_name: func},
            n_runs=n_runs,
            record_convergence=True,
            max_evals=max_evals
        )
        
        # Store results
        results[func_name] = function_results[func_name]
        
        # Collect data for summary dataframe
        for opt_name, opt_results in function_results[func_name].items():
            for run, result in enumerate(opt_results):
                all_results_data.append({
                    'function': func_name,
                    'dimension': dim,
                    'optimizer': opt_name,
                    'run': run,
                    'best_score': result.best_score,
                    'execution_time': result.execution_time,
                    'convergence_length': len(result.convergence_curve)
                })
        
        # Generate plots for this function
        if save_plots:
            analyzer.plot_convergence_comparison()
            for opt_name in all_optimizers:
                try:
                    analyzer.plot_parameter_adaptation(opt_name, func_name)
                except:
                    logging.warning(f"Could not plot parameter adaptation for {opt_name}")
                
                try:
                    analyzer.plot_diversity_analysis(opt_name, func_name)
                except:
                    logging.warning(f"Could not plot diversity analysis for {opt_name}")
    
    # Create overall performance heatmap
    if save_plots:
        analyzer.plot_performance_heatmap()
    
    # Create and save summary dataframe
    results_df = pd.DataFrame(all_results_data)
    results_df.to_csv('results/data/benchmark_results.csv', index=False)
    
    # Create summary statistics
    summary_stats = results_df.groupby(['function', 'optimizer']).agg({
        'best_score': ['mean', 'std', 'min'],
        'execution_time': ['mean', 'std']
    }).reset_index()
    
    summary_stats.to_csv('results/data/benchmark_summary.csv')
    
    # Create additional visualizations
    if save_plots:
        plt.figure(figsize=(14, 10))
        sns.boxplot(data=results_df, x='optimizer', y='best_score', hue='function')
        plt.yscale('log')
        plt.title('Performance Comparison Across Functions')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig('results/plots/performance_boxplot.png')
        plt.close()
    
    return results, results_df

# Run meta-learner with drift detection
def run_meta_learner_with_drift(n_samples=1000, n_features=10, drift_points=None, 
                              window_size=50, drift_threshold=0.5, significance_level=0.05, 
                              min_drift_interval=30, ema_alpha=0.3, visualize=False):
    """Run meta-learner with drift detection."""
    logging.info("Running meta-learner with drift detection")
    
    # Generate synthetic data
    X_full, y_full, drift_points = generate_synthetic_data_with_drift(
        n_samples=n_samples,
        n_features=n_features,
        drift_points=[n_samples//4, n_samples//2, 3*n_samples//4]  # Fixed drift points
    )
    logging.info(f"Generated synthetic data with drift points at: {drift_points}")
    
    # Initialize meta-learner
    meta_learner = MetaLearner()
    
    # Import the correct DriftDetector class from drift_detection
    from drift_detection.drift_detector import DriftDetector
    
    # Initialize drift detector with enhanced parameters
    detector = DriftDetector(
        window_size=window_size,
        drift_threshold=drift_threshold,
        significance_level=significance_level,
        min_drift_interval=min_drift_interval,
        ema_alpha=ema_alpha,
        max_history_size=100  # Add max_history_size parameter
    )
    
    # Set algorithms
    from optimizers.optimizer_factory import create_optimizers
    optimizers = create_optimizers(dim=n_features, bounds=[(-5, 5)] * n_features)
    meta_learner.set_algorithms(list(optimizers.values()))
    
    # Set drift detector with sensitive parameters
    meta_learner.drift_detector = detector
    
    # Initialize with first 200 samples
    X_init, y_init = X_full[:200], y_full[:200]
    meta_learner.fit(X_init, y_init)
    
    # Get best configuration
    best_config = meta_learner.best_config
    
    # Process data in chunks and check for drift
    chunk_size = 50
    detected_drifts = []
    all_predictions = []
    all_true_values = []
    all_errors = []
    all_severities = []
    
    for i in range(4, n_samples // chunk_size):
        start_idx = i * chunk_size
        end_idx = min((i + 1) * chunk_size, n_samples)
        X_chunk, y_chunk = X_full[start_idx:end_idx], y_full[start_idx:end_idx]
        
        # Check for drift
        try:
            drift_detected = meta_learner.update(X_chunk, y_chunk)
            if drift_detected:
                detected_drifts.append(start_idx)
                logging.info(f"Drift detected at sample {start_idx}")
        except Exception as e:
            logging.error(f"Error in update: {str(e)}")
        
        # Log statistics every 100 samples
        if i % 2 == 0:
            try:
                # Make predictions
                y_pred = meta_learner.predict(X_chunk)
                all_predictions.extend(y_pred)
                all_true_values.extend(y_chunk)
                
                # Calculate statistics
                from scipy import stats
                errors = y_chunk - y_pred
                all_errors.extend(errors)
                
                mean_shift = np.abs(np.mean(errors))
                
                # KS test with uniform distribution as reference
                ks_stat, p_value = stats.kstest(errors, 'uniform')
                
                # Calculate severity
                severity = mean_shift * ks_stat
                all_severities.append((start_idx, severity))
                
                logging.info(f"Sample {start_idx}: Mean shift={mean_shift:.4f}, KS stat={ks_stat:.4f}, p-value={p_value:.4f}, Severity={severity:.4f}")
            except Exception as e:
                logging.error(f"Prediction error: {str(e)}")
    
    # Manually check for drift at known drift points
    manual_drift_checks = []
    for drift_point in drift_points:
        logging.info(f"Manually checking for drift at point {drift_point}...")
        
        # Get data before and after drift point
        before_start = max(0, drift_point - window_size * 2)
        after_end = min(n_samples, drift_point + window_size * 2)
        
        X_before = X_full[before_start:drift_point]
        y_before = y_full[before_start:drift_point]
        X_after = X_full[drift_point:after_end]
        y_after = y_full[drift_point:after_end]
        
        # Check for drift
        drift_detected, mean_shift, ks_statistic, p_value = check_drift_at_point(
            meta_learner, X_before, y_before, X_after, y_after, 
            window_size=window_size, drift_threshold=drift_threshold, 
            significance_level=significance_level
        )
        
        manual_drift_checks.append({
            'point': drift_point,
            'detected': drift_detected,
            'mean_shift': mean_shift,
            'ks_stat': ks_statistic,
            'p_value': p_value,
            'severity': mean_shift * ks_statistic
        })
        
        if drift_detected:
            logging.info(f"Drift check at {drift_point}: Detected")
        else:
            logging.info(f"Drift check at {drift_point}: Not detected")
            
        logging.info(f"Stats: Mean shift={mean_shift:.4f}, KS stat={ks_statistic:.4f}, p-value={p_value:.4f}")
    
    # Get feature importance
    try:
        feature_importance = meta_learner.get_feature_importance()
        feature_names = [f"feature_{i}" for i in range(n_features)]
        feature_importance_dict = dict(zip(feature_names, feature_importance))
    except Exception as e:
        logging.warning(f"Could not retrieve feature importance")
        feature_importance_dict = {}
    
    # Visualize results if requested
    if visualize:
        try:
            import matplotlib.pyplot as plt
            import os
            
            # Create results directory if it doesn't exist
            os.makedirs('results', exist_ok=True)
            
            # Plot 1: True values vs Predictions
            plt.figure(figsize=(12, 6))
            plt.plot(all_true_values, label='True Values')
            plt.plot(all_predictions, label='Predictions')
            for drift_point in drift_points:
                plt.axvline(x=drift_point - 200, color='r', linestyle='--', alpha=0.5)
            for detected in detected_drifts:
                plt.axvline(x=detected - 200, color='g', linestyle=':', alpha=0.5)
            plt.legend()
            plt.title('True Values vs Predictions')
            plt.xlabel('Sample Index')
            plt.ylabel('Value')
            save_plot(plt.gcf(), 'drift_true_vs_pred.png', plot_type='drift')
            plt.close()
            
            # Plot 2: Prediction Errors
            plt.figure(figsize=(12, 6))
            plt.plot(all_errors)
            for drift_point in drift_points:
                plt.axvline(x=drift_point - 200, color='r', linestyle='--', alpha=0.5, label='Known Drift' if drift_point == drift_points[0] else None)
            for detected in detected_drifts:
                plt.axvline(x=detected - 200, color='g', linestyle=':', alpha=0.5, label='Detected Drift' if detected == detected_drifts[0] else None)
            plt.legend()
            plt.title('Prediction Errors')
            plt.xlabel('Sample Index')
            plt.ylabel('Error')
            save_plot(plt.gcf(), 'drift_errors.png', plot_type='drift')
            plt.close()
            
            # Plot 3: Drift Severity
            plt.figure(figsize=(12, 6))
            indices = [x[0] for x in all_severities]
            severities = [x[1] for x in all_severities]
            plt.plot(indices, severities, marker='o')
            plt.axhline(y=drift_threshold, color='r', linestyle='--', label='Threshold')
            for drift_point in drift_points:
                plt.axvline(x=drift_point, color='r', linestyle='--', alpha=0.5, label='Known Drift' if drift_point == drift_points[0] else None)
            for detected in detected_drifts:
                plt.axvline(x=detected, color='g', linestyle=':', alpha=0.5, label='Detected Drift' if detected == detected_drifts[0] else None)
            plt.legend()
            plt.title('Drift Severity Over Time')
            plt.xlabel('Sample Index')
            plt.ylabel('Severity')
            save_plot(plt.gcf(), 'drift_severity.png', plot_type='drift')
            plt.close()
            
            # Plot 4: Feature Importance
            if feature_importance_dict:
                plt.figure(figsize=(12, 6))
                features = list(feature_importance_dict.keys())
                importances = list(feature_importance_dict.values())
                plt.bar(features, importances)
                plt.title('Feature Importance')
                plt.xlabel('Feature')
                plt.ylabel('Importance')
                plt.xticks(rotation=45)
                plt.tight_layout()
                save_plot(plt.gcf(), 'feature_importance.png', plot_type='explainability')
                plt.close()
            
            logging.info("Visualizations saved to results directory")
        except Exception as e:
            logging.error(f"Error generating visualizations: {str(e)}")
    
    # Return results
    results = {
        "detected_drifts": detected_drifts,
        "known_drift_points": drift_points,
        "feature_importance": feature_importance_dict,
        "best_config": best_config,
        "manual_drift_checks": manual_drift_checks
    }
    
    return results

def generate_synthetic_data_with_drift(n_samples=1000, n_features=10, drift_points=None, noise_level=0.1):
    """Generate synthetic data with concept drift at specified points."""
    # Generate random drift points if not specified
    if drift_points is None:
        drift_points = sorted(np.random.choice(
            range(100, n_samples - 100), 
            size=3, 
            replace=False
        ))
    
    logging.info(f"Generated synthetic data with drift points at: {drift_points}")
    
    # Generate features
    X = np.random.randn(n_samples, n_features)
    
    # Generate target with concept drift
    y = np.zeros(n_samples)
    
    # Initial coefficients
    coef = np.random.randn(n_features)
    
    # Generate data with different coefficients for each segment
    start_idx = 0
    for drift_point in drift_points:
        # Apply current coefficients to this segment
        y[start_idx:drift_point] = np.dot(X[start_idx:drift_point], coef) + noise_level * np.random.randn(drift_point - start_idx)
        
        # Create more pronounced drift by significantly changing coefficients
        # Increase the magnitude of change at drift points
        coef = coef + 1.5 * np.random.randn(n_features)  # Increased from 0.5 to 1.5
        
        # Add an abrupt shift at the drift point
        shift_magnitude = 0.5 + 0.5 * np.random.random()  # Random shift between 0.5 and 1.0
        if drift_point < n_samples:
            # Apply an abrupt shift to a small window after the drift point
            post_drift_window = min(20, n_samples - drift_point)
            y[drift_point:drift_point+post_drift_window] += shift_magnitude
        
        start_idx = drift_point
    
    # Last segment
    y[start_idx:] = np.dot(X[start_idx:], coef) + noise_level * np.random.randn(n_samples - start_idx)
    
    return X, y, drift_points

def check_drift_at_point(meta_learner, X_before, y_before, X_after, y_after, window_size=10, drift_threshold=0.01, significance_level=0.9):
    """Check for drift at a specific point using data before and after the point."""
    # Create a separate drift detector with sensitive parameters
    drift_detector = DriftDetector(
        window_size=window_size,
        drift_threshold=drift_threshold,
        significance_level=significance_level,
        min_drift_interval=1,  # Allow immediate detection
        ema_alpha=0.9  # High alpha for quick response
    )
    
    # Make predictions
    try:
        y_pred_before = meta_learner.predict(X_before)
        y_pred_after = meta_learner.predict(X_after)
    except Exception as e:
        logging.error(f"Prediction error in drift check: {str(e)}")
        # Skip the meta_learner fallback and go straight to a simple model
        # This avoids any potential issues with invalid parameters
        from sklearn.ensemble import RandomForestRegressor
        simple_model = RandomForestRegressor(
            n_estimators=10,
            max_depth=5,
            min_samples_split=2,
            min_samples_leaf=1,
            max_features='sqrt',
            random_state=42
        )
        try:
            simple_model.fit(X_before, y_before)
            y_pred_before = simple_model.predict(X_before)
            y_pred_after = simple_model.predict(X_after)
        except Exception as e:
            logging.error(f"Simple model error: {str(e)}")
            # If all else fails, use a very basic prediction
            y_pred_before = np.mean(y_before) * np.ones_like(y_before)
            y_pred_after = np.mean(y_before) * np.ones_like(y_after)
    
    # Calculate errors
    errors_before = y_before - y_pred_before
    errors_after = y_after - y_pred_after
    
    # Directly compare error distributions
    mean_shift = abs(np.mean(errors_after) - np.mean(errors_before))
    std_before = np.std(errors_before) if np.std(errors_before) > 0 else 1.0
    mean_shift_normalized = mean_shift / std_before
    
    # Perform KS test
    try:
        ks_statistic, p_value = stats.ks_2samp(errors_before, errors_after)
    except Exception as e:
        logging.error(f"KS test error: {str(e)}")
        ks_statistic, p_value = 0.0, 1.0
    
    # Enhanced drift detection logic
    # 1. Check with drift detector
    drift_detector.set_reference_window(errors_before)
    drift_detected = drift_detector.detect_drift(errors_after, errors_before)[0]
    
    # 2. Additional direct checks for more sensitivity
    if not drift_detected:
        # Check for significant mean shift
        if mean_shift_normalized > drift_threshold * 0.8:
            drift_detected = True
            logging.info(f"Drift detected by direct mean shift check: {mean_shift_normalized:.4f}")
        
        # Check for significant distribution change
        elif ks_statistic > 0.15 and p_value < significance_level:
            drift_detected = True
            logging.info(f"Drift detected by direct KS test: statistic={ks_statistic:.4f}, p-value={p_value:.6f}")
        
        # Check for variance change
        elif abs(np.std(errors_after) - np.std(errors_before)) / std_before > 0.3:
            drift_detected = True
            logging.info(f"Drift detected by variance change")
    
    return drift_detected, mean_shift_normalized, ks_statistic, p_value

def save_plot(fig, filename, plot_type='general'):
    """
    Save a plot to the appropriate directory based on its type
    
    Args:
        fig: Matplotlib figure to save
        filename: Name of the file
        plot_type: Type of plot (drift, performance, explainability, benchmarks)
    """
    # Create base results directory if it doesn't exist
    results_dir = Path('results')
    results_dir.mkdir(exist_ok=True)
    
    # Create subdirectory based on plot type
    if plot_type == 'drift':
        subdir = results_dir / 'drift'
    elif plot_type == 'performance':
        subdir = results_dir / 'performance'
    elif plot_type == 'explainability':
        subdir = results_dir / 'explainability'
    elif plot_type == 'benchmarks':
        subdir = results_dir / 'benchmarks'
    else:
        subdir = results_dir
    
    # Create subdirectory if it doesn't exist
    subdir.mkdir(exist_ok=True, parents=True)
    
    # Save the figure
    fig_path = subdir / filename
    fig.savefig(fig_path)
    plt.close(fig)
    logging.info(f"Saved plot to {fig_path}")
    return fig_path

def run_explainability_analysis(
    model, X, y, explainer_type='shap', plot_types=None, 
    generate_plots=True, n_samples=5, **kwargs
):
    """
    Run explainability analysis on a trained model
    
    Args:
        model: Trained model to explain
        X: Input features
        y: Target values
        explainer_type: Type of explainer to use ('shap', 'lime', 'feature_importance', 'optimizer')
        plot_types: List of plot types to generate
        generate_plots: Whether to generate plots
        n_samples: Number of samples to use for explanation
        **kwargs: Additional parameters for the explainer
    
    Returns:
        Dictionary containing explanation results
    """
    try:
        from explainability.explainer_factory import ExplainerFactory
        import datetime
        import os
        
        logging.info(f"Running explainability analysis with {explainer_type}")
        
        # Create explainer factory
        explainer_factory = ExplainerFactory()
        
        # Create parameters dictionary based on explainer type
        explainer_params = {}
        
        # Common parameters
        if hasattr(model, 'feature_names_in_'):
            explainer_params['feature_names'] = model.feature_names_in_
        
        # Explainer-specific parameters
        if explainer_type.lower() == 'shap':
            explainer_params['n_samples'] = n_samples
        elif explainer_type.lower() == 'lime':
            explainer_params['n_samples'] = n_samples
            explainer_params['mode'] = 'regression'  # Default to regression
        # No specific parameters for feature_importance or optimizer
        
        # Add any additional parameters
        explainer_params.update(kwargs)
        
        # Create explainer
        explainer = explainer_factory.create_explainer(
            explainer_type, model, **explainer_params
        )
        
        # Generate explanation
        explanation = explainer.explain(X, y, **kwargs)
        
        # Generate plots if requested
        if generate_plots:
            # Get timestamp for filenames
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Get available plot types
            available_plot_types = explainer.supported_plot_types
            logging.info(f"Available plot types for {explainer_type} explainer: {', '.join(available_plot_types)}")
            
            # Validate plot types
            if plot_types:
                valid_plot_types = [pt for pt in plot_types if pt in available_plot_types]
                if not valid_plot_types:
                    logging.warning(f"None of the specified plot types {plot_types} are valid for {explainer_type} explainer. "
                                   f"Valid types are: {available_plot_types}")
                    plot_types = [available_plot_types[0]]  # Default to first available
                else:
                    plot_types = valid_plot_types
            else:
                # Default to first available plot type
                plot_types = [available_plot_types[0]]
            
            # Generate plots
            generated_plots = []
            plot_paths = {}
            
            for plot_type in plot_types:
                try:
                    fig = explainer.plot(plot_type)
                    plot_filename = f"{explainer_type}_{plot_type}_{timestamp}.png"
                    plot_path = save_plot(fig, plot_filename, plot_type='explainability')
                    generated_plots.append(plot_type)
                    plot_paths[plot_type] = str(plot_path)
                except Exception as e:
                    logging.error(f"Error generating {plot_type} plot: {str(e)}")
            
            logging.info(f"Generated plots: {', '.join(generated_plots)}")
            logging.info(f"Plots saved in: results/explainability")
        else:
            plot_paths = {}
        
        # Get feature importance
        feature_importance = explainer.get_feature_importance()
        
        # Return results
        return {
            'explainer_type': explainer_type,
            'feature_importance': feature_importance,
            'explanation': explanation,
            'plot_paths': plot_paths if generate_plots else {}
        }
    
    except Exception as e:
        logging.error(f"Error in run_explainability_analysis: {str(e)}")
        traceback.print_exc()
        return None

def run_optimizer_explainability(optimizer, plot_types=None, generate_plots=True, **kwargs):
    """
    Run explainability analysis on an optimizer
    
    Args:
        optimizer: Optimizer instance to explain
        plot_types: List of plot types to generate
        generate_plots: Whether to generate plots
        **kwargs: Additional parameters for the explainer
        
    Returns:
        Dictionary containing explanation data and plot paths
    """
    try:
        logging.info("Running optimizer explainability analysis")
        
        # Import necessary modules
        from explainability.explainer_factory import ExplainerFactory
        import datetime
        
        # Create explainer
        factory = ExplainerFactory()
        explainer = factory.create_explainer('optimizer', optimizer)
        
        # Generate explanation
        explanation = explainer.explain()
        
        # Generate plots if requested
        if generate_plots and plot_types:
            # Get timestamp for filenames
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Get available plot types
            available_plot_types = explainer.supported_plot_types
            logging.info(f"Available plot types for optimizer explainer: {', '.join(available_plot_types)}")
            
            # Validate plot types
            if plot_types:
                valid_plot_types = [pt for pt in plot_types if pt in available_plot_types]
                if not valid_plot_types:
                    logging.warning(f"None of the specified plot types {plot_types} are valid for optimizer explainer. "
                                   f"Valid types are: {available_plot_types}")
                    plot_types = [available_plot_types[0]]  # Default to first available
                else:
                    plot_types = valid_plot_types
            else:
                # Default to first available plot type
                plot_types = [available_plot_types[0]]
            
            # Generate plots
            generated_plots = []
            plot_paths = {}
            
            for plot_type in plot_types:
                try:
                    logging.info(f"Generating {plot_type} plot")
                    fig = explainer.plot(plot_type)
                    
                    # Save plot
                    optimizer_name = optimizer.__class__.__name__
                    plot_filename = f"{optimizer_name}_{plot_type}_{timestamp}.png"
                    plot_path = save_plot(fig, plot_filename, plot_type='explainability')
                    generated_plots.append(plot_type)
                    plot_paths[plot_type] = str(plot_path)
                except Exception as e:
                    logging.error(f"Error generating {plot_type} plot: {str(e)}")
            
            logging.info(f"Generated plots: {', '.join(generated_plots)}")
            logging.info(f"Plots saved in: results/explainability")
        else:
            plot_paths = {}
        
        # Add plot paths to explanation
        explanation['plot_paths'] = plot_paths
        explanation['explainer_type'] = 'optimizer'  # Add explainer type
        
        return explanation
    
    except Exception as e:
        logging.error(f"Error in run_optimizer_explainability: {str(e)}")
        traceback.print_exc()
        return None

def run_meta_learning(method='bayesian', surrogate=None, selection=None, exploration=0.2, history_weight=0.7):
    """Run meta-learning process to find the best optimizer for a given problem."""
    logging.info(f"Running meta-learning with method={method}, surrogate={surrogate}, selection={selection}")
    
    # Create directories if they don't exist
    os.makedirs('results/meta_learning', exist_ok=True)
    
    # Define test functions
    test_functions = {
        'sphere': lambda x: np.sum(x**2),
        'rosenbrock': lambda x: np.sum(100.0 * (x[1:] - x[:-1]**2)**2 + (1 - x[:-1])**2),
        'rastrigin': lambda x: 10 * len(x) + np.sum(x**2 - 10 * np.cos(2 * np.pi * x)),
        'ackley': lambda x: -20 * np.exp(-0.2 * np.sqrt(np.sum(x**2) / len(x))) - np.exp(np.sum(np.cos(2 * np.pi * x)) / len(x)) + 20 + np.e
    }
    
    # Initialize optimizers
    dim = 30
    bounds = [(-5, 5)] * dim
    
    # Initialize optimizer wrappers
    aco_opt = AntColonyOptimizer(dim=dim, bounds=bounds)
    gwo_opt = GreyWolfOptimizer(dim=dim, bounds=bounds)
    de_opt = DifferentialEvolutionOptimizer(dim=dim, bounds=bounds)
    es_opt = EvolutionStrategyOptimizer(dim=dim, bounds=bounds)
    de_adaptive_opt = DifferentialEvolutionOptimizer(dim=dim, bounds=bounds, adaptive=True)
    es_adaptive_opt = EvolutionStrategyOptimizer(dim=dim, bounds=bounds, adaptive=True)
    
    # Create dictionary of optimizers
    optimizers = {
        'ACO': aco_opt,
        'GWO': gwo_opt,
        'DE': de_opt,
        'ES': es_opt,
        'DE (Adaptive)': de_adaptive_opt,
        'ES (Adaptive)': es_adaptive_opt
    }
    
    # Initialize meta-optimizer with the correct parameters
    meta_opt = MetaOptimizer(
        dim=dim,
        bounds=bounds,
        optimizers=optimizers,
        n_parallel=2,
        history_file='results/meta_learning/history.json',
        selection_file='results/meta_learning/selection.json'
    )
    
    # Run meta-optimizer on each benchmark function
    results = {}
    best_algorithms = {}
    performance_metrics = {}
    
    for func_name, func in test_functions.items():
        logging.info(f"Running meta-learning on {func_name} function")
        
        # Run meta-optimizer
        meta_opt.reset()  # Reset optimizer state
        
        # Use the correct parameters for optimize
        result = meta_opt.optimize(
            func, 
            max_evals=1000,  # Reduced for quicker results
            context={"function_name": func_name}
        )
        
        # Extract results
        best_score = meta_opt.best_score
        best_solution = meta_opt.best_solution
        history = meta_opt.optimization_history
        
        # Store results
        results[func_name] = {
            'best_score': best_score,
            'best_solution': best_solution,
            'history': history
        }
        
        # Track which optimizer was selected most often
        optimizer_counts = {}
        for entry in history:
            optimizer = entry.get('selected_optimizer', 'unknown')
            optimizer_counts[optimizer] = optimizer_counts.get(optimizer, 0) + 1
        
        # Determine best algorithm
        if optimizer_counts:
            best_algorithm = max(optimizer_counts.items(), key=lambda x: x[1])[0]
        else:
            logging.warning(f"No optimizer selections recorded for {func_name}. Using default.")
            best_algorithm = "default"
        best_algorithms[func_name] = best_algorithm
        
        # Calculate performance metrics
        performance_metrics[func_name] = {
            'best_score': best_score,
            'optimizer_selections': optimizer_counts,
            'convergence_rate': len(history) / 1000  # Simple metric
        }
    
    # Determine overall best algorithm
    all_selections = {}
    for func_name, algorithm in best_algorithms.items():
        all_selections[algorithm] = all_selections.get(algorithm, 0) + 1
    
    if all_selections:
        overall_best_algorithm = max(all_selections.items(), key=lambda x: x[1])[0]
    else:
        logging.warning("No algorithm selections recorded. Using default as best algorithm.")
        overall_best_algorithm = "default"
    
    # Create summary plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot optimizer selection frequency
    algorithms = list(all_selections.keys())
    frequencies = [all_selections[alg] for alg in algorithms]
    
    ax.bar(algorithms, frequencies)
    ax.set_xlabel('Optimizer')
    ax.set_ylabel('Selection Frequency')
    ax.set_title('Meta-Learner Optimizer Selection Frequency')
    
    # Save plot
    save_plot(fig, 'meta_learner_selection_frequency', plot_type='meta')
    
    return {
        'best_algorithm': overall_best_algorithm,
        'algorithm_selections': best_algorithms,
        'performance': performance_metrics,
        'results': results
    }

def run_evaluation(model=None, X_test=None, y_test=None):
    """
    Evaluate a trained model on test data.
    
    Args:
        model: Trained model to evaluate (if None, creates a default model)
        X_test: Test features (if None, creates synthetic data)
        y_test: Test targets (if None, creates synthetic data)
        
    Returns:
        Dictionary with evaluation results
    """
    logging.info("Running model evaluation")
    
    # Create results directory
    results_dir = Path('results')
    results_dir.mkdir(exist_ok=True, parents=True)
    
    # Create model and data if not provided
    if model is None or X_test is None or y_test is None:
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.datasets import make_regression
        from sklearn.model_selection import train_test_split
        
        # Create synthetic data
        X, y = make_regression(n_samples=1000, n_features=10, noise=0.1, random_state=42)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Create and train model
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
    
    # Evaluate model
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    
    # Create evaluation plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot actual vs predicted
    ax.scatter(y_test, y_pred, alpha=0.5)
    
    # Add perfect prediction line
    min_val = min(np.min(y_test), np.min(y_pred))
    max_val = max(np.max(y_test), np.max(y_pred))
    ax.plot([min_val, max_val], [min_val, max_val], 'r--')
    
    ax.set_xlabel('Actual')
    ax.set_ylabel('Predicted')
    ax.set_title('Model Evaluation: Actual vs Predicted')
    
    # Add metrics to plot
    ax.text(0.05, 0.95, f'MSE: {mse:.4f}\nRMSE: {rmse:.4f}\nR²: {r2:.4f}',
            transform=ax.transAxes, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Save plot
    save_plot(fig, 'model_evaluation', plot_type='evaluation')
    
    return {
        'score': r2,
        'metrics': {
            'mse': mse,
            'rmse': rmse,
            'r2': r2
        },
        'predictions': y_pred.tolist()
    }

def run_optimization_and_evaluation(data_path: str, 
                                  save_dir: str = 'results',
                                  n_runs: int = 30,
                                  max_evals: int = 1000):
    """Run complete optimization and evaluation pipeline with visualizations."""
    
    # Create directories
    os.makedirs(save_dir, exist_ok=True)
    plots_dir = os.path.join(save_dir, 'plots')
    os.makedirs(plots_dir, exist_ok=True)
    
    # Initialize components
    evaluator = FrameworkEvaluator()
    drift_detector = DriftDetector(
        window_size=50,
        drift_threshold=1.8,
        significance_level=0.01
    )
    optimizer_analyzer = OptimizerAnalyzer(optimizers={
        'differential_evolution': DifferentialEvolutionOptimizer,
        'evolution_strategy': EvolutionStrategyOptimizer,
        'ant_colony': AntColonyOptimizer,
        'grey_wolf': GreyWolfOptimizer
    })
    
    # Load and preprocess data
    data = pd.read_csv(data_path)
    X = data.drop('target', axis=1)
    y = data['target']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    
    # Run optimization and get best model
    best_model = run_optimization(X_train, y_train, n_runs=n_runs, max_evals=max_evals)
    
    # Generate predictions
    y_pred = best_model.predict(X_test)
    y_prob = best_model.predict_proba(X_test)[:, 1]
    
    # Track performance and drift
    for i in range(len(X_test)):
        # Evaluate prediction performance
        evaluator.evaluate_prediction_performance(
            y_test[i:i+1], 
            y_pred[i:i+1], 
            y_prob[i:i+1]
        )
        
        # Track feature importance
        evaluator.track_feature_importance(
            X.columns, 
            best_model.feature_importances_
        )
        
        # Detect drift
        drift_detected = drift_detector.detect_drift(
            y_test[i:i+1],
            y_pred[i:i+1]
        )
        if drift_detected:
            evaluator.track_drift_event(drift_detector.get_statistics())
    
    # Generate visualizations
    
    # 1. Drift Detection Results
    drift_detector.plot_detection_results(
        save_path=os.path.join(plots_dir, 'drift_detection_results.png')
    )
    
    # 2. Drift Analysis
    drift_detector.plot_drift_analysis(
        save_path=os.path.join(plots_dir, 'drift_analysis.png')
    )
    drift_detector.save_analysis(plots_dir)  # Saves additional drift plots
    
    # 3. Framework Performance
    evaluator.plot_framework_performance(
        save_path=os.path.join(plots_dir, 'framework_performance.png')
    )
    
    # 4. Pipeline Performance
    evaluator.plot_pipeline_performance(
        save_path=os.path.join(plots_dir, 'pipeline_performance.png')
    )
    
    # 5. Performance Boxplot
    evaluator.plot_performance_boxplot(
        save_path=os.path.join(plots_dir, 'performance_boxplot.png')
    )
    
    # 6. Model Evaluation
    plot_model_evaluation(
        y_test, y_pred, y_prob,
        save_path=os.path.join(plots_dir, 'model_evaluation.png')
    )
    
    # 7. Optimizer Analysis
    optimizer_analyzer.plot_landscape_analysis(
        save_path=os.path.join(plots_dir, 'optimizer_landscape.png')
    )
    optimizer_analyzer.plot_gradient_analysis(
        save_path=os.path.join(plots_dir, 'optimizer_gradient.png')
    )
    optimizer_analyzer.plot_parameter_adaptation(
        save_path=os.path.join(plots_dir, 'optimizer_parameters.png')
    )
    
    # Save model
    model_path = os.path.join(save_dir, 'best_model.pkl')
    with open(model_path, 'wb') as f:
        pickle.dump(best_model, f)
        
    # Generate summary report
    with open(os.path.join(save_dir, 'optimization_report.txt'), 'w') as f:
        f.write("Optimization and Evaluation Report\n")
        f.write("================================\n\n")
        
        # Model Performance
        f.write("Model Performance\n")
        f.write("-----------------\n")
        metrics = evaluator.generate_performance_report()
        for metric, value in metrics.items():
            f.write(f"{metric}: {value:.4f}\n")
        f.write("\n")
        
        # Drift Analysis
        f.write("Drift Analysis\n")
        f.write("--------------\n")
        drift_stats = drift_detector.get_statistics()
        for stat, value in drift_stats.items():
            f.write(f"{stat}: {value:.4f}\n")
        f.write("\n")
        
        # Optimization Summary
        f.write("Optimization Summary\n")
        f.write("-------------------\n")
        opt_stats = optimizer_analyzer.get_statistics() if hasattr(optimizer_analyzer, 'get_statistics') else {}
        for stat, value in opt_stats.items():
            f.write(f"{stat}: {value}\n")
            
    return {
        'model': best_model,
        'evaluator': evaluator,
        'drift_detector': drift_detector,
        'optimizer_analyzer': optimizer_analyzer,
        'performance_metrics': evaluator.generate_performance_report()
    }

def main():
    """Main function."""
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Run optimization framework')
    parser.add_argument('--config', type=str, help='Path to configuration file')
    parser.add_argument('--optimize', action='store_true', help='Run optimization')
    parser.add_argument('--evaluate', action='store_true', help='Evaluate model')
    parser.add_argument('--meta', action='store_true', help='Run meta-learning')
    parser.add_argument('--drift', action='store_true', help='Run drift detection')
    parser.add_argument('--run-meta-learner-with-drift', action='store_true', help='Run meta-learner with drift detection')
    parser.add_argument('--explain-drift', action='store_true', help='Explain drift when detected')
    
    # Explainability arguments
    parser.add_argument('--explain', action='store_true', help='Run explainability analysis')
    parser.add_argument('--explainer', type=str, default='shap', choices=['shap', 'lime', 'feature_importance', 'optimizer'], 
                        help='Explainer type to use')
    parser.add_argument('--explain-plots', action='store_true', help='Generate and save explainability plots')
    parser.add_argument('--explain-plot-types', type=str, nargs='+', 
                        help='Specific plot types to generate (e.g., summary waterfall force dependence)')
    parser.add_argument('--explain-samples', type=int, default=50, help='Number of samples to use for explainability')
    parser.add_argument('--explain-optimizer', action='store_true', help='Run explainability analysis on optimizer')
    
    # Optimizer explainability arguments
    parser.add_argument('--optimizer-type', type=str, default='differential_evolution', 
                        choices=['differential_evolution', 'evolution_strategy', 'ant_colony', 'grey_wolf'],
                        help='Type of optimizer to explain')
    parser.add_argument('--optimizer-dim', type=int, default=10, help='Dimension for optimizer')
    parser.add_argument('--optimizer-bounds', type=float, nargs=2, default=[-5, 5], 
                        help='Bounds for optimizer (min max)')
    parser.add_argument('--optimizer-plot-types', type=str, nargs='+', 
                        default=['convergence', 'parameter_adaptation', 'diversity', 'landscape_analysis',
                                'decision_process', 'exploration_exploitation', 'gradient_estimation', 'performance_comparison'],
                        help='Plot types to generate for optimizer explainability')
    parser.add_argument('--test-functions', type=str, nargs='+', default=['sphere', 'rosenbrock'],
                        help='Test functions to run optimizer on')
    parser.add_argument('--max-evals', type=int, default=500, 
                        help='Maximum number of function evaluations')
    
    # Meta-learner and optimization parameters
    parser.add_argument('--method', type=str, default='bayesian', help='Method for meta-learner')
    parser.add_argument('--surrogate', type=str, default=None, help='Surrogate model for meta-learner')
    parser.add_argument('--selection', type=str, default=None, help='Selection strategy for meta-learner')
    parser.add_argument('--exploration', type=float, default=0.2, help='Exploration factor for meta-learner')
    parser.add_argument('--history', type=float, default=0.7, help='History weight for meta-learner')
    
    # Drift detection parameters
    parser.add_argument('--drift-window', type=int, default=50, help='Window size for drift detection')
    parser.add_argument('--drift-threshold', type=float, default=0.5, help='Threshold for drift detection')
    parser.add_argument('--drift-significance', type=float, default=0.05, help='Significance level for drift detection')
    parser.add_argument('--drift-min-interval', type=int, default=30, help='Minimum interval between drift detections')
    parser.add_argument('--drift-history-size', type=int, default=100, help='Maximum history size for drift detection')
    
    # Visualization and summary options
    parser.add_argument('--visualize', action='store_true', help='Visualize results')
    parser.add_argument('--summary', action='store_true', help='Print summary of results')
    
    # Parse arguments, handling unknown arguments gracefully
    args, unknown = parser.parse_known_args()
    
    # Warn about unknown arguments
    if unknown:
        logging.warning(f"Unknown arguments: {unknown}")
    
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    try:
        # Run drift detection if requested
        if args.drift or args.run_meta_learner_with_drift:
            logging.info("Running meta-learner with drift detection")
            results = run_meta_learner_with_drift(
                n_samples=1000,
                n_features=10,
                drift_points=[250, 500, 750],
                window_size=args.drift_window,
                drift_threshold=args.drift_threshold,
                significance_level=args.drift_significance,
                min_drift_interval=args.drift_min_interval,
                ema_alpha=0.3,
                visualize=args.visualize
            )
            
            # Log results
            logging.info(f"Drift detection complete.")
            logging.info(f"Detected drifts: {results['detected_drifts']}")
            logging.info(f"Known drift points: {results['known_drift_points']}")
            
            # Print summary if requested
            if args.summary:
                print("\nDrift Detection Summary:")
                print(f"  Window Size: {args.drift_window}")
                print(f"  Threshold: {args.drift_threshold}")
                print(f"  Significance Level: {args.drift_significance}")
                print(f"  Detected Drifts: {results['detected_drifts']}")
                print(f"  Known Drift Points: {results['known_drift_points']}")
                print(f"  Detection Accuracy: {results.get('detection_accuracy', 'N/A')}")
            return
        
        # Run meta-learner if requested
        if args.meta:
            logging.info("Running meta-learning")
            meta_params = {
                'method': args.method,
                'surrogate': args.surrogate,
                'selection': args.selection,
                'exploration': args.exploration,
                'history_weight': args.history
            }
            results = run_meta_learning(**meta_params)
            logging.info(f"Meta-learning complete. Best algorithm: {results['best_algorithm']}")
            
            # Print summary if requested
            if args.summary:
                print("\nMeta-Learning Summary:")
                print(f"  Method: {args.method}")
                print(f"  Surrogate Model: {args.surrogate or 'Default'}")
                print(f"  Selection Strategy: {args.selection or 'Default'}")
                print(f"  Exploration Factor: {args.exploration}")
                print(f"  History Weight: {args.history}")
                print(f"  Best Algorithm: {results['best_algorithm']}")
                print(f"  Performance: {results.get('performance', 'N/A')}")
            return
            
        # Run optimization if requested
        if args.optimize:
            logging.info("Running optimization")
            results, results_df = run_optimization()
            logging.info(f"Optimization complete. Best score: {results.get('best_score', 'N/A')}")
            
            # Save results
            results_df.to_csv('results/optimization_results.csv', index=False)
            
            # Print summary if requested
            if args.summary:
                print("\nOptimization Summary:")
                print(f"  Best Score: {results.get('best_score', 'N/A')}")
                print("\nTop Performing Optimizers:")
                for i, (optimizer, score) in enumerate(results.get('top_optimizers', {}).items()):
                    print(f"  {i+1}. {optimizer}: {score:.6f}")
            return
            
        # Run evaluation if requested
        if args.evaluate:
            logging.info("Evaluating model")
            results = run_evaluation()
            logging.info(f"Evaluation complete. Score: {results.get('score', 'N/A')}")
            
            # Print summary if requested
            if args.summary:
                print("\nEvaluation Summary:")
                print(f"  Score: {results['score']:.6f}")
                print(f"  Metrics: {results['metrics']}")
                print(f"  Predictions: {results['predictions']}")
            return
        
        # Run explainability analysis if requested
        if args.explain:
            logging.info("Running explainability analysis")
            
            # Create a model and data for explanation
            from sklearn.ensemble import RandomForestRegressor
            from sklearn.datasets import make_regression
            from sklearn.model_selection import train_test_split
            
            # Create synthetic data
            X, y = make_regression(n_samples=500, n_features=10, noise=0.1, random_state=42)
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            # Create and train model
            model = RandomForestRegressor(n_estimators=100, random_state=42)
            model.fit(X_train, y_train)
            
            # Run explainability analysis
            explainer_params = {
                'model': model,
                'X': X_test,
                'y': y_test,
                'explainer_type': args.explainer,
                'plot_types': args.explain_plot_types,
                'generate_plots': args.explain_plots,
                'n_samples': args.explain_samples
            }
            
            if args.explain_optimizer:
                # Run optimizer explainability
                logging.info("Running optimizer explainability")
                
                # Import necessary modules
                from optimizers.optimizer_factory import OptimizerFactory
                from benchmarking.test_functions import ClassicalTestFunctions
                
                # Create optimizer factory
                factory = OptimizerFactory()
                
                # Set bounds for optimizer
                bounds = [(args.optimizer_bounds[0], args.optimizer_bounds[1])] * args.optimizer_dim
                
                # Create optimizer
                optimizer = factory.create_optimizer(
                    args.optimizer_type, 
                    dim=args.optimizer_dim, 
                    bounds=bounds
                )
                
                # Get test functions
                test_functions = {}
                for func_name in args.test_functions:
                    if hasattr(ClassicalTestFunctions, func_name):
                        test_functions[func_name] = getattr(ClassicalTestFunctions, func_name)
                
                # Run optimizations on test functions
                results = {}
                for func_name, func in test_functions.items():
                    logging.info(f"Running {args.optimizer_type} on {func_name}...")
                    optimizer.reset()  # Reset optimizer state
                    result = optimizer.run(func, max_evals=args.max_evals)
                    results[func_name] = result
                
                # Run optimizer explainability
                explainer_params = {
                    'optimizer': optimizer,
                    'plot_types': args.optimizer_plot_types,
                    'generate_plots': args.explain_plots
                }
                
                explanation = run_optimizer_explainability(**explainer_params)
                
                # Print summary if requested
                if args.summary:
                    print("\nOptimizer Explainability Summary:")
                    print(f"  Optimizer Type: {args.optimizer_type}")
                    print(f"  Dimension: {args.optimizer_dim}")
                    print(f"  Bounds: {args.optimizer_bounds}")
                    print(f"  Test Functions: {args.test_functions}")
                    print(f"  Max Evaluations: {args.max_evals}")
                    print(f"  Plot Types: {args.optimizer_plot_types}")
                    
                    # Print performance on each test function
                    print("\n  Performance on Test Functions:")
                    for func_name, result in results.items():
                        print(f"    {func_name}: {result['best_score']:.6f} (evaluations: {result['evaluations']})")
                    
                    # Print plot paths
                    if 'plot_paths' in explanation:
                        print("\n  Generated Plots:")
                        for plot_type, plot_path in explanation['plot_paths'].items():
                            print(f"    {plot_type}: {plot_path}")
            else:
                # Run model explainability
                results = run_explainability_analysis(**explainer_params)
            
            logging.info(f"Explainability analysis complete.")
            
            # Print summary if requested
            if args.summary and results and not args.explain_optimizer:
                print("\nExplainability Summary:")
                print(f"  Explainer Type: {results['explainer_type']}")
                
                if 'feature_importance' in results and results['feature_importance']:
                    print("  Feature Importance:")
                    for feature, importance in sorted(results['feature_importance'].items(), key=lambda x: x[1], reverse=True)[:10]:
                        print(f"    {feature}: {importance:.6f}")
                
                if 'explanation' in results:
                    print(f"  Explanation: {results['explanation']}")
                
                if 'plot_paths' in results:
                    print("  Plot Paths:")
                    for plot_type, path in results['plot_paths'].items():
                        print(f"    {plot_type}: {path}")
            return
        
        # Default to running optimization
        logging.info("Running optimization by default")
        results, results_df = run_optimization()
        logging.info(f"Optimization complete. Best score: {results.get('best_score', 'N/A')}")
        
        # Save results
        results_df.to_csv('results/optimization_results.csv', index=False)
        
        # Print summary if requested
        if args.summary:
            print("\nOptimization Summary:")
            print(f"  Best Score: {results.get('best_score', 'N/A')}")
            print("\nTop Performing Optimizers:")
            for i, (optimizer, score) in enumerate(results.get('top_optimizers', {}).items()):
                print(f"  {i+1}. {optimizer}: {score:.6f}")
        return
        
    except Exception as e:
        logging.error(f"Error: {str(e)}")
        traceback.print_exc()

if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        logging.error(f"Error: {str(e)}")
        traceback.print_exc()
