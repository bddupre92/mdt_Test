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
import traceback
from pathlib import Path

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import local modules
from meta.meta_learner import MetaLearner
from meta.drift_detector import DriftDetector
from models.model_factory import ModelFactory
from explainability.explainer_factory import ExplainerFactory
from optimizers.optimizer_factory import create_optimizers

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
    DifferentialEvolutionWrapper, 
    EvolutionStrategyWrapper,
    AntColonyWrapper,
    GreyWolfWrapper,
    create_optimizers
)

# Import meta-optimizer
from meta.meta_optimizer import MetaOptimizer

# Import drift detection
from drift_detection.detector import DriftDetector as FullDriftDetector

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
def create_optimizers(dim: int, bounds: List[Tuple[float, float]]) -> Dict[str, Any]:
    """Create all optimizer instances for benchmarking"""
    return {
        'ACO': AntColonyWrapper(dim=dim, bounds=bounds, name="ACO"),
        'GWO': GreyWolfWrapper(dim=dim, bounds=bounds, name="GWO"),
        'DE': DifferentialEvolutionWrapper(dim=dim, bounds=bounds, name="DE"),
        'ES': EvolutionStrategyWrapper(dim=dim, bounds=bounds, name="ES"),
        'DE (Adaptive)': DifferentialEvolutionWrapper(dim=dim, bounds=bounds, adaptive=True, name="DE (Adaptive)"),
        'ES (Adaptive)': EvolutionStrategyWrapper(dim=dim, bounds=bounds, adaptive=True, name="ES (Adaptive)")
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
def run_meta_learner_with_drift(n_samples=1000, n_features=10, drift_points=None, window_size=10, drift_threshold=0.01, significance_level=0.9, visualize=False):
    """Run meta-learner with drift detection.
    
    Args:
        n_samples: Number of samples
        n_features: Number of features
        drift_points: List of drift points
        window_size: Window size for drift detection
        drift_threshold: Threshold for drift detection
        significance_level: Significance level for drift detection
        visualize: Whether to visualize the results
        
    Returns:
        Dictionary with results
    """
    # Generate synthetic data with drift
    X, y, drift_points = generate_synthetic_data_with_drift(
        n_samples=n_samples, n_features=n_features, drift_points=drift_points
    )
    
    # Initialize meta-learner
    meta_learner = MetaLearner(
        method='bayesian',
        selection_strategy='bayesian',
        exploration_factor=0.3
    )
    
    # Set algorithms
    from optimizers.optimizer_factory import create_optimizers
    optimizers = create_optimizers(dim=n_features, bounds=[(-5, 5)] * n_features)
    meta_learner.set_algorithms(list(optimizers.values()))
    
    # Set drift detector with sensitive parameters
    drift_detector = DriftDetector(
        window_size=window_size, 
        drift_threshold=drift_threshold, 
        significance_level=significance_level
    )
    meta_learner.drift_detector = drift_detector
    
    # Initialize with first 200 samples
    X_init, y_init = X[:200], y[:200]
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
        X_chunk, y_chunk = X[start_idx:end_idx], y[start_idx:end_idx]
        
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
        
        X_before = X[before_start:drift_point]
        y_before = y[before_start:drift_point]
        X_after = X[drift_point:after_end]
        y_after = y[drift_point:after_end]
        
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

def generate_synthetic_data_with_drift(n_samples=1000, n_features=10, n_drift_points=3, drift_points=None, noise_level=0.1):
    """Generate synthetic data with concept drift at specified points.
    
    Args:
        n_samples: Number of samples
        n_features: Number of features
        n_drift_points: Number of drift points to generate if drift_points is None
        drift_points: List of specific drift points to use (overrides n_drift_points)
        noise_level: Noise level for data generation
        
    Returns:
        Tuple of (X, y, drift_points)
    """
    # Generate random drift points if not specified
    if drift_points is None:
        drift_points = sorted(np.random.choice(
            range(100, n_samples - 100), 
            size=n_drift_points, 
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
        
        # Change coefficients for next segment (concept drift)
        coef = coef + 0.5 * np.random.randn(n_features)
        
        start_idx = drift_point
    
    # Last segment
    y[start_idx:] = np.dot(X[start_idx:], coef) + noise_level * np.random.randn(n_samples - start_idx)
    
    return X, y, drift_points

def check_drift_at_point(meta_learner, X_before, y_before, X_after, y_after, window_size=10, drift_threshold=0.01, significance_level=0.9):
    """Check for drift at a specific point using data before and after the point.
    
    Args:
        meta_learner: MetaLearner instance
        X_before: Features before drift point
        y_before: Target before drift point
        X_after: Features after drift point
        y_after: Target after drift point
        window_size: Window size for drift detection
        drift_threshold: Threshold for drift detection
        significance_level: Significance level for drift detection
        
    Returns:
        Tuple of (drift_detected, mean_shift, ks_statistic, p_value)
    """
    # Create a separate drift detector with sensitive parameters
    drift_detector = DriftDetector(
        window_size=window_size,
        drift_threshold=drift_threshold,
        significance_level=significance_level,
        min_drift_interval=1,
        ema_alpha=0.9
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
    
    # First establish baseline with pre-drift data
    drift_detector.detect_drift(y_before, y_pred_before)
    
    # Then check for drift with post-drift data
    drift_detected = drift_detector.detect_drift(y_after, y_pred_after)
    
    # Get statistics
    stats = drift_detector.get_statistics()
    mean_shift = stats.get('mean_shift', 0.0)
    ks_statistic = stats.get('ks_statistic', 0.0)
    p_value = stats.get('p_value', 1.0)
    
    return drift_detected, mean_shift, ks_statistic, p_value

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
        explainer_type: Type of explainer to use ('shap', 'lime', 'feature_importance')
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
        # No specific parameters for feature_importance
        
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
    parser.add_argument('--explainer', type=str, default='shap', choices=['shap', 'lime', 'feature_importance'], 
                        help='Explainer type to use')
    parser.add_argument('--explain-plots', action='store_true', help='Generate and save explainability plots')
    parser.add_argument('--explain-plot-types', type=str, nargs='+', 
                        help='Specific plot types to generate (e.g., summary waterfall force dependence)')
    parser.add_argument('--explain-samples', type=int, default=50, help='Number of samples to use for explainability')
    
    parser.add_argument('--method', type=str, default='bayesian', help='Method for meta-learner')
    parser.add_argument('--surrogate', type=str, default=None, help='Surrogate model for meta-learner')
    parser.add_argument('--selection', type=str, default=None, help='Selection strategy for meta-learner')
    parser.add_argument('--exploration', type=float, default=0.2, help='Exploration factor for meta-learner')
    parser.add_argument('--history', type=float, default=0.7, help='History weight for meta-learner')
    parser.add_argument('--drift-window', type=int, default=10, help='Window size for drift detection')
    parser.add_argument('--drift-threshold', type=float, default=0.01, help='Threshold for drift detection')
    parser.add_argument('--drift-significance', type=float, default=0.9, help='Significance level for drift detection')
    parser.add_argument('--visualize', action='store_true', help='Visualize results')
    args = parser.parse_args()
    
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
                visualize=args.visualize
            )
            
            # Log results
            logging.info(f"Drift detection complete.")
            logging.info(f"Detected drifts: {results['detected_drifts']}")
            logging.info(f"Known drift points: {results['known_drift_points']}")
            return
        
        # Run meta-learner if requested
        if args.meta:
            logging.info("Running meta-learning")
            results = run_meta_learning()
            logging.info(f"Meta-learning complete. Best algorithm: {results['best_algorithm']}")
            return
            
        # Run optimization if requested
        if args.optimize:
            logging.info("Running optimization")
            results, results_df = run_optimization()
            logging.info(f"Optimization complete. Best score: {results['best_score']:.4f}")
            
            # Save results
            results_df.to_csv('results/optimization_results.csv', index=False)
            return
            
        # Run evaluation if requested
        if args.evaluate:
            logging.info("Evaluating model")
            results = run_evaluation()
            logging.info(f"Evaluation complete. Score: {results['score']:.4f}")
            return
            
        # Run explainability analysis if requested
        if args.explain:
            logging.info("Running explainability analysis")
            
            # Create a model and data for explanation
            from sklearn.ensemble import RandomForestRegressor
            from sklearn.datasets import make_regression
            
            X, y = make_regression(n_samples=500, n_features=10, noise=0.1, random_state=42)
            model = RandomForestRegressor(n_estimators=100, random_state=42)
            model.fit(X, y)
            
            # Set up explainability parameters
            explainability_params = {}
            
            # Run explainability analysis
            explanation_results = run_explainability_analysis(
                model=model,
                X=X,
                y=y,
                explainer_type=args.explainer,
                plot_types=args.explain_plot_types,
                generate_plots=args.explain_plots,
                n_samples=args.explain_samples
            )
            
            if explanation_results:
                # Print feature importance
                print("\nFeature Importance:")
                for feature, importance in sorted(
                    explanation_results['feature_importance'].items(),
                    key=lambda x: abs(x[1]),
                    reverse=True
                ):
                    print(f"{feature}: {importance:.4f}")
        
            logging.info("Explainability analysis complete")
            return
        
        # Run default optimization if no specific action is requested
        logging.info("Running default optimization")
        results, results_df = run_optimization()
        logging.info(f"Optimization complete. Best score: {results['best_score']:.4f}")
        
        # Save results
        results_df.to_csv('results/optimization_results.csv', index=False)
            
    except Exception as e:
        logging.error(f"Error in main: {str(e)}", exc_info=True)
        traceback.print_exc()

if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        logging.error(f"Error in main: {str(e)}", exc_info=True)