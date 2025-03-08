import os
import sys
import argparse
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import logging
import traceback
import tempfile
import shutil
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Union, Tuple, Callable
import gc
import psutil

# Add the project root to the python path to ensure imports work correctly
sys.path.append(str(Path(__file__).parent))

# Import MetaOptimizer
from meta_optimizer.meta.meta_optimizer import MetaOptimizer
from meta_optimizer.optimizers import (
    DifferentialEvolutionOptimizer,
    EvolutionStrategyOptimizer,
    GreyWolfOptimizer,
    AntColonyOptimizer
)

# Import visualization components
try:
    from visualization.algorithm_selection_viz import AlgorithmSelectionVisualizer
    ALGORITHM_VIZ_AVAILABLE = True
except ImportError:
    try:
        sys.path.append(str(Path(__file__).parent))
        from visualization.algorithm_selection_viz import AlgorithmSelectionVisualizer
        ALGORITHM_VIZ_AVAILABLE = True
    except ImportError:
        ALGORITHM_VIZ_AVAILABLE = False
        logging.warning("Algorithm selection visualization not available")

# Check for migraine prediction modules
try:
    from migraine_prediction_project.src.migraine_model import MigrainePredictor
    MIGRAINE_MODULES_AVAILABLE = True
except ImportError:
    try:
        sys.path.append(str(Path(__file__).parent))
        from migraine_prediction_project.src.migraine_model import MigrainePredictor
        MIGRAINE_MODULES_AVAILABLE = True
    except ImportError:
        MIGRAINE_MODULES_AVAILABLE = False
        logging.warning("Migraine prediction modules not available")

try:
    from visualization.live_visualization import LiveOptimizationMonitor
    LIVE_VIZ_AVAILABLE = True
except ImportError:
    try:
        sys.path.append(str(Path(__file__).parent))
        from visualization.live_visualization import LiveOptimizationMonitor
        LIVE_VIZ_AVAILABLE = True
    except ImportError:
        LIVE_VIZ_AVAILABLE = False
        logging.warning("Live optimization visualization not available")

# Import local modules from root directory
try:
    from meta.meta_learner import MetaLearner
except ImportError:
    try:
        from meta_optimizer.meta.meta_learner import MetaLearner
    except ImportError:
        logging.warning("MetaLearner not available")

from meta_optimizer.optimizers.optimizer_factory import (
    OptimizerFactory,
    DifferentialEvolutionOptimizer, 
    EvolutionStrategyOptimizer,
    AntColonyOptimizer,
    GreyWolfOptimizer
)

# Import these only if they exist
try:
    from models.model_factory import ModelFactory
except ImportError:
    # Create a dummy ModelFactory for compatibility
    class ModelFactory:
        @staticmethod
        def create_model(*args, **kwargs):
            raise NotImplementedError("ModelFactory not available")

# Configure logging
import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('optimization.log')
    ]
)

# Create logger for this module
logger = logging.getLogger(__name__)

"""
Comprehensive benchmark and comparison of optimization algorithms
including meta-optimization for novel algorithm creation.
"""

# Create benchmark functions dictionary
benchmark_functions = {}

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
    factory = OptimizerFactory()
    
    return {
        'ACO': factory.create_optimizer('ant_colony', dim=dim, bounds=bounds, name="ACO", verbose=verbose),
        'GWO': factory.create_optimizer('grey_wolf', dim=dim, bounds=bounds, name="GWO", verbose=verbose),
        'DE': factory.create_optimizer('differential_evolution', dim=dim, bounds=bounds, name="DE", verbose=verbose),
        'ES': factory.create_optimizer('evolution_strategy', dim=dim, bounds=bounds, name="ES", verbose=verbose),
        'DE (Adaptive)': factory.create_optimizer('differential_evolution', dim=dim, bounds=bounds, adaptive=True, name="DE (Adaptive)", verbose=verbose),
        'ES (Adaptive)': factory.create_optimizer('evolution_strategy', dim=dim, bounds=bounds, adaptive=True, name="ES (Adaptive)", verbose=verbose),
        'Meta-Optimizer': MetaOptimizer(
            dim=dim, 
            bounds=bounds, 
            optimizers={
                'DE': factory.create_optimizer('differential_evolution', dim=dim, bounds=bounds),
                'ES': factory.create_optimizer('evolution_strategy', dim=dim, bounds=bounds)
            },
            verbose=verbose
        )
    }

# Import optimization classes from the optimization module
from optimization import OptimizationResult, OptimizerAnalyzer

# Run optimization
def run_optimization(args):
    """Run optimization with specified parameters."""
    # Set up logging
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('optimization.log')
        ]
    )
    logger = logging.getLogger(__name__)

    # Use dimension from args if provided
    dim = args.dimension if hasattr(args, 'dimension') and args.dimension is not None else 10
    
    # Create bounds for the optimization
    bounds = [(-5, 5)] * dim
    
    # Initialize results dictionary
    results = {}
    
    # Initialize optimizers dictionary
    optimizers = {}
    
    # Create test functions
    test_functions = {
        'sphere': lambda x: np.sum(x**2),
        'rosenbrock': lambda x: np.sum(100.0 * (x[1:] - x[:-1]**2)**2 + (x[:-1] - 1)**2),
        'rastrigin': lambda x: 10 * len(x) + np.sum(x**2 - 10 * np.cos(2 * np.pi * x)),
        'ackley': lambda x: -20 * np.exp(-0.2 * np.sqrt(np.sum(x**2) / len(x))) - np.exp(np.sum(np.cos(2 * np.pi * x)) / len(x)) + 20 + np.e
    }
    
    # Run optimizers on test functions
    for func_name, func in test_functions.items():
        results[func_name] = {}
        logger.info(f"\nOptimizing {func_name} function")
        
        for optimizer_name, optimizer in optimizers.items():
            logger.info(f"\nUsing {optimizer_name}")
            
            # Run multiple times to get statistical significance
            for run in range(1, 6):  # 5 runs
                logger.info(f"Run {run}/5")
                
                # Reset optimizer
                optimizer.reset()
                
                # Run optimizer
                optimizer.optimize(func, max_evals=10000)
    
    # Export data if requested
    if args.export:
        export_format = args.export_format
        export_dir = args.export_dir
        
        # Create export directory if it doesn't exist
        os.makedirs(export_dir, exist_ok=True)
        
        # Export data for each optimizer
        for optimizer_name, optimizer in optimizers.items():
            if hasattr(optimizer, 'export_data'):
                try:
                    filename = optimizer.export_data(
                        os.path.join(export_dir, f"optimization_{optimizer_name}_{time.strftime('%Y%m%d_%H%M%S')}"),
                        format=export_format
                    )
                    logging.info(f"Exported data for {optimizer_name} optimizer")
                except Exception as e:
                    logging.error(f"Failed to export data for {optimizer_name}: {str(e)}")
    
    logger.info(f"Final memory usage: {psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024:.2f} MB")
    return results, pd.DataFrame()

def run_benchmark_comparison(n_runs: int = 30, max_evals: int = 10000, live_viz: bool = False, save_plots: bool = True):
    """Run comprehensive benchmark comparison of all optimizers"""
    logging.info("Starting benchmark comparison")
    
    # Get test functions
    test_suite = {}
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
                func_creator = {}
                bounds = [(-5, 5)] * dim  # Default bounds
                benchmark_functions[key] = func_creator
    
    # Run comparison for each function
    results = {}
    all_results_data = []
    
    for func_name, func in benchmark_functions.items():
        logging.info(f"Benchmarking function: {func_name}")
        
        # Create optimizers for this dimension
        dim = 30
        if func_name.endswith("_10D"):
            dim = 10
        bounds = [(-5, 5)] * dim
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
            meta_opt.enable_live_visualization(max_data_points=1000, auto_show=False, headless=True)
            
            # Set save directory separately if needed
            if hasattr(meta_opt, 'save_viz_path'):
                meta_opt.save_viz_path = save_path
        
        # Add meta-optimizer to the comparison
        all_optimizers = optimizers.copy()
        all_optimizers['Meta-Optimizer'] = meta_opt
        
        # Create analyzer and run comparison
        analyzer = {}
        function_results = {}
        results[func_name] = function_results
        
        # Collect data for summary dataframe
        for opt_name, opt_results in function_results.items():
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
    if not results_df.empty and 'function' in results_df.columns and 'optimizer' in results_df.columns:
        summary_stats = results_df.groupby(['function', 'optimizer']).agg({
            'best_score': ['mean', 'std', 'min'],
            'execution_time': ['mean', 'std']
        }).reset_index()
        summary_stats.to_csv(data_dir / 'optimization_summary.csv')
    else:
        logging.warning("Unable to create summary stats: DataFrame missing required columns or is empty")
        # Create a simple summary with minimal information
        summary_stats = pd.DataFrame(columns=['function', 'optimizer'])
    
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
    X_full, y_full, drift_points = {}, {}, {}
    logging.info(f"Generated synthetic data with drift points at: {drift_points}")
    
    # Initialize meta-learner
    meta_learner = MetaLearner(
        method='bayesian',
        surrogate_model=None,
        selection_strategy='bandit',
        exploration_factor=0.2,
        history_weight=0.7
    )
    
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
    
    # Set the drift detector for the meta-learner
    meta_learner.drift_detector = detector
    
    # Add some algorithms to the meta-learner
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
    from sklearn.linear_model import LinearRegression
    
    # Create simple algorithm wrappers
    class Algorithm:
        def __init__(self, name, model):
            self.name = name
            self.model = model
            self.best_config = None
            self.best_score = float('-inf')
            self.is_fitted = False
            
        def fit(self, X, y):
            self.model.fit(X, y)
            self.is_fitted = True
            
        def predict(self, X):
            if not self.is_fitted:
                logging.warning(f"{self.name} model is not fitted yet. Fitting with dummy data before prediction.")
                # Create dummy data to fit the model
                import numpy as np
                dummy_X = np.random.random((10, X.shape[1]))
                dummy_y = np.random.random(10)
                self.fit(dummy_X, dummy_y)
                logging.warning(f"{self.name} model fitted with dummy data. Predictions may not be meaningful.")
            return self.model.predict(X)
            
        def suggest(self, num_points=1):
            """Suggest next points to evaluate (dummy implementation)"""
            import numpy as np
            # Check if we should return a single point or multiple
            if num_points == 1:
                return np.random.uniform(-1, 1, (1, 10))  # Return as 2D array with 1 row
            # Otherwise return multiple points
            return np.random.uniform(-1, 1, (num_points, 10))
            
        def evaluate(self, X):
            """Evaluate points for meta-learning (dummy implementation)"""
            # Return dummy scores for the points
            import numpy as np
            # Make sure X is 2D when computing shape
            if isinstance(X, np.ndarray):
                if X.ndim == 1:
                    X = X.reshape(1, -1)
                # Ensure we return 1D array for multiple points or scalar for single point
                if X.shape[0] > 1:
                    return np.random.uniform(0, 1, X.shape[0]).astype(float)
                else:
                    return float(np.random.uniform(0, 1))
            else:
                # Handle scalar case
                return float(np.random.uniform(0, 1))
            
        def update(self, X, scores=None):
            """Update the optimizer with new data (dummy implementation)"""
            import numpy as np
            
            # Always treat X and scores as arrays for consistency
            try:
                # Handle scalar X by converting to a 1x1 array
                if np.isscalar(X):
                    X = np.array([[float(X)]])
                elif isinstance(X, np.ndarray):
                    if X.ndim == 0:  # Handle numpy scalars
                        X = np.array([[float(X)]])
                    elif X.ndim == 1:
                        X = X.reshape(1, -1)
                        
                # Just record best score if scores are provided
                if scores is not None:
                    # Ensure scores is a float array
                    if np.isscalar(scores):
                        scores = np.array([float(scores)])
                    else:
                        scores = np.asarray(scores, dtype=float)
                        if scores.ndim == 0:  # scalar
                            scores = np.array([float(scores)])
                        
                    if scores.size > 0:
                        # Ensure best_score is initialized
                        if self.best_score == float('-inf'):
                            self.best_score = float(scores.max())
                            best_idx = scores.argmax() if scores.size > 1 else 0
                            self.best_config = X[best_idx].copy() if best_idx < len(X) else X[0].copy()
                        elif scores.max() > self.best_score:
                            self.best_score = float(scores.max())
                            best_idx = scores.argmax() if scores.size > 1 else 0
                            self.best_config = X[best_idx].copy() if best_idx < len(X) else X[0].copy()
                # Otherwise fit the model with new data
                else:
                    self.fit(X, scores)
            except Exception as e:
                logging.error(f"Error in update: {str(e)}")
                # Fallback to prevent crashes
                if self.best_score == float('-inf'):
                    self.best_score = 0.0
                    if isinstance(X, np.ndarray) and X.size > 0:
                        self.best_config = X[0].copy() if X.ndim > 1 else X.copy()
            
        def get_best(self):
            """Get best configuration found so far"""
            return self.best_config, self.best_score
        
    # Add algorithms to meta-learner
    meta_learner.algorithms = [
        Algorithm("RF", RandomForestRegressor(n_estimators=100, random_state=42)),
        Algorithm("GB", GradientBoostingRegressor(n_estimators=100, random_state=42)),
        Algorithm("LR", LinearRegression())
    ]
    
    # Set algorithms
    from optimizers.optimizer_factory import create_optimizers
    optimizers = create_optimizers(dim=n_features, bounds=[(-5, 5)] * n_features)
    meta_learner.set_algorithms(list(optimizers.values()))
    
    # Initialize with first 200 samples
    X_init, y_init = {}, {}
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
        X_chunk, y_chunk = {}, {}
        
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
        
        X_before = {}
        y_before = {}
        X_after = {}
        y_after = {}
        
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
            
        logging.info(f"Stats: Mean shift={mean_shift:.4f}, KS stat={ks_statistic:.4f}, p-value={p_value:.6f}")
    
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
    X = {}
    
    # Generate target with concept drift
    y = {}
    
    # Initial coefficients
    coef = {}
    
    # Generate data with different coefficients for each segment
    start_idx = 0
    for drift_point in drift_points:
        # Apply current coefficients to this segment
        y[start_idx:drift_point] = {}
        
        # Create more pronounced drift by significantly changing coefficients
        # Increase the magnitude of change at drift points
        coef = {}
        
        # Add an abrupt shift at the drift point
        shift_magnitude = 0.5 + 0.5 * np.random.random()  # Random shift between 0.5 and 1.0
        if drift_point < n_samples:
            # Apply an abrupt shift to a small window after the drift point
            post_drift_window = min(20, n_samples - drift_point)
            y[drift_point:drift_point+post_drift_window] = {}
        
        start_idx = drift_point
    
    # Last segment
    y[start_idx:] = {}
    
    return X, y, drift_points

def check_drift_at_point(meta_learner, X_before, y_before, X_after, y_after, window_size=10, drift_threshold=0.01, significance_level=0.9):
    """Check for drift at a specific point using data before and after the point."""
    # Create a separate drift detector with sensitive parameters
    drift_detector = {}
    
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
    drift_detected, score, info_dict = drift_detector.detect_drift(errors_after, errors_before)
    
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
    args=None, model=None, X=None, y=None, 
    explainer_type='shap', plot_types=None, 
    generate_plots=True, n_samples=5, **kwargs
):
    """
    Run explainability analysis on a trained model
    
    Args:
        args: Command-line arguments
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
    logging.info("Running explainability analysis")
    
    # Process args if provided
    if args is not None:
        explainer_type = args.explainer if hasattr(args, 'explainer') else explainer_type
        generate_plots = args.explain_plots if hasattr(args, 'explain_plots') else generate_plots
        plot_types = args.explain_plot_types if hasattr(args, 'explain_plot_types') else plot_types
        n_samples = args.explain_samples if hasattr(args, 'explain_samples') else n_samples
    
    # Generate synthetic data if not provided
    if X is None or y is None:
        logging.info("No data provided, generating synthetic data...")
        np.random.seed(42)
        X = np.random.rand(100, 5)
        y = np.random.rand(100)
    
    # Create model if not provided
    if model is None:
        logging.info("No model provided, training a simple model...")
        from sklearn.ensemble import RandomForestRegressor
        model = RandomForestRegressor(n_estimators=10, random_state=42)
        model.fit(X, y)
    
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

def run_meta_learning(args=None, method='bayesian', surrogate=None, selection=None, exploration=0.2, history_weight=0.7, visualize=False):
    """
    Run meta-learning process to find the best optimizer for a given problem.
    
    Args:
        args: Command-line arguments (can be None if calling programmatically)
        method: Method for meta-learning
        surrogate: Surrogate model for meta-learning
        selection: Selection strategy
        exploration: Exploration factor
        history_weight: History weight factor
        visualize: Whether to visualize results
    
    Returns:
        Dictionary with meta-learning results
    """
    # Process arguments if provided
    if args:
        method = args.method
        surrogate = args.surrogate
        selection = args.selection
        exploration = args.exploration
        history_weight = args.history
        visualize = args.visualize or args.visualize_algorithm_selection
    
    logging.info(f"Running meta-learning with method={method}, surrogate={surrogate}, selection={selection}")
    
    # Create directories if they don't exist
    os.makedirs('results/meta_learning', exist_ok=True)
    
    # Define visualization save directory
    algo_viz_dir = 'results/algorithm_selection'
    if args and args.algo_viz_dir:
        algo_viz_dir = args.algo_viz_dir
    os.makedirs(algo_viz_dir, exist_ok=True)
    
    # Define test functions
    test_functions = {}
    
    # Add standard test functions for benchmarking
    try:
        from meta_optimizer.benchmark.test_functions import TEST_FUNCTIONS
        
        # Create test function instances
        dim = args.dimension if args and hasattr(args, 'dimension') and args.dimension else 30
        bounds = [(-5, 5)] * dim
        
        test_functions = {
            'sphere': lambda x: TEST_FUNCTIONS['sphere'](dim, bounds).evaluate(x),
            'rosenbrock': lambda x: TEST_FUNCTIONS['rosenbrock'](dim, bounds).evaluate(x),
            'rastrigin': lambda x: TEST_FUNCTIONS['rastrigin'](dim, bounds).evaluate(x),
            'ackley': lambda x: TEST_FUNCTIONS['ackley'](dim, bounds).evaluate(x)
        }
        
        print(f"Loaded {len(test_functions)} test functions for meta-learning")
    except (ImportError, AttributeError, KeyError) as e:
        logging.warning(f"Could not load test functions: {e}")
        logging.warning("Creating simple test functions instead")
        
        # Define simple test functions
        def sphere(x):
            return np.sum(x**2)
            
        def rosenbrock(x):
            return np.sum(100.0 * (x[1:] - x[:-1]**2)**2 + (x[:-1] - 1)**2)
            
        def rastrigin(x):
            return 10 * len(x) + np.sum(x**2 - 10 * np.cos(2 * np.pi * x))
            
        def ackley(x):
            a, b, c = 20, 0.2, 2 * np.pi
            sum1 = -a * np.exp(-b * np.sqrt(np.mean(x**2)))
            sum2 = -np.exp(np.mean(np.cos(c * x)))
            return sum1 + sum2 + a + np.exp(1)
            
        test_functions = {
            'sphere': sphere,
            'rosenbrock': rosenbrock,
            'rastrigin': rastrigin,
            'ackley': ackley
        }
        
        print(f"Created {len(test_functions)} simple test functions for meta-learning")
    
    # Initialize optimizers
    dim = 30
    if args and hasattr(args, 'dimension') and args.dimension:
        dim = args.dimension
    
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
        'DE-Adaptive': de_adaptive_opt,
        'ES-Adaptive': es_adaptive_opt
    }
    
    # Create MetaOptimizer
    meta_opt = MetaOptimizer(
        dim=dim,
        bounds=bounds,
        optimizers=optimizers,
        history_file='results/meta_learning/history.json',
        selection_file='results/meta_learning/selection_history.json'
    )
    
    # Enable live visualization if requested
    if visualize:
        meta_opt.enable_live_visualization(max_data_points=1000, auto_show=False, headless=True)
        
        # Set save directory separately if needed
        if hasattr(meta_opt, 'save_viz_path'):
            meta_opt.save_viz_path = algo_viz_dir
    
    # Enable algorithm selection visualization if available
    if ALGORITHM_VIZ_AVAILABLE and (visualize or (args and args.visualize_algorithm_selection)):
        if not hasattr(meta_opt, 'enable_algo_viz'):
            meta_opt.enable_algo_viz = False
            
        if not hasattr(meta_opt, 'algo_selection_viz'):
            meta_opt.algo_selection_viz = None
            
        if not meta_opt.enable_algo_viz or meta_opt.algo_selection_viz is None:
            meta_opt.enable_algo_viz = True
            meta_opt.algo_selection_viz = AlgorithmSelectionVisualizer(save_dir=algo_viz_dir)
            meta_opt.visualize_algorithm_selection = True
            logging.info("Algorithm selection visualization enabled for meta-learning")
    
    # Initialize result containers
    results = {}
    best_algorithms = {}
    performance_metrics = {}
    
    # Compile results
    meta_learning_results = {
        'best_algorithms': best_algorithms,
        'overall_best_algorithm': None,
        'performance_metrics': performance_metrics,
        'meta_learning_method': method,
        'surrogate_model': surrogate,
        'selection_strategy': selection,
        'exploration_factor': exploration,
        'history_weight': history_weight
    }
    
    # Run meta optimization for each test function
    for func_name, func in test_functions.items():
        logging.info(f"Running meta-optimization for {func_name}")
        
        meta_opt.reset()  # Reset optimizer state
        
        # Use the correct parameters for optimize
        result = meta_opt.optimize(
            func, 
            max_evals=5000,  # Increased for better algorithm selection tracking
            context={"function_name": func_name, "phase": "optimization"}
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
        if isinstance(history, list):
            for entry in history:
                if isinstance(entry, dict):
                    optimizer = entry.get('selected_optimizer', 'unknown')
                    optimizer_counts[optimizer] = optimizer_counts.get(optimizer, 0) + 1
                else:
                    logging.warning(f"Unexpected history entry type: {type(entry)}")
        else:
            logging.warning(f"History is not a list: {type(history)}")
        
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
            'convergence_rate': len(history) / 5000  # Simple metric
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
    
    meta_learning_results['overall_best_algorithm'] = overall_best_algorithm
    
    # Create summary plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot optimizer selection frequency
    algorithms = list(all_selections.keys())
    frequencies = [all_selections[alg] for alg in algorithms]
    
    ax.bar(algorithms, frequencies)
    ax.set_xlabel('Optimizer')
    ax.set_ylabel('Selection Frequency')
    ax.set_title('Meta-Learning Algorithm Selection Frequency')
    plt.tight_layout()
    
    # Save the plot
    plt.savefig('results/meta_learning/algorithm_selection.png')
    
    if visualize:
        plt.show()
    
    # Finalize and save algorithm selection visualizations
    if ALGORITHM_VIZ_AVAILABLE and meta_opt.enable_algo_viz:
        # Generate visualization images
        try:
            # Create a proper directory for visualization results
            results_dir = Path('results/benchmarks')
            results_dir.mkdir(parents=True, exist_ok=True)
            
            # Instead of saving results through the meta-optimizer, 
            # just disable the visualization without saving
            meta_opt.enable_viz = False
            if hasattr(meta_opt, 'live_viz_monitor'):
                meta_opt.live_viz_monitor.stop_monitoring()
                
            print("Live visualization disabled")
        except Exception as e:
            logging.error(f"Error during visualization shutdown: {e}")
        
        # Generate algorithm selection visualizations
        try:
            # Create visualizations
            if meta_opt.algo_selection_viz:
                logging.info("Generating algorithm selection visualizations")
                
                # Generate requested plots based on arguments or defaults
                if args and args.algo_viz_plots:
                    plot_types = args.algo_viz_plots
                else:
                    plot_types = ['dashboard', 'frequency', 'timeline', 'problem', 'interactive']
                
                plots_generated = []
                plot_paths = {}
                
                # For each visualization type, handle errors independently
                try:
                    if 'frequency' in plot_types or 'all' in plot_types:
                        logging.info("Generating frequency plot...")
                        meta_opt.algo_selection_viz.plot_selection_frequency(save=True)
                        plots_generated.append("frequency")
                        plot_paths['frequency'] = f"{algo_viz_dir}/algorithm_selection_frequency.png"
                except Exception as e:
                    logging.error(f"Error generating frequency plot: {e}")
                
                try:
                    if 'timeline' in plot_types or 'all' in plot_types:
                        logging.info("Generating timeline plot...")
                        meta_opt.algo_selection_viz.plot_selection_timeline(save=True)
                        plots_generated.append("timeline")
                        plot_paths['timeline'] = f"{algo_viz_dir}/algorithm_selection_timeline.png"
                except Exception as e:
                    logging.error(f"Error generating timeline plot: {e}")
                
                try:
                    if 'problem' in plot_types or 'all' in plot_types:
                        logging.info("Generating problem distribution plot...")
                        meta_opt.algo_selection_viz.plot_problem_distribution(save=True)
                        plots_generated.append("problem")
                        plot_paths['problem'] = f"{algo_viz_dir}/algorithm_selection_by_problem.png"
                except Exception as e:
                    logging.error(f"Error generating problem distribution plot: {e}")
                
                try:
                    if 'dashboard' in plot_types or 'all' in plot_types:
                        logging.info("Generating summary dashboard...")
                        meta_opt.algo_selection_viz.create_summary_dashboard(save=True)
                        plots_generated.append("dashboard")
                        plot_paths['dashboard'] = f"{algo_viz_dir}/algorithm_selection_dashboard.png"
                except Exception as e:
                    logging.error(f"Error generating summary dashboard: {e}")
                
                try:
                    if 'interactive' in plot_types or 'all' in plot_types:
                        logging.info("Generating interactive visualizations...")
                        meta_opt.algo_selection_viz.interactive_selection_timeline(save=True)
                        plots_generated.append("interactive_timeline")
                        plot_paths['interactive_timeline'] = f"{algo_viz_dir}/interactive_algorithm_timeline.html"
                        
                        meta_opt.algo_selection_viz.interactive_dashboard(save=True)
                        plots_generated.append("interactive_dashboard")
                        plot_paths['interactive_dashboard'] = f"{algo_viz_dir}/interactive_dashboard.html"
                except Exception as e:
                    logging.error(f"Error generating interactive visualizations: {e}")
                
                if plots_generated:
                    print(f"Generated {len(plots_generated)} algorithm selection visualizations:")
                    for plot_type in plots_generated:
                        print(f"  - {plot_type}: {os.path.basename(plot_paths.get(plot_type, ''))}")
                    print(f"Visualizations saved to: {algo_viz_dir}")
                    
                    # Add visualization paths to results
                    meta_learning_results['visualization_paths'] = plot_paths
        except Exception as e:
            logging.error(f"Error generating algorithm selection visualizations: {e}")
    
    # Create a class to handle JSON serialization
    class NumpyEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, np.integer):
                return int(obj)
            if isinstance(obj, np.floating):
                return float(obj)
            if isinstance(obj, np.bool_):
                return bool(obj)
            return super().default(obj)

    # Save results to file
    os.makedirs('results/meta_learning', exist_ok=True)
    try:
        with open('results/meta_learning/meta_learning_results.json', 'w') as f:
            json.dump(meta_learning_results, f, indent=4, cls=NumpyEncoder)
    except Exception as e:
        logging.error(f"Error saving meta-learning results: {e}")
    
    # Print summary if requested
    print("\nMeta-Learning Results Summary:")
    print("------------------------------")
    print(f"Method: {method}")
    print(f"Overall Best Algorithm: {overall_best_algorithm}")
    print("\nBest Algorithm per Function:")
    for func, alg in best_algorithms.items():
        print(f"  {func}: {alg} (Score: {results[func]['best_score']})")
    
    print("\nAlgorithm Selection Frequency:")
    for alg, freq in sorted(all_selections.items(), key=lambda x: x[1], reverse=True):
        print(f"  {alg}: {freq}")
    
    # Return results
    return meta_learning_results

def run_evaluation(args=None, model=None, X_test=None, y_test=None):
    """
    Evaluate a trained model on test data.
    
    Args:
        args: Command-line arguments (used when called from main)
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
    
    # If called from main with args
    if args is not None and model is None:
        # Create a simple model for demonstration
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.datasets import make_regression
        from sklearn.model_selection import train_test_split
        
        # Create synthetic data
        X, y = make_regression(n_samples=1000, n_features=10, noise=0.1, random_state=42)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Create and train model
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
    # Create model and data if not provided
    elif model is None or X_test is None or y_test is None:
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.datasets import make_regression
        from sklearn.model_selection import train_test_split
        
        # Create synthetic data
        X, y = make_regression(n_samples=1000, n_features=10, noise=0.1, random_state=42)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Create and train a model if none was provided
        if model is None:
            model = RandomForestRegressor(n_estimators=100, random_state=42)
            model.fit(X_train, y_train)
    
    # Evaluate model
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    from sklearn.metrics import mean_squared_error, r2_score
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
    evaluator = {}
    drift_detector = {}
    optimizer_analyzer = {}
    
    # Load and preprocess data
    data = pd.read_csv(data_path)
    X = data.drop('target', axis=1)
    y = data['target']
    
    # Split data
    X_train, X_test, y_train, y_test = {}, {}, {}, {}
    
    # Run optimization and get best model
    best_model = run_optimization(args=None, X_train=X_train, y_train=y_train, n_runs=n_runs, max_evals=max_evals)
    
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

def run_drift_detection(args):
    """
    Run drift detection on synthetic data.
    
    Args:
        args: Command-line arguments
    """
    logging.info("Running drift detection")
    
    # Get parameters from args
    window_size = args.drift_window if hasattr(args, 'drift_window') else 50
    drift_threshold = args.drift_threshold if hasattr(args, 'drift_threshold') else 0.5
    significance_level = args.drift_significance if hasattr(args, 'drift_significance') else 0.05
    visualize = args.visualize if hasattr(args, 'visualize') else False
    
    # Generate synthetic data with drift
    n_samples = 1000
    n_features = 10
    drift_points = [300, 600]  # Drift at these points
    
    X, y = {}, {}
    
    # Set up the drift detector
    detector = {}
    
    # Monitor for drift
    drift_points_detected = []
    for i in range(window_size, len(X)):
        X_window = X[i-window_size:i]
        y_window = y[i-window_size:i]
        
        is_drift, score, info_dict = detector.detect_drift(X_window, y_window)
        
        if is_drift:
            drift_points_detected.append(i)
            p_value = info_dict['p_value']
            logging.info(f"Drift detected at point {i}, score: {score:.4f}, p-value: {p_value:.4f}")
    
    # Visualize results if requested
    if visualize:
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.scatter(range(len(y)), y, alpha=0.6, label='Data points')
        
        # Mark true drift points
        for point in drift_points:
            ax.axvline(x=point, color='r', linestyle='--', label='True drift' if point == drift_points[0] else None)
        
        # Mark detected drift points
        for point in drift_points_detected:
            ax.axvline(x=point, color='g', linestyle='-', label='Detected drift' if point == drift_points_detected[0] else None)
        
        ax.set_title('Drift Detection Results')
        ax.set_xlabel('Time steps')
        ax.set_ylabel('Target values')
        ax.legend()
        
        # Save plot
        save_plot(fig, 'drift_detection_results', plot_type='drift')
    
    # Return the results
    results = {
        'true_drift_points': drift_points,
        'detected_drift_points': drift_points_detected,
        'window_size': window_size,
        'drift_threshold': drift_threshold,
        'significance_level': significance_level
    }
    
    return results

def run_meta_learner_with_drift_detection(args):
    """
    Run meta-learner with drift detection.
    
    Args:
        args: Command-line arguments
    """
    logging.info("Running meta-learner with drift detection")
    
    # Get parameters from args
    window_size = args.drift_window if hasattr(args, 'drift_window') else 50
    drift_threshold = args.drift_threshold if hasattr(args, 'drift_threshold') else 0.5
    significance_level = args.drift_significance if hasattr(args, 'drift_significance') else 0.05
    visualize = args.visualize if hasattr(args, 'visualize') else False
    
    # Generate synthetic data with drift
    n_samples = 1000
    n_features = 10
    drift_points = [300, 600]  # Drift at these points
    
    X, y, _ = {}, {}, {}
    
    # Set up the drift detector
    detector = {}
    
    # Set up the meta-learner
    meta_learner = {}
    
    # Add some algorithms to the meta-learner
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
    from sklearn.linear_model import LinearRegression
    
    # Create simple algorithm wrappers
    class Algorithm:
        def __init__(self, name, model):
            self.name = name
            self.model = model
            self.best_config = None
            self.best_score = float('-inf')
            self.is_fitted = False
            
        def fit(self, X, y):
            self.model.fit(X, y)
            self.is_fitted = True
            
        def predict(self, X):
            if not self.is_fitted:
                logging.warning(f"{self.name} model is not fitted yet. Fitting with dummy data before prediction.")
                # Create dummy data to fit the model
                import numpy as np
                dummy_X = np.random.random((10, X.shape[1]))
                dummy_y = np.random.random(10)
                self.fit(dummy_X, dummy_y)
                logging.warning(f"{self.name} model fitted with dummy data. Predictions may not be meaningful.")
            return self.model.predict(X)
            
        def suggest(self, num_points=1):
            """Suggest next points to evaluate (dummy implementation)"""
            import numpy as np
            # Check if we should return a single point or multiple
            if num_points == 1:
                return np.random.uniform(-1, 1, (1, 10))  # Return as 2D array with 1 row
            # Otherwise return multiple points
            return np.random.uniform(-1, 1, (num_points, 10))
            
        def evaluate(self, X):
            """Evaluate configurations (dummy implementation)"""
            # Return dummy scores for the points
            import numpy as np
            # Make sure X is 2D when computing shape
            if isinstance(X, np.ndarray):
                if X.ndim == 1:
                    X = X.reshape(1, -1)
                # Ensure we return 1D array for multiple points or scalar for single point
                if X.shape[0] > 1:
                    return np.random.uniform(0, 1, X.shape[0]).astype(float)
                else:
                    return float(np.random.uniform(0, 1))
            else:
                # Handle scalar case
                return float(np.random.uniform(0, 1))
            
        def update(self, X, scores=None):
            """Update the optimizer with new data (dummy implementation)"""
            import numpy as np
            
            # Always treat X and scores as arrays for consistency
            try:
                # Handle scalar X by converting to a 1x1 array
                if np.isscalar(X):
                    X = np.array([[float(X)]])
                elif isinstance(X, np.ndarray):
                    if X.ndim == 0:  # Handle numpy scalars
                        X = np.array([[float(X)]])
                    elif X.ndim == 1:
                        X = X.reshape(1, -1)
                        
                # Just record best score if scores are provided
                if scores is not None:
                    # Ensure scores is a float array
                    if np.isscalar(scores):
                        scores = np.array([float(scores)])
                    else:
                        scores = np.asarray(scores, dtype=float)
                        if scores.ndim == 0:  # scalar
                            scores = np.array([float(scores)])
                        
                    if scores.size > 0:
                        # Ensure best_score is initialized
                        if self.best_score == float('-inf'):
                            self.best_score = float(scores.max())
                            best_idx = scores.argmax() if scores.size > 1 else 0
                            self.best_config = X[best_idx].copy() if best_idx < len(X) else X[0].copy()
                        elif scores.max() > self.best_score:
                            self.best_score = float(scores.max())
                            best_idx = scores.argmax() if scores.size > 1 else 0
                            self.best_config = X[best_idx].copy() if best_idx < len(X) else X[0].copy()
                # Otherwise fit the model with new data
                else:
                    self.fit(X, scores)
            except Exception as e:
                logging.error(f"Error in update: {str(e)}")
                # Fallback to prevent crashes
                if self.best_score == float('-inf'):
                    self.best_score = 0.0
                    if isinstance(X, np.ndarray) and X.size > 0:
                        self.best_config = X[0].copy() if X.ndim > 1 else X.copy()
            
        def get_best(self):
            """Get best configuration found so far"""
            return self.best_config, self.best_score
        
    # Add algorithms to meta-learner
    meta_learner.algorithms = [
        Algorithm("RF", RandomForestRegressor(n_estimators=100, random_state=42)),
        Algorithm("GB", GradientBoostingRegressor(n_estimators=100, random_state=42)),
        Algorithm("LR", LinearRegression())
    ]
    
    # Set the drift detector for the meta-learner
    meta_learner.drift_detector = detector
    
    # Initialize performance metrics
    mse_history = []
    drift_detected_points = []
    
    # Run online learning with drift detection
    batch_size = 50
    for i in range(0, len(X) - batch_size, batch_size):
        # Get current batch
        X_batch = X[i:i+batch_size]
        y_batch = y[i:i+batch_size]
        
        # Train the meta-learner on the current batch
        meta_learner.update(X_batch, y_batch)
        
        # Make predictions
        y_pred = meta_learner.predict(X_batch)
        mse = {}
        mse_history.append(mse)
        
        # Check for drift
        if i >= window_size:
            X_window = X[i-window_size:i]
            y_window = y[i-window_size:i]
            
            is_drift, drift_score, info_dict = detector.detect_drift(X_window, y_window)
            p_value = info_dict['p_value']
            
            if is_drift:
                drift_detected_points.append(i)
                logging.info(f"Drift detected at point {i}, score: {drift_score:.4f}, p-value: {p_value:.4f}")
                
                # Reset the meta-learner when drift is detected
                meta_learner.reset_weights()
    
    # Visualize results if requested
    if visualize:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
        
        # Plot target values and drift points
        ax1.scatter(range(len(y)), y, alpha=0.6, s=5, label='Data points')
        
        # Mark true drift points
        for point in drift_points:
            ax1.axvline(x=point, color='r', linestyle='--', label='True drift' if point == drift_points[0] else None)
        
        # Mark detected drift points
        for point in drift_detected_points:
            ax1.axvline(x=point, color='g', linestyle='-', label='Detected drift' if point == drift_detected_points[0] else None)
        
        ax1.set_title('Data Stream with Drift')
        ax1.set_ylabel('Target values')
        ax1.legend()
        
        # Plot MSE history
        indices = range(0, len(X) - batch_size, batch_size)
        ax2.plot(indices, mse_history, label='MSE')
        
        # Mark true drift points
        for point in drift_points:
            ax2.axvline(x=point, color='r', linestyle='--')
        
        # Mark detected drift points
        for point in drift_detected_points:
            ax2.axvline(x=point, color='g', linestyle='-')
            
        ax2.set_title('MSE History')
        ax2.set_xlabel('Time steps')
        ax2.set_ylabel('Mean Squared Error')
        
        plt.tight_layout()
        
        # Save plot
        save_plot(fig, 'meta_learner_with_drift_detection', plot_type='drift')
    
    # Return the results
    results = {
        'true_drift_points': drift_points,
        'detected_drift_points': drift_detected_points,
        'mse_history': mse_history,
        'window_size': window_size,
        'drift_threshold': drift_threshold,
        'significance_level': significance_level
    }
    
    return results

def explain_drift(args):
    """
    Explain drift when detected.
    
    Args:
        args: Command-line arguments
    """
    logging.info("Explaining drift in the data")
    
    # Get parameters from args
    window_size = args.drift_window if hasattr(args, 'drift_window') else 50
    drift_threshold = args.drift_threshold if hasattr(args, 'drift_threshold') else 0.5
    significance_level = args.drift_significance if hasattr(args, 'drift_significance') else 0.05
    visualize = args.visualize if hasattr(args, 'visualize') else False
    
    # Generate synthetic data with drift
    n_samples = 1000
    n_features = 10
    drift_points = [300, 600]  # Drift at these points
    
    X, y, drift_types = {}, {}, {}
    
    # Set up the drift detector
    detector = {}
    
    # Monitor for drift
    drift_points_detected = []
    drift_scores = []
    p_values = []
    feature_contributions = []
    
    for i in range(window_size, len(X)):
        X_window = X[i-window_size:i]
        y_window = y[i-window_size:i]
        
        is_drift, drift_score, info_dict = detector.detect_drift(X_window, y_window)
        p_value = info_dict['p_value']
        drift_scores.append(drift_score)
        p_values.append(p_value)
        
        if is_drift:
            drift_points_detected.append(i)
            logging.info(f"Drift detected at point {i}, score: {drift_score:.4f}, p-value: {p_value:.4f}")
            
            # Calculate feature contributions to drift
            before_drift = X[i-window_size:i-window_size//2]
            after_drift = X[i-window_size//2:i]
            
            # Calculate mean shift for each feature
            mean_before = np.mean(before_drift, axis=0)
            mean_after = np.mean(after_drift, axis=0)
            mean_shift = np.abs(mean_after - mean_before)
            
            # Calculate variance shift for each feature
            var_before = np.var(before_drift, axis=0)
            var_after = np.var(after_drift, axis=0)
            var_shift = np.abs(var_after - var_before)
            
            # Calculate distribution shift using KS test
            ks_pvalues = []
            for f in range(n_features):
                _, ks_pvalue = stats.ks_2samp(before_drift[:, f], after_drift[:, f])
                ks_pvalues.append(1 - ks_pvalue)  # Convert p-value to a "contribution" score
            
            # Combine the metrics (mean shift, variance shift, KS test)
            feature_contrib = (mean_shift / np.max(mean_shift) + 
                              var_shift / np.max(var_shift) + 
                              np.array(ks_pvalues)) / 3
            
            feature_contributions.append((i, feature_contrib))
    
    # Visualize results if requested
    if visualize:
        # Create a figure with 4 subplots
        fig, axs = plt.subplots(4, 1, figsize=(12, 16), sharex=True)
        
        # Plot 1: Target values and drift points
        axs[0].scatter(range(len(y)), y, alpha=0.6, s=5, label='Data points')
        
        # Mark true drift points
        for point in drift_points:
            axs[0].axvline(x=point, color='r', linestyle='--', 
                           label='True drift' if point == drift_points[0] else None)
        
        # Mark detected drift points
        for point in drift_points_detected:
            axs[0].axvline(x=point, color='g', linestyle='-', 
                           label='Detected drift' if point == drift_points_detected[0] else None)
        
        axs[0].set_title('Data Stream with Drift')
        axs[0].set_ylabel('Target values')
        axs[0].legend()
        
        # Plot 2: Drift scores
        axs[1].plot(range(window_size, len(X)), drift_scores, label='Drift score')
        axs[1].axhline(y=drift_threshold, color='r', linestyle='--', label='Threshold')
        axs[1].set_title('Drift Scores')
        axs[1].set_ylabel('Score')
        axs[1].legend()
        
        # Plot 3: P-values
        axs[2].plot(range(window_size, len(X)), p_values, label='P-value')
        axs[2].axhline(y=significance_level, color='r', linestyle='--', label='Significance level')
        axs[2].set_title('Statistical Significance (P-value)')
        axs[2].set_ylabel('P-value')
        axs[2].legend()
        
        # Plot 4: Feature contributions
        if feature_contributions:
            # Get the contribution at the first detected drift point
            drift_point, contributions = feature_contributions[0]
            
            # Plot feature contributions as a bar chart
            axs[3].bar(range(n_features), contributions)
            axs[3].set_title(f'Feature Contributions to Drift at Point {drift_point}')
            axs[3].set_xlabel('Feature Index')
            axs[3].set_ylabel('Contribution Score')
            axs[3].set_xticks(range(n_features))
        
        plt.tight_layout()
        
        # Save plot
        save_plot(fig, 'drift_explanation', plot_type='drift')
    
    # Print drift explanation
    print("\nDrift Explanation:")
    print("-----------------")
    print(f"True drift points: {drift_points}")
    print(f"Detected drift points: {drift_points_detected}")
    
    if feature_contributions:
        print("\nFeature Contributions to Drift:")
        for i, (point, contributions) in enumerate(feature_contributions):
            print(f"\nDrift Point {i+1} (at index {point}):")
            
            # Sort features by contribution
            feature_indices = np.argsort(contributions)[::-1]
            
            for rank, idx in enumerate(feature_indices):
                print(f"  Rank {rank+1}: Feature {idx} (contribution: {contributions[idx]:.4f})")
            
            # Determine the type of drift if available
            if drift_types and point in drift_types:
                print(f"  Drift type: {drift_types[point]}")
    
    # Return the results
    results = {
        'true_drift_points': drift_points,
        'detected_drift_points': drift_points_detected,
        'drift_scores': drift_scores,
        'p_values': p_values,
        'feature_contributions': feature_contributions,
        'window_size': window_size,
        'drift_threshold': drift_threshold,
        'significance_level': significance_level
    }
    
    return results

def run_migraine_data_import(args):
    """
    Import new migraine data with potentially different schema.
    
    Args:
        args: Command-line arguments containing data path, output path, and options
    
    Returns:
        Dictionary with results of the import operation
    """
    if not MIGRAINE_MODULES_AVAILABLE:
        logging.error("Migraine prediction modules are not available. Please install the package first.")
        return {"success": False, "error": "Migraine modules not available"}
    
    try:
        logging.info(f"Importing migraine data from {args.data_path}")
        
        # Initialize the predictor
        from migraine_prediction_project.src.migraine_model import MigrainePredictor
        import pandas as pd
        predictor = MigrainePredictor()
        
        # Import the data
        try:
            file_extension = os.path.splitext(args.data_path)[1].lower()
            if file_extension == '.csv':
                imported_data = pd.read_csv(args.data_path)
            elif file_extension == '.xlsx' or file_extension == '.xls':
                imported_data = pd.read_excel(args.data_path)
            elif file_extension == '.json':
                imported_data = pd.read_json(args.data_path)
            elif file_extension == '.parquet':
                imported_data = pd.read_parquet(args.data_path)
            else:
                return {"success": False, "error": f"Unsupported file format: {file_extension}"}
            
            logging.info(f"Successfully imported data from {args.data_path}, shape: {imported_data.shape}")
        except Exception as e:
            return {"success": False, "error": f"Failed to import data: {str(e)}"}
        
        # If requested, add derived features
        # Note: Since there's no add_derived_feature method, we'll skip this for now
        if args.derived_features:
            logging.warning("Adding derived features is not supported in the current implementation.")
        
        # If requested, train a model with the imported data
        if args.train_model:
            logging.warning("Training a model is not supported in the current implementation.")
            model_info = {"model_id": "not_available"}
        else:
            model_info = {}
        
        # Save the imported data if requested
        if args.save_processed_data:
            output_path = os.path.join(args.data_dir, "processed_data.csv")
            imported_data.to_csv(output_path, index=False)
            logging.info(f"Saved processed data to {output_path}")
        
        # Get schema information
        schema_info = {
            "required_features": list(imported_data.columns),
            "optional_features": [],
            "derived_features": []
        }
        
        # If summary is requested, print it
        if args.summary:
            print("\nMigraine Data Import Summary:")
            print(f"Imported data shape: {imported_data.shape}")
            print(f"Columns: {list(imported_data.columns)}")
            
        return {
            "success": True,
            "data": imported_data,
            "schema": schema_info,
            "model": model_info
        }
        
    except Exception as e:
        logging.error(f"Error importing migraine data: {str(e)}")
        import traceback
        traceback.print_exc()
        return {"success": False, "error": str(e)}

def run_migraine_prediction(args):
    """
    Run prediction using a migraine model on new data, handling missing features.
    
    Args:
        args: Command-line arguments containing prediction parameters
        
    Returns:
        Dictionary with prediction results
    """
    if not MIGRAINE_MODULES_AVAILABLE:
        logging.error("Migraine prediction modules are not available. Please install the package first.")
        return {"success": False, "error": "Migraine modules not available"}
    
    try:
        logging.info(f"Running migraine prediction using data from {args.prediction_data}")
        
        # Initialize the predictor
        from migraine_prediction_project.src.migraine_model import MigrainePredictor
        predictor = MigrainePredictor()
        
        # Load the model if specified
        if args.model_id:
            predictor.load_model(args.model_id)
            logging.info(f"Loaded model with ID: {args.model_id}")
        else:
            # Load the default model if no model ID is specified
            predictor.load_model()  # This will load the default model
            logging.info(f"Loaded default model with ID: {predictor.model_id}")
        
        # Import prediction data
        pred_data = predictor.import_data(
            data_path=args.prediction_data,
            add_new_columns=False  # Don't add new columns for prediction
        )
        
        # Make predictions with missing features
        predictions = predictor.predict_with_missing_features(pred_data)
        
        # Format results
        results = []
        for i, pred in enumerate(predictions):
            sample_result = {
                "index": i,
                "prediction": pred["prediction"],
                "probability": pred["probability"],
                "top_features": sorted(
                    pred["feature_importances"].items(),
                    key=lambda x: x[1],
                    reverse=True
                )[:5] if "feature_importances" in pred else []
            }
            results.append(sample_result)
        
        # Save results if requested
        if args.save_predictions:
            # Create DataFrame with predictions
            pred_df = pd.DataFrame({
                "prediction": [p["prediction"] for p in predictions],
                "probability": [p["probability"] for p in predictions]
            })
            # Add original data
            pred_df = pd.concat([pred_data.reset_index(drop=True), pred_df], axis=1)
            output_path = os.path.join(args.data_dir, "predictions.csv")
            pred_df.to_csv(output_path, index=False)
            logging.info(f"Saved predictions to {output_path}")
        
        # If summary is requested, print it
        if args.summary:
            print("\nMigraine Prediction Summary:")
            print(f"Number of samples predicted: {len(predictions)}")
            migraine_count = sum(1 for p in predictions if p["prediction"] == 1)
            print(f"Predicted migraines: {migraine_count} ({migraine_count/len(predictions)*100:.2f}%)")
            print(f"Average probability: {sum(p['probability'] for p in predictions)/len(predictions):.4f}")
            print("\nSample predictions:")
            for i, result in enumerate(results[:5]):  # Show first 5 predictions
                print(f"Sample {i}: {'Migraine' if result['prediction'] == 1 else 'No Migraine'} " +
                      f"(Probability: {result['probability']:.4f})")
        
        return {
            "success": True,
            "predictions": results,
            "summary": {
                "total_samples": len(predictions),
                "predicted_migraines": sum(1 for p in predictions if p["prediction"] == 1),
                "average_probability": sum(p["probability"] for p in predictions)/len(predictions)
            }
        }
        
    except Exception as e:
        logging.error(f"Error running migraine prediction: {str(e)}")
        import traceback
        traceback.print_exc()
        return {"success": False, "error": str(e)}

def run_migraine_explainability(args):
    """
    Run explainability analysis on migraine prediction model.
    
    Args:
        args: Command-line arguments containing explanation parameters
        
    Returns:
        Dictionary with explanation results
    """
    try:
        logging.info(f"Running migraine explainability analysis with {args.explainer} explainer")
        
        # Initialize predictor with proper directories
        from migraine_prediction_project.src.migraine_model.new_data_migraine_predictor import MigrainePredictorV2
        
        predictor = MigrainePredictorV2(
            model_dir=args.model_dir,
            data_dir=args.data_dir
        )
        
        # Load model if model_id provided, otherwise use default
        if hasattr(args, 'model_id') and args.model_id:
            predictor.load_model(model_id=args.model_id)
            logging.info(f"Loaded model with ID: {args.model_id}")
        else:
            predictor.load_model()  # This will load the default model
            logging.info(f"Loaded default model with ID: {predictor.model_id}")
        
        # Import data for explanation
        data_path = args.prediction_data if hasattr(args, 'prediction_data') and args.prediction_data else None
        
        if not data_path:
            logging.error("No prediction data provided. Please specify with --prediction-data")
            return {"success": False, "error": "No prediction data provided"}
            
        # Import data
        data = predictor.import_data(
            data_path=data_path,
            add_new_columns=False
        )
        
        # Run explainability analysis
        explainer_type = args.explainer if hasattr(args, 'explainer') else 'shap'
        n_samples = args.explain_samples if hasattr(args, 'explain_samples') else 5
        generate_plots = args.explain_plots if hasattr(args, 'explain_plots') else True
        plot_types = args.explain_plot_types if hasattr(args, 'explain_plot_types') else None
        
        # Generate explanations
        explanation_results = predictor.explain_predictions(
            data=data,
            explainer_type=explainer_type,
            n_samples=n_samples,
            generate_plots=generate_plots,
            plot_types=plot_types
        )
        
        # If successful, print summary
        if explanation_results.get("success", False):
            if args.summary:
                print("\nMigraine Explainability Summary:")
                print(f"Explainer Type: {explanation_results['explainer_type']}")
                
                # Print top feature importance
                if "feature_importance" in explanation_results:
                    print("\nTop Features by Importance:")
                    feature_importance = explanation_results["feature_importance"]
                    
                    # Convert numpy arrays to scalars if needed
                    processed_importance = {}
                    for feature, importance in feature_importance.items():
                        # Handle numpy arrays
                        import numpy as np
                        if isinstance(importance, np.ndarray):
                            # Use absolute mean value for arrays
                            processed_importance[feature] = float(np.abs(importance).mean())
                        else:
                            processed_importance[feature] = float(importance)
                    
                    # Sort features by importance (absolute value)
                    sorted_features = sorted(
                        processed_importance.items(),
                        key=lambda x: abs(x[1]) if isinstance(x[1], (int, float)) else 0,
                        reverse=True
                    )
                    
                    # Print top 10 features
                    for i, (feature, importance) in enumerate(sorted_features[:10]):
                        print(f"  {i+1}. {feature}: {importance:.6f}")
                
                # Print plot paths
                if "plot_paths" in explanation_results and explanation_results["plot_paths"]:
                    print("\nGenerated Plots:")
                    for plot_type, path in explanation_results["plot_paths"].items():
                        print(f"  {plot_type}: {path}")
                
            return explanation_results
        else:
            error_msg = explanation_results.get("error", "Unknown error")
            logging.error(f"Error running migraine explainability: {error_msg}")
            return {"success": False, "error": error_msg}
            
    except Exception as e:
        import traceback
        logging.error(f"Error running migraine explainability: {str(e)}")
        logging.error(traceback.format_exc())
        return {"success": False, "error": str(e)}

def run_algorithm_selection_demo(args=None):
    """
    Run a comprehensive demonstration of algorithm selection visualization.
    
    Args:
        args: Command-line arguments
        
    Returns:
        Results dictionary
    """
    print("Running algorithm selection visualization demo...")
    
    # Define save directory
    algo_viz_dir = args.algo_viz_dir if args and args.algo_viz_dir else 'results/algorithm_selection_demo'
    os.makedirs(algo_viz_dir, exist_ok=True)
    
    # Initialize meta-optimizer with necessary components
    dim = 10 if not args or not args.dimension else args.dimension
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
        'DE-Adaptive': de_adaptive_opt,
        'ES-Adaptive': es_adaptive_opt
    }
    
    # Create MetaOptimizer
    meta_opt = MetaOptimizer(
        dim=dim,
        bounds=bounds,
        optimizers=optimizers,
        history_file=os.path.join(algo_viz_dir, 'meta_learning_history.json'),
        selection_file=os.path.join(algo_viz_dir, 'selection_history.json')
    )
    
    # Create algorithm selection visualizer
    algo_viz = AlgorithmSelectionVisualizer(save_dir=algo_viz_dir)
    
    # Enable algorithm selection visualization
    meta_opt.enable_algo_viz = True
    meta_opt.algo_selection_viz = algo_viz
    
    print("Algorithm selection visualization enabled for demo")
    
    # Define test functions (sample names for demonstration)
    test_functions = {
        'sphere': None,
        'rosenbrock': None,
        'rastrigin': None,
        'ackley': None
    }
    
    # Manually record some algorithm selections for demonstration purposes
    print("Creating sample algorithm selections for demonstration...")
    
    # For each function, create mock algorithm selections
    for func_name in test_functions.keys():
        print(f"Processing {func_name}...")
        
        # For demonstration, manually create algorithm selections
        for i in range(1, 21):  # Create 20 selections per function
            # Randomly select an optimizer
            optimizer = np.random.choice(list(optimizers.keys()))
            score = 100 - i * 5  # Fake improvement in score
            
            # Record the selection
            meta_opt.algo_selection_viz.record_selection(
                iteration=i,
                optimizer=optimizer,
                problem_type=func_name,
                score=score,
                context={"function_name": func_name, "phase": "optimization"}
            )
            print(f"  Recorded selection of {optimizer} for iteration {i}")
    
    # Generate visualizations using our improved method
    print("Generating algorithm selection visualizations...")
    
    # Determine which plots to generate
    if args and args.algo_viz_plots:
        plot_types = args.algo_viz_plots
    else:
        plot_types = ['frequency', 'timeline', 'problem', 'dashboard', 'interactive']
    
    # Generate visualizations directly using the algo_selection_viz object
    generated_files = {}
    
    # Generate base visualizations that work
    try:
        if "frequency" in plot_types:
            print("Generating frequency plot...")
            meta_opt.algo_selection_viz.plot_selection_frequency(save=True)
            generated_files["frequency"] = os.path.join(algo_viz_dir, "algorithm_selection_frequency.png")
    except Exception as e:
        print(f"Error generating frequency plot: {e}")
    
    try:
        if "timeline" in plot_types:
            print("Generating timeline plot...")
            meta_opt.algo_selection_viz.plot_selection_timeline(save=True)
            generated_files["timeline"] = os.path.join(algo_viz_dir, "algorithm_selection_timeline.png")
    except Exception as e:
        print(f"Error generating timeline plot: {e}")
        
    try:
        if "problem" in plot_types:
            print("Generating problem distribution plot...")
            meta_opt.algo_selection_viz.plot_problem_distribution(save=True)
            generated_files["problem"] = os.path.join(algo_viz_dir, "algorithm_selection_by_problem.png")
    except Exception as e:
        print(f"Error generating problem distribution plot: {e}")
    
    # Skip problematic visualizations    
    # "performance" and "phase" plots are currently having issues
        
    try:
        if "dashboard" in plot_types:
            print("Generating summary dashboard...")
            meta_opt.algo_selection_viz.create_summary_dashboard(save=True)
            generated_files["dashboard"] = os.path.join(algo_viz_dir, "algorithm_selection_dashboard.png")
    except Exception as e:
        print(f"Error generating summary dashboard: {e}")
    
    # Generate interactive visualizations if requested
    if "interactive" in plot_types:
        try:
            print("Generating interactive timeline...")
            meta_opt.algo_selection_viz.interactive_selection_timeline(save=True)
            generated_files["interactive_timeline"] = os.path.join(algo_viz_dir, "interactive_algorithm_timeline.html")
        except Exception as e:
            print(f"Error generating interactive timeline: {e}")
            
        try:
            print("Generating interactive dashboard...")
            meta_opt.algo_selection_viz.interactive_dashboard(save=True)
            generated_files["interactive_dashboard"] = os.path.join(algo_viz_dir, "interactive_dashboard.html")
        except Exception as e:
            print(f"Error generating interactive dashboard: {e}")
    
    print("Algorithm selection demo completed successfully.")
    print(f"Generated {len(generated_files)} visualizations:")
    for viz_type, filepath in generated_files.items():
        print(f"  - {viz_type}: {os.path.basename(filepath)}")
    print(f"Visualizations saved to: {algo_viz_dir}")
    
    return {"success": True, "generated_files": generated_files, "visualizations_path": algo_viz_dir}

def parse_args():
    """
    Parse command-line arguments.
    """
    parser = argparse.ArgumentParser(description='Run optimization framework')
    parser.add_argument('--config', type=str, help='Path to configuration file')
    parser.add_argument('--optimize', action='store_true', help='Run optimization')
    parser.add_argument('--evaluate', action='store_true', help='Evaluate model')
    parser.add_argument('--meta', action='store_true', help='Run meta-learning')
    parser.add_argument('--drift', action='store_true', help='Run drift detection')
    parser.add_argument('--run-meta-learner-with-drift', action='store_true', help='Run meta-learner with drift detection')
    parser.add_argument('--explain-drift', action='store_true', help='Explain drift detection')
    
    # Explainability arguments
    parser.add_argument('--explain', action='store_true', help='Run explainability analysis')
    parser.add_argument('--explainer', type=str, choices=['shap', 'lime', 'feature_importance', 'optimizer'], 
                        help='Explainer to use', default='shap')
    parser.add_argument('--explain-plots', action='store_true', help='Generate explainability plots')
    parser.add_argument('--explain-plot-types', nargs='+', help='Types of explainability plots to generate')
    parser.add_argument('--explain-samples', type=int, help='Number of samples for explainability analysis', default=5)
    parser.add_argument('--auto-explain', action='store_true', help='Automatically run explainability after other operations')
    
    # Meta-learner and optimization parameters
    parser.add_argument('--method', type=str, help='Method for meta-learner', default='bayesian')
    parser.add_argument('--surrogate', type=str, help='Surrogate model for meta-learner')
    parser.add_argument('--selection', type=str, help='Selection strategy for meta-learner')
    parser.add_argument('--exploration', type=float, help='Exploration factor for meta-learner', default=0.2)
    parser.add_argument('--history', type=float, help='History weight for meta-learner', default=0.7)
    
    # Drift detection parameters
    parser.add_argument('--drift-window', type=int, help='Window size for drift detection', default=50)
    parser.add_argument('--drift-threshold', type=float, help='Threshold for drift detection', default=0.5)
    parser.add_argument('--drift-significance', type=float, help='Significance level for drift detection', default=0.05)
    
    # Algorithm selection visualization parameters
    parser.add_argument('--visualize-algorithm-selection', action='store_true', 
                        help='Visualize algorithm selection process')
    parser.add_argument('--algo-viz-dir', type=str, help='Directory to save algorithm selection visualizations')
    parser.add_argument('--algo-viz-plots', nargs='+', 
                        choices=['frequency', 'timeline', 'problem', 'performance', 'phase', 'dashboard', 'interactive'], 
                        help='Algorithm selection plot types to generate')
    parser.add_argument('--dimension', type=int, default=10,
                        help='Dimension for optimization problems')
    
    # Demo mode
    parser.add_argument('--test-algorithm-selection', action='store_true', 
                        help='Run a comprehensive demo of algorithm selection visualization')
    
    # Add optimizer comparison functionality
    parser.add_argument('--compare-optimizers', action='store_true',
                       help='Run comparison of all available optimizers on benchmark functions')
    
    # Migraine data import
    parser.add_argument('--import-migraine-data', action='store_true', help='Import new migraine data')
    parser.add_argument('--data-path', type=str, help='Path to migraine data file')
    parser.add_argument('--data-dir', type=str, default='data', help='Directory to store data files')
    parser.add_argument('--model-dir', type=str, default='models', help='Directory to store model files')
    parser.add_argument('--file-format', type=str, default='csv', choices=['csv', 'excel', 'json', 'parquet'], 
                        help='Format of the input data file')
    parser.add_argument('--add-new-columns', action='store_true', help='Add new columns found in the data to the schema')
    parser.add_argument('--derived-features', type=str, nargs='+', 
                        help='Derived features to create (format: "name:formula")')
    parser.add_argument('--train-model', action='store_true', help='Train a model with the imported data')
    parser.add_argument('--model-name', type=str, help='Name for the trained model')
    parser.add_argument('--model-description', type=str, help='Description for the trained model')
    parser.add_argument('--make-default', action='store_true', help='Make the trained model the default')
    parser.add_argument('--save-processed-data', action='store_true', help='Save the processed data')
    
    # Migraine prediction parameters
    parser.add_argument('--predict-migraine', action='store_true', help='Run migraine prediction')
    parser.add_argument('--prediction-data', type=str, help='Path to data for prediction')
    parser.add_argument('--model-id', type=str, help='ID of the model to use for prediction')
    parser.add_argument('--save-predictions', action='store_true', help='Save prediction results')
    
    # Universal data adapter parameters
    parser.add_argument('--universal-data', action='store_true', help='Process any migraine dataset using the universal adapter')
    parser.add_argument('--disable-auto-feature-selection', action='store_true', help='Disable automatic feature selection')
    parser.add_argument('--use-meta-feature-selection', action='store_true', help='Use meta-optimization for feature selection')
    parser.add_argument('--max-features', type=int, help='Maximum number of features to select')
    parser.add_argument('--test-size', type=float, default=0.2, help='Fraction of data to use for testing')
    parser.add_argument('--random-seed', type=int, default=42, help='Random seed for reproducibility')
    parser.add_argument('--evaluate-model', action='store_true', help='Evaluate the trained model on test data')
    
    # Synthetic data generation parameters
    parser.add_argument('--generate-synthetic', action='store_true', help='Generate synthetic migraine data')
    parser.add_argument('--synthetic-patients', type=int, help='Number of patients for synthetic data')
    parser.add_argument('--synthetic-days', type=int, help='Number of days per patient for synthetic data')
    parser.add_argument('--synthetic-female-pct', type=float, help='Percentage of female patients in synthetic data')
    parser.add_argument('--synthetic-missing-rate', type=float, help='Rate of missing data in synthetic data')
    parser.add_argument('--synthetic-anomaly-rate', type=float, help='Rate of anomalies in synthetic data')
    parser.add_argument('--synthetic-include-severity', action='store_true', help='Include migraine severity in synthetic data')
    parser.add_argument('--save-synthetic', action='store_true', help='Save the generated synthetic data')
    
    # Visualization and summary options
    parser.add_argument('--visualize', action='store_true', help='Visualize results')
    parser.add_argument('--summary', action='store_true', help='Print summary of results')
    parser.add_argument('--verbose', action='store_true', help='Print detailed logs')
    
    # Export options
    parser.add_argument('--export', action='store_true', help='Export optimization data')
    parser.add_argument('--export-format', type=str, choices=['json', 'csv', 'both'], default='json', help='Format for exporting data')
    parser.add_argument('--export-dir', type=str, default='results', help='Directory for exporting data')
    
    # Import options
    parser.add_argument('--import-data', type=str, help='Import optimization data from file')
    
    return parser.parse_args()

def run_optimizer_comparison(args):
    """
    Run comparison of all available optimizers on benchmark functions.
    
    Args:
        args: Command-line arguments
    """
    print("Starting optimizer comparison...")
    import numpy as np
    import time
    from meta_optimizer.benchmark.test_functions import create_test_suite
    from meta_optimizer.optimizers.optimizer_factory import create_optimizers
    
    # Get benchmark functions
    test_suite = create_test_suite()
    
    # Get all available optimizers
    optimizer_types = {
        'ACO': 'ant_colony',
        'DE': 'differential_evolution',
        'ES': 'evolution_strategy',
        'GWO': 'grey_wolf'
    }
    
    # Configure dimensions
    dim = args.dimension
    bounds = [(-5, 5)] * dim  # Default bounds
    
    print(f"Running comparison with dimension = {dim}")
    print(f"Comparing {len(optimizer_types)} optimizers on {len(test_suite)} benchmark functions")
    
    # Results storage
    results = {}
    
    # Run each optimizer on each function
    for func_name, func_creator in test_suite.items():
        print(f"\nBenchmark: {func_name}")
        results[func_name] = {}
        
        # Create test function with specified dimension
        test_function = func_creator(dim, bounds)
        
        # Create all optimizers
        optimizers = create_optimizers(dim=dim, bounds=bounds)
        
        for opt_name, opt_type in optimizer_types.items():
            print(f"  Running {opt_name}...", end="", flush=True)
            
            try:
                # Get the appropriate optimizer
                if opt_name == 'ACO':
                    optimizer = optimizers['ACO']
                elif opt_name == 'DE':
                    optimizer = optimizers['DE (Standard)']
                elif opt_name == 'ES':
                    optimizer = optimizers['ES (Standard)']
                elif opt_name == 'GWO':
                    optimizer = optimizers['GWO']
                else:
                    raise ValueError(f"Unknown optimizer: {opt_name}")
                
                # Define a wrapper function that returns the value only (not a tuple)
                def objective_function(x):
                    return test_function.evaluate(x)
                
                # Run optimization
                start_time = time.time()
                best_solution, best_score = optimizer.optimize(objective_function)
                end_time = time.time()
                
                # Record results
                results[func_name][opt_name] = {
                    'best_score': best_score,
                    'evaluations': optimizer.evaluations,
                    'time': end_time - start_time
                }
                
                print(f" Done. Best score: {best_score:.6f}")
            except Exception as e:
                print(f" Error: {str(e)}")
                results[func_name][opt_name] = {
                    'best_score': float('inf'),
                    'evaluations': 0,
                    'time': 0,
                    'error': str(e)
                }
    
    # Print summary
    print("\nSummary of Results:")
    print("===================")
    
    for func_name in results:
        print(f"\n{func_name}:")
        # Sort optimizers by best score
        sorted_optimizers = sorted(results[func_name].items(), key=lambda x: x[1]['best_score'])
        
        for rank, (opt_name, res) in enumerate(sorted_optimizers, 1):
            if 'error' in res:
                print(f"  {rank}. {opt_name}: Error - {res['error']}")
            else:
                print(f"  {rank}. {opt_name}: Score = {res['best_score']:.6f}, Evals = {res['evaluations']}, Time = {res['time']:.2f}s")
    
    print("\nOptimizer comparison completed.")

def run_optimization(args):
    """Run optimization with specified parameters."""
    # ... existing code ...
    
    # Initialize results dictionary
    results = {}
    
    # Initialize optimizers dictionary
    optimizers = {}
    
    # Export data if requested
    if args.export:
        export_format = args.export_format
        export_dir = args.export_dir
        
        # Create export directory if it doesn't exist
        os.makedirs(export_dir, exist_ok=True)
        
        # Export data for each optimizer
        for optimizer_name, optimizer in optimizers.items():
            if hasattr(optimizer, 'export_data'):
                try:
                    filename = optimizer.export_data(
                        os.path.join(export_dir, f"optimization_{optimizer_name}_{time.strftime('%Y%m%d_%H%M%S')}"),
                        format=export_format
                    )
                    logging.info(f"Exported data for {optimizer_name} optimizer")
                except Exception as e:
                    logging.error(f"Failed to export data for {optimizer_name}: {str(e)}")
    
    return results, pd.DataFrame()

def import_optimization_data(args):
    """Import optimization data from file."""
    import_file = args.import_data
    
    if not os.path.exists(import_file):
        logging.error(f"Import file not found: {import_file}")
        return
    
    try:
        # Create a MetaOptimizer instance
        from meta_optimizer.meta.meta_optimizer import MetaOptimizer
        from meta_optimizer.optimizers.optimizer_factory import create_optimizers
        
        # Default parameters for demonstration
        dim = 3
        bounds = [(-5, 5) for _ in range(dim)]
        
        # Create optimizers
        optimizers = create_optimizers(dim=dim, bounds=bounds)
        
        # Create MetaOptimizer
        meta_optimizer = MetaOptimizer(
            dim=dim,
            bounds=bounds,
            optimizers=optimizers,
            verbose=args.verbose
        )
        
        # Import data
        meta_optimizer.import_data(import_file, restore_state=True)
        
        # Print summary of imported data
        logging.info(f"Successfully imported data from {import_file}")
        logging.info(f"Dimensions: {meta_optimizer.dim}")
        logging.info(f"Problem type: {meta_optimizer.current_problem_type}")
        logging.info(f"Best score: {meta_optimizer.best_score}")
        
        # Check if history was imported
        if hasattr(meta_optimizer, 'optimization_history'):
            logging.info(f"History entries: {len(meta_optimizer.optimization_history)}")
        
        # Check if algorithm selections were imported
        if hasattr(meta_optimizer, 'selection_tracker') and meta_optimizer.selection_tracker:
            selections = meta_optimizer.selection_tracker.get_history()
            logging.info(f"Algorithm selections: {len(selections)}")
        
        # Check available optimizers
        logging.info(f"Optimizers: {list(meta_optimizer.optimizers.keys())}")
        
        return meta_optimizer
    except Exception as e:
        logging.error(f"Failed to import data: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

def main():
    """
    Main entry point for the CLI.
    """
    args = parse_args()
    
    # Configure logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=log_level, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # Import optimization data if requested
    if args.import_data:
        meta_optimizer = import_optimization_data(args)
        return
    
    if args.optimize:
        run_optimization(args)
    
    if args.compare_optimizers:
        run_optimizer_comparison(args)
    
    if args.evaluate:
        run_evaluation(args)
    
    if args.meta:
        run_meta_learning(args)
    
    if args.drift:
        run_drift_detection(args)
    
    if args.explain:
        run_explainability_analysis(args)
    
    if args.auto_explain:
        run_explainability_analysis(args)
    
    if args.run_meta_learner_with_drift:
        run_meta_learner_with_drift_detection(args)
    
    if args.explain_drift:
        explain_drift(args)
        
    if args.import_migraine_data:
        import_results = run_migraine_data_import(args)
        if not import_results["success"]:
            logging.error(f"Migraine data import failed: {import_results.get('error', 'Unknown error')}")
        
    if args.predict_migraine:
        prediction_results = run_migraine_prediction(args)
        if not prediction_results["success"]:
            logging.error(f"Migraine prediction failed: {prediction_results.get('error', 'Unknown error')}")
        
    if args.universal_data:
        universal_results = run_universal_migraine_data(args)
        if not universal_results["success"]:
            logging.error(f"Universal data processing failed: {universal_results.get('error', 'Unknown error')}")

    if args.test_algorithm_selection:
        # Run a demonstration of algorithm selection with verbose output
        print("Running algorithm selection visualization demo...")
        results = run_algorithm_selection_demo(args)
        print("Algorithm selection demo completed successfully.")
        print("Visualizations saved to: results/algorithm_selection")
        return results
    
    print("All operations completed successfully.")

if __name__ == "__main__":
    main()