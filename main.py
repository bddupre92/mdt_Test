"""
Comprehensive benchmark and comparison of optimization algorithms
including meta-optimization for novel algorithm creation.
"""

import numpy as np
import pandas as pd
import logging
from pathlib import Path
import time
from typing import Dict, Any, List, Tuple, Optional
import seaborn as sns
import argparse
import time
import os

# Set matplotlib backend before importing pyplot
import matplotlib
matplotlib.use('TkAgg')  # Try 'MacOSX' or 'Qt5Agg' if this doesn't work
import matplotlib.pyplot as plt

# Import benchmark functions
from benchmarking.test_functions import TEST_FUNCTIONS, create_test_suite

# Import optimizers
from optimizers.optimizer_factory import (
    DifferentialEvolutionWrapper, 
    EvolutionStrategyWrapper,
    AntColonyWrapper,
    GreyWolfWrapper,
    create_optimizers as factory_create_optimizers
)

# Import meta-optimization components
from meta.meta_optimizer import MetaOptimizer
from meta.meta_learner import MetaLearner

# Import drift detection
from drift_detection.detector import DriftDetector

# Import visualization tools
from visualization.optimizer_analysis import OptimizerAnalyzer
from visualization.live_visualization import LiveOptimizationMonitor

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
    directories = ['results', 'results/plots', 'results/data']
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

# Run benchmark comparison
def run_benchmark_comparison(n_runs: int = 30, max_evals: int = 10000, live_viz: bool = False, save_plots: bool = False):
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
            record_convergence=True
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
def run_meta_learner_with_drift():
    """Run meta-learner with drift detection on synthetic data"""
    from scripts.test_framework import generate_temporal_data
    
    logging.info("Running meta-learner with drift detection")
    
    # Generate synthetic data with drift
    n_samples = 1000
    drift_points = [300, 600]
    X, y, feature_names = generate_temporal_data(
        n_samples=n_samples,
        n_features=5,
        drift_points=drift_points
    )
    
    # Split data
    train_size = 150
    X_init, y_init = X[:train_size], y[:train_size]
    X_stream, y_stream = X[train_size:], y[train_size:]
    
    # Create meta-learner
    meta_learner = MetaLearner(
        optimizers=create_optimizers(5, [(-1, 1)] * 5),
        drift_detector=DriftDetector(window_size=50),
        history_file='results/data/meta_learner_history.json'
    )
    
    # Initial training
    meta_learner.optimize(X_init, y_init, feature_names=feature_names)
    
    # Process streaming data
    predictions = []
    true_values = []
    drift_detections = []
    algorithm_selections = []
    accuracies = []
    window_size = 50
    
    for i in range(len(X_stream)):
        x_i = X_stream[i:i+1]
        y_i = y_stream[i:i+1]
        
        # Make prediction
        pred = meta_learner.predict(x_i)
        predictions.append(pred[0])
        true_values.append(y_i[0])
        
        # Track algorithm selection
        if hasattr(meta_learner, 'current_algorithm'):
            algorithm_selections.append(meta_learner.current_algorithm)
        
        # Update with true label
        meta_learner.update(x_i, y_i)
        
        # Check if drift was detected
        if meta_learner.drift_detector.last_drift is not None:
            drift_detections.append(train_size + i)
        
        # Calculate running accuracy
        if i >= window_size:
            window_acc = np.mean(np.array(predictions[-window_size:]) == np.array(true_values[-window_size:]))
            accuracies.append(window_acc)
    
    # Calculate overall metrics
    accuracy = np.mean(np.array(predictions) == np.array(true_values))
    
    # Plot results
    plt.figure(figsize=(15, 12))
    
    # Plot 1: Predictions vs True Values
    plt.subplot(4, 1, 1)
    plt.plot(true_values, 'b-', label='True Values')
    plt.plot(predictions, 'r-', alpha=0.7, label='Predictions')
    for drift in drift_points:
        plt.axvline(drift - train_size, color='g', linestyle='--', label='True Drift' if drift == drift_points[0] else "")
    for drift in drift_detections:
        plt.axvline(drift - train_size, color='m', linestyle=':', label='Detected Drift' if drift == drift_detections[0] else "")
    plt.title('Predictions vs True Values')
    plt.legend()
    
    # Plot 2: Running Accuracy
    plt.subplot(4, 1, 2)
    plt.plot(range(window_size, len(accuracies) + window_size), accuracies, 'g-')
    plt.axhline(accuracy, color='r', linestyle='--', label=f'Overall Accuracy: {accuracy:.4f}')
    for drift in drift_points:
        plt.axvline(drift - train_size, color='g', linestyle='--')
    for drift in drift_detections:
        plt.axvline(drift - train_size, color='m', linestyle=':')
    plt.ylim(0, 1)
    plt.title('Running Accuracy (Window Size: 50)')
    plt.legend()
    
    # Plot 3: Algorithm Selection
    if algorithm_selections:
        plt.subplot(4, 1, 3)
        unique_algorithms = list(set(algorithm_selections))
        algo_indices = [unique_algorithms.index(algo) for algo in algorithm_selections]
        plt.plot(algo_indices, 'b-')
        for drift in drift_points:
            plt.axvline(drift - train_size, color='g', linestyle='--')
        for drift in drift_detections:
            plt.axvline(drift - train_size, color='m', linestyle=':')
        plt.yticks(range(len(unique_algorithms)), unique_algorithms)
        plt.title('Algorithm Selection Over Time')
    
    # Plot 4: Feature Importance
    plt.subplot(4, 1, 4)
    feature_importance = meta_learner.get_feature_importance()
    plt.bar(feature_names, feature_importance)
    plt.title('Feature Importance')
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.savefig('results/plots/meta_learner_drift_analysis.png')
    
    # Save drift detection metrics
    drift_metrics = {
        'true_drifts': drift_points,
        'detected_drifts': drift_detections,
        'detection_delay': [min([abs(d - td) for td in drift_points]) for d in drift_detections] if drift_detections else [],
        'false_positives': len([d for d in drift_detections if min([abs(d - td) for td in drift_points]) > 50]),
        'false_negatives': len([td for td in drift_points if min([abs(td - d) for d in drift_detections]) > 50]) if drift_detections else len(drift_points)
    }
    
    logging.info(f"Meta-learner accuracy: {accuracy:.4f}")
    logging.info(f"True drift points: {drift_points}")
    logging.info(f"Detected drift points: {drift_detections}")
    logging.info(f"Drift detection metrics: {drift_metrics}")
    
    return {
        'accuracy': accuracy,
        'true_drifts': drift_points,
        'detected_drifts': drift_detections,
        'algorithm_selection': algorithm_selections,
        'feature_importance': dict(zip(feature_names, feature_importance)),
        'drift_metrics': drift_metrics
    }

def main():
    """Main entry point"""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Optimization Framework Benchmarking')
    parser.add_argument('--live-viz', action='store_true', help='Enable live visualization')
    parser.add_argument('--n-runs', type=int, default=5, help='Number of runs per optimizer')
    parser.add_argument('--max-evals', type=int, default=2000, help='Maximum function evaluations')
    parser.add_argument('--save-plots', action='store_true', help='Save visualization plots')
    parser.add_argument('--max-data-points', type=int, default=1000, help='Maximum data points to store per optimizer')
    parser.add_argument('--no-auto-show', action='store_true', help='Disable automatic plot display')
    parser.add_argument('--headless', action='store_true', help='Run in headless mode (no display, save plots only)')
    args = parser.parse_args()
    
    # Setup environment
    setup_environment()
    logging.info("Starting optimization framework demonstration")
    
    # Run benchmark comparison
    logging.info("Running benchmark comparison...")
    benchmark_results, results_df = run_benchmark_comparison(
        n_runs=args.n_runs, 
        max_evals=args.max_evals, 
        live_viz=args.live_viz,
        save_plots=args.save_plots
    )
    
    # Run meta-learner with drift detection
    logging.info("Running meta-learner with drift detection...")
    meta_results = run_meta_learner_with_drift()
    
    # Save combined results
    results = {
        'benchmark': benchmark_results,
        'meta_learner': meta_results
    }
    
    # Generate final report
    logging.info("Generating final report")
    
    # Print summary
    print("\n=== Optimization Framework Results ===")
    print("\nBenchmark Functions Performance:")
    
    # Group by function and optimizer
    summary = results_df.groupby(['function', 'optimizer']).agg({
        'best_score': ['mean', 'std', 'min'],
        'execution_time': ['mean']
    }).reset_index()
    
    # Print formatted summary
    for func_name in summary['function'].unique():
        print(f"\n{func_name}:")
        func_data = summary[summary['function'] == func_name]
        for _, row in func_data.iterrows():
            opt_name = row['optimizer']
            mean_score = row[('best_score', 'mean')]
            std_score = row[('best_score', 'std')]
            min_score = row[('best_score', 'min')]
            mean_time = row[('execution_time', 'mean')]
            print(f"  {opt_name}: {mean_score:.3e} ± {std_score:.3e} (min: {min_score:.3e}, time: {mean_time:.2f}s)")
    
    print("\nMeta-Learner Performance:")
    print(f"  Accuracy: {meta_results['accuracy']:.4f}")
    print(f"  Drift Detection Rate: {len(meta_results['detected_drifts'])}/{len(meta_results['true_drifts'])}")
    
    if 'drift_metrics' in meta_results:
        metrics = meta_results['drift_metrics']
        print(f"  False Positives: {metrics['false_positives']}")
        print(f"  False Negatives: {metrics['false_negatives']}")
        if metrics['detection_delay']:
            print(f"  Average Detection Delay: {np.mean(metrics['detection_delay']):.2f} samples")
    
    if 'feature_importance' in meta_results:
        print("\nFeature Importance:")
        for feature, importance in meta_results['feature_importance'].items():
            print(f"  {feature}: {importance:.4f}")
    
    logging.info("Completed optimization framework demonstration")
    print("\nAll results have been saved to the 'results' directory.")

    # Run optimization
    meta_optimizer = MetaOptimizer(
        dim=10,
        bounds=[(-5, 5)] * 10,
        optimizers=create_optimizers(10, [(-5, 5)] * 10),
        history_file='results/data/meta_history.json'
    )
    
    # Ensure results directory exists
    os.makedirs('results', exist_ok=True)
    
    results = meta_optimizer.optimize(
        objective_func=benchmark_functions['sphere_10D'],
        max_evals=args.max_evals,
        live_viz=args.live_viz,
        save_viz_results=args.save_plots,
        viz_results_path='results/optimization_progress.png',
        viz_data_path='results/optimization_data.csv' if args.save_plots else None,
        max_data_points=args.max_data_points,
        headless=args.headless
    )
    
    # If live visualization is enabled, keep the program running until user interrupts
    if args.live_viz:
        print("\nLive visualization running. Press Ctrl+C to stop.")
        try:
            # Keep the main thread alive to maintain the visualization
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\nVisualization stopped by user.")

if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        logging.error(f"Error in main: {str(e)}", exc_info=True)