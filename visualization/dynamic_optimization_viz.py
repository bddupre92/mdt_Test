#!/usr/bin/env python3
"""
Dynamic Optimization Visualization Module

This module provides functions to visualize the performance of different optimizers
on dynamic optimization problems with concept drift.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import time
import logging
from typing import Dict, List, Tuple, Callable, Any, Optional

# Set up logger
logger = logging.getLogger(__name__)

def create_dynamic_optimization_visualization(
    results: Dict[str, Any],
    function_name: str,
    drift_type: str,
    save_dir: str = 'results/visualizations',
    show_plot: bool = False
) -> str:
    """
    Create visualization of optimizer performance on dynamic problems.
    
    Parameters:
    -----------
    results : Dict[str, Any]
        Results from dynamic optimization runs
    function_name : str
        Name of the test function
    drift_type : str
        Type of drift ('sudden', 'oscillatory', 'gradual', etc.)
    save_dir : str
        Directory to save visualizations
    show_plot : bool
        Whether to display the plot (in addition to saving it)
    
    Returns:
    --------
    str
        Path to the saved visualization
    """
    import matplotlib.pyplot as plt
    import numpy as np
    
    # Ensure save directory exists
    os.makedirs(save_dir, exist_ok=True)
    
    # Extract optimizers and their performance data
    optimizers = ['DE', 'ES', 'GWO', 'ACO']  # Add or remove as needed
    evaluations = results.get('evaluations', [])
    optimal_values = results.get('optimal_values', [])
    
    # Create figure
    plt.figure(figsize=(12, 8))
    
    # Plot optimal value
    plt.plot(evaluations, optimal_values, 'k--', linewidth=2, label='Optimal Value')
    
    # Plot each optimizer's performance
    markers = {'DE': 'o', 'ES': 's', 'GWO': '^', 'ACO': 'd'}  # Circle, square, triangle, diamond
    colors = {'DE': 'blue', 'ES': 'green', 'GWO': 'red', 'ACO': 'purple'}
    
    for optimizer in optimizers:
        if optimizer in results:
            optimizer_data = results[optimizer]
            if 'best_scores' in optimizer_data and len(optimizer_data['best_scores']) > 0:
                scores = optimizer_data['best_scores']
                # Make sure we have matching x values for these scores
                if 'evaluations' in optimizer_data and len(optimizer_data['evaluations']) == len(scores):
                    # Use the optimizer-specific evaluation points if available
                    opt_evals = optimizer_data['evaluations']
                    plt.plot(opt_evals, scores, marker=markers.get(optimizer, 'o'), 
                             color=colors.get(optimizer, 'blue'), linewidth=1.5,
                             label=f'{optimizer} Best Score')
                else:
                    # If dimensions mismatch, log warning and skip this optimizer
                    logger.warning(f"Dimension mismatch for {optimizer}: evaluations {len(evaluations)}, scores {len(scores)}")
    
    # Customize plot
    drift_description = drift_type.capitalize() + ' Drift'
    plt.title(f'Optimizer Performance on Dynamic {function_name.capitalize()} Function ({drift_description})')
    plt.xlabel('Function Evaluations')
    plt.ylabel('Function Value')
    plt.grid(True)
    plt.legend()
    
    # Save figure
    filename = f'dynamic_optimization_{function_name.lower()}_{drift_type}.png'
    filepath = os.path.join(save_dir, filename)
    plt.savefig(filepath, dpi=150)
    
    if show_plot:
        plt.show()
    else:
        plt.close()
    
    logger.info(f"Dynamic optimization visualization saved to {filepath}")
    return filepath

def run_dynamic_optimization_experiment(
    function_name: str,
    drift_type: str,
    dim: int = 2,
    bounds: Optional[List[Tuple[float, float]]] = None,
    drift_rate: float = 0.1,
    drift_interval: int = 20,
    severity: float = 1.0,
    max_iterations: int = 500,
    reoptimize_interval: int = 50,
    save_dir: str = 'results/visualizations',
    show_plot: bool = False
) -> Dict[str, Any]:
    """
    Run dynamic optimization experiment and create visualization.
    
    Parameters:
    -----------
    function_name : str
        Name of the test function ('ackley', 'rastrigin', etc.)
    drift_type : str
        Type of drift ('sudden', 'oscillatory', 'gradual', etc.)
    dim : int
        Problem dimensionality
    bounds : Optional[List[Tuple[float, float]]]
        Bounds for each dimension (if None, defaults to [(-5, 5)] * dim)
    drift_rate : float
        Rate of drift (0.0 to 1.0)
    drift_interval : int
        Interval between drift events (in function evaluations)
    severity : float
        Severity of drift (0.0 to 1.0)
    max_iterations : int
        Maximum number of iterations
    reoptimize_interval : int
        Re-optimize after this many function evaluations
    save_dir : str
        Directory to save visualizations
    show_plot : bool
        Whether to display the plot (in addition to saving it)
    
    Returns:
    --------
    Dict[str, Any]
        Results of the experiment including visualizations
    """
    try:
        # Import necessary components
        from meta_optimizer.benchmark.dynamic_benchmark import create_dynamic_benchmark
        from meta_optimizer.benchmark.test_functions import create_test_suite
        from meta_optimizer.optimizers.optimizer_factory import create_optimizers
        
        # Create output directory
        os.makedirs(save_dir, exist_ok=True)
        
        # Set bounds if not provided
        if bounds is None:
            bounds = [(-5, 5)] * dim
        
        # Get test functions
        test_suite = create_test_suite()
        
        # Check if function exists
        if function_name not in test_suite:
            logger.error(f"Function '{function_name}' not found in test suite")
            return {"success": False, "error": f"Function '{function_name}' not found"}
        
        # Create the test function instance
        test_func = test_suite[function_name](dim, bounds)
        
        # Create a wrapper function that calls test_func.evaluate
        def func_wrapper(x):
            return test_func.evaluate(x)
        
        # Create dynamic benchmark
        dynamic_benchmark = create_dynamic_benchmark(
            base_function=func_wrapper,
            dim=dim,
            bounds=bounds,
            drift_type=drift_type,
            drift_rate=drift_rate,
            drift_interval=drift_interval,
            severity=severity
        )
        
        # Create optimizers
        optimizers = create_optimizers(dim=dim, bounds=bounds)
        selected_optimizers = {
            'DE': optimizers['DE (Standard)'],
            'ES': optimizers['ES (Standard)'],
            'GWO': optimizers['GWO'],
            'ACO': optimizers['ACO']
        }
        
        # Data structures to store results
        optimal_values = []
        optimizer_tracks = {name: [] for name in selected_optimizers.keys()}
        optimizer_evals = {name: [] for name in selected_optimizers.keys()}
        evaluation_points = []
        current_eval = 0
        
        # Reset the benchmark
        dynamic_benchmark.reset()
        
        # Function to evaluate with tracking
        def evaluate_with_tracking(x):
            nonlocal current_eval
            result = dynamic_benchmark.evaluate(x)
            
            # Record data at regular intervals
            if current_eval % 5 == 0:
                evaluation_points.append(current_eval)
                optimal_values.append(dynamic_benchmark.current_optimal)
            
            current_eval += 1
            return result
        
        # Run the experiment
        logger.info(f"Running dynamic optimization experiment on {function_name} with {drift_type} drift")
        logger.info(f"Dimension: {dim}, Drift rate: {drift_rate}, Drift interval: {drift_interval}")
        
        for iteration in range(max_iterations // reoptimize_interval):
            logger.info(f"Iteration {iteration+1}/{max_iterations // reoptimize_interval}")
            
            # Run each optimizer for a short period
            for name, optimizer in selected_optimizers.items():
                logger.info(f"  Running {name}...")
                
                # Reset evaluation counter for this run
                optimizer.evaluations = 0
                
                # Run optimization for a short period
                start_time = time.time()
                best_solution, best_score = optimizer.optimize(
                    evaluate_with_tracking, 
                    max_evals=reoptimize_interval
                )
                end_time = time.time()
                
                # Record the best solution found
                optimizer_tracks[name].append((current_eval, best_score))
                
                logger.info(f"    Best score: {best_score:.6f}, Current optimal: {dynamic_benchmark.current_optimal:.6f}")
                logger.info(f"    Time: {end_time - start_time:.2f}s, Evaluations: {optimizer.evaluations}")
        
        # Prepare results for visualization
        results = {
            'evaluations': evaluation_points,
            'optimal_values': optimal_values
        }
        
        # Add optimizer tracks to results
        for name, track in optimizer_tracks.items():
            if track:
                # Extract evaluation points and scores from tracks
                evals, scores = zip(*track)
                results[name] = {
                    'evaluations': evals,
                    'best_scores': scores
                }
        
        # Create visualization
        viz_path = create_dynamic_optimization_visualization(
            results=results,
            function_name=function_name,
            drift_type=drift_type,
            save_dir=save_dir,
            show_plot=show_plot
        )
        
        # Get drift characteristics
        drift_info = dynamic_benchmark.get_drift_characteristics()
        
        # Return results
        return {
            "success": True,
            "visualization_path": viz_path,
            "function_name": function_name,
            "drift_type": drift_type,
            "dimension": dim,
            "drift_rate": drift_rate,
            "drift_interval": drift_interval,
            "severity": severity,
            "drift_characteristics": {
                k: v for k, v in drift_info.items() if k != 'drift_history'
            }
        }
    
    except Exception as e:
        logger.error(f"Error in dynamic optimization experiment: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return {"success": False, "error": str(e)}

if __name__ == "__main__":
    # Example usage when run directly
    logging.basicConfig(level=logging.INFO)
    
    # Run experiment for the Ackley function with sudden drift
    ackley_results = run_dynamic_optimization_experiment(
        function_name="ackley",
        drift_type="sudden",
        dim=2,
        drift_rate=0.1,
        drift_interval=20,
        severity=1.0
    )
    
    # Run experiment for the Rastrigin function with oscillatory drift
    rastrigin_results = run_dynamic_optimization_experiment(
        function_name="rastrigin",
        drift_type="oscillatory",
        dim=2,
        drift_rate=0.1,
        drift_interval=20,
        severity=1.0
    )
    
    print("Dynamic optimization experiments completed.")
    print(f"Ackley visualization: {ackley_results.get('visualization_path', 'Error')}")
    print(f"Rastrigin visualization: {rastrigin_results.get('visualization_path', 'Error')}") 