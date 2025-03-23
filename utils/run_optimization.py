#!/usr/bin/env python3
"""
run_optimization.py
----------------
Utility for running optimization with history recording.
"""

import os
import sys
import argparse
from pathlib import Path
import time
import numpy as np

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from optimizers.de import DifferentialEvolutionOptimizer
from optimizers.es import EvolutionStrategyOptimizer
from optimizers.aco import AntColonyOptimizer
from optimizers.gwo import GreyWolfOptimizer
from meta_optimizer.meta.meta_optimizer import MetaOptimizer

def create_test_function(name: str = 'rastrigin', dim: int = 2):
    """
    Create a test function.
    
    Args:
        name: Name of the test function
        dim: Dimensionality
        
    Returns:
        Function and bounds
    """
    if name == 'sphere':
        def func(x):
            return np.sum(x**2)
        bounds = [(-5.12, 5.12)] * dim
        
    elif name == 'rosenbrock':
        def func(x):
            return np.sum(100 * (x[1:] - x[:-1]**2)**2 + (1 - x[:-1])**2)
        bounds = [(-2.048, 2.048)] * dim
        
    elif name == 'rastrigin':
        def func(x):
            A = 10
            return A * dim + np.sum(x**2 - A * np.cos(2 * np.pi * x))
        bounds = [(-5.12, 5.12)] * dim
        
    elif name == 'ackley':
        def func(x):
            a, b, c = 20, 0.2, 2 * np.pi
            sum1 = np.sum(x**2)
            sum2 = np.sum(np.cos(c * x))
            term1 = -a * np.exp(-b * np.sqrt(sum1 / dim))
            term2 = -np.exp(sum2 / dim)
            return term1 + term2 + a + np.exp(1)
        bounds = [(-32.768, 32.768)] * dim
        
    else:
        raise ValueError(f"Unknown test function: {name}")
        
    return func, bounds

def run_optimization(function_name: str, dim: int, max_evals: int, output_dir: str, 
                    optimizer_names: list, save_history: bool = True, verbose: bool = True):
    """
    Run optimization and save results.
    
    Args:
        function_name: Name of the test function
        dim: Dimensionality
        max_evals: Maximum function evaluations
        output_dir: Directory to save results
        optimizer_names: List of optimizer names to use
        save_history: Whether to save optimization history
        verbose: Whether to show progress bars
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Create test function
    func, bounds = create_test_function(function_name, dim)
    
    # Create optimizers
    optimizers = {}
    for name in optimizer_names:
        if name == 'DE':
            optimizers[name] = DifferentialEvolutionOptimizer(dim=dim, bounds=bounds, verbose=verbose)
        elif name == 'ES':
            optimizers[name] = EvolutionStrategyOptimizer(dim=dim, bounds=bounds, verbose=verbose)
        elif name == 'ACO':
            optimizers[name] = AntColonyOptimizer(dim=dim, bounds=bounds, verbose=verbose)
        elif name == 'GWO':
            optimizers[name] = GreyWolfOptimizer(dim=dim, bounds=bounds, verbose=verbose)
        elif name == 'DE-Adaptive':
            optimizers[name] = DifferentialEvolutionOptimizer(dim=dim, bounds=bounds, adaptive=True, verbose=verbose)
        elif name == 'ES-Adaptive':
            optimizers[name] = EvolutionStrategyOptimizer(dim=dim, bounds=bounds, adaptive=True, verbose=verbose)
    
    # Create meta-optimizer
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    history_file = os.path.join(output_dir, f"optimization_history_{function_name}_dim{dim}_{timestamp}.json")
    selection_file = os.path.join(output_dir, f"selection_history_{function_name}_dim{dim}_{timestamp}.json")
    
    meta_opt = MetaOptimizer(
        dim=dim,
        bounds=bounds,
        optimizers=optimizers,
        n_parallel=2,
        budget_per_iteration=100,
        default_max_evals=max_evals,
        verbose=verbose,
        history_file=history_file if save_history else None,
        selection_file=selection_file if save_history else None
    )
    
    # Enable visualization
    meta_opt.enable_live_visualization(
        save_path=os.path.join(output_dir, f"visualization_{function_name}_dim{dim}_{timestamp}.png"),
        auto_show=False
    )
    
    # Run optimization
    start_time = time.time()
    result = meta_opt.run(
        objective_func=func, 
        max_evals=max_evals,
        export_data=True,
        export_format='both',
        export_path=os.path.join(output_dir, f"optimization_data_{function_name}_dim{dim}_{timestamp}")
    )
    runtime = time.time() - start_time
    
    # Print results
    print(f"Optimization completed in {runtime:.2f} seconds")
    print(f"Best solution: {result['solution']}")
    print(f"Best score: {result['score']}")
    print(f"Total evaluations: {result['evaluations']}")
    print(f"Data exported to: {result.get('export_path', 'Not exported')}")
    
    return result

def main():
    """Main function for the CLI."""
    parser = argparse.ArgumentParser(description="Run optimization with history recording")
    
    # Optimization parameters
    parser.add_argument("--function", type=str, choices=['sphere', 'rosenbrock', 'rastrigin', 'ackley'], 
                      default='rastrigin', help="Test function to optimize")
    parser.add_argument("--dim", type=int, default=2, help="Dimensionality")
    parser.add_argument("--max-evals", type=int, default=1000, help="Maximum function evaluations")
    
    # Output options
    parser.add_argument("--output-dir", type=str, default="results/optimization", 
                      help="Directory to save results")
    
    # Optimizer options
    parser.add_argument("--optimizers", nargs='+', 
                      default=['DE', 'ES', 'ACO', 'GWO', 'DE-Adaptive', 'ES-Adaptive'],
                      help="Optimizers to use")
    
    # Other options
    parser.add_argument("--no-history", action="store_true", help="Disable history recording")
    parser.add_argument("--verbose", action="store_true", help="Show progress bars")
    
    # Parse arguments
    args = parser.parse_args()
    
    # Run optimization
    run_optimization(
        function_name=args.function,
        dim=args.dim,
        max_evals=args.max_evals,
        output_dir=args.output_dir,
        optimizer_names=args.optimizers,
        save_history=not args.no_history,
        verbose=args.verbose
    )
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 