#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Visualization generator for the meta-optimizer framework.
This script generates publication-quality visualizations for the optimization results.
"""

import os
import sys
import time
import logging
import argparse
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from threading import Timer
import signal
import gc
from typing import Dict, List, Tuple, Any, Optional
from datetime import datetime

# Add parent directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from optimizers.base_optimizer import BaseOptimizer
from optimizers.aco import AntColonyOptimizer
from optimizers.gwo import GreyWolfOptimizer
from optimizers.differential_evolution import DifferentialEvolutionOptimizer
from optimizers.evolution_strategy import EvolutionStrategyOptimizer
from meta_optimizer.meta.meta_optimizer import MetaOptimizer

def create_test_functions():
    """Create benchmark test functions."""
    def sphere(x):
        return np.sum(x**2)
    
    def rosenbrock(x):
        return np.sum(100.0 * (x[1:] - x[:-1]**2.0)**2.0 + (1 - x[:-1])**2.0)
    
    def rastrigin(x):
        return 10 * len(x) + np.sum(x**2 - 10 * np.cos(2 * np.pi * x))
    
    def ackley(x):
        n = len(x)
        return -20 * np.exp(-0.2 * np.sqrt(np.sum(x**2) / n)) - \
               np.exp(np.sum(np.cos(2 * np.pi * x)) / n) + 20 + np.e
    
    return {
        'sphere': sphere,
        'rosenbrock': rosenbrock,
        'rastrigin': rastrigin,
        'ackley': ackley
    }

def create_optimizers(dim: int, bounds: List[Tuple[float, float]], 
                     max_evals: int, population_size: int = 30,
                     verbose: bool = False) -> Dict[str, BaseOptimizer]:
    """Create all optimizers with consistent parameters."""
    common_params = {
        'dim': dim,
        'bounds': bounds,
        'population_size': population_size,
        'max_evals': max_evals,
        'verbose': verbose,
        'timeout': None,
        'iteration_timeout': None
    }
    
    base_optimizers = {
        'DE': DifferentialEvolutionOptimizer(**common_params),
        'ES': EvolutionStrategyOptimizer(**common_params),
        'ACO': AntColonyOptimizer(**common_params),
        'GWO': GreyWolfOptimizer(**common_params)
    }
    
    # Create adaptive variants with additional parameters
    adaptive_params = {**common_params, 'adaptive': True}
    adaptive_optimizers = {
        'DE-Adaptive': DifferentialEvolutionOptimizer(**adaptive_params),
        'ES-Adaptive': EvolutionStrategyOptimizer(**adaptive_params)
    }
    
    # Combine all optimizers
    return {**base_optimizers, **adaptive_optimizers}

def generate_convergence_plots(convergence_data: Dict, output_dir: str):
    """Generate convergence plots from the optimization results."""
    if not convergence_data:
        logging.warning("No convergence data available")
        return

    for dim in convergence_data:
        for func_name in convergence_data[dim]:
            plt.figure(figsize=(12, 8))
            
            for opt_name, results in convergence_data[dim][func_name].items():
                # Collect all convergence curves for this optimizer
                all_curves = []
                for run in results:
                    if 'convergence' in run and run['convergence']:
                        all_curves.append(run['convergence'])
                
                if all_curves:
                    # Convert to numpy array for easier manipulation
                    curves = np.array(all_curves)
                    
                    # Calculate mean and std
                    mean_curve = np.mean(curves, axis=0)
                    std_curve = np.std(curves, axis=0)
                    
                    # Plot mean with std shading
                    x = np.arange(len(mean_curve))
                    plt.plot(x, mean_curve, label=f"{opt_name}", linewidth=2)
                    plt.fill_between(x, mean_curve - std_curve, mean_curve + std_curve, alpha=0.2)
            
            plt.xlabel('Iterations')
            plt.ylabel('Best Score')
            plt.title(f'{func_name} (dim={dim})')
            plt.legend()
            plt.yscale('log')
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f'convergence_{func_name}_dim{dim}.png'), dpi=300, bbox_inches='tight')
            plt.close()

def generate_performance_plots(performance_data: Dict, output_dir: str):
    """Generate performance comparison plots."""
    if not performance_data:
        logging.warning("No performance data available")
        return
        
    # Create performance boxplots
    plt.figure(figsize=(15, 10))
    data = []
    labels = []
    optimizers = []
    
    for opt_name in performance_data:
        for func_name in performance_data[opt_name]:
            scores = performance_data[opt_name][func_name]
            data.extend(scores)
            labels.extend([func_name] * len(scores))
            optimizers.extend([opt_name] * len(scores))
    
    df = pd.DataFrame({
        'Score': data,
        'Problem': labels,
        'Optimizer': optimizers
    })
    
    sns.boxplot(x='Optimizer', y='Score', hue='Problem', data=df)
    plt.xticks(rotation=45)
    plt.title('Optimizer Performance Comparison')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'performance_boxplot.png'), dpi=300, bbox_inches='tight')
    plt.close()

def run_benchmark_with_visualization(
    dim: int = 2,
    bounds: List[Tuple[float, float]] = None,
    max_evals: int = 200,
    population_size: int = 30,
    save_path: str = None,
    verbose: bool = True,
    interactive: bool = True
):
    """Run benchmark with visualization.
    
    Args:
        dim: Problem dimensionality
        bounds: Parameter bounds (if None, uses [-5, 5] for each dimension)
        max_evals: Maximum function evaluations
        population_size: Population size for the optimizers
        save_path: Directory to save results (if None, uses default path)
        verbose: Whether to show progress and logging
        interactive: Whether to enable interactive visualization
    """
    # Set default bounds if not provided
    if bounds is None:
        bounds = [(-5, 5)] * dim
        
    # Set up proper paths
    if save_path is None:
        # Use absolute path
        base_dir = "/Users/blair.dupre/Documents/migrineDT/mdt_Test"
        save_path = os.path.join(base_dir, "results", "visualizations", "benchmarks")
    
    # Create output directories
    os.makedirs(save_path, exist_ok=True)
    
    # Create optimizers with proper settings
    optimizers = create_optimizers(
        dim=dim,
        bounds=bounds,
        max_evals=max_evals,
        population_size=population_size,
        verbose=verbose
    )
    
    # Create meta-optimizer with proper initialization
    meta_optimizer = MetaOptimizer(
        dim=dim,
        bounds=bounds,
        optimizers={name: opt for name, opt in optimizers.items() if name not in ['Meta']},
        verbose=verbose,
        n_parallel=2,
        budget_per_iteration=100,
        default_max_evals=max_evals
    )
    
    # Enable visualization with proper paths
    meta_optimizer.enable_live_visualization(
        save_path=save_path,
        max_data_points=1000,
        auto_show=interactive,
        headless=not interactive  # Set headless mode based on interactive flag
    )
    
    # Define test function (Rastrigin)
    def func(x):
        A = 10
        return A * len(x) + sum((xi**2 - A * np.cos(2 * np.pi * xi)) for xi in x)
    
    try:
        # Run optimization
        result = meta_optimizer.optimize(func)
        
        # Save results with timestamp
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        results_path = os.path.join(save_path, f"optimization_results_dim{dim}_{timestamp}.png")
        data_path = os.path.join(save_path, f"optimization_data_dim{dim}_{timestamp}.json")
        
        # Ensure the paths are absolute
        results_path = os.path.abspath(results_path)
        data_path = os.path.abspath(data_path)
        
        meta_optimizer.disable_live_visualization(
            save_results=True,
            results_path=results_path,
            data_path=data_path
        )
        
        return result
        
    except KeyboardInterrupt:
        print("\nOptimization interrupted by user")
        meta_optimizer.disable_live_visualization()
        return None
    except Exception as e:
        print(f"\nError during optimization: {str(e)}")
        meta_optimizer.disable_live_visualization()
        return None

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate optimization visualizations')
    parser.add_argument('--dims', nargs='+', type=int, default=[2, 5],
                      help='Problem dimensions to test')
    parser.add_argument('--noise-levels', nargs='+', type=float, default=[0.0],
                      help='Noise levels to test')
    parser.add_argument('--runs', type=int, default=3,
                      help='Number of runs per configuration')
    parser.add_argument('--max-evals', type=int, default=200,
                      help='Maximum function evaluations per run')
    parser.add_argument('--output-dir', type=str,
                      default="/Users/blair.dupre/Documents/migrineDT/mdt_Test/results/visualizations/benchmarks",
                      help='Output directory for results')
    parser.add_argument('--verbose', action='store_true',
                      help='Enable verbose output')
    parser.add_argument('--interactive', action='store_true',
                      help='Enable interactive visualization')
    
    args = parser.parse_args()
    
    # Create output directory and ensure it's absolute
    args.output_dir = os.path.abspath(args.output_dir)
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Run benchmark for each dimension
    for dim in args.dims:
        result = run_benchmark_with_visualization(
            dim=dim,
            max_evals=args.max_evals,
            save_path=args.output_dir,
            verbose=args.verbose,
            interactive=args.interactive
        )