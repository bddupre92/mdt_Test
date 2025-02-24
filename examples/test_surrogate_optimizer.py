"""
test_surrogate_optimizer.py
-------------------------
Test script for the Surrogate Model-Based Optimizer
"""

import os
import sys
from pathlib import Path

# Add project root to Python path
project_root = str(Path(__file__).parent.parent)
if project_root not in sys.path:
    sys.path.append(project_root)

import numpy as np
from typing import Dict, Any, Optional, List
import matplotlib.pyplot as plt

from optimizers.ml_optimizers.surrogate_optimizer import SurrogateOptimizer
from optimizers.de import DifferentialEvolutionOptimizer
from benchmarking.test_functions import TEST_FUNCTIONS

def sphere(x):
    """Sphere function"""
    return np.sum(x**2)

def rastrigin(x):
    """Rastrigin function"""
    A = 10
    n = len(x)
    return A * n + np.sum(x**2 - A * np.cos(2 * np.pi * x))

def rosenbrock(x):
    """Rosenbrock function"""
    return np.sum(100.0 * (x[1:] - x[:-1]**2)**2 + (1 - x[:-1])**2)

def ackley(x):
    """Ackley function"""
    a = 20
    b = 0.2
    c = 2 * np.pi
    d = len(x)
    sum_sq_term = -a * np.exp(-b * np.sqrt(np.sum(x**2) / d))
    cos_term = -np.exp(np.sum(np.cos(c * x) / d))
    return sum_sq_term + cos_term + a + np.exp(1)

def griewank(x):
    """Griewank function"""
    sum_term = np.sum(x**2) / 4000
    prod_term = np.prod(np.cos(x / np.sqrt(np.arange(1, len(x) + 1))))
    return 1 + sum_term - prod_term

def run_comparison(dim: int = 2,
                  n_runs: int = 10,
                  max_evals: int = 1000,
                  test_functions: Optional[List[str]] = None):
    """
    Run comparison between Surrogate Optimizer and standard DE
    
    Args:
        dim: Problem dimensionality
        n_runs: Number of independent runs
        max_evals: Maximum function evaluations per run
        test_functions: List of test functions to use (default: sphere, rastrigin)
    """
    if test_functions is None:
        test_functions = ['sphere', 'rastrigin', 'rosenbrock', 'ackley', 'griewank']
        
    functions = {
        'sphere': sphere,
        'rastrigin': rastrigin,
        'rosenbrock': rosenbrock,
        'ackley': ackley,
        'griewank': griewank
    }
    
    bounds = [(-100, 100)] * dim
    
    # Create optimizers
    optimizers = {
        'Surrogate': lambda: SurrogateOptimizer(
            dim=dim,
            bounds=bounds,
            pop_size=20,
            n_initial=10,
            exploitation_ratio=0.7
        ),
        'DE': lambda: DifferentialEvolutionOptimizer(
            dim=dim,
            bounds=bounds,
            pop_size=20
        )
    }
    
    # Store results
    results = {func_name: {opt_name: [] for opt_name in optimizers}
              for func_name in test_functions}
    
    # Run optimization
    for func_name in test_functions:
        print(f"\nOptimizing {func_name}")
        objective_func = functions[func_name]
        
        if func_name in ['ackley', 'griewank']:
            bounds = [(-32.768, 32.768)] * dim
        elif func_name == 'rastrigin':
            bounds = [(-5.12, 5.12)] * dim
        else:
            bounds = [(-5, 5)] * dim
        
        for opt_name, opt_factory in optimizers.items():
            print(f"  Using {opt_name}")
            
            for run in range(n_runs):
                optimizer = opt_factory()
                best_solution, best_score = optimizer.optimize(
                    objective_func,
                    max_evals=max_evals,
                    record_history=True
                )
                results[func_name][opt_name].append({
                    'best_score': best_score,
                    'solution': best_solution,
                    'history': optimizer.history,
                    'diversity': optimizer.diversity_history,
                    'params': optimizer.param_history
                })
    
    # Print results
    print("\nResults:")
    for func_name in test_functions:
        print(f"\n{func_name}:")
        for opt_name in optimizers:
            scores = [r['best_score'] for r in results[func_name][opt_name]]
            print(f"  {opt_name}:")
            print(f"    Mean: {np.mean(scores):.2e}")
            print(f"    Std:  {np.std(scores):.2e}")
            print(f"    Best: {np.min(scores):.2e}")
            print(f"    Worst: {np.max(scores):.2e}")
            print(f"    Median: {np.median(scores):.2e}")
    
    return results

if __name__ == '__main__':
    # Run with increased evaluations
    results = run_comparison(dim=2, n_runs=10, max_evals=1000)
