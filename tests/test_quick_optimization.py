"""
Quick optimization test to measure performance and completion time.
"""
import pytest
import time
import numpy as np
from pathlib import Path

# Import necessary components
from optimizers.optimizer_factory import (
    DifferentialEvolutionWrapper, 
    GreyWolfWrapper,
    create_optimizers
)
from benchmarking.test_functions import TEST_FUNCTIONS
from meta.meta_optimizer import MetaOptimizer

def test_single_optimizer_performance():
    """Test a single optimizer's performance on a simple function."""
    # Setup
    dim = 10
    bounds = [(-5, 5)] * dim
    
    # Create test function (sphere is simple and fast to evaluate)
    sphere_func = TEST_FUNCTIONS['sphere'](dim, bounds)
    
    # Create optimizer
    optimizer = GreyWolfWrapper(dim=dim, bounds=bounds, name="GWO")
    
    # Measure performance
    start_time = time.time()
    
    # Run optimization with limited evaluations
    solution = optimizer.optimize(sphere_func, max_evals=1000)
    
    # Calculate runtime
    runtime = time.time() - start_time
    
    # The optimizer might return (solution, score) tuple or just solution
    if isinstance(solution, tuple) and len(solution) == 2:
        solution_array, score = solution
    else:
        solution_array = solution
        score = sphere_func(solution_array)
    
    # Print results
    print(f"\nOptimizer: GWO")
    print(f"Runtime: {runtime:.2f} seconds")
    print(f"Score: {score:.6e}")
    print(f"Evaluations: {optimizer.evaluations}")
    
    # Assertions
    assert runtime < 10, "Optimization took too long"
    assert score < 1.0, "Score should be close to zero"
    
def test_meta_optimizer_quick():
    """Test meta-optimizer with minimal settings for quick execution."""
    # Setup
    dim = 10
    bounds = [(-5, 5)] * dim
    
    # Create test function
    sphere_func = TEST_FUNCTIONS['sphere'](dim, bounds)
    
    # Create optimizers with minimal settings
    optimizers = {
        'GWO': GreyWolfWrapper(dim=dim, bounds=bounds, name="GWO"),
        'DE': DifferentialEvolutionWrapper(dim=dim, bounds=bounds, name="DE")
    }
    
    # Create meta-optimizer
    meta_opt = MetaOptimizer(
        dim=dim,
        bounds=bounds,
        optimizers=optimizers,
        n_parallel=2
    )
    
    # Measure performance
    start_time = time.time()
    
    # Run optimization with very limited evaluations
    results = meta_opt.optimize(
        objective_func=sphere_func,
        max_evals=500,  # Very small number for quick testing
        live_viz=False,
        headless=True
    )
    
    # Calculate runtime
    runtime = time.time() - start_time
    
    # Print results
    print(f"\nMeta-Optimizer Quick Test")
    print(f"Runtime: {runtime:.2f} seconds")
    print(f"Best score: {results['best_score']:.6e}")
    print(f"Total evaluations: {results['total_evaluations']}")
    
    # Assertions
    assert runtime < 15, "Meta-optimization took too long"
    assert results['best_score'] < 1.0, "Score should be close to zero"

if __name__ == "__main__":
    # Create results directory if it doesn't exist
    Path("results").mkdir(exist_ok=True)
    
    # Run tests
    test_single_optimizer_performance()
    test_meta_optimizer_quick()
