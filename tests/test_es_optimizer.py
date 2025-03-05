#!/usr/bin/env python3
"""
Test script for Evolution Strategy Optimizer
"""

import numpy as np
import time
import logging
from typing import List, Tuple
from tqdm.auto import tqdm  # tqdm is already in requirements.txt
from optimizers.es import EvolutionStrategyOptimizer

# Configure logging
logging.basicConfig(level=logging.DEBUG,
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# Define simple objective function (sphere function)
def sphere(x: np.ndarray) -> float:
    """Sphere function - simple test function with global minimum at origin"""
    return np.sum(x**2)

def main():
    # Problem setup
    dim = 10  # Lower dimension for quicker testing
    bounds = [(-5, 5)] * dim
    
    # Create optimizer with conservative settings
    es = EvolutionStrategyOptimizer(
        dim=dim,
        bounds=bounds,
        population_size=30,    # Larger population
        max_evals=3000,        # More evaluations
        timeout=30.0,          # Longer timeout
        initial_step_size=0.5, # Start with a reasonable step size
        adaptive=True,
        verbose=True           # Enable verbose mode for progress bars
    )
    
    print("\n🚀 Testing Evolution Strategy Optimizer 🚀")
    print(f"Dimensions: {dim}")
    print(f"Max evaluations: {es.max_evals}")
    
    # Run optimization
    start_time = time.time()
    best_solution, best_score = es.optimize(sphere)
    end_time = time.time()
    
    # Print results
    print("\n📊 Optimization Results:")
    print(f"✨ Best score: {best_score:.8f}")
    print(f"🎯 Best solution: {best_solution}")
    print(f"🔢 Evaluations used: {es.evaluations}")
    print(f"⏱️ Time taken: {end_time - start_time:.2f} seconds")
    print(f"Convergence at iteration: {es._current_iteration}")
    
    # Verify result is close to the origin (optimal solution)
    distance_to_origin = np.linalg.norm(best_solution)
    print(f"Distance to optimal solution: {distance_to_origin:.8f}")
    
    # Success rate and strategy parameters
    if hasattr(es, 'param_history') and 'success_rate' in es.param_history and es.param_history['success_rate']:
        print(f"Final success rate: {es.param_history['success_rate'][-1]:.4f}")
    if hasattr(es, 'param_history') and 'step_size' in es.param_history and es.param_history['step_size']:
        print(f"Final step size: {es.param_history['step_size'][-1]:.8f}")
    
    if distance_to_origin < 0.1:
        print("TEST PASSED: Solution is close to the optimal point")
    else:
        print("TEST FAILED: Solution is too far from the optimal point")
    
    # Analyze convergence
    if hasattr(es, 'convergence_curve') and es.convergence_curve:
        print("\nConvergence analysis:")
        n_points = min(10, len(es.convergence_curve))
        for i in range(n_points):
            idx = i * (len(es.convergence_curve) // n_points)
            print(f"Iteration {idx}: score={es.convergence_curve[idx]:.8f}")
    
if __name__ == "__main__":
    main()
