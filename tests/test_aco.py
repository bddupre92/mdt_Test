#!/usr/bin/env python3
# Test script for ACO optimizer

import numpy as np
import sys
import os
from pathlib import Path

# Add the parent directory to the path if needed
parent_dir = str(Path(__file__).parent.parent.absolute())
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from optimizers.aco import AntColonyOptimizer

def sphere(x):
    """Simple sphere function for testing"""
    return np.sum(x**2)

def main():
    # Problem parameters
    dim = 5
    bounds = [(-5, 5) for _ in range(dim)]
    
    # Create ACO optimizer
    aco = AntColonyOptimizer(
        dim=dim,
        bounds=bounds,
        population_size=20,
        max_evals=1000,
        num_points=20,
        verbose=True
    )
    
    # Run optimization
    print("Running ACO optimization...")
    best_solution, best_score = aco.optimize(sphere)
    
    # Print results
    print(f"Best solution: {best_solution}")
    print(f"Best score: {best_score}")
    
    # Verify that the pheromone matrix has the correct shape
    print(f"Pheromone matrix shape: {aco.pheromone.shape}")
    print(f"Expected shape: ({dim}, {aco.num_points})")
    
    # Check if the shape is correct
    assert aco.pheromone.shape == (dim, aco.num_points), "Pheromone matrix has incorrect shape!"
    
    print("ACO optimizer test passed!")

if __name__ == "__main__":
    main() 