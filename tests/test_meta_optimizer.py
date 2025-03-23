"""
Quick test script for the Meta-Optimizer
"""
import numpy as np
from meta.meta_optimizer import MetaOptimizer
from optimizers.aco import AntColonyOptimizer
from optimizers.gwo import GreyWolfOptimizer


def main():
    """Test the Meta-Optimizer with a simple function"""
    print("Testing Meta-Optimizer...")
    
    # Define a simple test function (sphere function)
    def sphere(x):
        return np.sum(x**2)
    
    # Create a Meta-Optimizer with a single sub-optimizer
    dim = 5
    bounds = [(-5, 5)] * dim
    
    meta_opt = MetaOptimizer(
        dim=dim,
        bounds=bounds,
        optimizers={
            'ACO': AntColonyOptimizer(dim=dim, bounds=bounds),
            'GWO': GreyWolfOptimizer(dim=dim, bounds=bounds)
        },
        verbose=True
    )
    
    # Run the optimizer
    print("Running optimization...")
    max_evals = 1000
    result = meta_opt.run(sphere, max_evals=max_evals)
    
    # Print results
    print("\nOptimization Results:")
    print(f"Best solution: {result['solution']}")
    print(f"Best score: {result['score']:.10f}")
    print(f"Total evaluations: {result['evaluations']}")
    print(f"Runtime: {result['runtime']:.3f} seconds")
    
    # Check that convergence curve is being recorded
    if 'convergence_curve' in result and result['convergence_curve']:
        print(f"Convergence curve has {len(result['convergence_curve'])} points")
    else:
        print("No convergence curve recorded")
    
    # Test optimizer with multiple runs
    print("\nTesting multiple runs...")
    for i in range(3):
        print(f"\nRun {i+1}/3:")
        meta_opt.reset()  # Reset the optimizer
        result = meta_opt.run(sphere, max_evals=max_evals)
        print(f"Best score: {result['score']:.10f}")


if __name__ == "__main__":
    main()
