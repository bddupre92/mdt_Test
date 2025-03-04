"""
Benchmark script specifically for testing the Meta-Optimizer
"""
import numpy as np
import matplotlib.pyplot as plt
from meta.meta_optimizer import MetaOptimizer
from optimizers.aco import AntColonyOptimizer
from optimizers.gwo import GreyWolfOptimizer
from optimizers.de import DifferentialEvolutionOptimizer


def sphere(x):
    """Sphere test function"""
    return np.sum(x**2)


def main():
    """Run a simple benchmark of the Meta-Optimizer"""
    print("Benchmarking Meta-Optimizer...")
    
    # Setup parameters
    dim = 5
    bounds = [(-5, 5)] * dim
    n_runs = 3
    max_evals = 1000
    
    # Create MetaOptimizer with multiple optimizers
    meta_opt = MetaOptimizer(
        dim=dim,
        bounds=bounds,
        optimizers={
            'ACO': AntColonyOptimizer(dim=dim, bounds=bounds),
            'GWO': GreyWolfOptimizer(dim=dim, bounds=bounds),
            'DE': DifferentialEvolutionOptimizer(dim=dim, bounds=bounds)
        },
        verbose=True
    )
    
    # Run multiple benchmarks
    results = []
    
    for i in range(n_runs):
        print(f"\nRun {i+1}/{n_runs}:")
        meta_opt.reset()
        start_time = np.datetime64('now')
        result = meta_opt.run(sphere, max_evals=max_evals)
        end_time = np.datetime64('now')
        
        elapsed = (end_time - start_time) / np.timedelta64(1, 's')
        
        print(f"  Best score: {result['score']:.10f}")
        print(f"  Runtime: {elapsed:.3f} seconds")
        
        results.append(result['score'])
    
    # Show summary
    print("\nSummary Statistics:")
    print(f"  Mean score: {np.mean(results):.10f}")
    print(f"  Min score: {np.min(results):.10f}")
    print(f"  Max score: {np.max(results):.10f}")
    print(f"  Std Dev: {np.std(results):.10f}")


if __name__ == "__main__":
    main()
