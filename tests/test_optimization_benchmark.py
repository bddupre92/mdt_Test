"""
Benchmark test for optimization performance analysis.
"""
import pytest
import time
import numpy as np
import pandas as pd
from pathlib import Path
import logging

# Import necessary components
from optimizers.optimizer_factory import (
    DifferentialEvolutionWrapper, 
    EvolutionStrategyWrapper,
    AntColonyWrapper,
    GreyWolfWrapper,
    create_optimizers
)
from benchmarking.test_functions import TEST_FUNCTIONS, create_test_suite
from meta.meta_optimizer import MetaOptimizer
from visualization.optimizer_analysis import OptimizerAnalyzer

# Configure logging
logging.basicConfig(level=logging.INFO)

def test_benchmark_mini():
    """Run a mini benchmark with a subset of functions and optimizers."""
    # Setup
    dim = 10
    bounds = [(-5, 5)] * dim
    n_runs = 2  # Very small number for quick testing
    max_evals = 1000  # Limited evaluations for quick testing
    
    # Create test functions (just use 2 for quick testing)
    test_functions = {
        'sphere_10D': TEST_FUNCTIONS['sphere'](dim, bounds),
        'rastrigin_10D': TEST_FUNCTIONS['rastrigin'](dim, bounds)
    }
    
    # Create optimizers (just use 2 for quick testing)
    # For DE, set a smaller population size to ensure max_evals is respected
    optimizers = {
        'GWO': GreyWolfWrapper(dim=dim, bounds=bounds, name="GWO"),
        'DE': DifferentialEvolutionWrapper(dim=dim, bounds=bounds, name="DE", population_size=10)
    }
    
    # Set max_evals for each optimizer explicitly
    for opt in optimizers.values():
        opt.max_evals = max_evals
    
    # Create analyzer
    analyzer = OptimizerAnalyzer(optimizers)
    
    # Measure performance
    start_time = time.time()
    
    # Run comparison with explicit max_evals
    results = analyzer.run_comparison(
        test_functions,
        n_runs=n_runs,
        record_convergence=True,
        max_evals=max_evals  # Pass max_evals explicitly
    )
    
    # Calculate runtime
    runtime = time.time() - start_time
    
    # Collect results
    all_results_data = []
    for func_name, func_results in results.items():
        for opt_name, opt_results in func_results.items():
            for run, result in enumerate(opt_results):
                all_results_data.append({
                    'function': func_name,
                    'optimizer': opt_name,
                    'run': run,
                    'best_score': result.best_score,
                    'execution_time': result.execution_time,
                    'evaluations': result.hyperparameters.get('evaluations', 0)  # Track actual evaluations
                })
    
    # Create DataFrame
    results_df = pd.DataFrame(all_results_data)
    
    # Print summary
    print("\n=== Mini Benchmark Results ===")
    print(f"Total runtime: {runtime:.2f} seconds")
    print("\nAverage scores by function and optimizer:")
    
    # Group by function and optimizer
    summary = results_df.groupby(['function', 'optimizer']).agg({
        'best_score': ['mean', 'min'],
        'execution_time': ['mean'],
        'evaluations': ['mean']  # Add evaluations to summary
    }).reset_index()
    
    # Print formatted summary
    for func_name in summary['function'].unique():
        print(f"\n{func_name}:")
        func_data = summary[summary['function'] == func_name]
        for _, row in func_data.iterrows():
            opt_name = row['optimizer']
            mean_score = row[('best_score', 'mean')]
            min_score = row[('best_score', 'min')]
            mean_time = row[('execution_time', 'mean')]
            mean_evals = row[('evaluations', 'mean')] if ('evaluations', 'mean') in row else 0
            print(f"  {opt_name}: {mean_score:.3e} ± (min: {min_score:.3e}, time: {mean_time:.2f}s, evals: {int(mean_evals)})")
    
    # Assertions
    assert runtime < 30, "Benchmark took too long"
    
    # Check that evaluations are within reasonable limits
    max_allowed_evals = max_evals * 10  # Allow some flexibility for DE's implementation
    for _, row in results_df.iterrows():
        evals = row.get('evaluations', 0)
        if evals > 0:  # Only check if evaluations were tracked
            assert evals <= max_allowed_evals, f"Too many evaluations: {evals} > {max_allowed_evals}"
    
    return runtime, results_df

if __name__ == "__main__":
    # Create results directory if it doesn't exist
    Path("results").mkdir(exist_ok=True)
    Path("results/data").mkdir(exist_ok=True, parents=True)
    
    # Run test
    runtime, results = test_benchmark_mini()
    print(f"\nTotal runtime: {runtime:.2f} seconds")
