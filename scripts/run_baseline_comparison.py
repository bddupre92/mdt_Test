#!/usr/bin/env python3
"""
Script to run baseline comparison between Meta Optimizer and SATzilla-inspired algorithm selection.
"""

import os
import numpy as np
import pandas as pd
import argparse
import time
from pathlib import Path

from meta_optimizer.meta.meta_optimizer import MetaOptimizer
from meta_optimizer.optimizers import OptimizerFactory
from baseline_comparison import BaselineComparison, ComparisonVisualizer
from baseline_comparison.benchmark_utils import get_benchmark_function, get_function_bounds


class BenchmarkProblem:
    """Wrapper for benchmark functions with standardized interface."""
    
    def __init__(self, function_name, dimension=10, seed=None):
        """
        Initialize a benchmark problem.
        
        Args:
            function_name: Name of the benchmark function
            dimension: Problem dimensionality
            seed: Random seed for reproducibility
        """
        self.function_name = function_name
        self.dimension = dimension
        self.benchmark_fn = get_benchmark_function(function_name)
        self.seed = seed
        
        # Set random seed if provided
        if seed is not None:
            np.random.seed(seed)
        
        # Set bounds based on the benchmark function
        if function_name in ['ackley', 'rastrigin', 'griewank']:
            self.lower_bounds = np.array([-5.0] * dimension)
            self.upper_bounds = np.array([5.0] * dimension)
        elif function_name == 'rosenbrock':
            self.lower_bounds = np.array([-2.0] * dimension)
            self.upper_bounds = np.array([2.0] * dimension)
        elif function_name == 'sphere':
            self.lower_bounds = np.array([-10.0] * dimension)
            self.upper_bounds = np.array([10.0] * dimension)
        else:
            self.lower_bounds = np.array([-10.0] * dimension)
            self.upper_bounds = np.array([10.0] * dimension)
    
    def evaluate(self, x):
        """Evaluate the benchmark function at a point."""
        return self.benchmark_fn(x)
    
    def __str__(self):
        return f"{self.function_name.capitalize()} function ({self.dimension}D)"


def create_problem_generator(function_names, dimensions, seed=None):
    """
    Create a problem generator function.
    
    Args:
        function_names: List of benchmark function names
        dimensions: List of dimensions to use
        seed: Random seed for reproducibility
    
    Returns:
        Function that generates random benchmark problems
    """
    if seed is not None:
        np.random.seed(seed)
    
    # Track which problems have been generated
    generated_problems = set()
    
    def generator():
        # Generate a problem configuration that hasn't been used yet
        while True:
            function_name = np.random.choice(function_names)
            dimension = np.random.choice(dimensions)
            
            problem_key = f"{function_name}_{dimension}"
            if problem_key not in generated_problems:
                generated_problems.add(problem_key)
                problem_seed = np.random.randint(0, 1000)
                return BenchmarkProblem(function_name, dimension, seed=problem_seed)
    
    return generator


def main():
    """Run baseline comparison between Meta Optimizer and baseline methods."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Run baseline comparison')
    parser.add_argument('--training-problems', type=int, default=50,
                      help='Number of problems for training (default: 50)')
    parser.add_argument('--test-problems', type=int, default=20,
                      help='Number of problems for testing (default: 20)')
    parser.add_argument('--export-dir', type=str, default='results/baseline_comparison',
                      help='Directory to export results (default: results/baseline_comparison)')
    parser.add_argument('--seed', type=int, default=42,
                      help='Random seed for reproducibility (default: 42)')
    parser.add_argument('--verbose', action='store_true',
                      help='Enable verbose output')
    args = parser.parse_args()
    
    # Create export directory
    export_dir = Path(args.export_dir)
    export_dir.mkdir(parents=True, exist_ok=True)
    
    if args.verbose:
        print(f"Using seed: {args.seed}")
        print(f"Exporting results to: {export_dir}")
    
    # Set random seed
    np.random.seed(args.seed)
    
    # Define benchmark problems and dimensions
    function_names = ['ackley', 'rastrigin', 'rosenbrock', 'sphere', 'griewank']
    dimensions = [2, 5, 10, 20]
    
    # Create problem generator
    problem_generator = create_problem_generator(function_names, dimensions, seed=args.seed)
    
    # Define optimization algorithms
    algorithms = ['DE', 'ES', 'ACO', 'GWO']
    
    # Create Meta Optimizer
    if args.verbose:
        print("Initializing Meta Optimizer...")
    
    # Create optimizers using OptimizerFactory
    factory = OptimizerFactory()
    optimizers = {}
    
    for algo in algorithms:
        if args.verbose:
            print(f"  Creating {algo} optimizer...")
        optimizers[algo] = factory.create_optimizer(
            algo,
            dim=dimensions[0],  # Start with first dimension, will adapt to problem
            bounds=[(-5, 5)] * dimensions[0]
        )
    
    # Create Meta Optimizer with optimizers
    dim = dimensions[0]  # Use first dimension as default
    bounds = [(-5, 5)] * dim
    meta_optimizer = MetaOptimizer(
        dim=dim,
        bounds=bounds,
        optimizers=optimizers
    )
    
    # Create comparison framework
    if args.verbose:
        print("Creating baseline comparison framework...")
    comparison = BaselineComparison(meta_optimizer, problem_generator, algorithms)
    
    # Run training phase
    if args.verbose:
        print(f"Running training phase with {args.training_problems} problems...")
        start_time = time.time()
    
    comparison.run_training_phase(n_problems=args.training_problems)
    
    if args.verbose:
        training_time = time.time() - start_time
        print(f"Training completed in {training_time:.2f} seconds")
    
    # Run comparison
    if args.verbose:
        print(f"Running comparison with {args.test_problems} problems...")
        start_time = time.time()
    
    results = comparison.run_comparison(n_problems=args.test_problems, verbose=args.verbose)
    
    if args.verbose:
        comparison_time = time.time() - start_time
        print(f"Comparison completed in {comparison_time:.2f} seconds")
    
    # Save results
    results_df = comparison.get_summary_dataframe()
    results_df.to_csv(export_dir / 'summary_results.csv', index=False)
    
    # Save raw results
    raw_results = pd.DataFrame({method: pd.Series(performances) for method, performances in results.items()})
    raw_results.to_csv(export_dir / 'raw_results.csv', index=False)
    
    # Create visualizations
    if args.verbose:
        print("Generating visualizations...")
    
    visualizer = ComparisonVisualizer(results, export_dir=str(export_dir))
    visualizer.create_all_visualizations()
    
    if args.verbose:
        print(f"Results and visualizations saved to: {export_dir}")
        print("Done!")


if __name__ == '__main__':
    main() 