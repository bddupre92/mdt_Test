#!/usr/bin/env python3
"""
Demonstration of the baseline comparison framework for evaluating Meta Optimizer against
SATzilla-inspired algorithm selection.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Any

# Import Meta Optimizer components
from meta_optimizer.meta.meta_optimizer import MetaOptimizer
from meta_optimizer.benchmark.test_functions import get_benchmark_function
from meta_optimizer.optimizers import OptimizerFactory

# Import baseline comparison framework
from baseline_comparison import BaselineComparison, ComparisonVisualizer

class BenchmarkProblem:
    """Simple wrapper for benchmark functions to standardize the interface."""
    
    def __init__(self, function_name: str, dimension: int = 10):
        """
        Initialize a benchmark problem.
        
        Args:
            function_name: Name of the benchmark function
            dimension: Dimensionality of the problem
        """
        self.function_name = function_name
        self.dimension = dimension
        self.benchmark_fn = get_benchmark_function(function_name)
        
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
    
    def evaluate(self, x: np.ndarray) -> float:
        """Evaluate the benchmark function at a point."""
        return self.benchmark_fn(x)
    
    def __str__(self) -> str:
        return f"{self.function_name.capitalize()} function ({self.dimension}D)"


def problem_generator():
    """Generate random benchmark problems for testing."""
    function_names = ['ackley', 'rastrigin', 'rosenbrock', 'sphere', 'griewank']
    dimensions = [2, 5, 10, 20]
    
    function_name = np.random.choice(function_names)
    dimension = np.random.choice(dimensions)
    
    return BenchmarkProblem(function_name, dimension)


def main():
    """Run the baseline comparison demonstration."""
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Define available optimization algorithms
    algorithms = ['DE', 'ES', 'ACO', 'GWO']
    
    # Create Meta Optimizer instance
    # Create optimizers using OptimizerFactory
    factory = OptimizerFactory()
    optimizers = {}
    dim = 10
    bounds = [(-5, 5)] * dim
    
    for algo in algorithms:
        optimizers[algo] = factory.create_optimizer(
            algo,
            dim=dim,
            bounds=bounds
        )
    
    # Create Meta Optimizer with optimizers
    meta_optimizer = MetaOptimizer(
        dim=dim,
        bounds=bounds,
        optimizers=optimizers
    )
    
    # Create the comparison framework
    comparison = BaselineComparison(meta_optimizer, problem_generator, algorithms)
    
    # Run training phase for SATzilla-inspired selector
    comparison.run_training_phase(n_problems=50)
    
    # Run comparison
    results = comparison.run_comparison(n_problems=20)
    
    # Create visualizations
    os.makedirs('results/comparison', exist_ok=True)
    visualizer = ComparisonVisualizer(results, export_dir='results/comparison')
    visualizer.create_all_visualizations()
    
    # Display summary table
    summary_df = comparison.get_summary_dataframe()
    print("\nSummary Table:")
    print(summary_df.to_string(index=False))
    
    print("\nResults saved to results/comparison/")


if __name__ == "__main__":
    main() 