#!/usr/bin/env python3
"""
Test benchmark script to verify the baseline comparison framework
This script runs a small benchmark test to ensure the comparison framework
is working correctly before running the full benchmark suite.
"""

import os
import sys
import json
import logging
import datetime
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from collections import Counter

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add the parent directory to the path if needed
parent_dir = str(Path(__file__).parent.parent.absolute())
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

try:
    # Import the baseline comparison framework
    from baseline_comparison.comparison_runner import BaselineComparison
    from baseline_comparison.baseline_algorithms.satzilla_inspired import SatzillaInspiredSelector
    from baseline_comparison.benchmark_utils import get_benchmark_function
    
    # Import MetaOptimizer if it exists in a known location
    try:
        # Assuming MetaOptimizer is in a module at the same level as baseline_comparison
        from meta_optimizer import MetaOptimizer
    except ImportError:
        logger.warning("Could not import MetaOptimizer. Using mock instead.")
        # Create a mock MetaOptimizer class for testing if the real one can't be imported
        class MetaOptimizer:
            def __init__(self, *args, **kwargs):
                self.name = "MockMetaOptimizer"
                
            def optimize(self, problem, *args, **kwargs):
                # Simple random optimization for testing
                best_x = np.random.uniform(-5, 5, problem.dims)
                best_y = problem.evaluate(best_x)
                return best_x, best_y, 100

except ImportError as e:
    logger.error(f"Import error: {e}")
    sys.exit(1)

def run_test_benchmark():
    """Run a small test benchmark to verify the framework is working"""
    logger.info("Starting test benchmark...")
    
    # Test parameters
    dimensions = 2
    max_evaluations = 1000
    num_trials = 3  # Small number for quick testing
    
    # Create a list of simple benchmark functions
    benchmark_functions = [
        get_benchmark_function("sphere", dimensions),
        get_benchmark_function("rosenbrock", dimensions),
    ]
    
    # Initialize the SATzilla-inspired selector
    selector = SatzillaInspiredSelector()
    
    # Initialize the Meta Optimizer
    meta_optimizer = MetaOptimizer()
    
    # Initialize the comparison framework
    comparison = BaselineComparison(
        baseline_selector=selector,
        meta_optimizer=meta_optimizer,
        max_evaluations=max_evaluations,
        num_trials=num_trials
    )
    
    # Run the comparison
    try:
        logger.info("Running comparison...")
        results = comparison.run_comparison(benchmark_functions)
        
        # Generate some basic plots
        logger.info("Generating visualizations...")
        comparison.plot_performance_comparison(results)
        comparison.plot_algorithm_selection_frequency(results)
        
        # Save the results
        output_dir = Path("results/baseline_comparison")
        output_dir.mkdir(exist_ok=True, parents=True)
        
        # Save the plots
        try:
            plt.figure()
            comparison.plot_performance_comparison(results)
            plt.savefig(output_dir / "performance_comparison.png")
        finally:
            plt.close('all')
        
        try:
            plt.figure()
            comparison.plot_algorithm_selection_frequency(results)
            plt.savefig(output_dir / "algorithm_selection.png")
        finally:
            plt.close('all')
        
        # Save results as JSON
        # Convert numpy types to Python types for JSON serialization
        def convert_for_json(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            else:
                return obj
        
        json_results = {}
        for func_name, func_results in results.items():
            json_results[func_name] = {k: convert_for_json(v) for k, v in func_results.items()}
        
        with open(output_dir / "benchmark_results.json", "w") as f:
            json.dump(json_results, f, indent=2)
        
        # Save a summary report as text
        with open(output_dir / "benchmark_summary.txt", "w") as f:
            f.write("Baseline Comparison Benchmark Summary\n")
            f.write("===================================\n\n")
            f.write(f"Date: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Dimensions: {dimensions}\n")
            f.write(f"Max Evaluations: {max_evaluations}\n")
            f.write(f"Number of Trials: {num_trials}\n\n")
            
            f.write("Results by Function:\n")
            f.write("-----------------\n\n")
            
            for func_name, func_results in results.items():
                f.write(f"Function: {func_name}\n")
                f.write(f"  Baseline best fitness: {func_results['baseline_best_fitness_avg']:.6f} ± {func_results['baseline_best_fitness_std']:.6f}\n")
                f.write(f"  Meta Optimizer best fitness: {func_results['meta_best_fitness_avg']:.6f} ± {func_results['meta_best_fitness_std']:.6f}\n")
                improvement = (func_results['baseline_best_fitness_avg'] - func_results['meta_best_fitness_avg']) / abs(func_results['baseline_best_fitness_avg']) * 100
                f.write(f"  Improvement: {improvement:.2f}%\n")
                f.write(f"  Baseline algorithm selections: {dict(sorted(Counter(func_results['baseline_selected_algorithms']).items()))}\n")
                f.write(f"  Meta algorithm selections: {dict(sorted(Counter(func_results['meta_selected_algorithms']).items()))}\n\n")
        
        logger.info(f"Test results saved to {output_dir}")
        
        # Print a summary of the results
        print("\nTest Benchmark Results Summary:")
        print("===============================")
        for func_name, func_results in results.items():
            print(f"\nFunction: {func_name}")
            print(f"  Baseline best fitness: {func_results['baseline_best_fitness_avg']:.6f}")
            print(f"  Meta Optimizer best fitness: {func_results['meta_best_fitness_avg']:.6f}")
            improvement = (func_results['baseline_best_fitness_avg'] - func_results['meta_best_fitness_avg']) / abs(func_results['baseline_best_fitness_avg']) * 100
            print(f"  Improvement: {improvement:.2f}%")
        
        logger.info("Test benchmark completed successfully!")
        return True
        
    except Exception as e:
        logger.error(f"Error during comparison: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = run_test_benchmark()
    sys.exit(0 if success else 1) 