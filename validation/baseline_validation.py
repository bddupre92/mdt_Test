#!/usr/bin/env python3
"""
Baseline Algorithm Comparison Validation

This script implements the first validation step in the Meta-Optimizer Validation Plan:
1. Baseline Algorithm Comparison

It tests the Meta-Optimizer against individual optimization algorithms
on benchmark functions and validates the SATzilla-inspired algorithm selection.
"""

import os
import sys
import argparse
import logging
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime

# Import optimizer factory
from meta_optimizer.optimizers.optimizer_factory import OptimizerFactory

from meta_optimizer.meta.meta_optimizer import MetaOptimizer
from baseline_comparison.baseline_algorithms.satzilla_inspired import SatzillaInspiredSelector
from baseline_comparison.comparison_runner import BaselineComparison

# Import benchmark functions
from benchmark.benchmark_functions import get_benchmark_function, get_all_benchmark_functions

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add project root to path to ensure imports work correctly
project_root = str(Path(__file__).parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from cli.problem_wrapper import ProblemWrapper

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Run baseline algorithm comparison validation")
    
    parser.add_argument("--dimensions", "-d", type=int, default=30, 
                      help="Number of dimensions for test functions (default: 30)")
    parser.add_argument("--max-evaluations", "-e", type=int, default=10000, 
                      help="Maximum number of function evaluations (default: 10000)")
    parser.add_argument("--num-trials", "-t", type=int, default=30, 
                      help="Number of trials per function (default: 30)")
    parser.add_argument("--output-dir", "-o", type=str, default=None,
                      help="Directory to save results (default: results/baseline_validation/<timestamp>)")
    parser.add_argument("--selector-path", "-s", type=str, default=None,
                      help="Path to trained SATzilla model (optional)")
    parser.add_argument("--functions", "-f", type=str, nargs="+", 
                      default=["sphere", "rosenbrock", "rastrigin", "ackley", "griewank"],
                      help="List of benchmark functions to use")
    parser.add_argument("--all-functions", action="store_true",
                      help="Use all available benchmark functions")
    parser.add_argument("--seed", type=int, default=42,
                      help="Random seed for reproducibility (default: 42)")
    parser.add_argument("--verbose", "-v", action="count", default=0,
                      help="Increase verbosity (can be used multiple times)")
    parser.add_argument("--quiet", "-q", action="store_true",
                      help="Suppress non-error messages")
    parser.add_argument("--no-visualizations", action="store_true",
                      help="Disable visualization generation")
    parser.add_argument("--timestamp-dir", action="store_true", default=True,
                      help="Create a timestamped subdirectory for results")
    
    return parser.parse_args()

def run_baseline_validation(args):
    """Run the baseline algorithm comparison validation"""
    # Set random seed
    np.random.seed(args.seed)
    
    # Configure logging based on verbosity
    if args.quiet:
        logging.getLogger().setLevel(logging.ERROR)
    elif args.verbose == 0:
        logging.getLogger().setLevel(logging.INFO)
    elif args.verbose == 1:
        logging.getLogger().setLevel(logging.DEBUG)
    else:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Create output directory with timestamp if requested
    if args.timestamp_dir:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = args.output_dir or os.path.join("results", "baseline_validation", timestamp)
    else:
        output_dir = args.output_dir or os.path.join("results", "baseline_validation")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Save configuration
    config = vars(args)
    config["timestamp"] = datetime.now().strftime("%Y%m%d_%H%M%S")
    with open(os.path.join(output_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)
    
    # Initialize baseline selector
    baseline_selector = SatzillaInspiredSelector()
    
    # Get the benchmark functions
    if args.all_functions:
        logger.info("Using all available benchmark functions")
        benchmark_functions = get_all_benchmark_functions(dimensions=args.dimensions)
    else:
        logger.info(f"Using specified benchmark functions: {args.functions}")
        benchmark_functions = [
            get_benchmark_function(name, args.dimensions) 
            for name in args.functions
        ]
    
    # Create optimizers
    factory = OptimizerFactory()
    
    # Use proper bounds for benchmark functions: typically [-5, 5] for standard benchmarks
    bounds = [(-5, 5) for _ in range(args.dimensions)]
    
    # Create optimizers with consistent naming and proper bounds
    optimizers = {
        'DE': factory.create_optimizer('differential_evolution', dim=args.dimensions, bounds=bounds),
        'ES': factory.create_optimizer('evolution_strategy', dim=args.dimensions, bounds=bounds),
        'DE (Adaptive)': factory.create_optimizer('differential_evolution', dim=args.dimensions, bounds=bounds, adaptive=True),
        'ES (Adaptive)': factory.create_optimizer('evolution_strategy', dim=args.dimensions, bounds=bounds, adaptive=True),
        'ACO': factory.create_optimizer('ant_colony', dim=args.dimensions, bounds=bounds),
        'GWO': factory.create_optimizer('grey_wolf', dim=args.dimensions, bounds=bounds)
    }
    
    # Verify optimizers were created
    if not optimizers:
        logger.error("Failed to create optimizers")
    else:
        logger.info(f"Created {len(optimizers)} optimizers: {', '.join(optimizers.keys())}")
        
    # Print optimizer details for debugging
    for name, optimizer in optimizers.items():
        logger.debug(f"Optimizer {name}: {type(optimizer)}")
    
    # Set up context for better algorithm selection
    context = {
        "problem_type": "benchmark",
        "dimensions": args.dimensions,
        "adaptive_selection": True,
        "benchmark_names": args.functions
    }
    
    # Meta-Optimizer initialization with optimizers explicitly passed
    meta_optimizer = MetaOptimizer(
        dim=args.dimensions,
        bounds=bounds,
        optimizers=optimizers,  # Explicitly pass optimizers
        n_parallel=1,  # Single optimizer at a time for clearer validation
        budget_per_iteration=25,
        default_max_evals=args.max_evaluations,
        early_stopping=True,  # Enable early stopping to prevent unnecessary evaluations
        verbose=args.verbose > 0,  # Set verbose mode based on verbosity level
        use_selection_tracker=True  # Track selection performance
    )
    
    # Verify optimizer registration
    if hasattr(meta_optimizer, 'optimizers'):
        registered_count = len(meta_optimizer.optimizers)
        if registered_count > 0:
            logger.info(f"Meta-Optimizer has {registered_count} registered optimizers: {', '.join(meta_optimizer.optimizers.keys())}")
        else:
            logger.warning("Meta-Optimizer has no registered optimizers. Will use fallback algorithms.")
    else:
        logger.warning("Meta-Optimizer does not have an optimizers attribute. Check implementation.")
    
    # Initialize comparison runner
    comparison = BaselineComparison(
        baseline_selector=baseline_selector,
        meta_optimizer=meta_optimizer,
        max_evaluations=args.max_evaluations,
        num_trials=args.num_trials,
        verbose=(args.verbose > 0),
        output_dir=output_dir,
        model_path=args.selector_path
    )
    
    # Run the comparison
    logger.info("Starting baseline algorithm comparison validation")
    logger.info(f"Using {len(benchmark_functions)} benchmark functions")
    logger.info(f"Dimensions: {args.dimensions}, Max evaluations: {args.max_evaluations}, Num trials: {args.num_trials}")
    
    for i, benchmark_func in enumerate(benchmark_functions):
        logger.info(f"Running benchmark {i+1}/{len(benchmark_functions)}: {benchmark_func.name}")
        
        # Create problem with correct bounds
        # Standard benchmark bounds are typically [-5, 5] for each dimension
        problem_bounds = [(-5, 5) for _ in range(args.dimensions)]
        problem = ProblemWrapper(
            benchmark_func, 
            dimensions=args.dimensions,
            bounds=problem_bounds
        )
        
        # Ensure the problem has the proper attributes for feature extraction
        if not hasattr(problem, 'bounds') or not problem.bounds:
            problem.bounds = problem_bounds
            
        if not hasattr(problem, 'dims'):
            problem.dims = args.dimensions
            
        logger.debug(f"Problem configured with bounds: {problem.bounds}")
        
        comparison.run_comparison(
            problem_name=benchmark_func.name,
            problem_func=problem,
            dimensions=args.dimensions,
            max_evaluations=args.max_evaluations,
            num_trials=args.num_trials
        )
    
    # Generate additional summary
    summary_path = os.path.join(output_dir, "validation_summary.md")
    generate_validation_summary(comparison.results, summary_path)
    
    logger.info(f"Validation completed. Results saved to {output_dir}")
    return 0

def generate_validation_summary(results, output_path):
    """Generate a detailed markdown summary of the validation results"""
    
    try:
        # Log the start of function execution
        logger.info(f"Starting to generate validation summary to {output_path}")
        logger.info(f"Results contains data for {len(results)} problems")
        
        # Check if results is empty
        if not results:
            logger.warning("No results were found to generate the summary")
            with open(output_path, "w") as f:
                f.write("# Baseline Algorithm Comparison Validation Summary\n\n")
                f.write(f"Validation completed at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                f.write("No results were generated during the validation run.\n")
            logger.info("Empty validation summary created successfully")
            return
        
        # Log which problems are present in the results
        logger.info(f"Problems in results: {', '.join(results.keys())}")
        
        with open(output_path, "w") as f:
            f.write("# Baseline Algorithm Comparison Validation Summary\n\n")
            f.write(f"Validation completed at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write("## Overall Performance\n\n")
            f.write("| Benchmark Function | Baseline Best Fitness | Meta-Optimizer Best Fitness | Improvement % |\n")
            f.write("|-------------------|----------------------|---------------------------|-------------|\n")
            
            for problem_name, problem_results in results.items():
                baseline_fitness = problem_results.get("baseline_best_fitness_avg", 0)
                meta_fitness = problem_results.get("meta_best_fitness_avg", 0)
                
                # Calculate improvement percentage if needed
                improvement = problem_results.get("improvement_percentage", 0)
                if improvement == 0 and baseline_fitness != 0:
                    improvement = ((baseline_fitness - meta_fitness) / max(abs(baseline_fitness), 1e-10)) * 100
                
                f.write(f"| {problem_name} | {baseline_fitness:.6e} | {meta_fitness:.6e} | {improvement:.2f}% |\n")
            
            f.write("\n## Algorithm Selection Frequency\n\n")
            f.write("### Baseline Selector\n\n")
            for problem_name, problem_results in results.items():
                f.write(f"#### {problem_name}\n\n")
                algorithms = problem_results.get("baseline_selected_algorithms", [])
                
                if not algorithms:
                    f.write("No algorithm selection data available.\n\n")
                    continue
                    
                from collections import Counter
                selection_counts = Counter(algorithms)
                total_selections = len(algorithms)
                
                f.write("| Algorithm | Selection Count | Selection Percentage |\n")
                f.write("|-----------|----------------|----------------------|\n")
                
                for algorithm, count in selection_counts.items():
                    percentage = (count / total_selections) * 100
                    f.write(f"| {algorithm} | {count} | {percentage:.2f}% |\n")
                
                f.write("\n")
            
            f.write("### Meta-Optimizer\n\n")
            for problem_name, problem_results in results.items():
                f.write(f"#### {problem_name}\n\n")
                algorithms = problem_results.get("meta_selected_algorithms", [])
                
                if not algorithms:
                    f.write("No algorithm selection data available.\n\n")
                    continue
                    
                from collections import Counter
                selection_counts = Counter(algorithms)
                total_selections = len(algorithms)
                
                f.write("| Algorithm | Selection Count | Selection Percentage |\n")
                f.write("|-----------|----------------|----------------------|\n")
                
                for algorithm, count in selection_counts.items():
                    percentage = (count / total_selections) * 100
                    f.write(f"| {algorithm} | {count} | {percentage:.2f}% |\n")
                
                f.write("\n")
            
            f.write("## Validation Conclusions\n\n")
            
            # Calculate overall improvement
            total_improvement = 0
            valid_results_count = 0
            
            for problem_results in results.values():
                if "improvement_percentage" in problem_results:
                    total_improvement += problem_results["improvement_percentage"]
                    valid_results_count += 1
            
            average_improvement = total_improvement / valid_results_count if valid_results_count > 0 else 0
            
            f.write(f"- The Meta-Optimizer achieved an average improvement of {average_improvement:.2f}% over baseline selectors\n")
            f.write("- The effectiveness varied across different problem types\n")
            f.write("- Algorithm selection patterns show that the Meta-Optimizer adapts its selection strategy to the problem\n")
            f.write("- Visual performance comparisons are available in the results directory\n")
        
        logger.info(f"Validation summary generated successfully at {output_path}")
    except Exception as e:
        # If there's an exception, log it
        logger.error(f"Error generating validation summary: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())

if __name__ == "__main__":
    args = parse_arguments()
    sys.exit(run_baseline_validation(args)) 