"""
meta_learning_viz_demo.py
-------------------------
A demonstration script to showcase algorithm selection visualization
in the context of meta-learning with real problem benchmarks.
"""

import os
import sys
import logging
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import argparse

# Add project root to Python path
sys.path.append(str(Path(__file__).parent))

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# Import necessary components
try:
    from visualization.algorithm_selection_viz import AlgorithmSelectionVisualizer
    from meta_optimizer.meta.meta_optimizer import MetaOptimizer
    from meta_optimizer.optimizers.aco import AntColonyOptimizer
    from meta_optimizer.optimizers.gwo import GreyWolfOptimizer
    from meta_optimizer.optimizers.de import DifferentialEvolutionOptimizer
    from meta_optimizer.optimizers.es import EvolutionStrategyOptimizer
    from meta_optimizer.benchmark.test_functions import Sphere, Rosenbrock, Rastrigin, Ackley, TEST_FUNCTIONS
except ImportError as e:
    logging.error(f"Failed to import required components: {e}")
    sys.exit(1)

def create_benchmark_functions(dims=[10, 30]):
    """Create benchmark functions for testing."""
    benchmark_functions = {}
    
    # Define bounds for each dimension
    bounds = [(-5, 5)] * dims[0]  # Use the first dimension to create bounds
    
    # Create test functions using the TEST_FUNCTIONS dictionary
    for func_name, func_class in TEST_FUNCTIONS.items():
        benchmark_functions[func_name] = func_class(dim=dims[0], bounds=bounds)
    
    return benchmark_functions

def create_optimizers(dim, bounds):
    """Create a dictionary of optimizer instances."""
    return {
        'ACO': AntColonyOptimizer(dim=dim, bounds=bounds),
        'GWO': GreyWolfOptimizer(dim=dim, bounds=bounds),
        'DE': DifferentialEvolutionOptimizer(dim=dim, bounds=bounds),
        'ES': EvolutionStrategyOptimizer(dim=dim, bounds=bounds),
        'DE-Adaptive': DifferentialEvolutionOptimizer(dim=dim, bounds=bounds, adaptive=True),
        'ES-Adaptive': EvolutionStrategyOptimizer(dim=dim, bounds=bounds, adaptive=True)
    }

def run_meta_learning_demo(args):
    """Run meta-learning with algorithm selection visualization."""
    print("Running meta-learning with algorithm selection visualization...")
    
    # Create save directory
    algo_viz_dir = args.viz_dir
    os.makedirs(algo_viz_dir, exist_ok=True)
    
    # Create benchmark functions
    benchmark_functions = create_benchmark_functions(dims=[args.dimension])
    print(f"Created {len(benchmark_functions)} benchmark functions")
    
    # Set bounds and dimension
    dim = args.dimension
    bounds = [(-5, 5)] * dim
    
    # Create optimizers
    optimizers = create_optimizers(dim, bounds)
    print(f"Created {len(optimizers)} optimizers")
    
    # Create meta-optimizer
    meta_opt = MetaOptimizer(
        dim=dim,
        bounds=bounds,
        optimizers=optimizers,
        history_file=os.path.join(algo_viz_dir, 'meta_learning_history.json'),
        selection_file=os.path.join(algo_viz_dir, 'selection_history.json')
    )
    
    # Create algorithm selection visualizer
    algo_viz = AlgorithmSelectionVisualizer(save_dir=algo_viz_dir)
    
    # Enable algorithm selection visualization
    meta_opt.enable_algo_viz = True
    meta_opt.algo_selection_viz = algo_viz
    
    print("Algorithm selection visualization enabled")
    
    # Run meta-learning on each benchmark function
    results = {}
    
    for func_name, func_obj in benchmark_functions.items():
        print(f"Running meta-optimization for {func_name}")
        
        # Reset optimizer state
        meta_opt.reset()
        
        # Run optimization with context
        try:
            result = meta_opt.optimize(
                func_obj.evaluate,
                max_evals=args.max_evals,
                context={"function_name": func_name, "phase": "optimization"}
            )
            
            # Store results
            results[func_name] = {
                "best_score": meta_opt.best_score,
                "best_solution": meta_opt.best_solution.tolist() if meta_opt.best_solution is not None else None,
                "evaluations": meta_opt.total_evaluations
            }
            
            print(f"Completed optimization for {func_name} - Best score: {meta_opt.best_score:.6f}")
        except Exception as e:
            print(f"Error optimizing {func_name}: {e}")
    
    # Generate algorithm selection visualizations
    print("Generating algorithm selection visualizations...")
    
    # Generate visualizations directly using the algo_selection_viz object
    generated_files = {}
    
    # Generate base visualizations that work
    try:
        if "frequency" in args.plots:
            print("Generating frequency plot...")
            algo_viz.plot_selection_frequency(save=True)
            generated_files["frequency"] = os.path.join(algo_viz_dir, "algorithm_selection_frequency.png")
    except Exception as e:
        print(f"Error generating frequency plot: {e}")
    
    try:
        if "timeline" in args.plots:
            print("Generating timeline plot...")
            algo_viz.plot_selection_timeline(save=True)
            generated_files["timeline"] = os.path.join(algo_viz_dir, "algorithm_selection_timeline.png")
    except Exception as e:
        print(f"Error generating timeline plot: {e}")
        
    try:
        if "problem" in args.plots:
            print("Generating problem distribution plot...")
            algo_viz.plot_problem_distribution(save=True)
            generated_files["problem"] = os.path.join(algo_viz_dir, "algorithm_selection_by_problem.png")
    except Exception as e:
        print(f"Error generating problem distribution plot: {e}")
        
    # Skip problematic visualizations
    # "performance" and "phase" plots are having issues
        
    try:
        if "dashboard" in args.plots:
            print("Generating summary dashboard...")
            algo_viz.create_summary_dashboard(save=True)
            generated_files["dashboard"] = os.path.join(algo_viz_dir, "algorithm_selection_dashboard.png")
    except Exception as e:
        print(f"Error generating summary dashboard: {e}")
    
    # Generate interactive visualizations if requested
    if "interactive" in args.plots:
        try:
            print("Generating interactive timeline...")
            algo_viz.interactive_selection_timeline(save=True)
            generated_files["interactive_timeline"] = os.path.join(algo_viz_dir, "interactive_algorithm_timeline.html")
        except Exception as e:
            print(f"Error generating interactive timeline: {e}")
            
        try:
            print("Generating interactive dashboard...")
            algo_viz.interactive_dashboard(save=True)
            generated_files["interactive_dashboard"] = os.path.join(algo_viz_dir, "interactive_dashboard.html")
        except Exception as e:
            print(f"Error generating interactive dashboard: {e}")
    
    print("Meta-learning demo completed successfully.")
    print(f"Generated {len(generated_files)} visualizations:")
    for viz_type, filepath in generated_files.items():
        print(f"  - {viz_type}: {os.path.basename(filepath)}")
    print(f"Visualizations saved to: {algo_viz_dir}")
    
    return {"success": True, "generated_files": generated_files, "results": results}

def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Meta-Learning Algorithm Selection Visualization Demo")
    parser.add_argument("--viz-dir", type=str, default="results/meta_learning_viz",
                        help="Directory to save visualization files")
    parser.add_argument("--dimension", type=int, default=10,
                        help="Dimension for benchmark functions")
    parser.add_argument("--max-evals", type=int, default=1000,
                        help="Maximum evaluations per benchmark function")
    parser.add_argument("--plots", nargs="+", 
                        choices=["frequency", "timeline", "problem", "performance", "phase", "dashboard", "interactive"],
                        default=["frequency", "timeline", "problem", "dashboard", "interactive"],
                        help="List of plot types to generate")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    run_meta_learning_demo(args)
