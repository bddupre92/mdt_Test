import os
import numpy as np
import logging
import json
import matplotlib.pyplot as plt
from typing import Dict, List, Any, Tuple, Optional, Callable
import argparse
from pathlib import Path
import time

from utils.json_utils import save_json
from utils.plotting import save_plot, setup_plot_style

def create_objective_function(function_name: str, dim: int) -> Callable:
    """
    Create an objective function for optimization
    
    Parameters:
    -----------
    function_name : str
        Name of the function ('sphere', 'rosenbrock', 'rastrigin', 'ackley')
    dim : int
        Dimensionality of the function
        
    Returns:
    --------
    Callable
        Objective function that takes a vector of size dim and returns a scalar
    """
    def sphere(x):
        """Sphere function"""
        return np.sum(x**2)
    
    def rosenbrock(x):
        """Rosenbrock function"""
        return np.sum(100.0 * (x[1:] - x[:-1]**2)**2 + (x[:-1] - 1)**2)
    
    def rastrigin(x):
        """Rastrigin function"""
        return 10 * len(x) + np.sum(x**2 - 10 * np.cos(2 * np.pi * x))
    
    def ackley(x):
        """Ackley function"""
        a, b, c = 20, 0.2, 2*np.pi
        sum1 = -a * np.exp(-b * np.sqrt(np.sum(x**2) / len(x)))
        sum2 = -np.exp(np.sum(np.cos(c * x)) / len(x))
        return sum1 + sum2 + a + np.exp(1)
    
    functions = {
        'sphere': sphere,
        'rosenbrock': rosenbrock,
        'rastrigin': rastrigin,
        'ackley': ackley
    }
    
    if function_name not in functions:
        raise ValueError(f"Unknown function: {function_name}. Available functions: {list(functions.keys())}")
    
    return functions[function_name]

class StubOptimizer:
    """Stub optimizer for testing"""
    def __init__(self, dim, bounds, population_size, name, **kwargs):
        self.dim = dim
        self.bounds = bounds
        self.population_size = population_size
        self.params = kwargs
        self.name = name
        self.best_score = float('inf')
        self.best_solution = np.zeros(dim)
        
    def optimize(self, objective_func, max_evals=1000, verbose=False):
        """Stub optimization method"""
        # Simply return a random point within bounds
        lower_bounds = np.array([b[0] for b in self.bounds])
        upper_bounds = np.array([b[1] for b in self.bounds])
        
        # Generate some sample points
        points = np.random.uniform(
            lower_bounds, upper_bounds, 
            size=(min(10, max_evals), self.dim)
        )
        
        # Evaluate points
        scores = np.array([objective_func(p) for p in points])
        
        # Find best
        best_idx = np.argmin(scores)
        self.best_score = scores[best_idx]
        self.best_solution = points[best_idx]
        
        return self.best_solution, self.best_score
        
    def export_data(self, filename, format='json'):
        """Stub for exporting data"""
        data = {
            'optimizer': self.name,
            'best_score': float(self.best_score),
            'best_solution': self.best_solution.tolist(),
            'params': self.params
        }
        
        # Save to file
        with open(f"{filename}.{format}", 'w') as f:
            json.dump(data, f, indent=2)
            
        return f"{filename}.{format}"

def create_optimizer(optimizer_type: str, dim: int, bounds: List[Tuple[float, float]], 
                    population_size: int = 50, **kwargs) -> Any:
    """
    Create an optimizer
    
    Parameters:
    -----------
    optimizer_type : str
        Type of optimizer ('DE', 'PSO', 'ACO', 'GWO', 'ES', etc.)
    dim : int
        Dimensionality of the optimization problem
    bounds : List[Tuple[float, float]]
        Bounds for each dimension
    population_size : int, optional
        Population size for population-based optimizers
    **kwargs
        Additional parameters for the optimizer
    
    Returns:
    --------
    Any
        Optimizer instance
    """
    # Try to import actual optimizer implementations
    try:
        from meta_optimizer.optimizers.optimizer_factory import create_optimizer
        
        # Map simplified naming to actual optimizer types
        optimizer_map = {
            'DE': 'differential_evolution',
            'PSO': 'particle_swarm',
            'ACO': 'ant_colony',
            'GWO': 'grey_wolf',
            'ES': 'evolution_strategy',
            'GA': 'genetic_algorithm',
            'SA': 'simulated_annealing'
        }
        
        # Get the actual optimizer type
        actual_type = optimizer_map.get(optimizer_type, optimizer_type)
        
        # Create and return the optimizer
        return create_optimizer(
            optimizer_type=actual_type,
            dim=dim,
            bounds=bounds,
            population_size=population_size,
            **kwargs
        )
        
    except ImportError:
        # Create a simple stub optimizer for testing
        logging.warning(f"Could not import actual optimizer. Creating stub for {optimizer_type}")
        return StubOptimizer(dim, bounds, population_size, optimizer_type, **kwargs)

def run_optimization(args: argparse.Namespace) -> Dict[str, Any]:
    """
    Run optimization with specified parameters
    
    Parameters:
    -----------
    args : argparse.Namespace
        Command-line arguments
        
    Returns:
    --------
    Dict[str, Any]
        Results of the optimization process
    """
    # Parse arguments with defaults
    optimizer_type = args.optimizer if hasattr(args, 'optimizer') else "DE"
    problem = args.problem if hasattr(args, 'problem') else "sphere"
    dim = args.dimension if hasattr(args, 'dimension') else 10
    max_evals = args.max_evaluations if hasattr(args, 'max_evaluations') else 1000
    population_size = args.population_size if hasattr(args, 'population_size') else 50
    visualize = args.visualize if hasattr(args, 'visualize') else False
    
    logging.info(f"Running optimization with {optimizer_type} on {problem} problem in {dim} dimensions")
    logging.info(f"Parameters: max_evals={max_evals}, population_size={population_size}")
    
    # Create directories for results
    results_dir = 'results/optimization'
    os.makedirs(results_dir, exist_ok=True)
    
    # Create bounds for optimization
    bounds = [(-5, 5)] * dim
    
    # Create objective function
    objective_func = create_objective_function(problem, dim)
    
    # Create optimizer
    optimizer = create_optimizer(
        optimizer_type=optimizer_type, 
        dim=dim, 
        bounds=bounds,
        population_size=population_size
    )
    
    # Initialize results
    results = {
        "optimizer": optimizer_type,
        "problem": problem,
        "dimension": dim,
        "max_evaluations": max_evals,
        "population_size": population_size
    }
    
    # Run optimization
    start_time = time.time()
    
    try:
        best_solution, best_score = optimizer.optimize(
            objective_func, 
            max_evals=max_evals,
            verbose=True
        )
        
        elapsed_time = time.time() - start_time
        
        # Record results
        results["best_score"] = float(best_score)
        results["best_solution"] = best_solution.tolist() if hasattr(best_solution, 'tolist') else best_solution
        results["elapsed_time"] = elapsed_time
        results["success"] = True
        
        logging.info(f"Optimization completed in {elapsed_time:.2f} seconds with best score: {best_score}")
    
    except Exception as e:
        elapsed_time = time.time() - start_time
        
        # Record error
        results["elapsed_time"] = elapsed_time
        results["success"] = False
        results["error"] = str(e)
        
        logging.error(f"Error during optimization: {str(e)}")
    
    # Save results
    results_file = os.path.join(results_dir, f"optimization_{optimizer_type}_{problem}_{time.strftime('%Y%m%d_%H%M%S')}.json")
    save_json(results, results_file)
    logging.info(f"Results saved to {results_file}")
    
    return results

def run_optimizer_comparison(args: argparse.Namespace) -> Dict[str, Any]:
    """
    Run comparison of optimizers on benchmark functions
    
    Parameters:
    -----------
    args : argparse.Namespace
        Command-line arguments
        
    Returns:
    --------
    Dict[str, Any]
        Results of the optimizer comparison
    """
    logging.info("Starting optimizer comparison...")
    
    # Parse arguments with defaults
    dim = args.dimension if hasattr(args, 'dimension') else 10
    max_evals = args.max_evaluations if hasattr(args, 'max_evaluations') else 1000
    n_runs = getattr(args, 'benchmark_repetitions', 5)
    
    # Create output directory
    results_dir = 'results/optimizer_comparison'
    os.makedirs(results_dir, exist_ok=True)
    
    # Get benchmark functions
    benchmark_functions = {
        'sphere': create_objective_function('sphere', dim),
        'rosenbrock': create_objective_function('rosenbrock', dim),
        'rastrigin': create_objective_function('rastrigin', dim),
        'ackley': create_objective_function('ackley', dim)
    }
    
    # Define optimizers to compare
    optimizer_types = ['DE', 'ES', 'ACO', 'GWO', 'PSO']
    
    # Initialize results
    comparison_results = {
        "dimension": dim,
        "max_evaluations": max_evals,
        "runs_per_optimizer": n_runs,
        "results": {}
    }
    
    # Run comparison
    start_time = time.time()
    
    # Define bounds
    bounds = [(-5, 5)] * dim
    
    for func_name, objective_func in benchmark_functions.items():
        logging.info(f"Testing function: {func_name}")
        comparison_results["results"][func_name] = {}
        
        for opt_name in optimizer_types:
            logging.info(f"  Testing optimizer: {opt_name}")
            comparison_results["results"][func_name][opt_name] = {
                "scores": [],
                "times": []
            }
            
            for run in range(n_runs):
                try:
                    # Create optimizer
                    optimizer = create_optimizer(
                        optimizer_type=opt_name, 
                        dim=dim, 
                        bounds=bounds,
                        population_size=50
                    )
                    
                    # Run optimization
                    run_start = time.time()
                    best_solution, best_score = optimizer.optimize(
                        objective_func, 
                        max_evals=max_evals,
                        verbose=False
                    )
                    run_time = time.time() - run_start
                    
                    # Record results
                    comparison_results["results"][func_name][opt_name]["scores"].append(float(best_score))
                    comparison_results["results"][func_name][opt_name]["times"].append(run_time)
                    
                except Exception as e:
                    logging.error(f"Error running {opt_name} on {func_name}: {str(e)}")
            
            # Calculate statistics
            scores = comparison_results["results"][func_name][opt_name]["scores"]
            times = comparison_results["results"][func_name][opt_name]["times"]
            
            if scores:
                comparison_results["results"][func_name][opt_name]["mean_score"] = float(np.mean(scores))
                comparison_results["results"][func_name][opt_name]["std_score"] = float(np.std(scores))
                comparison_results["results"][func_name][opt_name]["mean_time"] = float(np.mean(times))
                
                logging.info(f"    Mean score: {np.mean(scores):.6f}, Mean time: {np.mean(times):.2f}s")
    
    # Determine best optimizer for each function
    comparison_results["best_optimizers"] = {}
    
    for func_name in benchmark_functions.keys():
        best_opt = None
        best_score = float('inf')
        
        for opt_name in optimizer_types:
            if (func_name in comparison_results["results"] and 
                opt_name in comparison_results["results"][func_name] and
                "mean_score" in comparison_results["results"][func_name][opt_name]):
                
                mean_score = comparison_results["results"][func_name][opt_name]["mean_score"]
                if mean_score < best_score:
                    best_score = mean_score
                    best_opt = opt_name
        
        comparison_results["best_optimizers"][func_name] = best_opt
    
    # Calculate total time
    comparison_results["total_time"] = time.time() - start_time
    
    # Save results
    results_file = os.path.join(results_dir, f"comparison_{time.strftime('%Y%m%d_%H%M%S')}.json")
    save_json(comparison_results, results_file)
    logging.info(f"Comparison results saved to {results_file}")
    
    # Print summary
    print("\nOptimizer Comparison Summary:")
    print("=============================")
    print("Best optimizer per function:")
    for func_name, best_opt in comparison_results["best_optimizers"].items():
        if best_opt:
            mean_score = comparison_results["results"][func_name][best_opt]["mean_score"]
            print(f"  {func_name}: {best_opt} (score: {mean_score:.6f})")
    
    return comparison_results
