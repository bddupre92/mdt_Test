"""
Comparison runner for baseline algorithm selectors and Meta Optimizer

This module provides a framework for benchmarking baseline algorithm
selection methods against the Meta Optimizer.
"""

import time
import logging
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Any, Tuple, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class BaselineComparison:
    """
    Framework for comparing baseline algorithm selectors against the Meta Optimizer
    """
    
    def __init__(
        self,
        baseline_selector,
        meta_optimizer,
        max_evaluations: int = 10000,
        num_trials: int = 10,
        verbose: bool = True
    ):
        """
        Initialize the comparison framework
        
        Args:
            baseline_selector: The baseline algorithm selector to benchmark
            meta_optimizer: The Meta Optimizer instance to compare against
            max_evaluations: Maximum number of function evaluations per algorithm
            num_trials: Number of trials to run per algorithm for statistical significance
            verbose: Whether to print progress information
        """
        self.baseline_selector = baseline_selector
        self.meta_optimizer = meta_optimizer
        self.max_evaluations = max_evaluations
        self.num_trials = num_trials
        self.verbose = verbose
        
        # Store the available optimization algorithms
        # These should be compatible with both the baseline selector and Meta Optimizer
        self.available_algorithms = self._get_available_algorithms()
        
        logger.info(f"Initialized BaselineComparison with {len(self.available_algorithms)} algorithms")
        logger.info(f"Max evaluations: {max_evaluations}, Num trials: {num_trials}")
    
    def _get_available_algorithms(self) -> List[str]:
        """
        Get the list of available optimization algorithms
        
        Returns:
            List of algorithm names
        """
        # This list should be consistent with what's available in both
        # the baseline selector and Meta Optimizer
        # Default algorithms that are commonly available
        algorithms = [
            "differential_evolution",
            "particle_swarm",
            "genetic_algorithm",
            "simulated_annealing",
            "cma_es"
        ]
        
        # If the baseline selector has a get_available_algorithms method, use that
        if hasattr(self.baseline_selector, "get_available_algorithms"):
            algorithms = self.baseline_selector.get_available_algorithms()
        
        return algorithms
    
    def run_comparison(self, benchmark_functions: List) -> Dict[str, Dict]:
        """
        Run a comparison between the baseline selector and Meta Optimizer
        
        Args:
            benchmark_functions: List of benchmark functions to test
            
        Returns:
            Dictionary with results for each benchmark function
        """
        results = {}
        
        for func in benchmark_functions:
            func_name = func.name if hasattr(func, "name") else str(func)
            
            if self.verbose:
                logger.info(f"Running comparison on {func_name}")
            
            # Initialize results for this function
            func_results = {
                "baseline_best_fitness": [],
                "baseline_evaluations": [],
                "baseline_time": [],
                "baseline_selected_algorithms": [],
                "meta_best_fitness": [],
                "meta_evaluations": [],
                "meta_time": [],
                "meta_selected_algorithms": []
            }
            
            # Run trials
            for trial in range(self.num_trials):
                if self.verbose:
                    logger.info(f"  Trial {trial+1}/{self.num_trials}")
                
                # Run baseline selector
                baseline_result = self._run_baseline(func, trial)
                
                # Run Meta Optimizer
                meta_result = self._run_meta_optimizer(func, trial)
                
                # Store results
                func_results["baseline_best_fitness"].append(baseline_result["best_fitness"])
                func_results["baseline_evaluations"].append(baseline_result["evaluations"])
                func_results["baseline_time"].append(baseline_result["time"])
                func_results["baseline_selected_algorithms"].append(baseline_result["selected_algorithm"])
                
                func_results["meta_best_fitness"].append(meta_result["best_fitness"])
                func_results["meta_evaluations"].append(meta_result["evaluations"])
                func_results["meta_time"].append(meta_result["time"])
                func_results["meta_selected_algorithms"].append(meta_result["selected_algorithm"])
            
            # Calculate average results
            # Create a list of keys first to avoid modifying the dictionary during iteration
            keys_to_average = [key for key in func_results if "algorithms" not in key]
            for key in keys_to_average:
                func_results[key + "_avg"] = np.mean(func_results[key])
                func_results[key + "_std"] = np.std(func_results[key])
            
            # Store results for this function
            results[func_name] = func_results
        
        return results
    
    def _run_baseline(self, func, trial: int) -> Dict[str, Any]:
        """
        Run the baseline selector on a benchmark function
        
        Args:
            func: The benchmark function
            trial: Trial number
            
        Returns:
            Dictionary with results
        """
        start_time = time.time()
        
        # Select algorithm using the baseline selector
        selected_algorithm = self.baseline_selector.select_algorithm(func)
        
        # Use the selected algorithm to optimize the function
        # This part depends on how your baseline selector interface works
        # You might need to adapt this to your specific implementation
        try:
            # Try to use the optimize method if available
            if hasattr(self.baseline_selector, "optimize"):
                best_x, best_y, evaluations = self.baseline_selector.optimize(
                    func, 
                    algorithm=selected_algorithm,
                    max_evaluations=self.max_evaluations
                )
            else:
                # Otherwise, use a simple optimization approach
                # This is a placeholder - you should implement the actual optimization
                best_x = np.random.uniform(func.bounds[0], func.bounds[1], func.dims)
                best_y = func.evaluate(best_x)
                evaluations = self.max_evaluations
                logger.warning(f"Using placeholder optimization for baseline selector")
        except Exception as e:
            logger.error(f"Error in baseline optimization: {e}")
            # Provide dummy results in case of error
            best_x = np.zeros(func.dims)
            best_y = float('inf')
            evaluations = 0
        
        end_time = time.time()
        
        return {
            "best_x": best_x,
            "best_fitness": best_y,
            "evaluations": evaluations,
            "time": end_time - start_time,
            "selected_algorithm": selected_algorithm
        }
    
    def _run_meta_optimizer(self, func, trial: int) -> Dict[str, Any]:
        """
        Run the Meta Optimizer on a benchmark function
        
        Args:
            func: The benchmark function
            trial: Trial number
            
        Returns:
            Dictionary with results
        """
        start_time = time.time()
        
        # Run Meta Optimizer
        try:
            # Adapt this to match the Meta Optimizer interface
            best_x, best_y, evaluations = self.meta_optimizer.optimize(
                func,
                max_evaluations=self.max_evaluations
            )
            
            # Try to get the selected algorithm if the Meta Optimizer exposes it
            if hasattr(self.meta_optimizer, "selected_algorithm"):
                selected_algorithm = self.meta_optimizer.selected_algorithm
            else:
                selected_algorithm = "unknown"
                
        except Exception as e:
            logger.error(f"Error in Meta Optimizer: {e}")
            # Provide dummy results in case of error
            best_x = np.zeros(func.dims)
            best_y = float('inf')
            evaluations = 0
            selected_algorithm = "error"
        
        end_time = time.time()
        
        return {
            "best_x": best_x,
            "best_fitness": best_y,
            "evaluations": evaluations,
            "time": end_time - start_time,
            "selected_algorithm": selected_algorithm
        }
    
    def plot_performance_comparison(self, results: Dict[str, Dict], 
                                   metric: str = "best_fitness_avg") -> None:
        """
        Plot a comparison of performance between baseline and Meta Optimizer
        
        Args:
            results: Dictionary with results from run_comparison
            metric: Which metric to plot (default: best_fitness_avg)
        """
        func_names = list(results.keys())
        baseline_values = [results[f]["baseline_{0}".format(metric)] for f in func_names]
        meta_values = [results[f]["meta_{0}".format(metric)] for f in func_names]
        
        # Create a bar chart
        fig, ax = plt.subplots(figsize=(10, 6))
        x = np.arange(len(func_names))
        width = 0.35
        
        ax.bar(x - width/2, baseline_values, width, label='Baseline')
        ax.bar(x + width/2, meta_values, width, label='Meta Optimizer')
        
        ax.set_ylabel(metric.replace("_", " ").title())
        ax.set_title(f'Performance Comparison ({metric})')
        ax.set_xticks(x)
        ax.set_xticklabels(func_names, rotation=45, ha='right')
        ax.legend()
        
        fig.tight_layout()
        
    def plot_algorithm_selection_frequency(self, results: Dict[str, Dict]) -> None:
        """
        Plot the frequency of algorithm selection
        
        Args:
            results: Dictionary with results from run_comparison
        """
        # Collect all selected algorithms
        baseline_algorithms = []
        meta_algorithms = []
        
        for func_name, func_results in results.items():
            baseline_algorithms.extend(func_results["baseline_selected_algorithms"])
            meta_algorithms.extend(func_results["meta_selected_algorithms"])
        
        # Count frequencies
        baseline_counts = {}
        for alg in baseline_algorithms:
            baseline_counts[alg] = baseline_counts.get(alg, 0) + 1
            
        meta_counts = {}
        for alg in meta_algorithms:
            meta_counts[alg] = meta_counts.get(alg, 0) + 1
        
        # Create a bar chart
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Baseline
        ax1.bar(baseline_counts.keys(), baseline_counts.values())
        ax1.set_title('Baseline Algorithm Selection')
        ax1.set_ylabel('Frequency')
        ax1.set_xticklabels(baseline_counts.keys(), rotation=45, ha='right')
        
        # Meta Optimizer
        ax2.bar(meta_counts.keys(), meta_counts.values())
        ax2.set_title('Meta Optimizer Algorithm Selection')
        ax2.set_ylabel('Frequency')
        ax2.set_xticklabels(meta_counts.keys(), rotation=45, ha='right')
        
        fig.tight_layout() 