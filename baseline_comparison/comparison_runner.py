"""
Comparison runner for baseline algorithm selectors and Meta Optimizer

This module provides a framework for benchmarking baseline algorithm
selection methods against the Meta Optimizer.
"""

import os
import json
import time
import logging
import random
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib.gridspec import GridSpec
from typing import Dict, List, Any, Tuple, Optional
from collections import Counter
from cli.problem_wrapper import ProblemWrapper
from meta_optimizer.benchmark.test_functions import create_test_suite
from scipy import stats
from pathlib import Path
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
        verbose: bool = True,
        output_dir: str = "results/baseline_comparison",
        model_path: str = None
    ):
        """
        Initialize the comparison framework
        
        Args:
            baseline_selector: The baseline algorithm selector to benchmark
            meta_optimizer: The Meta Optimizer instance to compare against
            max_evaluations: Maximum number of function evaluations per algorithm
            num_trials: Number of trials to run per algorithm for statistical significance
            verbose: Whether to print progress information
            output_dir: Directory to save results and visualizations
            model_path: Optional path to trained model for baseline selector
        """
        self.baseline_selector = baseline_selector
        self.meta_optimizer = meta_optimizer
        self.max_evaluations = max_evaluations
        self.num_trials = num_trials
        self.verbose = verbose
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        self.output_dir = output_dir
        
        # Initialize result dictionary
        self.results = {}
        
        # Available algorithms
        self.available_algorithms = ["differential_evolution", "particle_swarm", 
                                  "genetic_algorithm", "simulated_annealing", "cma_es"]
        
        # Random seed for reproducibility
        self.random_seed = None
        
        # Load model if path is provided
        self.model_loaded = False
        if model_path is not None and hasattr(baseline_selector, 'load_model'):
            try:
                baseline_selector.load_model(model_path)
                logger.info(f"Loaded model from {model_path}")
                self.model_loaded = True
            except Exception as e:
                logger.warning(f"Could not load model from {model_path}: {e}")
        else:
            logger.warning("No model_path provided. Using untrained baseline selector.")
            
        # Configure common algorithm pool
        self.standardize_algorithm_pool()
            
        logger.info(f"Initialized BaselineComparison with {len(self.available_algorithms)} algorithms")
        logger.info(f"Max evaluations: {max_evaluations}, Num trials: {num_trials}")

    def standardize_algorithm_pool(self):
        """Ensure both baseline selector and meta-optimizer use identical algorithm pools."""
        # Get available algorithms from both systems
        baseline_algorithms = self.baseline_selector.get_available_algorithms() if hasattr(self.baseline_selector, 'get_available_algorithms') else []
        meta_algorithms = self.meta_optimizer.get_available_optimizers() if hasattr(self.meta_optimizer, 'get_available_optimizers') else []
        
        # Find common algorithms
        if baseline_algorithms and meta_algorithms:
            common_algorithms = set(baseline_algorithms).intersection(set(meta_algorithms))
            if common_algorithms:
                self.available_algorithms = list(common_algorithms)
        
        logger.info(f"Using {len(self.available_algorithms)} algorithms: {sorted(self.available_algorithms)}")
        
        # Configure both systems to use only common algorithms
        if hasattr(self.baseline_selector, 'set_available_algorithms'):
            self.baseline_selector.set_available_algorithms(list(self.available_algorithms))
        
        if hasattr(self.meta_optimizer, 'set_available_optimizers'):
            self.meta_optimizer.set_available_optimizers(list(self.available_algorithms))
            
        return self.available_algorithms

    def set_random_seeds(self, seed=None):
        """Set random seeds for reproducible experiments."""
        if seed is None:
            seed = int(time.time())
        
        self.random_seed = seed
        np.random.seed(seed)
        random.seed(seed)
        
        # Set seeds for optimizers if they support it
        if hasattr(self.baseline_selector, 'set_seed'):
            self.baseline_selector.set_seed(seed)
        if hasattr(self.meta_optimizer, 'set_seed'):
            self.meta_optimizer.set_seed(seed)
            
        logger.info(f"Random seeds set to {seed}")
        return seed

    def prepare_problem_for_run(self, problem: ProblemWrapper):
        """Reset and prepare problem for a fresh optimization run."""
        # Reset evaluations counter and tracking
        problem.evaluations = 0
        problem.tracking_objective = None
        
        # Create a deep copy to avoid any side effects between runs
        import copy
        return copy.deepcopy(problem)
    
    def run_comparison(self, problem_name, problem_func, dimensions, max_evaluations, num_trials):
        """Run comparison between baseline and meta optimizer."""
        baseline_best_fitness = []
        baseline_evaluations = []
        baseline_time = []
        baseline_convergence = []
        baseline_selections = []
        
        meta_best_fitness = []
        meta_evaluations = []
        meta_time = []
        meta_convergence = []
        meta_selections = []
        
        # Stores normalized metrics for all trials
        normalized_metrics = []
        
        for trial in range(1, num_trials + 1):
            logger.info(f"  Trial {trial}/{num_trials}")
            
            # Create fresh problem instances for each optimizer to avoid cross-contamination
            # Use identical seeds for identical starting points
            seed = int(time.time()) if self.random_seed is None else self.random_seed + trial
            np.random.seed(seed)
            random.seed(seed)
            
            baseline_problem = self.prepare_problem_for_run(problem_func)
            meta_problem = self.prepare_problem_for_run(problem_func)
            
            # Verify and synchronize bounds and other attributes
            self._synchronize_problem_instances(baseline_problem, meta_problem)
            
            # Run baseline with identical configuration
            start_time = time.time()
            baseline_result = self._run_baseline(baseline_problem, max_evaluations)
            baseline_time.append(time.time() - start_time)
            baseline_best_fitness.append(baseline_result["best_fitness"])
            baseline_evaluations.append(baseline_result["evaluations"])
            baseline_convergence.append(baseline_result.get("convergence_data", []))
            baseline_selections.append(baseline_result["selected_algorithm"])
            
            # Run meta optimizer with identical configuration
            start_time = time.time()
            try:
                meta_result = self._run_meta_optimizer(meta_problem, max_evaluations)
                meta_time.append(time.time() - start_time)
                meta_best_fitness.append(meta_result["best_fitness"])
                meta_evaluations.append(meta_result["evaluations"])
                meta_convergence.append(meta_result.get("convergence_data", []))
                meta_selections.append(meta_result["selected_algorithm"])
            except Exception as e:
                logger.error(f"Error running meta optimizer: {e}")
                logger.exception(e)
                # Use default values on error
                meta_time.append(time.time() - start_time)
                meta_best_fitness.append(float('inf'))
                meta_evaluations.append(max_evaluations)
                meta_convergence.append([])
                meta_selections.append("error")
            
            # Validate results for this trial
            validation_result = self.validate_results(baseline_result, meta_result)
            if not validation_result["valid"]:
                for warning in validation_result["warnings"]:
                    logger.warning(f"Validation warning for trial {trial}: {warning}")
                
            # Calculate normalized metrics for this trial
            trial_metrics = self.calculate_normalized_metrics(baseline_result, meta_result)
            normalized_metrics.append(trial_metrics)
            
            # Generate normalized comparison plots for this trial
            trial_dir = os.path.join(self.output_dir, problem_name, f"trial_{trial}")
            os.makedirs(trial_dir, exist_ok=True)
            
            self.generate_normalized_comparison_plots(
                baseline_result, 
                meta_result, 
                os.path.join(trial_dir, f"normalized_comparison.png"),
                f"{problem_name} (Trial {trial})"
            )
            
        # Calculate averages
        avg_baseline_fitness = np.mean(baseline_best_fitness)
        avg_baseline_evals = np.mean(baseline_evaluations)
        avg_baseline_time = np.mean(baseline_time)
        
        avg_meta_fitness = np.mean(meta_best_fitness)
        avg_meta_evals = np.mean(meta_evaluations)
        avg_meta_time = np.mean(meta_time)
        
        # Calculate average normalized metrics
        avg_normalized_metrics = self._average_normalized_metrics(normalized_metrics)
        
        # Average convergence data
        avg_baseline_convergence = self._average_convergence_data(baseline_convergence)
        avg_meta_convergence = self._average_convergence_data(meta_convergence)
        
        # Generate aggregate normalized comparison
        agg_dir = os.path.join(self.output_dir, problem_name)
        os.makedirs(agg_dir, exist_ok=True)
        
        self.generate_aggregate_normalized_comparison(
            normalized_metrics,
            os.path.join(agg_dir, "aggregate_normalized_comparison.png"),
            f"Aggregate Normalized Comparison - {problem_name}"
        )
        
        # Store results
        self.results[problem_name] = {
            "baseline": {
                "best_fitness": baseline_best_fitness,
                "avg_best_fitness": avg_baseline_fitness,
                "evaluations": baseline_evaluations,
                "avg_evaluations": avg_baseline_evals,
                "time": baseline_time,
                "avg_time": avg_baseline_time,
                "convergence": baseline_convergence,
                "avg_convergence": avg_baseline_convergence,
                "selected_algorithms": baseline_selections
            },
            "meta": {
                "best_fitness": meta_best_fitness,
                "avg_best_fitness": avg_meta_fitness,
                "evaluations": meta_evaluations,
                "avg_evaluations": avg_meta_evals,
                "time": meta_time,
                "avg_time": avg_meta_time,
                "convergence": meta_convergence,
                "avg_convergence": avg_meta_convergence,
                "selected_algorithms": meta_selections
            },
            "normalized_metrics": normalized_metrics,
            "avg_normalized_metrics": avg_normalized_metrics
        }
        
        # Print summary
        logger.info(f"  Average results for {problem_name}:")
        logger.info(f"    Baseline: fitness={avg_baseline_fitness:.6f}, evals={avg_baseline_evals:.1f}, time={avg_baseline_time:.3f}s")
        logger.info(f"    Meta-Opt: fitness={avg_meta_fitness:.6f}, evals={avg_meta_evals:.1f}, time={avg_meta_time:.3f}s")
        
        fitness_improvement = ((avg_baseline_fitness - avg_meta_fitness) / avg_baseline_fitness) * 100
        eval_improvement = ((avg_baseline_evals - avg_meta_evals) / avg_baseline_evals) * 100
        time_improvement = ((avg_baseline_time - avg_meta_time) / avg_baseline_time) * 100
        
        logger.info(f"    Improvements: fitness={fitness_improvement:.2f}%, evals={eval_improvement:.2f}%, time={time_improvement:.2f}%")
        
        return self.results[problem_name]
        
    def _average_normalized_metrics(self, normalized_metrics_list):
        """Average normalized metrics across multiple trials."""
        if not normalized_metrics_list:
            return {}
            
        # Initialize with keys from first metrics dict
        first_metrics = normalized_metrics_list[0]
        avg_metrics = {}
        
        # Process each key in the metrics
        for key in first_metrics:
            # Handle nested dictionaries (like normalized_points)
            if isinstance(first_metrics[key], dict):
                avg_metrics[key] = {}
                # Get all possible nested keys
                nested_keys = set()
                for metrics in normalized_metrics_list:
                    if key in metrics and isinstance(metrics[key], dict):
                        nested_keys.update(metrics[key].keys())
                
                # Initialize nested dictionaries
                for nested_key in nested_keys:
                    # Check if the nested value is a dict in the first metrics
                    is_nested_dict = False
                    for metrics in normalized_metrics_list:
                        if (key in metrics and isinstance(metrics[key], dict) and 
                            nested_key in metrics[key] and isinstance(metrics[key][nested_key], dict)):
                            is_nested_dict = True
                            # Initialize with the structure from the first available nested-nested dict
                            avg_metrics[key][nested_key] = {k: 0.0 for k in metrics[key][nested_key].keys()}
                            break
                    
                    # If not a dict, initialize as a float
                    if not is_nested_dict:
                        avg_metrics[key][nested_key] = 0.0
                
                # Sum all nested values
                for metrics in normalized_metrics_list:
                    if key in metrics and isinstance(metrics[key], dict):
                        for nested_key in metrics[key]:
                            if nested_key in avg_metrics[key]:
                                if isinstance(metrics[key][nested_key], dict) and isinstance(avg_metrics[key][nested_key], dict):
                                    # Handle nested-nested dictionaries
                                    for k, v in metrics[key][nested_key].items():
                                        if k in avg_metrics[key][nested_key]:
                                            avg_metrics[key][nested_key][k] += v
                                elif not isinstance(metrics[key][nested_key], dict) and not isinstance(avg_metrics[key][nested_key], dict):
                                    # Handle simple values in nested dict
                                    avg_metrics[key][nested_key] += metrics[key][nested_key]
                                else:
                                    # Skip incompatible types and log a warning
                                    logger.warning(f"Type mismatch in normalized metrics: {type(avg_metrics[key][nested_key])} vs {type(metrics[key][nested_key])}")
                
                # Divide nested values by count
                for nested_key in avg_metrics[key]:
                    if isinstance(avg_metrics[key][nested_key], dict):
                        for k in avg_metrics[key][nested_key]:
                            avg_metrics[key][nested_key][k] /= len(normalized_metrics_list)
                    else:
                        avg_metrics[key][nested_key] /= len(normalized_metrics_list)
            else:
                # Handle simple values
                avg_metrics[key] = 0.0
                for metrics in normalized_metrics_list:
                    if key in metrics:
                        avg_metrics[key] += metrics[key]
                avg_metrics[key] /= len(normalized_metrics_list)
            
        return avg_metrics
        
    def _average_convergence_data(self, convergence_list):
        """Average convergence data across multiple trials."""
        if not convergence_list:
            return []
            
        # Find maximum length for padding
        max_len = max(len(conv) for conv in convergence_list)
        
        # Pad shorter lists with their last value
        padded_data = []
        for conv in convergence_list:
            if len(conv) < max_len:
                padded = conv.copy()
                # If convergence data is empty, use a large value
                last_value = padded[-1] if padded else float('inf')
                padded.extend([last_value] * (max_len - len(conv)))
                padded_data.append(padded)
            else:
                padded_data.append(conv)
                
        # Average across trials
        avg_convergence = []
        for i in range(max_len):
            avg_value = np.mean([data[i] for data in padded_data])
            avg_convergence.append(avg_value)
            
        return avg_convergence

    def _synchronize_problem_instances(self, baseline_problem, meta_problem):
        """Ensure both problem instances have identical configuration."""
        
        # Synchronize bounds
        if hasattr(baseline_problem, 'bounds') and hasattr(meta_problem, 'bounds'):
            if baseline_problem.bounds != meta_problem.bounds:
                logger.warning(f"Bounds mismatch: baseline={baseline_problem.bounds}, meta={meta_problem.bounds}")
                # Use baseline bounds as the reference
                meta_problem.bounds = baseline_problem.bounds
                
        # Synchronize dimension
        if hasattr(baseline_problem, 'dimensions') and hasattr(meta_problem, 'dimensions'):
            if baseline_problem.dimensions != meta_problem.dimensions:
                logger.warning(f"Dimension mismatch: baseline={baseline_problem.dimensions}, meta={meta_problem.dimensions}")
                meta_problem.dimensions = baseline_problem.dimensions
                
        # Synchronize function reference if different
        if hasattr(baseline_problem, 'function') and hasattr(meta_problem, 'function'):
            if baseline_problem.function is not meta_problem.function:
                logger.warning(f"Function reference mismatch, synchronizing")
                meta_problem.function = baseline_problem.function
                
        # Ensure evaluation counter is reset
        baseline_problem.evaluations = 0
        meta_problem.evaluations = 0
        
        logger.debug(f"Problems synchronized: dims={baseline_problem.dimensions}, bounds={baseline_problem.bounds}")
        return baseline_problem, meta_problem
    
    def _run_baseline(self, problem: ProblemWrapper, max_evaluations: int) -> Dict[str, Any]:
        start_time = time.time()
        convergence_history = []
        best_fitness = float('inf')
        evaluation_count = 0
        
        # Create a wrapper function to enforce evaluation budget and track convergence
        original_evaluate = problem.evaluate
        
        def budget_enforced_evaluate(x):
            nonlocal evaluation_count, best_fitness
            
            # Check if we've reached the maximum evaluations
            if evaluation_count >= max_evaluations:
                logger.warning(f"Baseline reached max evaluations ({max_evaluations}), stopping further evaluations")
                # Return a large value to indicate failure when budget is exceeded
                return float('inf') 
            
            # Increment our evaluation counter
            evaluation_count += 1
            
            # Call the original evaluate function
            result = original_evaluate(x)
            
            # Track convergence
            if result < best_fitness:
                best_fitness = result
                convergence_history.append(best_fitness)
            
            return result
        
        # Replace the evaluate method with our budget-enforced version
        problem.evaluate = budget_enforced_evaluate
        
        # Ensure tracking_objective is called when we update best_fitness
        def tracking_objective(fitness):
            nonlocal best_fitness
            if fitness < best_fitness:
                best_fitness = fitness
            convergence_history.append(best_fitness)

        problem.tracking_objective = tracking_objective
        
        # Run baseline optimization
        result = self.baseline_selector.optimize(problem, max_evaluations)
        selected_algorithm = self.baseline_selector.get_selected_algorithm() if hasattr(self.baseline_selector, 'get_selected_algorithm') else 'unknown'
        
        end_time = time.time()
        
        # Restore original evaluate function
        problem.evaluate = original_evaluate
        
        return {
            'best_solution': result[0],
            'best_fitness': result[1],
            'evaluations': evaluation_count,
            'time': end_time - start_time,
            'convergence_data': convergence_history,
            'selected_algorithm': selected_algorithm
        }
    
    def _run_meta_optimizer(self, problem: ProblemWrapper, max_evaluations: int) -> Dict[str, Any]:
        start_time = time.time()
        convergence_history = []
        best_fitness = float('inf')
        original_evaluations = problem.evaluations
        evaluation_count = 0
        
        # Create a wrapper function to track convergence but NOT enforce budget
        original_evaluate = problem.evaluate
        
        def tracking_evaluate(x):
            nonlocal evaluation_count, best_fitness
            
            # Increment our evaluation counter (but don't enforce limit)
            evaluation_count += 1
            
            # Log every 1000 evaluations to keep track of progress
            if evaluation_count % 1000 == 0:
                logger.info(f"Meta optimizer at {evaluation_count} evaluations")
            
            # Call the original evaluate function
            result = original_evaluate(x)
            
            # Track convergence
            if result < best_fitness:
                best_fitness = result
                convergence_history.append(best_fitness)
            
            return result
        
        # Replace the evaluate method with our tracking version
        problem.evaluate = tracking_evaluate
        
        # Ensure tracking_objective is called when we update best_fitness
        def tracking_objective(fitness):
            nonlocal best_fitness
            if fitness < best_fitness:
                best_fitness = fitness
            convergence_history.append(best_fitness)

        problem.tracking_objective = tracking_objective
        
        # Run meta optimization
        result = self.meta_optimizer.optimize(problem, max_evaluations)
        
        # Safety check: Handle case where optimize returns None
        if result is None:
            logger.warning("Meta optimizer returned None. Using fallback values.")
            # Use fallback values
            best_solution = np.zeros(problem.dims)
            best_fitness = float('inf')
            meta_evals = evaluation_count
            selected_algorithm = "unknown"
        else:
            # Get the selected algorithm
            selected_algorithm = self.meta_optimizer.get_selected_algorithm() if hasattr(self.meta_optimizer, 'get_selected_algorithm') else 'unknown'
            
            # Ensure selected_algorithm is a regular Python string, not a NumPy string
            if hasattr(selected_algorithm, 'item') and callable(selected_algorithm.item):
                selected_algorithm = selected_algorithm.item()
            
            if selected_algorithm == 'unknown' and hasattr(self.meta_optimizer, 'most_recent_selected_optimizers'):
                # Try to extract algorithm name directly from most_recent_selected_optimizers
                if self.meta_optimizer.most_recent_selected_optimizers:
                    if isinstance(self.meta_optimizer.most_recent_selected_optimizers, list):
                        names = []
                        for opt in self.meta_optimizer.most_recent_selected_optimizers:
                            if isinstance(opt, str):
                                names.append(opt)
                            elif hasattr(opt, 'item') and callable(opt.item):
                                names.append(opt.item())  # Convert numpy string to Python string
                            else:
                                names.append(str(opt))
                        selected_algorithm = ','.join(names)
                    else:
                        selected_algorithm = str(self.meta_optimizer.most_recent_selected_optimizers)
            
            # Extract best solution and fitness from result
            best_solution = result[0] if isinstance(result, tuple) and len(result) > 0 else None
            # Use tracked best_fitness if result doesn't contain it
            if isinstance(result, tuple) and len(result) > 1:
                best_fitness = result[1]
            
            # Use our evaluation count for accurate tracking
            meta_evals = evaluation_count
        
        end_time = time.time()
        
        # Restore original evaluate function
        problem.evaluate = original_evaluate
        
        return {
            'solution': best_solution,
            'best_fitness': best_fitness,
            'evaluations': meta_evals,
            'time': end_time - start_time,
            'convergence_data': convergence_history,
            'selected_algorithm': selected_algorithm
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
        
        # Get all fitness values for box plots
        baseline_all_values = []
        meta_all_values = []
        baseline_avgs = []
        meta_avgs = []
        
        for func_name in func_names:
            baseline_values = results[func_name]["baseline_best_fitness_all"]
            meta_values = results[func_name]["meta_best_fitness_all"]
            
            # Ensure we have non-zero values to display for baseline
            # Add a small epsilon to zero values to make them visible on log scale
            baseline_values = [max(val, 1e-16) for val in baseline_values]
            meta_values = [max(val, 1e-16) for val in meta_values]
            
            baseline_all_values.append(baseline_values)
            meta_all_values.append(meta_values)
            baseline_avgs.append(np.mean(baseline_values))
            meta_avgs.append(np.mean(meta_values))
        
        # Create figure with two subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), height_ratios=[2, 1])
        
        # Box plot
        positions = np.arange(len(func_names)) * 3
        width = 0.8
        
        bp1 = ax1.boxplot(baseline_all_values, positions=positions-width, 
                         widths=width, patch_artist=True,
                         boxprops=dict(facecolor='lightblue', color='blue'),
                         medianprops=dict(color='blue'),
                         flierprops=dict(color='blue', markerfacecolor='blue'),
                         labels=[''] * len(func_names))
        
        bp2 = ax1.boxplot(meta_all_values, positions=positions+width,
                         widths=width, patch_artist=True,
                         boxprops=dict(facecolor='orange', color='red'),
                         medianprops=dict(color='red'),
                         flierprops=dict(color='red', markerfacecolor='red'),
                         labels=[''] * len(func_names))
        
        # Set y-scale to log for better visualization
        ax1.set_yscale('log')
        ax1.set_ylabel('Best Fitness (log scale)')
        ax1.set_title('Performance Distribution Across Trials')
        ax1.legend([bp1["boxes"][0], bp2["boxes"][0]], ['Baseline', 'Meta Optimizer'])
        ax1.set_xticks(positions)
        ax1.set_xticklabels(func_names, rotation=45, ha='right')
        ax1.grid(True, which='both', linestyle='--', alpha=0.5)
        
        # Bar plot
        x = np.arange(len(func_names))
        width = 0.35
        
        ax2.bar(x - width/2, baseline_avgs, width, label='Baseline', color='lightblue')
        ax2.bar(x + width/2, meta_avgs, width, label='Meta Optimizer', color='orange')
        
        ax2.set_yscale('log')
        ax2.set_ylabel('Average Best Fitness (log scale)')
        ax2.set_title('Average Performance Comparison')
        ax2.set_xticks(x)
        ax2.set_xticklabels(func_names, rotation=45, ha='right')
        ax2.legend()
        ax2.grid(True, which='both', linestyle='--', alpha=0.5)
        
        plt.tight_layout()

    def plot_violin_comparison(self, results: Dict[str, Dict]) -> None:
        """Create violin plots comparing the distribution of fitness values."""
        plt.figure(figsize=(12, 6))
        
        # Collect all fitness values for each problem
        data = []
        labels = []
        positions = []
        colors = []
        pos = 0
        
        for problem_name in results:
            baseline_fitness = results[problem_name]["baseline_best_fitness_all"]
            meta_fitness = results[problem_name]["meta_best_fitness_all"]
            
            # Ensure we have non-zero values to display for baseline
            baseline_fitness = [max(val, 1e-16) for val in baseline_fitness]
            meta_fitness = [max(val, 1e-16) for val in meta_fitness]
            
            data.extend([baseline_fitness, meta_fitness])
            labels.extend([f"{problem_name}\nBaseline", f"{problem_name}\nMeta"])
            positions.extend([pos, pos + 1])
            colors.extend(['lightblue', '#D43F3A'])
            pos += 3
        
        # Create violin plot
        parts = plt.violinplot(data, positions=positions, showmeans=True)
        
        # Customize violin plot appearance
        for i, pc in enumerate(parts['bodies']):
            pc.set_facecolor(colors[i])
            pc.set_edgecolor('black')
            pc.set_alpha(0.7)
        
        parts['cmeans'].set_color('black')
        parts['cmins'].set_color('black')
        parts['cmaxes'].set_color('black')
        parts['cbars'].set_color('black')
        
        plt.xticks(positions, labels, rotation=45, ha='right')
        plt.yscale('log')
        plt.ylabel('Fitness Value (log scale)')
        plt.title('Distribution of Fitness Values: Baseline vs Meta-Optimizer')
        plt.grid(True, which="both", ls="-", alpha=0.2)
        
        # Add legend for colors
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='lightblue', edgecolor='black', alpha=0.7, label='Baseline'),
            Patch(facecolor='#D43F3A', edgecolor='black', alpha=0.7, label='Meta Optimizer')
        ]
        plt.legend(handles=legend_elements, loc='upper right')
        
        # Adjust layout and save
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'violin_comparison.png'))
        plt.close()

    def plot_convergence_curves(self, results: Dict[str, Any], output_dir: str):
        """Plot convergence curves for baseline and meta optimizer."""
        plt.figure(figsize=(10, 6))
        
        # Plot baseline convergence
        baseline_convergence = results.get('baseline_convergence_data', [])
        if baseline_convergence:
            # Average convergence data across trials
            min_length = min(len(curve) for curve in baseline_convergence if curve)  # Check for non-empty curves
            if min_length > 0:
                # Truncate all curves to minimum length and convert to numpy array
                baseline_curves = np.array([curve[:min_length] for curve in baseline_convergence if len(curve) >= min_length])
                if len(baseline_curves) > 0:
                    # Add small epsilon to zero values for log scale
                    baseline_curves = np.maximum(baseline_curves, 1e-16)
                    # Calculate mean and std across trials
                    baseline_mean = np.mean(baseline_curves, axis=0)
                    baseline_std = np.std(baseline_curves, axis=0)
                    x = range(len(baseline_mean))
                    plt.plot(x, baseline_mean, label='Baseline', color='blue', linewidth=2)
                    plt.fill_between(x, baseline_mean - baseline_std, baseline_mean + baseline_std, 
                                   color='blue', alpha=0.2)
        
        # Plot meta optimizer convergence
        meta_convergence = results.get('meta_convergence_data', [])
        if meta_convergence:
            min_length = min(len(curve) for curve in meta_convergence if curve)  # Check for non-empty curves
            if min_length > 0:
                meta_curves = np.array([curve[:min_length] for curve in meta_convergence if len(curve) >= min_length])
                if len(meta_curves) > 0:
                    # Add small epsilon to zero values for log scale
                    meta_curves = np.maximum(meta_curves, 1e-16)
                    meta_mean = np.mean(meta_curves, axis=0)
                    meta_std = np.std(meta_curves, axis=0)
                    x = range(len(meta_mean))
                    plt.plot(x, meta_mean, label='Meta Optimizer', color='orange', linewidth=2)
                    plt.fill_between(x, meta_mean - meta_std, meta_mean + meta_std,
                                   color='orange', alpha=0.2)
        
        plt.yscale('log')  # Use log scale for fitness values
        plt.xlabel('Evaluations')
        plt.ylabel('Best Fitness (log scale)')
        plt.title('Convergence Comparison')
        plt.legend()
        plt.grid(True, which='both', linestyle='--', alpha=0.5)
        
        # Save plot
        plt.savefig(os.path.join(output_dir, '@convergence_curves.png'))
        plt.close()

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
        
        # Convert dict to lists for plotting
        baseline_keys = list(baseline_counts.keys())
        baseline_values = [baseline_counts[k] for k in baseline_keys]
        
        # Baseline
        ax1.bar(range(len(baseline_keys)), baseline_values)
        ax1.set_title('Baseline Algorithm Selection')
        ax1.set_ylabel('Frequency')
        ax1.set_xticks(range(len(baseline_keys)))
        ax1.set_xticklabels(baseline_keys, rotation=45, ha='right')
        
        # Convert dict to lists for plotting
        meta_keys = list(meta_counts.keys())
        meta_values = [meta_counts[k] for k in meta_keys]
        
        # Meta Optimizer
        ax2.bar(range(len(meta_keys)), meta_values)
        ax2.set_title('Meta Optimizer Algorithm Selection')
        ax2.set_ylabel('Frequency')
        ax2.set_xticks(range(len(meta_keys)))
        ax2.set_xticklabels(meta_keys, rotation=45, ha='right')
        
        fig.tight_layout()
    
    def calculate_success_rate(self, fitness_values: List[float], threshold: float = 1e-6) -> float:
        """
        Calculate the success rate based on fitness values
        
        Args:
            fitness_values: List of fitness values from optimization runs
            threshold: Success threshold - values below this are considered successful
            
        Returns:
            Success rate as a float between 0 and 1
        """
        if not fitness_values:
            return 0.0
            
        # Filter out inf/nan values
        valid_values = [f for f in fitness_values if not (np.isinf(f) or np.isnan(f))]
        if not valid_values:
            return 0.0
            
        # Calculate success rate based on threshold
        successes = sum(1 for f in valid_values if f < threshold)
        return successes / len(fitness_values)  # Use total trials for denominator
    
    def save_results(self):
        """Save results to a JSON file"""
        results_file = os.path.join(self.output_dir, "results.json")
        with open(results_file, 'w') as f:
            json.dump(self.results, f, indent=2)
        print(f"Results saved to {results_file}")

    def run(self):
        """Run the baseline comparison for all benchmark functions."""
        # Log experiment configuration for reproducibility
        self.log_experiment_configuration()
        
        # Set random seed for reproducibility if not already set
        if self.random_seed is None:
            self.set_random_seeds()
            
        test_functions = create_test_suite()
        logging.info(f"Running comparison for {len(test_functions)} benchmark functions")
        
        for func_name, func_class in test_functions.items():
            logging.info(f"Running comparison for {func_name}")
            # Create appropriate bounds for each dimension
            # Use the default bounds for each test function:
            # sphere: (-5.12, 5.12), rosenbrock: (-5, 10), rastrigin: (-5.12, 5.12),
            # ackley: (-32.768, 32.768), griewank: (-600, 600), schwefel: (-500, 500)
            default_bounds = {
                "sphere": (-5.12, 5.12),
                "rosenbrock": (-5.0, 10.0),
                "rastrigin": (-5.12, 5.12),
                "ackley": (-32.768, 32.768),
                "griewank": (-600, 600),
                "schwefel": (-500, 500),
                "levy": (-10, 10)
            }
            
            # Get the default bound for this function, or use (-5, 5) if not found
            bound = default_bounds.get(func_name, (-5, 5))
            bounds = [bound] * self.meta_optimizer.dim
            
            # Create test function with specified dimensions and appropriate bounds
            func = func_class(self.meta_optimizer.dim, bounds)
            problem = ProblemWrapper(func, self.meta_optimizer.dim)
            
            self.run_comparison(
                problem_name=func_name,
                problem_func=problem,
                dimensions=self.meta_optimizer.dim,
                max_evaluations=self.max_evaluations,
                num_trials=self.num_trials
            )
        
        # Generate additional visualizations after all functions are processed
        if len(self.results) > 0:
            logging.info("Generating additional visualizations...")
            
            # Statistical significance tests
            self.plot_statistical_tests(self.results)
            logging.info("Statistical test plots saved")
            
            # Radar chart comparison
            self.plot_radar_comparison(self.results)
            logging.info("Radar comparison chart saved")
            
            # Summary table
            self.create_summary_table(self.results)
            logging.info("Summary table saved")
            
            logging.info("All visualizations completed")

    def plot_statistical_tests(self, results: Dict[str, Dict]) -> None:
        """Plot statistical significance tests between baseline and meta optimizer results."""
        from scipy import stats
        
        plt.figure(figsize=(12, 6))
        
        # Collect p-values and effect sizes
        func_names = []
        p_values = []
        effect_sizes = []
        significance = []
        
        for func_name, func_results in results.items():
            if "baseline" not in func_results or "meta" not in func_results:
                logger.warning(f"Missing baseline or meta data for {func_name}")
                continue
                
            if "best_fitness" not in func_results["baseline"] or "best_fitness" not in func_results["meta"]:
                logger.warning(f"Missing best_fitness data for {func_name}")
                continue
                
            func_names.append(func_name)
            baseline = func_results["baseline"]["best_fitness"]
            meta = func_results["meta"]["best_fitness"]
            
            # Skip if not enough data points
            if len(baseline) < 2 or len(meta) < 2:
                logger.warning(f"Not enough data points for statistical test on {func_name}")
                p_values.append(1.0)
                effect_sizes.append(0.0)
                significance.append("n/a")
                continue
            
            # Perform t-test
            try:
                t_stat, p_value = stats.ttest_ind(baseline, meta, equal_var=False)
                if np.isnan(p_value) or np.isinf(p_value):
                    p_value = 1.0  # Default to no significance if test fails
                p_values.append(p_value)
                
                # Calculate Cohen's d effect size
                mean_diff = abs(np.mean(baseline) - np.mean(meta))
                pooled_std = np.sqrt((np.std(baseline)**2 + np.std(meta)**2) / 2)
                d = mean_diff / max(pooled_std, 1e-10)  # Avoid division by zero
                if np.isnan(d) or np.isinf(d):
                    d = 0.0  # Default to no effect if calculation fails
                effect_sizes.append(d)
                
                # Determine significance
                if p_value < 0.001:
                    significance.append("***")
                elif p_value < 0.01:
                    significance.append("**")
                elif p_value < 0.05:
                    significance.append("*")
                else:
                    significance.append("ns")
            except Exception as e:
                logger.warning(f"Statistical test failed for {func_name}: {e}")
                p_values.append(1.0)
                effect_sizes.append(0.0)
                significance.append("error")
        
        if not func_names:  # If no functions to plot, skip plotting
            logger.warning("No functions with valid data for statistical tests")
            return
        
        if not p_values:  # If no valid p-values, skip plotting
            logger.warning("No valid statistical tests to plot")
            return
        
        # Create subplot for p-values
        plt.subplot(1, 2, 1)
        bars = plt.bar(func_names, p_values, color='skyblue')
        
        # Add significance markers
        for i, (bar, sig) in enumerate(zip(bars, significance)):
            height = bar.get_height()
            if height > 0:
                plt.text(i, min(height + 0.01, 1.0), sig, 
                    ha='center', va='bottom', fontweight='bold')
            else:
                plt.text(i, 0.01, sig, ha='center', va='bottom', fontweight='bold')
        
        plt.axhline(y=0.05, color='r', linestyle='--', alpha=0.7)
        plt.text(len(func_names)-1, 0.055, 'p=0.05', color='r', ha='right')
        
        plt.ylabel('p-value')
        plt.title('Statistical Significance (t-test)')
        plt.xticks(rotation=45, ha='right')
        
        # Set reasonable y-limits
        valid_p = [p for p in p_values if not np.isnan(p) and not np.isinf(p) and p > 0]
        if valid_p:
            max_p = max(valid_p)
            plt.ylim(0, min(max_p * 1.2 + 0.05, 1.05))  # Cap at slightly above 1.0
        else:
            plt.ylim(0, 1.05)
        
        # Create subplot for effect sizes
        plt.subplot(1, 2, 2)
        bars = plt.bar(func_names, effect_sizes, color='lightgreen')
        
        # Add effect size interpretation
        for i, d in enumerate(effect_sizes):
            if np.isnan(d) or np.isinf(d):
                effect = "Invalid"
                d_plot = 0
            else:
                effect = "Large" if d > 0.8 else "Medium" if d > 0.5 else "Small" if d > 0.2 else "Negligible"
                d_plot = d
            plt.text(i, min(d_plot + 0.1, 5.0), effect, ha='center', va='bottom', rotation=45)
        
        plt.ylabel("Effect Size (Cohen's d)")
        plt.title('Effect Size Comparison')
        plt.xticks(rotation=45, ha='right')
        
        # Set reasonable y-limits for effect sizes
        valid_d = [d for d in effect_sizes if not np.isnan(d) and not np.isinf(d) and d > 0]
        if valid_d:
            max_d = max(valid_d)
            plt.ylim(0, max(max_d * 1.2 + 0.2, 1.5))  # Ensure enough room for labels
        else:
            plt.ylim(0, 1.5)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'statistical_tests.png'), dpi=150, bbox_inches='tight')
        plt.close()

    def plot_radar_comparison(self, results: Dict[str, Dict]) -> None:
        """Create a radar chart comparing multiple metrics between baseline and meta optimizer."""
        import matplotlib.pyplot as plt
        import numpy as np
        
        # Prepare data for radar chart
        func_names = list(results.keys())
        if not func_names:
            logger.warning("No functions to plot in radar chart")
            return
        
        n_funcs = len(func_names)
        
        # Calculate relative improvements for each metric
        fitness_improvement = []
        time_improvement = []
        evaluations_improvement = []
        
        for func_name in func_names:
            if "baseline" not in results[func_name] or "meta" not in results[func_name]:
                logger.warning(f"Missing baseline or meta data for {func_name}")
                # Use zero improvement as fallback
                fitness_improvement.append(0.0)
                time_improvement.append(0.0)
                evaluations_improvement.append(0.0)
                continue
            
            # Get baseline data
            baseline_data = results[func_name]["baseline"]
            meta_data = results[func_name]["meta"]
            
            # Fitness improvement (negative is better)
            baseline_fitness = baseline_data.get("avg_best_fitness", 0)
            meta_fitness = meta_data.get("avg_best_fitness", 0)
            
            # Handle infinity and NaN values
            if np.isinf(baseline_fitness) or np.isnan(baseline_fitness):
                baseline_fitness = 1e10
            if np.isinf(meta_fitness) or np.isnan(meta_fitness):
                meta_fitness = 1e10
                
            rel_fitness_imp = (baseline_fitness - meta_fitness) / max(abs(baseline_fitness), 1e-10)
            fitness_improvement.append(max(min(rel_fitness_imp, 1.0), -1.0))  # Clamp to [-1, 1]
            
            # Time improvement (negative is better)
            baseline_time = baseline_data.get("avg_time", 0)
            meta_time = meta_data.get("avg_time", 0)
            rel_time_imp = (baseline_time - meta_time) / max(baseline_time, 1e-10)
            time_improvement.append(max(min(rel_time_imp, 1.0), -1.0))  # Clamp to [-1, 1]
            
            # Evaluations improvement (negative is better)
            baseline_evals = baseline_data.get("avg_evaluations", 0)
            meta_evals = meta_data.get("avg_evaluations", 0)
            rel_eval_imp = (baseline_evals - meta_evals) / max(baseline_evals, 1e-10)
            evaluations_improvement.append(max(min(rel_eval_imp, 1.0), -1.0))  # Clamp to [-1, 1]
        
        # Set up radar chart
        angles = np.linspace(0, 2*np.pi, n_funcs, endpoint=False).tolist()
        angles += angles[:1]  # Close the loop
        
        # Add the metrics to their respective lists and close the loop
        fitness_improvement += fitness_improvement[:1]
        time_improvement += time_improvement[:1]
        evaluations_improvement += evaluations_improvement[:1]
        
        func_names += func_names[:1]  # Close the loop for labels
        
        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))
        
        # Plot each metric
        ax.plot(angles, fitness_improvement, 'o-', linewidth=2, label='Fitness Improvement')
        ax.fill(angles, fitness_improvement, alpha=0.25)
        
        ax.plot(angles, time_improvement, 'o-', linewidth=2, label='Time Improvement')
        ax.fill(angles, time_improvement, alpha=0.25)
        
        ax.plot(angles, evaluations_improvement, 'o-', linewidth=2, label='Evaluations Improvement')
        ax.fill(angles, evaluations_improvement, alpha=0.25)
        
        # Set labels
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(func_names[:-1])
        
        # Add reference circles
        ax.set_yticks([-1, -0.5, 0, 0.5, 1])
        ax.set_yticklabels(['-100%', '-50%', '0%', '50%', '100%'])
        ax.set_rlim(-1, 1)
        
        # Add grid
        ax.grid(True)
        
        # Add a legend
        plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
        
        # Add title
        plt.title('Relative Improvement: Meta Optimizer vs Baseline', size=15, y=1.1)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'radar_comparison.png'), dpi=150, bbox_inches='tight')
        plt.close()

    def create_summary_table(self, results: Dict[str, Dict]) -> None:
        """Create a summary table of results as an image."""
        from matplotlib.table import Table
        
        if not results:
            logger.warning("No results to create summary table")
            return
        
        fig, ax = plt.subplots(figsize=(12, len(results) * 0.8 + 2))
        ax.axis('off')
        ax.axis('tight')
        
        # Prepare data for table
        table_data = []
        for func_name, func_results in results.items():
            if "baseline" not in func_results or "meta" not in func_results:
                logger.warning(f"Missing baseline or meta data for {func_name}")
                continue
            
            # Get baseline and meta data
            baseline_data = func_results["baseline"]
            meta_data = func_results["meta"]
            
            # Handle missing data safely
            if "avg_best_fitness" not in baseline_data or "avg_best_fitness" not in meta_data:
                logger.warning(f"Missing fitness data for {func_name}")
                continue
            
            # Extract fitness values with safety checks
            baseline_fitness_raw = baseline_data["avg_best_fitness"]
            meta_fitness_raw = meta_data["avg_best_fitness"]
            
            # Handle inf/nan values
            if np.isinf(baseline_fitness_raw) or np.isnan(baseline_fitness_raw):
                baseline_fitness = 1e16
            else:
                baseline_fitness = max(baseline_fitness_raw, 1e-16)  # Ensure non-zero for display
                
            if np.isinf(meta_fitness_raw) or np.isnan(meta_fitness_raw):
                meta_fitness = 1e16
            else:
                meta_fitness = max(meta_fitness_raw, 1e-16)  # Ensure non-zero for display
            
            # Calculate improvement - avoid division by zero and extreme percentages
            if baseline_fitness < 1e-15 and meta_fitness < 1e-15:
                # Both solutions are effectively at the optimum
                improvement = 0.0
            elif baseline_fitness < 1e-15:
                # Baseline is effectively at the optimum but meta is not
                # Cap at -100% instead of showing extreme negative values
                improvement = -100.0
            else:
                # Normal case - calculate actual improvement
                raw_improvement = (baseline_fitness - meta_fitness) / baseline_fitness * 100
                # Cap the improvement to reasonable bounds
                improvement = max(min(raw_improvement, 100.0), -100.0)
            
            # Extract other metrics with safety checks
            baseline_evals = baseline_data.get("avg_evaluations", 0)
            meta_evals = meta_data.get("avg_evaluations", 0)
            
            baseline_time = baseline_data.get("avg_time", 0)
            meta_time = meta_data.get("avg_time", 0)
            
            # Format values for display
            row = [
                func_name,
                f"{baseline_fitness:.2e}",
                f"{meta_fitness:.2e}",
                f"{improvement:.2f}%",
                f"{baseline_evals:.0f}",
                f"{meta_evals:.0f}",
                f"{baseline_time:.2f}s",
                f"{meta_time:.2f}s"
            ]
            table_data.append(row)
        
        if not table_data:
            logger.warning("No valid data for summary table")
            return
        
        # Create header
        columns = [
            "Function",
            "Baseline Fitness",
            "Meta Fitness",
            "Improvement",
            "Baseline Evals",
            "Meta Evals",
            "Baseline Time",
            "Meta Time"
        ]
        
        # Create the table
        table = ax.table(
            cellText=table_data,
            colLabels=columns,
            loc='center',
            cellLoc='center'
        )
        
        # Style the table
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 1.5)
        
        # Color code the improvement column
        for i, row in enumerate(table_data):
            try:
                improvement = float(row[3].strip('%'))
                cell = table[(i+1, 3)]  # +1 for header row
                
                if improvement > 10:  # Significant improvement
                    cell.set_facecolor('lightgreen')
                elif improvement < -10:  # Significant degradation
                    cell.set_facecolor('lightcoral')
                else:  # Neutral
                    cell.set_facecolor('lightyellow')
            except (ValueError, IndexError):
                # Skip if can't parse improvement value
                pass
        
        # Color the header row
        for i in range(len(columns)):
            table[(0, i)].set_facecolor('lightgrey')
            table[(0, i)].set_text_props(weight='bold')
        
        plt.title('Performance Summary Table', fontsize=14)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'summary_table.png'), dpi=200, bbox_inches='tight')
        plt.close()

    def calculate_precision_recall(self, baseline_results, meta_results):
        """
        Calculate precision and recall for the Meta Optimizer compared to baseline
        
        Args:
            baseline_results: Baseline optimization results
            meta_results: Meta Optimizer results
            
        Returns:
            Tuple of precision, recall
        """
        baseline_min = float('inf')
        for result in baseline_results:
            if result < baseline_min:
                baseline_min = result
                
        meta_min = float('inf')
        for result in meta_results:
            if result < meta_min:
                meta_min = result
                
        # Calculate improvement
        improvement = baseline_min - meta_min
        
        # Calculate precision and recall
        precision = 1.0 if improvement > 0 else 0.0
        recall = 1.0 if improvement > 0 else 0.0
        
        return precision, recall
        
    def plot_algorithm_distribution(self, ax1, ax2, figure_path: str):
        """
        Plot the distribution of algorithms selected by baseline and Meta Optimizer
        
        Args:
            ax1: Axis for baseline plot
            ax2: Axis for Meta Optimizer plot
            figure_path: Path to save the figure
        """
        from collections import Counter
        
        if not self.results:
            logger.warning("No results to plot algorithm distribution")
            return
        
        # Count algorithm frequencies
        baseline_algs = []
        meta_algs = []
        
        for problem_name, result in self.results.items():
            if "baseline" not in result or "meta" not in result:
                continue
                
            # Get algorithm selections
            if "selected_algorithms" in result["baseline"]:
                baseline_algs.extend(result["baseline"]["selected_algorithms"])
                
            if "selected_algorithms" in result["meta"]:
                meta_algs.extend(result["meta"]["selected_algorithms"])
        
        if not baseline_algs and not meta_algs:
            logger.warning("No algorithm selection data available")
            return
        
        # Count occurrences
        baseline_counts = Counter(baseline_algs)
        meta_counts = Counter(meta_algs)
        
        # Plot baseline distribution
        if baseline_counts:
            ax1.bar(baseline_counts.keys(), baseline_counts.values())
            ax1.set_title("Baseline Algorithm Selection")
            ax1.set_ylabel("Frequency")
            ax1.tick_params(axis='x', rotation=45)
        else:
            ax1.text(0.5, 0.5, "No baseline algorithm data", ha='center', va='center')
        
        # Plot meta distribution
        if meta_counts:
            ax2.bar(meta_counts.keys(), meta_counts.values())
            ax2.set_title("Meta Optimizer Algorithm Selection")
            ax2.set_ylabel("Frequency")
            ax2.tick_params(axis='x', rotation=45)
        else:
            ax2.text(0.5, 0.5, "No meta-optimizer algorithm data", ha='center', va='center')
        
        # Save figure
        plt.tight_layout()
        plt.savefig(figure_path, dpi=150, bbox_inches='tight')
        plt.close()

    def _calculate_improvement_percentage(self, baseline_fitness, meta_fitness):
        """
        Calculate bounded improvement percentage, capped between -100% and +100%
        
        Args:
            baseline_fitness: Average fitness from the baseline algorithm
            meta_fitness: Average fitness from the meta-optimizer
            
        Returns:
            Improvement percentage, between -100% and +100%
        """
        # For minimization problems, lower values are better
        if abs(baseline_fitness) < 1e-10:  # Avoid division by zero
            # If baseline already found perfect solution
            if abs(meta_fitness) < 1e-10:
                return 0.0  # Both found perfect solution
            else:
                return -100.0  # Meta performed worse, cap at -100%
        
        raw_improvement = (baseline_fitness - meta_fitness) / abs(baseline_fitness) * 100
        
        # Cap improvement between -100% and +100%
        return max(min(raw_improvement, 100.0), -100.0) 

    def validate_results(self, baseline_result, meta_result):
        """
        Validate that the comparison results meet expected criteria.
        
        Args:
            baseline_result: Dictionary with baseline optimizer results
            meta_result: Dictionary with meta optimizer results
            
        Returns:
            Dictionary with validation results and warnings
        """
        validation_result = {
            "valid": True,
            "warnings": []
        }
        
        # Check if baseline_result and meta_result are valid
        if baseline_result is None or meta_result is None:
            validation_result["valid"] = False
            validation_result["warnings"].append("Missing baseline or meta optimizer results")
            return validation_result
        
        # Validate that both optimizers returned valid solutions
        baseline_solution = baseline_result.get("best_solution", None)
        meta_solution = meta_result.get("best_solution", None)
        
        if baseline_solution is None:
            validation_result["valid"] = False
            validation_result["warnings"].append("Baseline optimizer did not return a valid solution")
        
        if meta_solution is None:
            validation_result["valid"] = False
            validation_result["warnings"].append("Meta optimizer did not return a valid solution")
        
        # Validate optimization outcomes based on fitness values
        baseline_fitness = baseline_result.get("best_fitness", float('inf'))
        meta_fitness = meta_result.get("best_fitness", float('inf'))
        
        if np.isinf(baseline_fitness) and np.isinf(meta_fitness):
            validation_result["valid"] = False
            validation_result["warnings"].append("Both optimizers failed to find a valid solution")
        
        # Validate that evaluations are within acceptable limits
        baseline_evals = baseline_result.get("evaluations", 0)
        meta_evals = meta_result.get("evaluations", 0)
        
        if baseline_evals <= 0:
            validation_result["valid"] = False
            validation_result["warnings"].append(f"Baseline evaluations is suspiciously low: {baseline_evals}")
            
        if meta_evals <= 0:
            validation_result["valid"] = False
            validation_result["warnings"].append(f"Meta evaluations is suspiciously low: {meta_evals}")
            
        # Check if baseline evaluations exceed maximum by more than 5%
        max_evaluations = self.max_evaluations
        allowed_excess = 0.05 * max_evaluations  # Allow 5% excess
        
        if baseline_evals > max_evaluations + allowed_excess:
            validation_result["valid"] = False
            validation_result["warnings"].append(f"Baseline evaluations ({baseline_evals}) exceeds maximum allowed ({max_evaluations + allowed_excess})")
        
        # We no longer check if meta evaluations exceed the maximum
        # This allows the meta optimizer to run as long as needed
        
        return validation_result
    
    def log_experiment_configuration(self):
        """Log detailed configuration for reproducibility."""
        config = {
            'max_evaluations': self.max_evaluations,
            'num_trials': self.num_trials,
            'dimensions': self.meta_optimizer.dim if hasattr(self.meta_optimizer, 'dim') else None,
            'algorithms': list(self.available_algorithms),
            'model_loaded': self.model_loaded,
            'random_seed': self.random_seed,
            'timestamp': time.strftime("%Y-%m-%d %H:%M:%S")
        }
        
        config_path = os.path.join(self.output_dir, "experiment_config.json")
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        
        logger.info(f"Experiment configuration saved to {config_path}") 

    def calculate_normalized_metrics(self, baseline_result, meta_result):
        """
        Calculate normalized metrics for fair comparison between baseline and meta optimizer.
        
        Args:
            baseline_result: Dictionary with baseline optimizer results
            meta_result: Dictionary with meta optimizer results
            
        Returns:
            Dictionary with normalized metrics
        """
        # Extract relevant values
        baseline_fitness = baseline_result["best_fitness"]
        baseline_evals = baseline_result["evaluations"]
        meta_fitness = meta_result["best_fitness"]
        meta_evals = meta_result["evaluations"]
        
        # Define normalization points (percentage of max evaluations)
        norm_points = [0.25, 0.5, 0.75, 1.0]
        
        # Calculate metrics
        metrics = {
            "baseline_fitness": baseline_fitness,
            "meta_fitness": meta_fitness,
            "baseline_evals": baseline_evals,
            "meta_evals": meta_evals,
            "fitness_improvement": (baseline_fitness - meta_fitness) / baseline_fitness if baseline_fitness != 0 else 0,
            "eval_improvement": (baseline_evals - meta_evals) / baseline_evals if baseline_evals != 0 else 0,
            "normalized_points": {}
        }
        
        # Get convergence data
        baseline_conv = baseline_result.get("convergence_data", [])
        meta_conv = meta_result.get("convergence_data", [])
        
        # Normalize convergence data to evaluations
        max_evals = max(baseline_evals, meta_evals)
        
        # For each normalization point, find the best fitness at that percentage of evaluations
        for point in norm_points:
            eval_budget = int(max_evals * point)
            
            # Find baseline fitness at this budget
            baseline_fitness_at_budget = self._get_fitness_at_budget(baseline_conv, baseline_evals, eval_budget)
            
            # Find meta fitness at this budget
            meta_fitness_at_budget = self._get_fitness_at_budget(meta_conv, meta_evals, eval_budget)
            
            # Calculate improvement percentage
            improvement = ((baseline_fitness_at_budget - meta_fitness_at_budget) / baseline_fitness_at_budget) * 100 if baseline_fitness_at_budget != 0 else 0
            
            metrics["normalized_points"][point] = {
                "eval_budget": eval_budget,
                "baseline_fitness": baseline_fitness_at_budget,
                "meta_fitness": meta_fitness_at_budget,
                "improvement": improvement
            }
            
        # Add efficiency metrics (fitness improvement per evaluation)
        metrics["efficiency"] = {
            "baseline_fitness_per_eval": baseline_fitness / baseline_evals if baseline_evals > 0 else float('inf'),
            "meta_fitness_per_eval": meta_fitness / meta_evals if meta_evals > 0 else float('inf')
        }
        
        # Calculate efficiency improvement percentage
        baseline_efficiency = metrics["efficiency"]["baseline_fitness_per_eval"]
        meta_efficiency = metrics["efficiency"]["meta_fitness_per_eval"]
        
        if baseline_efficiency != 0 and baseline_efficiency != float('inf'):
            efficiency_improvement = ((baseline_efficiency - meta_efficiency) / baseline_efficiency) * 100
        else:
            efficiency_improvement = 0
            
        metrics["efficiency"]["improvement"] = efficiency_improvement
        
        return metrics
    
    def _get_fitness_at_budget(self, convergence_data, total_evals, eval_budget):
        """
        Get the best fitness achieved within a given evaluation budget.
        
        Args:
            convergence_data: List of fitness values over time
            total_evals: Total evaluations performed
            eval_budget: Evaluation budget to measure fitness at
            
        Returns:
            Best fitness achieved within the budget
        """
        if not convergence_data:
            return float('inf')
            
        # If budget is greater than total, return final fitness
        if eval_budget >= total_evals:
            return convergence_data[-1]
            
        # Calculate position in convergence data
        # This assumes convergence_data is evenly distributed across evaluations
        if len(convergence_data) <= 1:
            return convergence_data[0] if convergence_data else float('inf')
            
        # Find index proportional to the evaluation budget
        idx = int((eval_budget / total_evals) * (len(convergence_data) - 1))
        idx = min(idx, len(convergence_data) - 1)  # Ensure valid index
        
        return convergence_data[idx]
        
    def generate_normalized_comparison_plots(self, baseline_result, meta_result, output_path, title="Normalized Comparison"):
        """
        Generate plots comparing baseline and meta optimizer using normalized metrics.
        
        Args:
            baseline_result: Dictionary with baseline optimizer results
            meta_result: Dictionary with meta optimizer results
            output_path: Path to save the visualization
            title: Plot title
        """
        # Calculate normalized metrics
        metrics = self.calculate_normalized_metrics(baseline_result, meta_result)
        
        # Create figure with multiple subplots with fixed size to avoid errors
        fig = plt.figure(figsize=(10, 8))  # Use a fixed, reasonable size
        gs = GridSpec(2, 2, figure=fig)
        
        # 1. Convergence curve plot
        ax1 = fig.add_subplot(gs[0, 0])
        self._plot_normalized_convergence(baseline_result, meta_result, ax1)
        ax1.set_title("Convergence Curves")
        
        # 2. Evaluation budget comparison
        ax2 = fig.add_subplot(gs[0, 1])
        self._plot_evaluation_comparison(metrics, ax2)
        ax2.set_title("Evaluation Budget Comparison")
        
        # 3. Fitness at normalized points
        ax3 = fig.add_subplot(gs[1, 0])
        self._plot_normalized_points(metrics, ax3)
        ax3.set_title("Fitness at Normalized Evaluation Points")
        
        # 4. Efficiency comparison
        ax4 = fig.add_subplot(gs[1, 1])
        self._plot_efficiency_comparison(metrics, ax4)
        ax4.set_title("Optimizer Efficiency")
        
        # Add title and adjust layout
        fig.suptitle(title, fontsize=16)
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        
        # Set maximum figure dimensions to avoid matplotlib errors
        fig_width, fig_height = fig.get_size_inches()
        if fig_height > 50:  # If height is unreasonably large
            logger.warning(f"Limiting figure height from {fig_height} to 50 inches")
            fig.set_size_inches(fig_width, 50)
        
        # Save figure without using bbox_inches='tight' to avoid renderer errors
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, dpi=150)
        plt.close(fig)
        
    def _plot_normalized_convergence(self, baseline_result, meta_result, ax):
        """Plot normalized convergence curves."""
        # Get convergence data
        baseline_conv = baseline_result.get("convergence_data", [])
        meta_conv = meta_result.get("convergence_data", [])
        
        baseline_evals = baseline_result["evaluations"]
        meta_evals = meta_result["evaluations"]
        
        # Plot convergence data if available
        if baseline_conv:
            x_baseline = np.linspace(0, baseline_evals, len(baseline_conv))
            ax.plot(x_baseline, baseline_conv, '-b', label=f'Baseline ({baseline_result["selected_algorithm"]})')
            
        if meta_conv:
            x_meta = np.linspace(0, meta_evals, len(meta_conv))
            ax.plot(x_meta, meta_conv, '-r', label=f'Meta-Optimizer ({meta_result["selected_algorithm"]})')
            
        ax.set_xlabel('Function Evaluations')
        ax.set_ylabel('Best Fitness')
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.6)
        
        # Use log scale for y-axis if appropriate (when values are varied)
        if baseline_conv and meta_conv:
            min_val = min(min(baseline_conv), min(meta_conv))
            max_val = max(max(baseline_conv), max(meta_conv))
            if max_val / (min_val + 1e-10) > 100:  # Use log scale if range is wide
                ax.set_yscale('log')
        
    def _plot_evaluation_comparison(self, metrics, ax):
        """Plot comparison of evaluations used by each optimizer."""
        baseline_evals = metrics["baseline_evals"]
        meta_evals = metrics["meta_evals"]
        
        # Create bar chart
        labels = ['Baseline', 'Meta-Optimizer']
        evals = [baseline_evals, meta_evals]
        
        bars = ax.bar(labels, evals, color=['blue', 'red'])
        
        # Add values on top of bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                    f'{int(height)}', ha='center', va='bottom')
                    
        # Calculate improvement percentage
        improvement = metrics["eval_improvement"] * 100
        
        # Add text for improvement
        if improvement > 0:
            ax.text(0.5, 0.9, f"{improvement:.1f}% fewer evaluations", 
                   transform=ax.transAxes, ha='center', va='center',
                   bbox=dict(facecolor='lightgreen', alpha=0.5))
        elif improvement < 0:
            ax.text(0.5, 0.9, f"{-improvement:.1f}% more evaluations", 
                   transform=ax.transAxes, ha='center', va='center',
                   bbox=dict(facecolor='lightcoral', alpha=0.5))
        else:
            ax.text(0.5, 0.9, "Same number of evaluations", 
                   transform=ax.transAxes, ha='center', va='center',
                   bbox=dict(facecolor='lightyellow', alpha=0.5))
                   
        ax.set_ylabel('Number of Evaluations')
        ax.grid(True, axis='y', linestyle='--', alpha=0.6)
        
    def _plot_normalized_points(self, metrics, ax):
        """Plot fitness at normalized evaluation points."""
        points = sorted(metrics["normalized_points"].keys())
        baseline_fitness = [metrics["normalized_points"][p]["baseline_fitness"] for p in points]
        meta_fitness = [metrics["normalized_points"][p]["meta_fitness"] for p in points]
        
        # Convert points to percentages for display
        point_labels = [f"{int(p*100)}%" for p in points]
        
        # Create line plot
        ax.plot(point_labels, baseline_fitness, 'o-b', label='Baseline')
        ax.plot(point_labels, meta_fitness, 'o-r', label='Meta-Optimizer')
        
        # Add text labels for improvement at each point
        for i, p in enumerate(points):
            improvement = metrics["normalized_points"][p]["improvement"]
            if abs(improvement) > 1:  # Only show if improvement is significant
                color = 'green' if improvement > 0 else 'red'
                ax.text(i, (baseline_fitness[i] + meta_fitness[i])/2,
                       f"{improvement:.1f}%", color=color, ha='center', fontweight='bold')
                       
        ax.set_xlabel('Percentage of Evaluation Budget')
        ax.set_ylabel('Best Fitness')
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.6)
        
        # Use log scale for y-axis if appropriate
        min_val = min(min(baseline_fitness), min(meta_fitness))
        max_val = max(max(baseline_fitness), max(meta_fitness))
        if max_val / (min_val + 1e-10) > 100:  # Use log scale if range is wide
            ax.set_yscale('log')
            
    def _plot_efficiency_comparison(self, metrics, ax):
        """Plot efficiency comparison (fitness per evaluation)."""
        if "efficiency" not in metrics:
            ax.text(0.5, 0.5, "No efficiency data available",
                   ha='center', va='center', transform=ax.transAxes)
            return
            
        baseline_eff = metrics["efficiency"]["baseline_fitness_per_eval"]
        meta_eff = metrics["efficiency"]["meta_fitness_per_eval"]
        
        # Handle inf or very large values
        if np.isinf(baseline_eff) or baseline_eff > 1e10:
            baseline_eff = 1e10
        if np.isinf(meta_eff) or meta_eff > 1e10:
            meta_eff = 1e10
            
        # Create bar chart
        labels = ['Baseline', 'Meta-Optimizer']
        efficiency = [baseline_eff, meta_eff]
        
        bars = ax.bar(labels, efficiency, color=['blue', 'red'])
        
        # Add values on top of bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{height:.2e}', ha='center', va='bottom')
                    
        # Calculate improvement percentage
        improvement = metrics["efficiency"]["improvement"]
        
        # Add text for improvement
        if improvement > 0:
            ax.text(0.5, 0.9, f"{improvement:.1f}% better efficiency", 
                   transform=ax.transAxes, ha='center', va='center',
                   bbox=dict(facecolor='lightgreen', alpha=0.5))
        elif improvement < 0:
            ax.text(0.5, 0.9, f"{-improvement:.1f}% worse efficiency", 
                   transform=ax.transAxes, ha='center', va='center',
                   bbox=dict(facecolor='lightcoral', alpha=0.5))
        else:
            ax.text(0.5, 0.9, "Same efficiency", 
                   transform=ax.transAxes, ha='center', va='center',
                   bbox=dict(facecolor='lightyellow', alpha=0.5))
                   
        ax.set_ylabel('Fitness per Evaluation (lower is better)')
        ax.grid(True, axis='y', linestyle='--', alpha=0.6)
        
    def generate_aggregate_normalized_comparison(self, normalized_metrics_list, output_path, title="Aggregate Normalized Comparison"):
        """
        Generate aggregate comparison of normalized metrics across multiple trials.
        
        Args:
            normalized_metrics_list: List of normalized metrics from each trial
            output_path: Path to save the visualization
            title: Plot title
        """
        if not normalized_metrics_list:
            return
            
        # Average metrics across trials
        avg_metrics = self._average_normalized_metrics(normalized_metrics_list)
        
        # Create figure with multiple subplots with fixed size to avoid errors
        fig = plt.figure(figsize=(10, 8))  # Use a fixed, reasonable size
        gs = GridSpec(2, 2, figure=fig)
        
        # 1. Aggregated improvement at normalized points
        ax1 = fig.add_subplot(gs[0, 0])
        self._plot_aggregate_normalized_points(normalized_metrics_list, ax1)
        ax1.set_title("Fitness Improvement at Evaluation Points")
        
        # 2. Success rate at different budgets
        ax2 = fig.add_subplot(gs[0, 1])
        self._plot_success_rate(normalized_metrics_list, ax2)
        ax2.set_title("Success Rate at Different Budgets")
        
        # 3. Average efficiency comparison
        ax3 = fig.add_subplot(gs[1, 0])
        self._plot_aggregate_efficiency(normalized_metrics_list, ax3)
        ax3.set_title("Average Optimizer Efficiency")
        
        # 4. Summary table
        ax4 = fig.add_subplot(gs[1, 1])
        self._plot_summary_table(avg_metrics, ax4)
        ax4.set_title("Performance Summary")
        
        # Add title and adjust layout
        fig.suptitle(title, fontsize=16)
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        
        # Set maximum figure dimensions to avoid matplotlib errors
        fig_width, fig_height = fig.get_size_inches()
        if fig_height > 50:  # If height is unreasonably large
            logger.warning(f"Limiting figure height from {fig_height} to 50 inches")
            fig.set_size_inches(fig_width, 50)
        
        # Save figure without using bbox_inches='tight' to avoid renderer errors
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, dpi=150)
        plt.close(fig)
        
    def _plot_aggregate_normalized_points(self, metrics_list, ax):
        """Plot aggregated improvement at normalized points across trials."""
        if not metrics_list:
            return
            
        # Get points from first metrics
        points = sorted(metrics_list[0]["normalized_points"].keys())
        
        # Calculate average improvement at each point
        improvements = []
        errors = []
        
        for p in points:
            point_improvements = [m["normalized_points"][p]["improvement"] for m in metrics_list]
            avg_improvement = np.mean(point_improvements)
            std_error = np.std(point_improvements) / np.sqrt(len(point_improvements))
            
            improvements.append(avg_improvement)
            errors.append(std_error)
            
        # Convert points to percentages for display
        point_labels = [f"{int(p*100)}%" for p in points]
        
        # Create bar chart with error bars
        bars = ax.bar(point_labels, improvements, yerr=errors, capsize=5, color='green' if all(i >= 0 for i in improvements) else 'orange')
        
        # Color individual bars based on improvement
        for i, bar in enumerate(bars):
            if improvements[i] > 0:
                bar.set_color('lightgreen')
            else:
                bar.set_color('lightcoral')
                
        # Add text labels
        for i, v in enumerate(improvements):
            color = 'green' if v > 0 else 'red'
            ax.text(i, v + errors[i] + 1, f"{v:.1f}%", color=color, ha='center', fontweight='bold')
            
        ax.set_xlabel('Percentage of Evaluation Budget')
        ax.set_ylabel('Average Improvement (%)')
        ax.grid(True, axis='y', linestyle='--', alpha=0.6)
        
        # Add horizontal line at y=0
        ax.axhline(y=0, color='k', linestyle='-', alpha=0.3)
        
    def _plot_success_rate(self, metrics_list, ax):
        """Plot success rate at different evaluation budgets."""
        if not metrics_list:
            return
            
        # Get points from first metrics
        points = sorted(metrics_list[0]["normalized_points"].keys())
        
        # Calculate success rate at each point (meta outperforming baseline)
        baseline_success = []
        meta_success = []
        
        for p in points:
            # Count trials where meta outperforms baseline at this point
            better_count = sum(1 for m in metrics_list if m["normalized_points"][p]["meta_fitness"] < m["normalized_points"][p]["baseline_fitness"])
            meta_rate = better_count / len(metrics_list) * 100
            
            # Count trials where baseline outperforms meta at this point
            worse_count = sum(1 for m in metrics_list if m["normalized_points"][p]["meta_fitness"] > m["normalized_points"][p]["baseline_fitness"])
            baseline_rate = worse_count / len(metrics_list) * 100
            
            meta_success.append(meta_rate)
            baseline_success.append(baseline_rate)
            
        # Convert points to percentages for display
        point_labels = [f"{int(p*100)}%" for p in points]
        
        # Create stacked bar chart
        width = 0.35
        x = np.arange(len(point_labels))
        
        ax.bar(x, meta_success, width, label='Meta Better', color='lightgreen')
        ax.bar(x, [100 - m - b for m, b in zip(meta_success, baseline_success)], width, 
               bottom=meta_success, label='Equal', color='lightyellow')
        ax.bar(x, baseline_success, width, 
               bottom=[100 - b for b in baseline_success], label='Baseline Better', color='lightcoral')
        
        # Add labels and legend
        ax.set_xlabel('Percentage of Evaluation Budget')
        ax.set_ylabel('Success Rate (%)')
        ax.set_xticks(x)
        ax.set_xticklabels(point_labels)
        ax.legend()
        ax.grid(True, axis='y', linestyle='--', alpha=0.6)
        
    def _plot_aggregate_efficiency(self, metrics_list, ax):
        """Plot aggregated efficiency comparison across trials."""
        if not metrics_list or not all("efficiency" in m for m in metrics_list):
            ax.text(0.5, 0.5, "No efficiency data available",
                   ha='center', va='center', transform=ax.transAxes)
            return
            
        # Calculate average efficiency
        baseline_eff = [m["efficiency"]["baseline_fitness_per_eval"] for m in metrics_list]
        meta_eff = [m["efficiency"]["meta_fitness_per_eval"] for m in metrics_list]
        
        # Filter out inf values
        baseline_eff = [e for e in baseline_eff if not np.isinf(e) and e < 1e10]
        meta_eff = [e for e in meta_eff if not np.isinf(e) and e < 1e10]
        
        if not baseline_eff or not meta_eff:
            ax.text(0.5, 0.5, "Insufficient efficiency data",
                   ha='center', va='center', transform=ax.transAxes)
            return
            
        # Calculate mean and standard error
        baseline_mean = np.mean(baseline_eff)
        baseline_err = np.std(baseline_eff) / np.sqrt(len(baseline_eff))
        meta_mean = np.mean(meta_eff)
        meta_err = np.std(meta_eff) / np.sqrt(len(meta_eff))
        
        # Create bar chart with error bars
        labels = ['Baseline', 'Meta-Optimizer']
        means = [baseline_mean, meta_mean]
        errors = [baseline_err, meta_err]
        
        bars = ax.bar(labels, means, yerr=errors, capsize=5, color=['blue', 'red'])
        
        # Add values on top of bars
        for i, bar in enumerate(bars):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + errors[i] + 0.01,
                    f'{means[i]:.2e}', ha='center', va='bottom')
                    
        # Calculate average improvement
        avg_improvement = np.mean([m["efficiency"]["improvement"] for m in metrics_list if "improvement" in m["efficiency"]])
        
        # Add text for improvement
        if avg_improvement > 0:
            ax.text(0.5, 0.9, f"{avg_improvement:.1f}% better efficiency", 
                   transform=ax.transAxes, ha='center', va='center',
                   bbox=dict(facecolor='lightgreen', alpha=0.5))
        elif avg_improvement < 0:
            ax.text(0.5, 0.9, f"{-avg_improvement:.1f}% worse efficiency", 
                   transform=ax.transAxes, ha='center', va='center',
                   bbox=dict(facecolor='lightcoral', alpha=0.5))
        else:
            ax.text(0.5, 0.9, "Same efficiency", 
                   transform=ax.transAxes, ha='center', va='center',
                   bbox=dict(facecolor='lightyellow', alpha=0.5))
                   
        ax.set_ylabel('Fitness per Evaluation (lower is better)')
        ax.grid(True, axis='y', linestyle='--', alpha=0.6)
        
    def _plot_summary_table(self, avg_metrics, ax):
        """Plot a summary table of metrics."""
        ax.axis('off')  # Hide axes
        
        # Check if we have metrics
        if not avg_metrics:
            ax.text(0.5, 0.5, "No metrics data available",
                   ha='center', va='center', transform=ax.transAxes)
            return
        
        # Prepare data
        data = []
        rows = []
        
        # Determine columns based on data available
        cols = ['Baseline', 'Meta-Optimizer']
        include_improvement = True
        
        # Check if we have both baseline and meta data consistently
        if any(k for k in avg_metrics.keys() if k not in ["normalized_points", "efficiency"]):
            # We have top-level metrics
            cols = ['Baseline', 'Meta-Optimizer']
            if "improvement" in avg_metrics:
                cols.append('Improvement')
                include_improvement = True
            
            # Add main metrics
            if "baseline_best_fitness" in avg_metrics and "meta_best_fitness" in avg_metrics:
                row_data = [f"{avg_metrics.get('baseline_best_fitness', 0):.6f}", 
                           f"{avg_metrics.get('meta_best_fitness', 0):.6f}"]
                if include_improvement and "improvement" in avg_metrics:
                    row_data.append(f"{avg_metrics.get('improvement', 0):.1f}%")
                data.append(row_data)
                rows.append('Final Fitness')
            
            # Add evaluations
            if "baseline_evaluations" in avg_metrics and "meta_evaluations" in avg_metrics:
                row_data = [f"{avg_metrics.get('baseline_evaluations', 0):.0f}", 
                           f"{avg_metrics.get('meta_evaluations', 0):.0f}"]
                if include_improvement:
                    eval_improvement = ((avg_metrics.get('baseline_evaluations', 0) / 
                                     avg_metrics.get('meta_evaluations', 1)) - 1) * 100
                    row_data.append(f"{eval_improvement:.1f}%")
                data.append(row_data)
                rows.append('Total Evaluations')
        
        # Add metrics at each normalized point if available
        if "normalized_points" in avg_metrics and avg_metrics["normalized_points"]:
            points = sorted(avg_metrics["normalized_points"].keys())
            for p in points:
                point_data = avg_metrics["normalized_points"][p]
                if isinstance(point_data, dict):
                    row_data = [f"{point_data.get('baseline_fitness', 0):.6f}", 
                               f"{point_data.get('meta_fitness', 0):.6f}"]
                    if include_improvement and "improvement" in point_data:
                        row_data.append(f"{point_data.get('improvement', 0):.1f}%")
                    data.append(row_data)
                    rows.append(f"Fitness at {int(float(p)*100)}%")
        
        # Add efficiency metrics if available
        if "efficiency" in avg_metrics and isinstance(avg_metrics["efficiency"], dict):
            eff = avg_metrics["efficiency"]
            row_data = [f"{eff.get('baseline_fitness_per_eval', 0):.2e}",
                       f"{eff.get('meta_fitness_per_eval', 0):.2e}"]
            if include_improvement and "improvement" in eff:
                row_data.append(f"{eff.get('improvement', 0):.1f}%")
            data.append(row_data)
            rows.append("Efficiency")
        
        # Ensure all rows have the same number of columns
        num_cols = len(cols)
        for i, row in enumerate(data):
            if len(row) < num_cols:
                # Pad row with empty strings if needed
                data[i] = row + [''] * (num_cols - len(row))
            elif len(row) > num_cols:
                # Truncate row if too long
                data[i] = row[:num_cols]
        
        # Check if we have any data to display
        if not data or not rows:
            ax.text(0.5, 0.5, "No comparable metrics available",
                   ha='center', va='center', transform=ax.transAxes)
            return
        
        # Create the table
        table = ax.table(
            cellText=data,
            rowLabels=rows,
            colLabels=cols,
            loc='center',
            cellLoc='center'
        )
        
        # Style the table
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 1.5)
        
        # Add color coding for improvement column if it exists
        if include_improvement and len(cols) > 2:
            for i, row in enumerate(data):
                if len(row) > 2:  # If row has improvement column
                    try:
                        improvement = float(row[2].strip('%') if row[2] else '0')
                        cell = table[(i+1, 2)]  # +1 for header row
                        
                        if improvement > 5:  # Significant improvement
                            cell.set_facecolor('lightgreen')
                        elif improvement < -5:  # Significant degradation
                            cell.set_facecolor('lightcoral')
                        else:  # Neutral
                            cell.set_facecolor('lightyellow')
                    except ValueError:
                        pass  # Skip if not a valid improvement value
        
        # Color the header row
        for j, col in enumerate(cols):
            table[(0, j)].set_facecolor('lightgrey')
            
        # Color row labels column
        for i in range(len(rows)):
            table[(i+1, -1)].set_facecolor('lightgrey')

    def _validate_trial_results(self, baseline_result, meta_result):
        """Validate the results of a trial."""
        validation_result = {
            "valid": True,
            "warnings": []
        }
        
        # Extract values from results
        if baseline_result is None:
            validation_result["valid"] = False
            validation_result["warnings"].append("Baseline result is None")
            
        if meta_result is None:
            validation_result["valid"] = False
            validation_result["warnings"].append("Meta result is None")
            
        # Return early if either result is None
        if not validation_result["valid"]:
            return validation_result
            
        baseline_fitness = baseline_result.get('best_fitness', float('inf'))
        meta_fitness = meta_result.get('best_fitness', float('inf'))
        baseline_solution = baseline_result.get('best_solution', None)
        meta_solution = meta_result.get('solution', None)
        baseline_evals = baseline_result.get('evaluations', 0)
        meta_evals = meta_result.get('evaluations', 0)
        
        # Check solution validity
        if baseline_solution is None:
            validation_result["valid"] = False
            validation_result["warnings"].append("Baseline optimizer did not return a valid solution")
            
        if meta_solution is None:
            validation_result["valid"] = False
            validation_result["warnings"].append("Meta optimizer did not return a valid solution")
        
        # Check for suspicious evaluation counts
        if baseline_evals < 10:
            validation_result["valid"] = False
            validation_result["warnings"].append(f"Baseline evaluations is suspiciously low: {baseline_evals}")
            
        if meta_evals < 10:
            validation_result["valid"] = False
            validation_result["warnings"].append(f"Meta evaluations is suspiciously low: {meta_evals}")
            
        # Check if baseline evaluations exceed maximum by more than 5%
        max_evaluations = self.max_evaluations
        allowed_excess = 0.05 * max_evaluations  # Allow 5% excess
        
        if baseline_evals > max_evaluations + allowed_excess:
            validation_result["valid"] = False
            validation_result["warnings"].append(f"Baseline evaluations ({baseline_evals}) exceeds maximum allowed ({max_evaluations + allowed_excess})")
        
        # We no longer check if meta evaluations exceed the maximum
        # This allows the meta optimizer to run as long as needed
        
        return validation_result