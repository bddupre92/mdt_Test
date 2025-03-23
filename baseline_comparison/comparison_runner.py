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
    Framework for comparing different algorithm selection approaches:
    1. Simple Baseline (random selection)
    2. Meta-Learner (basic bandit-based)
    3. Enhanced Meta-Optimizer (with feature extraction)
    4. SATzilla-inspired Selector (ML-based)
    """
    
    def __init__(
        self,
        simple_baseline,
        meta_learner,
        enhanced_meta,
        satzilla_selector,
        max_evaluations: int = 10000,
        num_trials: int = 10,
        verbose: bool = True,
        output_dir: str = "results/baseline_comparison",
        model_path: str = None
    ):
        """
        Initialize the comparison framework
        
        Args:
            simple_baseline: The simple baseline selector (random selection)
            meta_learner: The basic meta-learner
            enhanced_meta: The enhanced meta-optimizer with feature extraction
            satzilla_selector: The SATzilla-inspired selector
            max_evaluations: Maximum number of function evaluations per algorithm
            num_trials: Number of trials to run per algorithm
            verbose: Whether to print progress information
            output_dir: Directory to save results and visualizations
            model_path: Optional path to trained model for SATzilla selector
        """
        self.simple_baseline = simple_baseline
        self.meta_learner = meta_learner
        self.enhanced_meta = enhanced_meta
        self.satzilla_selector = satzilla_selector
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
        
        # Load SATzilla model if path is provided
        self.model_loaded = False
        if model_path is not None and hasattr(satzilla_selector, 'load_model'):
            try:
                satzilla_selector.load_model(model_path)
                logger.info(f"Loaded SATzilla model from {model_path}")
                self.model_loaded = True
            except Exception as e:
                logger.warning(f"Could not load SATzilla model from {model_path}: {e}")
        
        # Configure common algorithm pool
        self.standardize_algorithm_pool()
        
        logger.info(f"Initialized BaselineComparison with {len(self.available_algorithms)} algorithms")
        logger.info(f"Max evaluations: {max_evaluations}, Num trials: {num_trials}")
    
    def standardize_algorithm_pool(self):
        """Ensure all selectors use identical algorithm pools."""
        # Get available algorithms from all systems
        simple_algorithms = self.simple_baseline.get_available_algorithms() if hasattr(self.simple_baseline, 'get_available_algorithms') else []
        meta_algorithms = self.meta_learner.get_available_optimizers() if hasattr(self.meta_learner, 'get_available_optimizers') else []
        enhanced_algorithms = self.enhanced_meta.get_available_optimizers() if hasattr(self.enhanced_meta, 'get_available_optimizers') else []
        satzilla_algorithms = self.satzilla_selector.get_available_algorithms() if hasattr(self.satzilla_selector, 'get_available_algorithms') else []
        
        # Find common algorithms
        all_algorithms = [simple_algorithms, meta_algorithms, enhanced_algorithms, satzilla_algorithms]
        common_algorithms = set(all_algorithms[0])
        for algs in all_algorithms[1:]:
            if algs:
                common_algorithms.intersection_update(algs)
        
        if common_algorithms:
            self.available_algorithms = sorted(list(common_algorithms))
        
        logger.info(f"Using {len(self.available_algorithms)} algorithms: {sorted(self.available_algorithms)}")
        
        # Configure all systems to use only common algorithms
        for selector, name in [
            (self.simple_baseline, "Simple Baseline"),
            (self.meta_learner, "Meta Learner"),
            (self.enhanced_meta, "Enhanced Meta"),
            (self.satzilla_selector, "SATzilla")
        ]:
            if hasattr(selector, 'set_available_algorithms'):
                selector.set_available_algorithms(list(self.available_algorithms))
                logger.info(f"Set algorithms for {name}")
    
    def run_comparison(self, problem_name, problem_func, dimensions, max_evaluations, num_trials):
        """Run comparison between all four approaches."""
        results = {
            "simple": {"best_fitness": [], "evaluations": [], "time": [], "convergence": [], "selections": []},
            "meta": {"best_fitness": [], "evaluations": [], "time": [], "convergence": [], "selections": []},
            "enhanced": {"best_fitness": [], "evaluations": [], "time": [], "convergence": [], "selections": []},
            "satzilla": {"best_fitness": [], "evaluations": [], "time": [], "convergence": [], "selections": []}
        }
        
        # Stores normalized metrics for all trials
        normalized_metrics = []
        
        for trial in range(1, num_trials + 1):
            logger.info(f"  Trial {trial}/{num_trials}")
            
            # Create fresh problem instances for each optimizer
            seed = int(time.time()) if self.random_seed is None else self.random_seed + trial
            np.random.seed(seed)
            random.seed(seed)
            
            problems = {
                "simple": self.prepare_problem_for_run(problem_func),
                "meta": self.prepare_problem_for_run(problem_func),
                "enhanced": self.prepare_problem_for_run(problem_func),
                "satzilla": self.prepare_problem_for_run(problem_func)
            }
            
            # Create a directory for this trial
            trial_dir = os.path.join(self.output_dir, problem_name, f"trial_{trial}")
            os.makedirs(trial_dir, exist_ok=True)
            
            # Run each optimizer
            optimizers = {
                "simple": self.simple_baseline,
                "meta": self.meta_learner,
                "enhanced": self.enhanced_meta,
                "satzilla": self.satzilla_selector
            }
            
            trial_results = {
                "simple": {},
                "meta": {},
                "enhanced": {},
                "satzilla": {}
            }
            
            for name, optimizer in optimizers.items():
                start_time = time.time()
                try:
                    result = self._run_optimizer(optimizer, problems[name], max_evaluations)
                    results[name]["best_fitness"].append(result["best_fitness"])
                    results[name]["evaluations"].append(result["evaluations"])
                    results[name]["time"].append(time.time() - start_time)
                    results[name]["convergence"].append(result.get("convergence_data", []))
                    results[name]["selections"].append(result["selected_algorithm"])
                    
                    # Store result for this trial
                    trial_results[name] = {
                        "best_fitness": result["best_fitness"],
                        "evaluations": result["evaluations"],
                        "time": time.time() - start_time,
                        "convergence": result.get("convergence_data", []),
                        "selected_algorithm": result["selected_algorithm"]
                    }
                except Exception as e:
                    logger.error(f"Error running {name}: {e}")
                    results[name]["best_fitness"].append(float('inf'))
                    results[name]["evaluations"].append(max_evaluations)
                    results[name]["time"].append(0.0)
                    results[name]["convergence"].append([])
                    results[name]["selections"].append("error")
                    
                    # Store error result for this trial
                    trial_results[name] = {
                        "best_fitness": float('inf'),
                        "evaluations": max_evaluations,
                        "time": 0.0,
                        "convergence": [],
                        "selected_algorithm": "error"
                    }
            
            # Generate normalized comparison plot for this trial
            try:
                normalized_plot_path = os.path.join(trial_dir, "normalized_comparison.png")
                self.generate_normalized_comparison_plots(
                    trial_results["simple"],
                    trial_results["meta"],
                    trial_results["enhanced"],
                    trial_results["satzilla"],
                    normalized_plot_path,
                    title=f"{problem_name} (Trial {trial})"
                )
                logger.info(f"  Generated normalized comparison plot for trial {trial}")
            except Exception as e:
                logger.error(f"Error generating normalized comparison plot for trial {trial}: {e}")
            
            # Calculate normalized metrics for this trial
            trial_metrics = self._calculate_normalized_metrics(
                simple_result={"best_fitness": results["simple"]["best_fitness"][-1], "evaluations": results["simple"]["evaluations"][-1]},
                meta_result={"best_fitness": results["meta"]["best_fitness"][-1], "evaluations": results["meta"]["evaluations"][-1]},
                enhanced_result={"best_fitness": results["enhanced"]["best_fitness"][-1], "evaluations": results["enhanced"]["evaluations"][-1]},
                satzilla_result={"best_fitness": results["satzilla"]["best_fitness"][-1], "evaluations": results["satzilla"]["evaluations"][-1]}
            )
            
            # Add actual values to metrics for aggregate visualization
            trial_metrics["simple_fitness"] = results["simple"]["best_fitness"][-1]
            trial_metrics["meta_fitness"] = results["meta"]["best_fitness"][-1]
            trial_metrics["enhanced_fitness"] = results["enhanced"]["best_fitness"][-1]
            trial_metrics["satzilla_fitness"] = results["satzilla"]["best_fitness"][-1]
            
            trial_metrics["simple_evaluations"] = results["simple"]["evaluations"][-1]
            trial_metrics["meta_evaluations"] = results["meta"]["evaluations"][-1]
            trial_metrics["enhanced_evaluations"] = results["enhanced"]["evaluations"][-1]
            trial_metrics["satzilla_evaluations"] = results["satzilla"]["evaluations"][-1]
            
            normalized_metrics.append(trial_metrics)
        
        # Calculate averages
        avg_metrics = {}
        for name in results:
            avg_metrics[name] = {
                "avg_fitness": np.mean(results[name]["best_fitness"]),
                "avg_evaluations": np.mean(results[name]["evaluations"]),
                "avg_time": np.mean(results[name]["time"]),
                "std_fitness": np.std(results[name]["best_fitness"]),
                "std_evaluations": np.std(results[name]["evaluations"]),
                "std_time": np.std(results[name]["time"])
            }
        
        # Generate aggregate normalized comparison
        try:
            aggregate_plot_path = os.path.join(self.output_dir, problem_name, "aggregate_normalized_comparison.png")
            self.generate_aggregate_normalized_comparison(
                normalized_metrics,
                aggregate_plot_path,
                title=f"{problem_name} - Aggregate Results ({num_trials} trials)"
            )
            logger.info(f"  Generated aggregate normalized comparison")
        except Exception as e:
            logger.error(f"Error generating aggregate normalized comparison: {e}")
        
        # Store results
        self.results[problem_name] = {
            "simple_baseline": results["simple"],
            "meta_learner": results["meta"],
            "enhanced_meta": results["enhanced"],
            "satzilla": results["satzilla"],
            "averages": avg_metrics,
            "normalized_metrics": normalized_metrics
        }
        
        # Print summary
        logger.info(f"  Average results for {problem_name}:")
        for name in ["simple", "meta", "enhanced", "satzilla"]:
            logger.info(f"    {name.capitalize()}: fitness={avg_metrics[name]['avg_fitness']:.6f} "
                       f"(±{avg_metrics[name]['std_fitness']:.6f}), "
                       f"evals={avg_metrics[name]['avg_evaluations']:.1f} "
                       f"(±{avg_metrics[name]['std_evaluations']:.1f}), "
                       f"time={avg_metrics[name]['avg_time']:.3f}s "
                       f"(±{avg_metrics[name]['std_time']:.3f}s)")
        
        return self.results[problem_name]

    def _run_optimizer(self, optimizer, problem, max_evaluations):
        """Run a single optimizer on a problem."""
        start_time = time.time()
        convergence_history = []
        best_fitness = float('inf')
        evaluation_count = 0
        
        # Create a wrapper function to enforce evaluation budget and track convergence
        original_evaluate = problem.evaluate
        
        def budget_enforced_evaluate(x):
            nonlocal evaluation_count, best_fitness
            
            if evaluation_count >= max_evaluations:
                return float('inf')
            
            evaluation_count += 1
            result = original_evaluate(x)
            
            if result < best_fitness:
                best_fitness = result
            convergence_history.append(best_fitness)

            return result
        
        problem.evaluate = budget_enforced_evaluate
        
        try:
            result = optimizer.optimize(problem, max_evaluations)
            if isinstance(result, tuple):
                best_solution, best_fitness = result
            else:
                best_solution = result
            
            # Get selected algorithm
            if hasattr(optimizer, 'get_selected_algorithm'):
                selected_algorithm = optimizer.get_selected_algorithm()
            else:
                selected_algorithm = 'unknown'
            
            return {
                'best_solution': best_solution,
                'best_fitness': best_fitness,
                'evaluations': evaluation_count,
                'convergence_data': convergence_history,
                'selected_algorithm': selected_algorithm
            }
        finally:
            problem.evaluate = original_evaluate

    def _calculate_normalized_metrics(self, simple_result, meta_result, enhanced_result, satzilla_result):
        """Calculate normalized comparison metrics between all approaches."""
        # Implementation of normalized metric calculation
        # This would include relative improvements, efficiency metrics, etc.
        
        # Get fitness values (handling inf values)
        simple_fitness = simple_result["best_fitness"]
        meta_fitness = meta_result["best_fitness"]
        enhanced_fitness = enhanced_result["best_fitness"]
        satzilla_fitness = satzilla_result["best_fitness"]
        
        # Get evaluation counts
        simple_evals = max(1, simple_result["evaluations"])
        meta_evals = max(1, meta_result["evaluations"])
        enhanced_evals = max(1, enhanced_result["evaluations"])
        satzilla_evals = max(1, satzilla_result["evaluations"])
        
        # Calculate efficiency metrics (fitness per evaluation, normalized)
        if simple_fitness > 0 and simple_fitness != float('inf'):
            simple_efficiency = 1.0  # Baseline reference
            meta_efficiency = self._calculate_efficiency(simple_fitness, meta_fitness, simple_evals, meta_evals)
            enhanced_efficiency = self._calculate_efficiency(simple_fitness, enhanced_fitness, simple_evals, enhanced_evals)
            satzilla_efficiency = self._calculate_efficiency(simple_fitness, satzilla_fitness, simple_evals, satzilla_evals)
        else:
            # Just use evaluation ratio if fitness is zero or inf
            simple_efficiency = 1.0
            meta_efficiency = simple_evals / meta_evals if meta_evals > 0 else 0
            enhanced_efficiency = simple_evals / enhanced_evals if enhanced_evals > 0 else 0
            satzilla_efficiency = simple_evals / satzilla_evals if satzilla_evals > 0 else 0
            
        return {
            "fitness_improvements": {
                "meta_vs_simple": self._calculate_improvement(simple_fitness, meta_fitness),
                "enhanced_vs_simple": self._calculate_improvement(simple_fitness, enhanced_fitness),
                "satzilla_vs_simple": self._calculate_improvement(simple_fitness, satzilla_fitness)
            },
            "evaluation_improvements": {
                "meta_vs_simple": self._calculate_improvement(simple_evals, meta_evals),
                "enhanced_vs_simple": self._calculate_improvement(simple_evals, enhanced_evals),
                "satzilla_vs_simple": self._calculate_improvement(simple_evals, satzilla_evals)
            },
            "efficiency_metrics": {
                "simple": simple_efficiency,
                "meta": meta_efficiency,
                "enhanced": enhanced_efficiency,
                "satzilla": satzilla_efficiency
            }
        }

    def _calculate_improvement(self, baseline_value, comparison_value):
        """Calculate percentage improvement."""
        if baseline_value == 0:
            return 0.0
        if baseline_value == float('inf') and comparison_value == float('inf'):
            return 0.0
        if baseline_value == float('inf'):
            return 100.0  # Any finite value is 100% better than inf
        if comparison_value == float('inf'):
            return -100.0  # Inf is 100% worse than any finite value
        
        return ((baseline_value - comparison_value) / abs(baseline_value)) * 100
        
    def _calculate_efficiency(self, baseline_fitness, comparison_fitness, baseline_evals, comparison_evals):
        """Calculate efficiency ratio normalized to baseline."""
        # Handle special cases
        if comparison_fitness == float('inf'):
            return 0.0  # Infinitely bad efficiency
        if baseline_fitness == float('inf'):
            if comparison_fitness < float('inf'):
                return baseline_evals / comparison_evals  # Just compare evaluations
            return 0.0  # Both are inf
        if comparison_fitness <= 0:
            return (baseline_evals / comparison_evals)  # Just compare evaluations
            
        # Calculate efficiency: (relative fitness improvement) * (relative evaluation efficiency)
        return (baseline_fitness / comparison_fitness) * (baseline_evals / comparison_evals)

    def prepare_problem_for_run(self, problem: ProblemWrapper):
        """Reset and prepare problem for a fresh optimization run."""
        # Reset evaluations counter and tracking
        problem.evaluations = 0
        problem.tracking_objective = None
        
        # Create a deep copy to avoid any side effects between runs
        import copy
        return copy.deepcopy(problem)
    
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
            
        # Check if evaluations exceed maximum by more than 5%
        max_evaluations = self.max_evaluations
        allowed_excess = 0.05 * max_evaluations  # Allow 5% excess
        
        if baseline_evals > max_evaluations + allowed_excess:
            validation_result["valid"] = False
            validation_result["warnings"].append(f"Baseline evaluations ({baseline_evals}) exceeds maximum allowed ({max_evaluations + allowed_excess})")
            
        if meta_evals > max_evaluations + allowed_excess:
            validation_result["valid"] = False
            validation_result["warnings"].append(f"Meta evaluations ({meta_evals}) exceeds maximum allowed ({max_evaluations + allowed_excess})")
        
        return validation_result
    
    def log_experiment_configuration(self):
        """Log detailed configuration for reproducibility."""
        config = {
            'max_evaluations': self.max_evaluations,
            'num_trials': self.num_trials,
            'dimensions': self.meta_learner.dim if hasattr(self.meta_learner, 'dim') else None,
            'algorithms': list(self.available_algorithms),
            'model_loaded': self.model_loaded,
            'random_seed': self.random_seed,
            'timestamp': time.strftime("%Y-%m-%d %H:%M:%S")
        }
        
        config_path = os.path.join(self.output_dir, "experiment_config.json")
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        
        logger.info(f"Experiment configuration saved to {config_path}") 

    def plot_radar_comparison(self, results):
        """
        Create a radar chart comparing baseline and meta optimizer performance.
        
        Args:
            results: Dictionary of results, with function names as keys and function results as values
        """
        # Create directory for radar comparison
        radar_dir = os.path.join(self.output_dir, "radar_comparison")
        os.makedirs(radar_dir, exist_ok=True)
        
        # Metrics to compare
        metrics = [
            "best_fitness_avg",    # Lower is better, needs inversion
            "evaluations_avg",     # Lower is better, needs inversion
            "time_avg",            # Lower is better, needs inversion
            "success_rate"         # Higher is better
        ]
        
        # For plotting radar chart
        N = len(metrics)
        theta = np.linspace(0, 2*np.pi, N, endpoint=False).tolist()
        theta += theta[:1]  # Close the circle
        
        # Prepare data for radar chart
        baseline_values = []
        meta_values = []
        
        # Collect average values for each metric across all functions
        for metric in metrics:
            baseline_metric_values = []
            meta_metric_values = []
            
            if isinstance(results, dict):
                for func_name, func_data in results.items():
                    # Check if the function data contains average metrics
                    baseline_key = f"baseline_{metric}"
                    meta_key = f"meta_{metric}"
                    
                    # Average metrics might be stored directly in func_data
                    if isinstance(func_data, dict):
                        if baseline_key in func_data:
                            baseline_metric_values.append(func_data[baseline_key])
                        if meta_key in func_data:
                            meta_metric_values.append(func_data[meta_key])
            
            # If we collected any values, compute the average
            if baseline_metric_values:
                baseline_avg = np.mean(baseline_metric_values)
                baseline_values.append(baseline_avg)
            else:
                baseline_values.append(0)
            
            if meta_metric_values:
                meta_avg = np.mean(meta_metric_values)
                meta_values.append(meta_avg)
            else:
                meta_values.append(0)
        
        # Normalize values to 0-1 range for radar chart
        # We want all values to be higher = better
        normalized_baseline = []
        normalized_meta = []
        
        for i, metric in enumerate(metrics):
            baseline_val = baseline_values[i]
            meta_val = meta_values[i]
            
            # For success rate, higher is better; for others, lower is better
            if metric == "success_rate":
                # Normalize to 0-1, higher is better
                max_val = max(baseline_val, meta_val, 1)  # Avoid division by zero
                normalized_baseline.append(baseline_val / max_val)
                normalized_meta.append(meta_val / max_val)
            else:
                # For these metrics, lower is better, so invert
                # Add epsilon to avoid division by zero
                epsilon = 1e-10
                if baseline_val == 0 and meta_val == 0:
                    normalized_baseline.append(0)
                    normalized_meta.append(0)
                else:
                    max_val = max(baseline_val, meta_val, epsilon)
                    # Invert so that lower values are closer to 1
                    normalized_baseline.append(1 - (baseline_val / max_val))
                    normalized_meta.append(1 - (meta_val / max_val))
        
        # Close the loop for the radar chart
        normalized_baseline += normalized_baseline[:1]
        normalized_meta += normalized_meta[:1]
        
        # Create radar chart
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, polar=True)
        
        # Plot data
        ax.plot(theta, normalized_baseline, 'b-', label='Baseline', linewidth=2)
        ax.fill(theta, normalized_baseline, 'b', alpha=0.2)
        
        ax.plot(theta, normalized_meta, 'r-', label='Meta Optimizer', linewidth=2)
        ax.fill(theta, normalized_meta, 'r', alpha=0.2)
        
        # Set labels
        metric_labels = ["Fitness", "Evaluations", "Time", "Success Rate"]
        ax.set_xticks(theta[:-1])  # Exclude the last duplicate point
        ax.set_xticklabels(metric_labels)
        
        # Set radial ticks
        ax.set_rticks([0.25, 0.5, 0.75, 1])
        ax.set_rlabel_position(45)  # Move labels away from the center
        
        # Add title and legend
        plt.title("Performance Comparison (Higher is Better)")
        plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
        
        # Save the figure
        plt.tight_layout()
        plt.savefig(os.path.join(radar_dir, "radar_comparison.png"), dpi=300)
        plt.close()

    def plot_statistical_tests(self, results):
        """
        Perform and visualize statistical tests comparing baseline and meta optimizer.
        
        Args:
            results: Dictionary of results, with function names as keys and lists of trials as values
        """
        # Create directory for statistical test plots
        stat_test_dir = os.path.join(self.output_dir, "statistical_tests")
        os.makedirs(stat_test_dir, exist_ok=True)
        
        # Prepare data for statistical tests
        baseline_fitness = []
        meta_fitness = []
        baseline_evals = []
        meta_evals = []
        
        # Check if results is a dictionary with function names as keys
        if isinstance(results, dict):
            for func_name, func_results in results.items():
                # Check if the results for this function are stored as a list of trials
                if isinstance(func_results, list):
                    # Each func_results is a list of trial results
                    for trial_result in func_results:
                        if isinstance(trial_result, dict):
                            # Get baseline results
                            baseline_result = trial_result.get("baseline", {})
                            meta_result = trial_result.get("meta", {})
                            
                            # Extract metrics, using infinity for missing/failed results
                            if isinstance(baseline_result, dict):
                                baseline_fitness.append(baseline_result.get("best_fitness", float('inf')))
                                baseline_evals.append(baseline_result.get("evaluations", 0))
                            
                            if isinstance(meta_result, dict):
                                meta_fitness.append(meta_result.get("best_fitness", float('inf')))
                                meta_evals.append(meta_result.get("evaluations", 0))
        
        # Create figure with subplots
        fig = plt.figure(figsize=(15, 10))
        gs = GridSpec(2, 2, figure=fig)
        
        # 1. Overall Performance Comparison (Wilcoxon test)
        ax1 = fig.add_subplot(gs[0, 0])
        try:
            if len(baseline_fitness) > 0 and len(meta_fitness) > 0:
                stat, p_value = stats.wilcoxon(baseline_fitness, meta_fitness)
                ax1.boxplot([baseline_fitness, meta_fitness], labels=["Baseline", "Meta Optimizer"])
                ax1.set_title(f"Performance Comparison\n(Wilcoxon p-value: {p_value:.4f})")
                ax1.set_ylabel("Best Fitness")
            else:
                ax1.text(0.5, 0.5, "Insufficient data for statistical test", 
                        ha='center', va='center')
        except Exception as e:
            logger.warning(f"Could not perform Wilcoxon test: {e}")
            ax1.text(0.5, 0.5, "Insufficient data for statistical test", 
                    ha='center', va='center')
        
        # 2. Per-function Statistical Tests
        ax2 = fig.add_subplot(gs[0, 1])
        p_values = []
        func_names = []
        if isinstance(results, dict):
            for func_name, func_results in results.items():
                if isinstance(func_results, list):
                    func_baseline_fitness = []
                    func_meta_fitness = []
                    for trial in func_results:
                        if isinstance(trial, dict):
                            baseline = trial.get("baseline", {})
                            meta = trial.get("meta", {})
                            if isinstance(baseline, dict):
                                func_baseline_fitness.append(baseline.get("best_fitness", float('inf')))
                            if isinstance(meta, dict):
                                func_meta_fitness.append(meta.get("best_fitness", float('inf')))
                    
                    if len(func_baseline_fitness) > 0 and len(func_meta_fitness) > 0:
                        try:
                            _, p = stats.wilcoxon(func_baseline_fitness, func_meta_fitness)
                            p_values.append(p)
                            func_names.append(func_name)
                        except Exception as e:
                            logger.warning(f"Could not perform Wilcoxon test for {func_name}: {e}")
        
        if p_values:
            ax2.bar(func_names, p_values)
            ax2.axhline(y=0.05, color='r', linestyle='--', label='α=0.05')
            ax2.set_title("Statistical Significance by Function")
            ax2.set_xticklabels(func_names, rotation=45)
            ax2.set_ylabel("p-value")
            ax2.legend()
        else:
            ax2.text(0.5, 0.5, "Insufficient data for per-function tests", 
                   ha='center', va='center')
        
        # 3. Effect Size Analysis (Cohen's d)
        ax3 = fig.add_subplot(gs[1, 0])
        effect_sizes = []
        func_names_effect = []
        if isinstance(results, dict):
            for func_name, func_results in results.items():
                if isinstance(func_results, list):
                    func_baseline_fitness = []
                    func_meta_fitness = []
                    for trial in func_results:
                        if isinstance(trial, dict):
                            baseline = trial.get("baseline", {})
                            meta = trial.get("meta", {})
                            if isinstance(baseline, dict):
                                func_baseline_fitness.append(baseline.get("best_fitness", float('inf')))
                            if isinstance(meta, dict):
                                func_meta_fitness.append(meta.get("best_fitness", float('inf')))
                    
                    if len(func_baseline_fitness) > 1 and len(func_meta_fitness) > 1:
                        try:
                            baseline_mean = np.mean(func_baseline_fitness)
                            meta_mean = np.mean(func_meta_fitness)
                            pooled_std = np.sqrt((np.std(func_baseline_fitness, ddof=1) ** 2 + 
                                                  np.std(func_meta_fitness, ddof=1) ** 2) / 2)
                            
                            # Avoid division by zero
                            if pooled_std > 0:
                                cohens_d = (baseline_mean - meta_mean) / pooled_std
                                effect_sizes.append(cohens_d)
                                func_names_effect.append(func_name)
                        except Exception as e:
                            logger.warning(f"Could not calculate effect size for {func_name}: {e}")
        
        if effect_sizes:
            ax3.bar(func_names_effect, effect_sizes)
            ax3.axhline(y=0.2, color='g', linestyle='--', label='Small Effect')
            ax3.axhline(y=0.5, color='y', linestyle='--', label='Medium Effect')
            ax3.axhline(y=0.8, color='r', linestyle='--', label='Large Effect')
            ax3.set_title("Effect Size Analysis (Cohen's d)")
            ax3.set_xticklabels(func_names_effect, rotation=45)
            ax3.set_ylabel("Cohen's d")
            ax3.legend()
        else:
            ax3.text(0.5, 0.5, "Insufficient data for effect size analysis", 
                   ha='center', va='center')
        
        # 4. Success Rate Comparison
        ax4 = fig.add_subplot(gs[1, 1])
        baseline_success = []
        meta_success = []
        func_names_success = []
        if isinstance(results, dict):
            for func_name, func_results in results.items():
                if isinstance(func_results, list) and len(func_results) > 0:
                    # Calculate success rate for this function
                    baseline_success_count = 0
                    meta_success_count = 0
                    total_trials = 0
                    
                    for trial in func_results:
                        if isinstance(trial, dict):
                            total_trials += 1
                            baseline = trial.get("baseline", {})
                            meta = trial.get("meta", {})
                            
                            if isinstance(baseline, dict) and baseline.get("best_fitness", float('inf')) < float('inf'):
                                baseline_success_count += 1
                            
                            if isinstance(meta, dict) and meta.get("best_fitness", float('inf')) < float('inf'):
                                meta_success_count += 1
                    
                    if total_trials > 0:
                        baseline_success.append(baseline_success_count / total_trials * 100)
                        meta_success.append(meta_success_count / total_trials * 100)
                        func_names_success.append(func_name)
        
        if baseline_success and meta_success:
            x = np.arange(len(func_names_success))
            width = 0.35
            ax4.bar(x - width/2, baseline_success, width, label='Baseline')
            ax4.bar(x + width/2, meta_success, width, label='Meta Optimizer')
            ax4.set_xticks(x)
            ax4.set_xticklabels(func_names_success, rotation=45)
            ax4.set_title("Success Rate Comparison")
            ax4.set_ylabel("Success Rate (%)")
            ax4.legend()
        else:
            ax4.text(0.5, 0.5, "Insufficient data for success rate comparison", 
                   ha='center', va='center')
        
        # Adjust layout and save
        plt.tight_layout()
        plt.savefig(os.path.join(stat_test_dir, "statistical_tests.png"), dpi=300)
        plt.close()

    def create_summary_table(self, results):
        """
        Create a summary table of results and save it to a file.
        
        Args:
            results: Dictionary of results, with function names as keys and function results as values
        """
        # Create directory for summary table
        summary_dir = os.path.join(self.output_dir, "summary")
        os.makedirs(summary_dir, exist_ok=True)
        
        # Prepare data for summary table
        summary_data = []
        
        if isinstance(results, dict):
            for func_name, func_data in results.items():
                if isinstance(func_data, dict):
                    # Extract metrics
                    baseline_fitness = func_data.get("baseline_best_fitness_avg", float('inf'))
                    meta_fitness = func_data.get("meta_best_fitness_avg", float('inf'))
                    
                    baseline_evals = func_data.get("baseline_evaluations_avg", 0)
                    meta_evals = func_data.get("meta_evaluations_avg", 0)
                    
                    baseline_time = func_data.get("baseline_time_avg", 0)
                    meta_time = func_data.get("meta_time_avg", 0)
                    
                    baseline_success = func_data.get("baseline_success_rate", 0)
                    meta_success = func_data.get("meta_success_rate", 0)
                    
                    # Calculate improvements
                    if baseline_fitness > 0 and meta_fitness < float('inf'):
                        fitness_improvement = ((baseline_fitness - meta_fitness) / baseline_fitness) * 100
                    elif baseline_fitness == 0 and meta_fitness == 0:
                        fitness_improvement = 0
                    elif baseline_fitness == 0:
                        fitness_improvement = float('inf') if meta_fitness < 0 else -float('inf')
                    else:
                        fitness_improvement = float('-inf')
                    
                    evals_improvement = ((baseline_evals - meta_evals) / baseline_evals) * 100 if baseline_evals > 0 else 0
                    time_improvement = ((baseline_time - meta_time) / baseline_time) * 100 if baseline_time > 0 else 0
                    success_improvement = meta_success - baseline_success
                    
                    row = {
                        "Function": func_name,
                        "Baseline Fitness": f"{baseline_fitness:.6f}",
                        "Meta Fitness": f"{meta_fitness:.6f}",
                        "Fitness Improvement": f"{fitness_improvement:.2f}%",
                        "Baseline Evals": f"{baseline_evals:.1f}",
                        "Meta Evals": f"{meta_evals:.1f}",
                        "Evals Improvement": f"{evals_improvement:.2f}%",
                        "Baseline Time": f"{baseline_time:.3f}s",
                        "Meta Time": f"{meta_time:.3f}s",
                        "Time Improvement": f"{time_improvement:.2f}%",
                        "Baseline Success": f"{baseline_success:.1f}%",
                        "Meta Success": f"{meta_success:.1f}%",
                        "Success Improvement": f"{success_improvement:.1f}%"
                    }
                    
                    summary_data.append(row)
        
        # Create a DataFrame for the summary table
        if summary_data:
            df = pd.DataFrame(summary_data)
            
            # Save to CSV
            csv_path = os.path.join(summary_dir, "summary_table.csv")
            df.to_csv(csv_path, index=False)
            
            # Create a formatted HTML table
            html_path = os.path.join(summary_dir, "summary_table.html")
            
            # Apply conditional formatting
            def color_improvement(val):
                try:
                    if "%" in val:
                        val_num = float(val.strip("%"))
                        if val_num > 0:
                            return 'background-color: #d4f7d4'  # light green
                        elif val_num < 0:
                            return 'background-color: #f7d4d4'  # light red
                except:
                    pass
                return ''
            
            # Apply styling to the DataFrame
            styled_df = df.style.applymap(color_improvement, subset=[
                "Fitness Improvement", "Evals Improvement", 
                "Time Improvement", "Success Improvement"
            ])
            
            # Save to HTML
            styled_df.to_html(html_path)
            
            # Create a text summary
            txt_path = os.path.join(summary_dir, "summary_table.txt")
            with open(txt_path, 'w') as f:
                f.write("Summary of Baseline vs Meta-Optimizer Comparison\n")
                f.write("=" * 50 + "\n\n")
                
                for row in summary_data:
                    f.write(f"Function: {row['Function']}\n")
                    f.write("-" * 30 + "\n")
                    f.write(f"Fitness: {row['Baseline Fitness']} vs {row['Meta Fitness']} ({row['Fitness Improvement']})\n")
                    f.write(f"Evaluations: {row['Baseline Evals']} vs {row['Meta Evals']} ({row['Evals Improvement']})\n")
                    f.write(f"Time: {row['Baseline Time']} vs {row['Meta Time']} ({row['Time Improvement']})\n")
                    f.write(f"Success Rate: {row['Baseline Success']} vs {row['Meta Success']} ({row['Success Improvement']})\n\n")
            
            logger.info("Summary table saved to CSV, HTML, and TXT files")
        else:
            logger.warning("No data available for summary table")

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
            bounds = [bound] * self.meta_learner.dim
            
            # Create test function with specified dimensions and appropriate bounds
            func = func_class(self.meta_learner.dim, bounds)
            problem = ProblemWrapper(func, self.meta_learner.dim)
            
            self.run_comparison(
                problem_name=func_name,
                problem_func=problem,
                dimensions=self.meta_learner.dim,
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

    def _interpolate_convergence(self, convergence_data, max_evals, eval_points):
        """
        Interpolate convergence data to a common set of evaluation points.
        
        Args:
            convergence_data: The convergence data to interpolate
            max_evals: Maximum number of evaluations
            eval_points: Evaluation points to interpolate to
            
        Returns:
            Interpolated convergence data
        """
        if not convergence_data:
            return np.zeros_like(eval_points)
        
        # If convergence data is a single value, repeat it
        if len(convergence_data) == 1:
            return np.full_like(eval_points, convergence_data[0])
        
        # Generate original evaluation points
        orig_evals = np.linspace(0, max_evals, num=len(convergence_data))
        
        # Use linear interpolation
        interp_fitness = np.interp(
            eval_points, 
            orig_evals, 
            convergence_data,
            right=convergence_data[-1]  # Use last value for extrapolation
        )
        
        return interp_fitness

    def generate_normalized_comparison_plots(self, baseline_results, meta_results, enhanced_results, satzilla_results, output_path, title=None):
        """
        Generate normalized comparison plots between all optimizers.
        
        Args:
            baseline_results: Dictionary with baseline optimizer results
            meta_results: Dictionary with meta optimizer results
            enhanced_results: Dictionary with enhanced meta optimizer results
            satzilla_results: Dictionary with SATzilla selector results
            output_path: Path to save the plot
            title: Optional title for the plot
        """
        import matplotlib.pyplot as plt
        import numpy as np
        from matplotlib import gridspec
        
        # Extract key metrics
        baseline_best = baseline_results.get("best_fitness", float('inf'))
        meta_best = meta_results.get("best_fitness", float('inf'))
        enhanced_best = enhanced_results.get("best_fitness", float('inf'))
        satzilla_best = satzilla_results.get("best_fitness", float('inf'))
        
        baseline_evals = baseline_results.get("evaluations", 0)
        meta_evals = meta_results.get("evaluations", 0)
        enhanced_evals = enhanced_results.get("evaluations", 0)
        satzilla_evals = satzilla_results.get("evaluations", 0)
        
        # Extract convergence data
        baseline_convergence = baseline_results.get("convergence", [])
        meta_convergence = meta_results.get("convergence", [])
        enhanced_convergence = enhanced_results.get("convergence", [])
        satzilla_convergence = satzilla_results.get("convergence", [])
        
        # If no convergence data, create from best fitness
        if not baseline_convergence:
            baseline_convergence = [baseline_best]
        if not meta_convergence:
            meta_convergence = [meta_best]
        if not enhanced_convergence:
            enhanced_convergence = [enhanced_best]
        if not satzilla_convergence:
            satzilla_convergence = [satzilla_best]
        
        # Create evenly spaced evaluation points for comparison
        max_evals = max(baseline_evals, meta_evals, enhanced_evals, satzilla_evals)
        eval_points = np.linspace(0, max_evals, num=100)
        
        # Create the figure with a 2x2 grid
        fig = plt.figure(figsize=(12, 10))
        if title:
            fig.suptitle(title, fontsize=16)
        
        gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.3, wspace=0.3)
        
        # Subplot 1: Convergence Curves
        ax1 = fig.add_subplot(gs[0, 0])
        
        # Handle convergence data padding and interpolation
        if len(baseline_convergence) < len(eval_points):
            baseline_convergence = np.pad(baseline_convergence, 
                                    (0, len(eval_points) - len(baseline_convergence)), 
                                    'constant', 
                                    constant_values=baseline_convergence[-1] if baseline_convergence else baseline_best)
        if len(meta_convergence) < len(eval_points):
            meta_convergence = np.pad(meta_convergence, 
                                (0, len(eval_points) - len(meta_convergence)), 
                                'constant', 
                                constant_values=meta_convergence[-1] if meta_convergence else meta_best)
        if len(enhanced_convergence) < len(eval_points):
            enhanced_convergence = np.pad(enhanced_convergence, 
                                    (0, len(eval_points) - len(enhanced_convergence)), 
                                    'constant', 
                                    constant_values=enhanced_convergence[-1] if enhanced_convergence else enhanced_best)
        if len(satzilla_convergence) < len(eval_points):
            satzilla_convergence = np.pad(satzilla_convergence, 
                                    (0, len(eval_points) - len(satzilla_convergence)), 
                                    'constant', 
                                    constant_values=satzilla_convergence[-1] if satzilla_convergence else satzilla_best)
        
        # Ensure arrays aren't too long
        baseline_convergence = baseline_convergence[:len(eval_points)]
        meta_convergence = meta_convergence[:len(eval_points)]
        enhanced_convergence = enhanced_convergence[:len(eval_points)]
        satzilla_convergence = satzilla_convergence[:len(eval_points)]
        
        # Plot convergence curves with logarithmic y-axis
        ax1.set_yscale('log')
        if baseline_evals > 0:
            baseline_selected = baseline_results.get("selected_algorithm", "unknown")
            ax1.plot(eval_points, baseline_convergence, 'b-', label=f'Baseline ({baseline_selected})')
        if meta_evals > 0:
            meta_selected = meta_results.get("selected_algorithm", "unknown")
            ax1.plot(eval_points, meta_convergence, 'r-', label=f'Meta Learner')
        if enhanced_evals > 0:
            enhanced_selected = enhanced_results.get("selected_algorithm", "unknown")
            ax1.plot(eval_points, enhanced_convergence, 'g-', label=f'Enhanced Meta')
        if satzilla_evals > 0:
            satzilla_selected = satzilla_results.get("selected_algorithm", "unknown")
            ax1.plot(eval_points, satzilla_convergence, 'c-', label=f'SATzilla')
        
        ax1.set_xlabel('Function Evaluations')
        ax1.set_ylabel('Best Fitness')
        ax1.set_title('Convergence Curves')
        ax1.grid(True, alpha=0.3, which='both')
        ax1.legend()
        
        # Subplot 2: Evaluation Budget Comparison
        ax2 = fig.add_subplot(gs[0, 1])
        
        # Calculate evaluation metrics
        eval_data = [baseline_evals, meta_evals, enhanced_evals, satzilla_evals]
        eval_labels = ['Baseline', 'Meta-Learner', 'Enhanced', 'SATzilla']
        
        # Plot evaluation bar chart
        bars = ax2.bar(eval_labels, eval_data)
        
        # Add percentage labels for comparative evaluations
        if baseline_evals > 0:
            for i, bar in enumerate(bars[1:], 1):
                percent_diff = ((eval_data[i] - baseline_evals) / baseline_evals) * 100
                if percent_diff != 0:
                    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
                            f'{percent_diff:.1f}% {"more" if percent_diff > 0 else "fewer"} evaluations',
                            ha='center', va='bottom', rotation=0, bbox=dict(facecolor='red' if percent_diff > 0 else 'green', alpha=0.3))
        
        ax2.set_ylabel('Number of Evaluations')
        ax2.set_title('Evaluation Budget Comparison')
        
        # Subplot 3: Fitness at Normalized Evaluation Points
        ax3 = fig.add_subplot(gs[1, 0])
        
        # Normalize evaluation points to percentages
        percentage_points = [25, 50, 75, 100]  # 25%, 50%, 75%, 100% of budget
        normalized_points = [int(p * max_evals / 100) for p in percentage_points]
        
        # Get fitness values at these points
        baseline_fitness_at_points = []
        meta_fitness_at_points = []
        enhanced_fitness_at_points = []
        satzilla_fitness_at_points = []
        
        for point_idx, eval_point in enumerate(normalized_points):
            # Find index of closest evaluation point
            closest_idx = min(int(eval_point / max_evals * len(eval_points)), len(eval_points) - 1)
            
            # Get fitness at that point
            if closest_idx < len(baseline_convergence):
                baseline_fitness_at_points.append(baseline_convergence[closest_idx])
            else:
                baseline_fitness_at_points.append(baseline_best)
                
            if closest_idx < len(meta_convergence):
                meta_fitness_at_points.append(meta_convergence[closest_idx])
            else:
                meta_fitness_at_points.append(meta_best)
                
            if closest_idx < len(enhanced_convergence):
                enhanced_fitness_at_points.append(enhanced_convergence[closest_idx])
            else:
                enhanced_fitness_at_points.append(enhanced_best)
                
            if closest_idx < len(satzilla_convergence):
                satzilla_fitness_at_points.append(satzilla_convergence[closest_idx])
            else:
                satzilla_fitness_at_points.append(satzilla_best)
            
            # Add improvement percentages
            if point_idx > 0 and baseline_fitness_at_points[point_idx] > 0:
                # For Baseline
                baseline_improvement = ((baseline_fitness_at_points[0] - baseline_fitness_at_points[point_idx]) / 
                                      baseline_fitness_at_points[0]) * 100
                ax3.text(point_idx, baseline_fitness_at_points[point_idx] * 1.1, 
                        f"{baseline_improvement:.1f}%", color='blue', ha='center')
                
                # For Meta
                if meta_fitness_at_points[point_idx] > 0:
                    meta_improvement = ((meta_fitness_at_points[0] - meta_fitness_at_points[point_idx]) / 
                                       meta_fitness_at_points[0]) * 100
                    ax3.text(point_idx, meta_fitness_at_points[point_idx] * 0.9, 
                            f"{meta_improvement:.1f}%", color='red', ha='center')
                
                # For enhanced and SATzilla, add if needed
        
        # Plot fitness values
        ax3.set_yscale('log')
        ax3.plot(percentage_points, baseline_fitness_at_points, 'bo-', label='Baseline')
        ax3.plot(percentage_points, meta_fitness_at_points, 'ro-', label='Meta-Optimizer')
        if all(v > 0 for v in enhanced_fitness_at_points):
            ax3.plot(percentage_points, enhanced_fitness_at_points, 'go-', label='Enhanced')
        if all(v > 0 for v in satzilla_fitness_at_points):
            ax3.plot(percentage_points, satzilla_fitness_at_points, 'co-', label='SATzilla')
        
        ax3.set_xlabel('Percentage of Evaluation Budget')
        ax3.set_ylabel('Best Fitness')
        ax3.set_title('Fitness at Normalized Evaluation Points')
        ax3.set_xticks(percentage_points)
        ax3.set_xticklabels([f"{p}%" for p in percentage_points])
        ax3.grid(True, alpha=0.3)
        ax3.legend()
        
        # Subplot 4: Optimizer Efficiency (fitness improvement per evaluation)
        ax4 = fig.add_subplot(gs[1, 1])
        
        # Calculate efficiency metrics
        efficiency_baseline = baseline_best / max(1, baseline_evals)
        efficiency_meta = meta_best / max(1, meta_evals)
        efficiency_enhanced = enhanced_best / max(1, enhanced_evals)
        efficiency_satzilla = satzilla_best / max(1, satzilla_evals)
        
        efficiency_data = [efficiency_baseline, efficiency_meta, efficiency_enhanced, efficiency_satzilla]
        efficiency_labels = ['Baseline', 'Meta-Optimizer', 'Enhanced', 'SATzilla']
        
        # Plot efficiency bar chart
        bars = ax4.bar(efficiency_labels, efficiency_data)
        
        # Add comparative efficiency labels
        if efficiency_baseline > 0:
            for i, bar in enumerate(bars[1:], 1):
                rel_efficiency = efficiency_data[i] / efficiency_baseline
                label = "Same efficiency"
                if rel_efficiency < 0.95:
                    label = f"{100 * (1 - rel_efficiency):.1f}% more efficient"
                elif rel_efficiency > 1.05:
                    label = f"{100 * (rel_efficiency - 1):.1f}% less efficient"
                
                ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height()/2,
                        label, ha='center', va='center', 
                        bbox=dict(facecolor='green' if rel_efficiency <= 1 else 'red', alpha=0.3))
        
        ax4.set_yscale('log')
        ax4.set_ylabel('Fitness per Evaluation (lower is better)')
        ax4.set_title('Optimizer Efficiency')
        
        # Use constrained_layout instead of tight_layout for better spacing
        fig.set_constrained_layout(True)
        
        # Save the plot
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, dpi=150)
        plt.close()

    def generate_aggregate_normalized_comparison(self, normalized_metrics_list, output_path, title=None):
        """
        Generate aggregate normalized comparison plots across multiple trials.
        
        Args:
            normalized_metrics_list: List of normalized metrics from multiple trials
            output_path: Path to save the plot
            title: Optional title for the plot
        """
        import matplotlib.pyplot as plt
        import numpy as np
        
        # If no normalized metrics, skip plotting
        if not normalized_metrics_list:
            logger.warning("No normalized metrics to plot")
            return
        
        # Extract metrics from all trials
        simple_fitness = [m.get("simple_fitness", float('inf')) for m in normalized_metrics_list]
        meta_fitness = [m.get("meta_fitness", float('inf')) for m in normalized_metrics_list]
        enhanced_fitness = [m.get("enhanced_fitness", float('inf')) for m in normalized_metrics_list]
        satzilla_fitness = [m.get("satzilla_fitness", float('inf')) for m in normalized_metrics_list]
        
        simple_evals = [m.get("simple_evaluations", 0) for m in normalized_metrics_list]
        meta_evals = [m.get("meta_evaluations", 0) for m in normalized_metrics_list]
        enhanced_evals = [m.get("enhanced_evaluations", 0) for m in normalized_metrics_list]
        satzilla_evals = [m.get("satzilla_evaluations", 0) for m in normalized_metrics_list]
        
        # Create the plot with a grid of 2x2 subplots
        fig, axs = plt.subplots(2, 2, figsize=(12, 10))
        if title:
            fig.suptitle(title, fontsize=16)
        
        # Plot 1: Best Fitness Comparison
        ax1 = axs[0, 0]
        boxplot_data = [simple_fitness, meta_fitness, enhanced_fitness, satzilla_fitness]
        ax1.boxplot(boxplot_data)
        ax1.set_xticklabels(['Simple', 'Meta', 'Enhanced', 'SATzilla'])
        ax1.set_ylabel('Best Fitness')
        ax1.set_title('Best Fitness Comparison')
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Evaluations Comparison
        ax2 = axs[0, 1]
        boxplot_data = [simple_evals, meta_evals, enhanced_evals, satzilla_evals]
        ax2.boxplot(boxplot_data)
        ax2.set_xticklabels(['Simple', 'Meta', 'Enhanced', 'SATzilla'])
        ax2.set_ylabel('Evaluations')
        ax2.set_title('Evaluations Comparison')
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Fitness Improvement over Simple Baseline
        ax3 = axs[1, 0]
        
        # Calculate fitness improvement percentages
        fitness_improvements = []
        if simple_fitness and np.mean(simple_fitness) != 0:
            baseline_fitness = np.mean(simple_fitness)
            simple_imp = 0  # 0% improvement over itself
            meta_imp = 100 * (baseline_fitness - np.mean(meta_fitness)) / baseline_fitness if baseline_fitness != 0 else 0
            enhanced_imp = 100 * (baseline_fitness - np.mean(enhanced_fitness)) / baseline_fitness if baseline_fitness != 0 else 0
            satzilla_imp = 100 * (baseline_fitness - np.mean(satzilla_fitness)) / baseline_fitness if baseline_fitness != 0 else 0
            
            fitness_improvements = [simple_imp, meta_imp, enhanced_imp, satzilla_imp]
        
        # Plot fitness improvement bar chart
        if fitness_improvements:
            ax3.bar(['Simple', 'Meta', 'Enhanced', 'SATzilla'], fitness_improvements)
            ax3.set_ylabel('Fitness Improvement (%)')
            ax3.set_title('Fitness Improvement over Simple Baseline')
            ax3.grid(True, alpha=0.3)
        else:
            ax3.text(0.5, 0.5, 'No improvement data available', 
                   ha='center', va='center', transform=ax3.transAxes)
        
        # Plot 4: Evaluations Efficiency Comparison
        ax4 = axs[1, 1]
        
        # Calculate evaluations efficiency as a normalized ratio
        # Higher values indicate better efficiency (better fitness per evaluation)
        eval_efficiency = []
        if simple_evals and simple_fitness:
            # Calculate average fitness values, replacing inf with a very large number
            avg_simple_fitness = np.mean([f if f != float('inf') else 1e10 for f in simple_fitness])
            avg_meta_fitness = np.mean([f if f != float('inf') else 1e10 for f in meta_fitness])
            avg_enhanced_fitness = np.mean([f if f != float('inf') else 1e10 for f in enhanced_fitness]) 
            avg_satzilla_fitness = np.mean([f if f != float('inf') else 1e10 for f in satzilla_fitness])
            
            # Average evaluations
            avg_simple_evals = max(1, np.mean(simple_evals))
            avg_meta_evals = max(1, np.mean(meta_evals))
            avg_enhanced_evals = max(1, np.mean(enhanced_evals))
            avg_satzilla_evals = max(1, np.mean(satzilla_evals))
            
            # Fitness per evaluation ratio (normalized so Simple = 1.0)
            # For minimization problems, lower fitness is better
            if avg_simple_fitness > 0:  # For minimization problems with positive values
                simple_eff = 1.0  # Reference point
                meta_eff = (avg_simple_fitness / avg_meta_fitness) * (avg_simple_evals / avg_meta_evals) if avg_meta_fitness > 0 else 0
                enhanced_eff = (avg_simple_fitness / avg_enhanced_fitness) * (avg_simple_evals / avg_enhanced_evals) if avg_enhanced_fitness > 0 else 0
                satzilla_eff = (avg_simple_fitness / avg_satzilla_fitness) * (avg_simple_evals / avg_satzilla_evals) if avg_satzilla_fitness > 0 else 0
            else:  # For cases where fitness is 0 or negative
                simple_eff = 1.0
                meta_eff = avg_simple_evals / avg_meta_evals
                enhanced_eff = avg_simple_evals / avg_enhanced_evals
                satzilla_eff = avg_simple_evals / avg_satzilla_evals
            
            eval_efficiency = [simple_eff, meta_eff, enhanced_eff, satzilla_eff]
        
        # Plot evaluations efficiency bar chart
        if eval_efficiency:
            ax4.bar(['Simple', 'Meta', 'Enhanced', 'SATzilla'], eval_efficiency)
            ax4.set_ylabel('Efficiency Ratio (higher is better)')
            ax4.set_title('Efficiency Ratio (Fitness/Evaluations)')
            ax4.grid(True, alpha=0.3)
            # Horizontal line at 1.0 (the simple baseline)
            ax4.axhline(y=1.0, color='r', linestyle='--', alpha=0.5)
        else:
            ax4.text(0.5, 0.5, 'No efficiency data available', 
                   ha='center', va='center', transform=ax4.transAxes)
        
        # Use constrained_layout instead of tight_layout for better spacing
        fig.set_constrained_layout(True)
        
        # Save the plot
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, dpi=150)
        plt.close()