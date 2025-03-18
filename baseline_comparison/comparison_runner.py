"""
Comparison runner for baseline algorithm selectors and Meta Optimizer

This module provides a framework for benchmarking baseline algorithm
selection methods against the Meta Optimizer.
"""

import os
import json
import time
import logging
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Any, Tuple, Optional
from cli.problem_wrapper import ProblemWrapper
from meta_optimizer.benchmark.test_functions import create_test_suite
from scipy import stats
from pathlib import Path

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
            output_dir: Directory to save results
            model_path: Path to a trained SatzillaInspiredSelector model
        """
        self.baseline_selector = baseline_selector
        self.meta_optimizer = meta_optimizer
        self.max_evaluations = max_evaluations
        self.num_trials = num_trials
        self.verbose = verbose
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.results = {}
        
        # Load trained model if available
        self.model_loaded = False
        
        # First check if the selector is already trained (might have been loaded outside)
        if hasattr(self.baseline_selector, 'is_trained') and self.baseline_selector.is_trained:
            logger.info("Baseline selector is already trained, no need to load model.")
            # Set model_loaded to True
            self.model_loaded = True
        elif model_path and hasattr(self.baseline_selector, 'load_model'):
            try:
                logger.info(f"Attempting to load trained model from {model_path}")
                # Check if the model file exists
                model_path_obj = Path(model_path)
                
                if not model_path_obj.exists():
                    logger.warning(f"Model file {model_path} does not exist. Checking for alternate paths...")
                    
                    # Try with .joblib extension
                    if not str(model_path).endswith('.joblib'):
                        alt_path = f"{model_path}.joblib"
                        if Path(alt_path).exists():
                            model_path = alt_path
                            logger.info(f"Found model at alternate path: {model_path}")
                    
                    # Try looking in models subdirectory
                    if not Path(model_path).exists():
                        models_dir = Path(model_path).parent / "models"
                        if models_dir.exists():
                            model_name = Path(model_path).name
                            alt_path = models_dir / model_name
                            if alt_path.exists():
                                model_path = alt_path
                                logger.info(f"Found model in models subdirectory: {model_path}")
                            elif (models_dir / f"{model_name}.joblib").exists():
                                model_path = models_dir / f"{model_name}.joblib"
                                logger.info(f"Found model with joblib extension in models subdirectory: {model_path}")
                
                # Attempt to load the model - capture the return value to check success
                load_success = self.baseline_selector.load_model(str(model_path))
                
                # Verify the model is properly loaded - different verifications for different selector types
                if hasattr(self.baseline_selector, 'is_trained'):
                    # SatzillaInspiredSelector has an is_trained property
                    model_loaded_successfully = self.baseline_selector.is_trained
                    logger.info(f"Model loaded check via is_trained property: {model_loaded_successfully}")
                else:
                    # Other selectors might have different ways to check
                    model_loaded_successfully = load_success
                    logger.info(f"Model loaded check via load_model return value: {model_loaded_successfully}")
                
                # Additional validation for SatzillaInspiredSelector
                if hasattr(self.baseline_selector, 'models'):
                    has_models = bool(self.baseline_selector.models)
                    logger.info(f"Selector has models: {has_models} ({len(self.baseline_selector.models) if has_models else 0} models)")
                    
                    # Make sure at least some models actually exist
                    valid_models = []
                    if has_models:
                        for alg, model in self.baseline_selector.models.items():
                            if model is not None:
                                valid_models.append(alg)
                        logger.info(f"Valid models found for algorithms: {valid_models}")
                        
                    model_loaded_successfully = model_loaded_successfully and bool(valid_models)
                
                if model_loaded_successfully:
                    logger.info("Model loaded successfully!")
                    self.model_loaded = True
                else:
                    logger.warning("Model load attempt completed but verification failed")
                    self.model_loaded = False
            except Exception as e:
                logger.error(f"Failed to load model from {model_path}: {str(e)}")
                logger.error("Will proceed with untrained selector (random selection)")
        elif model_path:
            logger.warning(f"Baseline selector does not have a 'load_model' method. Ignoring model_path: {model_path}")
        else:
            logger.warning("No model_path provided. Using untrained baseline selector.")
        
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
        
        for trial in range(1, num_trials + 1):
            logger.info(f"  Trial {trial}/{num_trials}")
            
            # Run baseline
            start_time = time.time()
            baseline_result = self._run_baseline(problem_func, max_evaluations)
            baseline_time.append(time.time() - start_time)
            baseline_best_fitness.append(baseline_result["best_fitness"])
            baseline_evaluations.append(baseline_result["evaluations"])
            baseline_convergence.append(baseline_result.get("convergence_data", []))
            baseline_selections.append(baseline_result["selected_algorithm"])
            
            # Run meta optimizer
            start_time = time.time()
            meta_result = self._run_meta_optimizer(problem_func, max_evaluations)
            meta_time.append(time.time() - start_time)
            meta_best_fitness.append(meta_result["best_fitness"])
            meta_evaluations.append(meta_result["evaluations"])
            meta_convergence.append(meta_result.get("convergence_data", []))
            meta_selections.append(meta_result["selected_algorithm"])
        
        # Calculate averages and store results
        self.results[problem_name] = {
            "baseline_best_fitness_all": baseline_best_fitness,
            "baseline_evaluations_all": baseline_evaluations,
            "baseline_time_all": baseline_time,
            "baseline_convergence_data": baseline_convergence,
            "baseline_selected_algorithms": baseline_selections,
            "baseline_best_fitness_avg": np.mean(baseline_best_fitness),
            "baseline_evaluations_avg": np.mean(baseline_evaluations),
            "baseline_time_avg": np.mean(baseline_time),
            
            "meta_best_fitness_all": meta_best_fitness,
            "meta_evaluations_all": meta_evaluations,
            "meta_time_all": meta_time,
            "meta_convergence_data": meta_convergence,
            "meta_selected_algorithms": meta_selections,
            "meta_best_fitness_avg": np.mean(meta_best_fitness),
            "meta_evaluations_avg": np.mean(meta_evaluations),
            "meta_time_avg": np.mean(meta_time),
            
            "improvement_percentage": self._calculate_improvement_percentage(
                np.mean(baseline_best_fitness), 
                np.mean(meta_best_fitness)
            )
        }
        
        # Save results after each problem
        self.save_results()
        
        # Generate and save plots
        self.plot_performance_comparison(self.results)
        plt.savefig(os.path.join(self.output_dir, 'performance_comparison.png'))
        plt.close()
        
        self.plot_violin_comparison(self.results)
        
        self.plot_algorithm_selection_frequency(self.results)
        plt.savefig(os.path.join(self.output_dir, 'algorithm_selection.png'))
        plt.close()
        
        # Plot convergence curves for this problem
        self.plot_convergence_curves(self.results[problem_name], self.output_dir)
    
    def _run_baseline(self, problem: ProblemWrapper, max_evaluations: int) -> Dict[str, Any]:
        start_time = time.time()
        convergence_history = []
        best_fitness = float('inf')
        
        # Create a wrapper function to track convergence
        def tracking_objective(fitness):
            nonlocal best_fitness
            best_fitness = min(best_fitness, fitness)
            convergence_history.append(best_fitness)

        # Run baseline optimization with tracking
        problem.tracking_objective = tracking_objective
        result = self.baseline_selector.optimize(problem, max_evaluations)
        selected_algorithm = self.baseline_selector.get_selected_algorithm() if hasattr(self.baseline_selector, 'get_selected_algorithm') else 'unknown'
        
        end_time = time.time()
        
        return {
            'best_solution': result[0],
            'best_fitness': result[1],
            'evaluations': problem.evaluations,
            'time': end_time - start_time,
            'convergence_data': convergence_history,
            'selected_algorithm': selected_algorithm
        }
    
    def _run_meta_optimizer(self, problem: ProblemWrapper, max_evaluations: int) -> Dict[str, Any]:
        start_time = time.time()
        convergence_history = []
        best_fitness = float('inf')
        
        # Create a wrapper function to track convergence
        def tracking_objective(fitness):
            nonlocal best_fitness
            best_fitness = min(best_fitness, fitness)
            convergence_history.append(best_fitness)

        # Run meta optimization with tracking
        problem.tracking_objective = tracking_objective
        result = self.meta_optimizer.optimize(problem, max_evaluations)
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
        
        end_time = time.time()
        
        return {
            'best_solution': result[0],
            'best_fitness': result[1],
            'evaluations': problem.evaluations,
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
        test_functions = create_test_suite()
        logging.info(f"Running comparison for {len(test_functions)} benchmark functions")
        
        for func_name, func_class in test_functions.items():
            logging.info(f"Running comparison for {func_name}")
            # Create test function with specified dimensions and bounds
            bounds = [(0, 1)] * self.meta_optimizer.dim  # Default bounds [0,1] for each dimension
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
            func_names.append(func_name)
            baseline = func_results["baseline_best_fitness_all"]
            meta = func_results["meta_best_fitness_all"]
            
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
        
        if not p_values:  # If no valid p-values, skip plotting
            logger.warning("No valid statistical tests to plot")
            return
        
        # Create subplot for p-values
        plt.subplot(1, 2, 1)
        bars = plt.bar(func_names, p_values, color='skyblue')
        
        # Add significance markers
        for i, (bar, sig) in enumerate(zip(bars, significance)):
            plt.text(i, min(bar.get_height() + 0.01, 1.0), sig, 
                    ha='center', va='bottom', fontweight='bold')
        
        plt.axhline(y=0.05, color='r', linestyle='--', alpha=0.7)
        plt.text(len(func_names)-1, 0.055, 'p=0.05', color='r', ha='right')
        
        plt.ylabel('p-value')
        plt.title('Statistical Significance (t-test)')
        plt.xticks(rotation=45, ha='right')
        
        max_p = max(p for p in p_values if not np.isnan(p) and not np.isinf(p))
        plt.ylim(0, min(max_p * 1.2 + 0.05, 1.05))  # Cap at slightly above 1.0
        
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
        
        plt.ylabel("Cohen's d")
        plt.title('Effect Size')
        plt.xticks(rotation=45, ha='right')
        
        # Set reasonable y limits for effect sizes
        max_effect = max(d for d in effect_sizes if not np.isnan(d) and not np.isinf(d))
        plt.ylim(0, min(max_effect * 1.2, 5.0))  # Cap at 5.0
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'statistical_tests.png'))
        plt.close()

    def plot_radar_comparison(self, results: Dict[str, Dict]) -> None:
        """Create a radar chart comparing multiple metrics between baseline and meta optimizer."""
        import matplotlib.pyplot as plt
        import numpy as np
        
        # Prepare data for radar chart
        func_names = list(results.keys())
        n_funcs = len(func_names)
        
        # Calculate relative improvements for each metric
        fitness_improvement = []
        time_improvement = []
        evaluations_improvement = []
        
        for func_name in func_names:
            # Fitness improvement (negative is better)
            baseline_fitness = results[func_name]["baseline_best_fitness_avg"]
            meta_fitness = results[func_name]["meta_best_fitness_avg"]
            rel_fitness_imp = (baseline_fitness - meta_fitness) / max(abs(baseline_fitness), 1e-10)
            fitness_improvement.append(max(min(rel_fitness_imp, 1.0), -1.0))  # Clamp to [-1, 1]
            
            # Time improvement (negative is better)
            baseline_time = results[func_name]["baseline_time_avg"]
            meta_time = results[func_name]["meta_time_avg"]
            rel_time_imp = (baseline_time - meta_time) / max(baseline_time, 1e-10)
            time_improvement.append(max(min(rel_time_imp, 1.0), -1.0))  # Clamp to [-1, 1]
            
            # Evaluations improvement (negative is better)
            baseline_evals = results[func_name]["baseline_evaluations_avg"]
            meta_evals = results[func_name]["meta_evaluations_avg"]
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
        plt.savefig(os.path.join(self.output_dir, 'radar_comparison.png'))
        plt.close()

    def create_summary_table(self, results: Dict[str, Dict]) -> None:
        """Create a summary table of results as an image."""
        from matplotlib.table import Table
        
        fig, ax = plt.subplots(figsize=(12, len(results) * 0.8 + 2))
        ax.axis('off')
        ax.axis('tight')
        
        # Prepare data for table
        table_data = []
        for func_name, func_results in results.items():
            baseline_fitness = max(func_results["baseline_best_fitness_avg"], 1e-16)  # Ensure non-zero for display
            meta_fitness = max(func_results["meta_best_fitness_avg"], 1e-16)  # Ensure non-zero for display
            
            # Calculate improvement - avoid division by zero
            if baseline_fitness == 0 and meta_fitness == 0:
                improvement = 0.0  # If both are 0, no improvement
            elif baseline_fitness == 0:
                improvement = -100.0  # If baseline is 0 and meta is not, this is degradation
            else:
                improvement = (baseline_fitness - meta_fitness) / baseline_fitness * 100
            
            baseline_evals = func_results["baseline_evaluations_avg"]
            meta_evals = func_results["meta_evaluations_avg"]
            
            baseline_time = func_results["baseline_time_avg"]
            meta_time = func_results["meta_time_avg"]
            
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
            improvement = float(row[3].strip('%'))
            cell = table[(i+1, 3)]  # +1 for header row
            
            if improvement > 10:  # Significant improvement
                cell.set_facecolor('lightgreen')
            elif improvement < -10:  # Significant degradation
                cell.set_facecolor('lightcoral')
            else:  # Neutral
                cell.set_facecolor('lightyellow')
        
        # Color the header row
        for i in range(len(columns)):
            table[(0, i)].set_facecolor('lightblue')
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
        # Count algorithm frequencies
        baseline_counts = {}
        meta_counts = {}
        
        for problem_name, result in self.results.items():
            for trial_result in result['trials']:
                baseline_alg = trial_result['baseline']['selected_algorithm']
                meta_alg = trial_result['meta']['selected_algorithm']
                
                # Update baseline counts
                if baseline_alg not in baseline_counts:
                    baseline_counts[baseline_alg] = 0
                baseline_counts[baseline_alg] += 1
                
                # Update meta counts
                if meta_alg not in meta_counts:
                    meta_counts[meta_alg] = 0
                meta_counts[meta_alg] += 1
        
        # Create figure with two subplots
        fig = plt.figure(figsize=(12, 6))
        
        # Baseline
        baseline_keys = list(baseline_counts.keys())
        baseline_values = list(baseline_counts.values())
        ax1.bar(range(len(baseline_keys)), baseline_values)
        ax1.set_title('Baseline Algorithm Selection')
        ax1.set_ylabel('Frequency')
        ax1.set_xticks(range(len(baseline_keys)))
        ax1.set_xticklabels(baseline_keys, rotation=45, ha='right')
        
        # Meta Optimizer
        meta_keys = list(meta_counts.keys())
        meta_values = list(meta_counts.values())
        ax2.bar(range(len(meta_keys)), meta_values)
        ax2.set_title('Meta Optimizer Algorithm Selection')
        ax2.set_ylabel('Frequency')
        ax2.set_xticks(range(len(meta_keys)))
        ax2.set_xticklabels(meta_keys, rotation=45, ha='right')
        
        fig.tight_layout() 

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
                return -100.0  # Meta performed worse
        
        raw_improvement = (baseline_fitness - meta_fitness) / abs(baseline_fitness) * 100
        
        # Cap improvement between -100% and +100%
        return max(min(raw_improvement, 100.0), -100.0) 