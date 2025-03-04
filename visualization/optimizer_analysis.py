"""
optimizer_analysis.py
-------------------
Visualization and analysis tools for optimization algorithms.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Any, Tuple, Callable, Optional
from dataclasses import dataclass
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from concurrent.futures import ThreadPoolExecutor
import time
from pathlib import Path
from tqdm.auto import tqdm
import copy

# Import save_plot function
import sys
sys.path.append(str(Path(__file__).parent.parent))
from utils.plot_utils import save_plot

@dataclass
class OptimizationResult:
    """Store optimization run results"""
    optimizer_name: str
    function_name: str
    best_solution: np.ndarray
    best_score: float
    convergence_curve: List[float]
    execution_time: float
    hyperparameters: Dict[str, Any]
    success_rate: Optional[float] = None
    diversity_history: Optional[List[float]] = None
    param_history: Optional[Dict[str, List[float]]] = None
    landscape_metrics: Optional[Dict[str, float]] = None
    gradient_history: Optional[List[np.ndarray]] = None

class OptimizerAnalyzer:
    def __init__(self, optimizers: Dict[str, Any]):
        """
        Initialize optimizer analyzer.
        
        Args:
            optimizers: Dictionary mapping optimizer names to optimizer instances
        """
        self.optimizers = optimizers
        self.results = {}
        
    def run_optimizer(self, optimizer_name, optimizer, max_evals, num_runs):
        """Run a single optimizer multiple times and collect results."""
        run_results = []
        total_runtime = 0
        
        # Create progress bar for runs
        with tqdm(total=num_runs, desc=f"Run", unit="runs", 
                 disable=not self.verbose, leave=False, 
                 position=1) as run_pbar:
            
            for run in range(1, num_runs + 1):
                # Reset the optimizer if it has a reset method
                if hasattr(optimizer, 'reset') and callable(getattr(optimizer, 'reset')):
                    optimizer.reset()
                
                # Deep copy the objective function to avoid shared state issues
                # This is particularly important for stateful objective functions
                run_objective = copy.deepcopy(self.objective_func)
                
                try:
                    # Run the optimizer
                    start_time = time.time()
                    
                    # Handle different optimizer interfaces
                    if hasattr(optimizer, 'run') and callable(getattr(optimizer, 'run')):
                        # Meta-optimizer or similar with run method
                        results = optimizer.run(run_objective, max_evals)
                        best_solution = results.get('solution')
                        best_value = results.get('score')
                        evaluations = results.get('evaluations', max_evals)
                        # Get convergence curve if available
                        curve = results.get('convergence_curve', [])
                        # Ensure curve is not empty
                        if not curve:
                            curve = [(0, best_value), (evaluations, best_value)]
                    else:
                        # Standard optimizer with optimize method
                        best_solution, best_value = optimizer.optimize(run_objective, max_evals=max_evals)
                        evaluations = getattr(optimizer, 'evaluations', max_evals)
                        # Get convergence curve if available
                        curve = getattr(optimizer, 'convergence_curve', [])
                        # Ensure curve is not empty
                        if not curve:
                            curve = [(0, best_value), (evaluations, best_value)]
                    
                    runtime = time.time() - start_time
                    total_runtime += runtime
                    
                    # Ensure we have valid data
                    if best_solution is None:
                        best_solution = np.zeros(self.dimension)
                        best_value = float('inf')
                    
                    # Collect results
                    run_results.append({
                        'run': run,
                        'best_solution': best_solution,
                        'best_value': best_value,
                        'evaluations': evaluations,
                        'runtime': runtime,
                        'convergence_curve': curve
                    })
                    
                    # Update progress
                    run_pbar.set_postfix({'best_score': f"{best_value:.10f}", 'evals': f"{evaluations}"})
                    run_pbar.update(1)
                    
                except Exception as e:
                    print(f"Error running {optimizer_name} (Run {run}): {str(e)}")
                    import traceback
                    traceback.print_exc()
                    # Add a placeholder result to maintain run count
                    run_results.append({
                        'run': run,
                        'best_solution': np.zeros(self.dimension),
                        'best_value': float('inf'),
                        'evaluations': 0,
                        'runtime': 0,
                        'convergence_curve': [(0, float('inf')), (1, float('inf'))]
                    })
                    run_pbar.update(1)
                    
    def run_comparison(
            self,
            test_functions: Dict[str, Any],
            n_runs: int = 30,
            record_convergence: bool = True,
            max_evals: Optional[int] = None
        ) -> Dict[str, Dict[str, List[OptimizationResult]]]:
        """
        Run optimization comparison.
        
        Args:
            test_functions: Dictionary mapping function names to test functions
            n_runs: Number of independent runs per optimizer
            record_convergence: Whether to record convergence history
            max_evals: Maximum number of function evaluations per run
            
        Returns:
            Dictionary mapping function names to dictionaries mapping optimizer names to lists of results
        """
        print(f"Running comparison with {n_runs} independent runs per configuration")
        
        for func_name, func in test_functions.items():
            print(f"\nOptimizing {func_name}")
            self.results[func_name] = {}
            
            # Add tqdm for the optimizer loop
            optimizer_pbar = tqdm(self.optimizers.items(), desc="Optimizers", leave=False)
            for opt_name, optimizer in optimizer_pbar:
                optimizer_pbar.set_description(f"Optimizer: {opt_name}")
                print(f"  Using {opt_name}")
                results = []
                
                # Add tqdm for the run loop
                run_pbar = tqdm(range(n_runs), desc=f"Runs for {opt_name}", leave=False)
                for run in run_pbar:
                    run_pbar.set_description(f"Run {run+1}/{n_runs}")
                    start_time = time.time()
                    optimizer.reset()  # Reset optimizer state
                    
                    # Run optimization with specified max_evals
                    best_solution = optimizer.optimize(
                        func,
                        max_evals=max_evals if max_evals is not None else optimizer.max_evals
                    )
                    
                    # Get parameters including actual evaluations used
                    params = optimizer.get_parameters()
                    if hasattr(optimizer, 'evaluations'):
                        params['evaluations'] = optimizer.evaluations
                    
                    # Get landscape metrics and gradient history if available
                    landscape_metrics = None
                    if hasattr(optimizer, 'get_landscape_metrics'):
                        landscape_metrics = optimizer.get_landscape_metrics()
                    
                    gradient_history = None
                    if hasattr(optimizer, 'gradient_history'):
                        gradient_history = optimizer.gradient_history
                    
                    # Store results
                    results.append(OptimizationResult(
                        optimizer_name=opt_name,
                        function_name=func_name,
                        best_solution=best_solution,
                        best_score=optimizer.best_score,
                        convergence_curve=optimizer.convergence_curve if record_convergence else [],
                        execution_time=time.time() - start_time,
                        hyperparameters=params,
                        success_rate=optimizer.success_rate if hasattr(optimizer, 'success_rate') else None,
                        diversity_history=optimizer.diversity_history if hasattr(optimizer, 'diversity_history') else None,
                        param_history=optimizer.param_history if hasattr(optimizer, 'param_history') else None,
                        landscape_metrics=landscape_metrics,
                        gradient_history=gradient_history
                    ))
                
                self.results[func_name][opt_name] = results
                
        return self.results
    
    def plot_landscape_analysis(self, save_path: Optional[str] = None):
        """Plot landscape analysis metrics"""
        if not self.results:
            raise ValueError("No results available. Run comparison first.")
            
        fig = plt.figure(figsize=(15, 10))
        
        # Close previously opened figures to avoid warnings
        for fig_num in plt.get_fignums():
            if fig_num != fig.number:
                plt.close(fig_num)
        
        # Plot 1: Ruggedness Analysis
        plt.subplot(2, 2, 1)
        has_ruggedness_data = False
        for func_name in self.results:
            for opt_name, results in self.results[func_name].items():
                metrics = [r.landscape_metrics for r in results if r.landscape_metrics]
                if metrics:
                    ruggedness = [m.get('ruggedness', 0) for m in metrics if 'ruggedness' in m]
                    if ruggedness:
                        plt.boxplot(ruggedness, positions=[len(plt.gca().get_xticks())],
                                  labels=[f"{opt_name}\n{func_name}"])
                        has_ruggedness_data = True
        plt.title('Landscape Ruggedness')
        plt.xticks(rotation=45)
        
        # Plot 2: Local Optima Analysis
        plt.subplot(2, 2, 2)
        has_optima_data = False
        for func_name in self.results:
            for opt_name, results in self.results[func_name].items():
                metrics = [r.landscape_metrics for r in results if r.landscape_metrics]
                if metrics:
                    optima = [m.get('local_optima_count', 0) for m in metrics if 'local_optima_count' in m]
                    if optima:
                        plt.boxplot(optima, positions=[len(plt.gca().get_xticks())],
                                  labels=[f"{opt_name}\n{func_name}"])
                        has_optima_data = True
        plt.title('Local Optima Count')
        plt.xticks(rotation=45)
        
        # Plot 3: Fitness-Distance Correlation
        plt.subplot(2, 2, 3)
        has_fdc_data = False
        for func_name in self.results:
            for opt_name, results in self.results[func_name].items():
                metrics = [r.landscape_metrics for r in results if r.landscape_metrics]
                if metrics:
                    fdc = [m.get('fitness_distance_correlation', 0) for m in metrics if 'fitness_distance_correlation' in m]
                    if fdc:
                        plt.boxplot(fdc, positions=[len(plt.gca().get_xticks())],
                                  labels=[f"{opt_name}\n{func_name}"])
                        has_fdc_data = True
        plt.title('Fitness-Distance Correlation')
        plt.xticks(rotation=45)
        
        # Plot 4: Landscape Smoothness
        plt.subplot(2, 2, 4)
        has_smoothness_data = False
        for func_name in self.results:
            for opt_name, results in self.results[func_name].items():
                metrics = [r.landscape_metrics for r in results if r.landscape_metrics]
                if metrics:
                    smoothness = [m.get('smoothness', 0) for m in metrics if 'smoothness' in m]
                    if smoothness:
                        plt.boxplot(smoothness, positions=[len(plt.gca().get_xticks())],
                                  labels=[f"{opt_name}\n{func_name}"])
                        has_smoothness_data = True
        plt.title('Landscape Smoothness')
        plt.xticks(rotation=45)
        
        try:
            plt.tight_layout()
        except Exception as e:
            print(f"Warning: Could not apply tight_layout: {e}")
            
        if save_path:
            plt.savefig(save_path)
            
        plt.close()
        
    def plot_gradient_analysis(self, save_path: Optional[str] = None):
        """Plot gradient-based analysis"""
        if not self.results:
            raise ValueError("No results available. Run comparison first.")
            
        fig = plt.figure(figsize=(15, 10))
        
        # Close previously opened figures to avoid warnings
        for fig_num in plt.get_fignums():
            if fig_num != fig.number:
                plt.close(fig_num)
        
        # Plot 1: Gradient Magnitude Evolution
        ax1 = plt.subplot(2, 2, 1)
        has_magnitude_data = False
        
        for func_name in self.results:
            for opt_name, results in self.results[func_name].items():
                for result in results:
                    if result.gradient_history and len(result.gradient_history) > 0:
                        try:
                            magnitudes = [np.linalg.norm(g) for g in result.gradient_history]
                            ax1.plot(magnitudes, label=f"{opt_name}-{func_name}")
                            has_magnitude_data = True
                        except (ValueError, TypeError):
                            continue
                            
        ax1.set_title('Gradient Magnitude Evolution')
        ax1.set_xlabel('Iteration')
        ax1.set_ylabel('Gradient Magnitude')
        if has_magnitude_data:  # Only add legend if we have data
            ax1.legend()
        
        # Plot 2: Gradient Direction Change
        ax2 = plt.subplot(2, 2, 2)
        has_angle_data = False
        
        for func_name in self.results:
            for opt_name, results in self.results[func_name].items():
                for result in results:
                    if result.gradient_history and len(result.gradient_history) > 1:
                        try:
                            angles = []
                            for i in range(1, len(result.gradient_history)):
                                g1 = result.gradient_history[i-1]
                                g2 = result.gradient_history[i]
                                
                                # Check if vectors are valid
                                if not isinstance(g1, np.ndarray) or not isinstance(g2, np.ndarray):
                                    continue
                                
                                # Ensure non-zero vectors
                                norm1 = np.linalg.norm(g1)
                                norm2 = np.linalg.norm(g2)
                                if norm1 == 0 or norm2 == 0:
                                    continue
                                    
                                cos_angle = np.dot(g1, g2) / (norm1 * norm2)
                                angle = np.arccos(np.clip(cos_angle, -1.0, 1.0))
                                angles.append(angle)
                                
                            if angles:  # Only plot if we have angles
                                ax2.plot(angles, label=f"{opt_name}-{func_name}")
                                has_angle_data = True
                        except (ValueError, TypeError):
                            continue
                            
        ax2.set_title('Gradient Direction Change')
        ax2.set_xlabel('Iteration')
        ax2.set_ylabel('Angle (radians)')
        if has_angle_data:  # Only add legend if we have data
            ax2.legend()
        
        # Plot 3: Gradient Component Analysis
        ax3 = plt.subplot(2, 2, 3)
        has_component_data = False
        
        for func_name in self.results:
            for opt_name, results in self.results[func_name].items():
                for result in results:
                    if result.gradient_history and len(result.gradient_history) > 0:
                        try:
                            # Ensure all gradients have the same shape
                            components = []
                            for g in result.gradient_history:
                                if isinstance(g, np.ndarray) and g.size > 0:
                                    components.append(g)
                            
                            if components:
                                components = np.array(components)
                                ax3.boxplot(components, labels=[f"Dim{i+1}" for i in range(components.shape[1])])
                                has_component_data = True
                        except (ValueError, TypeError):
                            continue
                            
        ax3.set_title('Gradient Components Distribution')
        ax3.set_xlabel('Dimension')
        ax3.set_ylabel('Gradient Value')
        
        # Plot 4: Gradient Stability
        ax4 = plt.subplot(2, 2, 4)
        has_stability_data = False
        
        for func_name in self.results:
            for opt_name, results in self.results[func_name].items():
                for result in results:
                    if result.gradient_history and len(result.gradient_history) > 0:
                        try:
                            # Ensure all gradients have the same shape
                            components = []
                            for g in result.gradient_history:
                                if isinstance(g, np.ndarray) and g.size > 0:
                                    components.append(g)
                            
                            if components:
                                components = np.array(components)
                                stability = np.std(components, axis=0)
                                ax4.bar(range(len(stability)), stability, 
                                       label=f"{opt_name}-{func_name}", alpha=0.5)
                                has_stability_data = True
                        except (ValueError, TypeError):
                            continue
                            
        ax4.set_title('Gradient Stability (Standard Deviation)')
        ax4.set_xlabel('Dimension')
        ax4.set_ylabel('Standard Deviation')
        if has_stability_data:  # Only add legend if we have data
            ax4.legend()
        
        try:
            plt.tight_layout()
        except Exception as e:
            print(f"Warning: Could not apply tight_layout: {e}")
        
        if save_path:
            plt.savefig(save_path)
            
        plt.close()
    
    def plot_parameter_adaptation(self, optimizer_name=None, function_name=None, save_path: Optional[str] = None):
        """
        Plot parameter adaptation history.
        
        Args:
            optimizer_name: Optional name of the optimizer to plot
            function_name: Optional name of the function to plot
            save_path: Optional path to save the figure
        """
        if not self.results:
            raise ValueError("No results available. Run comparison first.")
            
        # Collect all unique parameter names
        param_names = set()
        for func_name, func_results in self.results.items():
            if function_name and func_name != function_name:
                continue
                
            for opt_name, opt_results in func_results.items():
                if optimizer_name and opt_name != optimizer_name:
                    continue
                    
                for result in opt_results:
                    if result.param_history:
                        param_names.update(result.param_history.keys())
        
        if not param_names:
            return
            
        n_params = len(param_names)
        fig = plt.figure(figsize=(15, max(5, 3*min(n_params, 10))))  # Limit figure height for many parameters
        
        # Close previously opened figures to avoid warnings
        for fig_num in plt.get_fignums():
            if fig_num != fig.number:
                plt.close(fig_num)
        
        # If too many parameters, only show a subset
        if n_params > 10:
            print(f"Warning: Too many parameters ({n_params}). Showing only the first 10.")
            param_names = sorted(param_names)[:10]
            n_params = 10
        
        has_data = [False] * n_params  # Track if each subplot has data
        
        for i, param_name in enumerate(sorted(param_names), 1):
            ax = plt.subplot(n_params, 1, i)
            
            # Track if any lines were added to this subplot
            has_lines = False
            
            for func_name, func_results in self.results.items():
                if function_name and func_name != function_name:
                    continue
                    
                for opt_name, results in func_results.items():
                    if optimizer_name and opt_name != optimizer_name:
                        continue
                        
                    for result in results:
                        if result.param_history and param_name in result.param_history:
                            history = result.param_history[param_name]
                            if len(history) > 0:  # Only plot if we have data
                                ax.plot(history, label=f"{opt_name}-{func_name}", alpha=0.7)
                                has_lines = True
                                has_data[i-1] = True
            
            ax.set_title(f'{param_name} Adaptation')
            ax.set_xlabel('Iteration')
            ax.set_ylabel('Parameter Value')
            if has_lines:  # Only add legend if we have data
                ax.legend()
            ax.grid(True)
        
        # Only apply tight_layout if we have data to plot
        if any(has_data):
            try:
                plt.tight_layout()
            except Exception as e:
                print(f"Warning: Could not apply tight_layout: {e}")
        
        if save_path:
            plt.savefig(save_path)
            
        # Close the figure to avoid memory leaks if not returning it
        if 'return_fig' not in locals() or not return_fig:
            plt.close(fig)
            
        return fig
    
    def plot_diversity_analysis(self, optimizer_name=None, function_name=None, save_path: Optional[str] = None):
        """
        Plot diversity analysis for population-based optimizers.
        
        Args:
            optimizer_name: Optional name of the optimizer to plot
            function_name: Optional name of the function to plot
            save_path: Optional path to save the figure
        """
        if not self.results:
            raise ValueError("No results available. Run comparison first.")
            
        fig = plt.figure(figsize=(15, 10))
        
        # Close previously opened figures to avoid warnings
        for fig_num in plt.get_fignums():
            if fig_num != fig.number:
                plt.close(fig_num)
        
        # Plot 1: Diversity Over Time
        ax1 = plt.subplot(2, 2, 1)
        has_diversity_data = False
        
        for func_name, func_results in self.results.items():
            if function_name and func_name != function_name:
                continue
                
            for opt_name, results in func_results.items():
                if optimizer_name and opt_name != optimizer_name:
                    continue
                    
                for i, result in enumerate(results):
                    if result.diversity_history and len(result.diversity_history) > 0:
                        ax1.plot(result.diversity_history, 
                                label=f"{opt_name}-{func_name}" if i == 0 else "_nolegend_",
                                alpha=0.7)
                        has_diversity_data = True
        
        ax1.set_title('Population Diversity Over Time')
        ax1.set_xlabel('Iteration')
        ax1.set_ylabel('Diversity Measure')
        if has_diversity_data:  # Only add legend if we have data
            ax1.legend()
        ax1.grid(True)
        
        # Plot 2: Diversity vs Fitness Improvement
        ax2 = plt.subplot(2, 2, 2)
        has_improvement_data = False
        
        for func_name, func_results in self.results.items():
            if function_name and func_name != function_name:
                continue
                
            for opt_name, results in func_results.items():
                if optimizer_name and opt_name != optimizer_name:
                    continue
                    
                for result in results:
                    if (result.diversity_history and len(result.diversity_history) > 1 and 
                        result.convergence_curve and len(result.convergence_curve) > 1):
                        # Calculate fitness improvements
                        curve = np.array(result.convergence_curve)
                        improvements = np.abs(np.diff(curve))
                        
                        # Match lengths (take min length)
                        diversity = np.array(result.diversity_history)
                        min_len = min(len(diversity)-1, len(improvements))
                        
                        if min_len > 0:
                            ax2.scatter(diversity[:min_len], improvements[:min_len], 
                                      label=f"{opt_name}-{func_name}", alpha=0.5)
                            has_improvement_data = True
        
        ax2.set_title('Diversity vs Fitness Improvement')
        ax2.set_xlabel('Diversity')
        ax2.set_ylabel('Fitness Improvement')
        if has_improvement_data:  # Only add legend if we have data
            ax2.legend()
        ax2.grid(True)
        
        # Plot 3: Average Diversity by Optimizer
        ax3 = plt.subplot(2, 2, 3)
        diversity_by_opt = {}
        
        for func_name, func_results in self.results.items():
            if function_name and func_name != function_name:
                continue
                
            for opt_name, results in func_results.items():
                if optimizer_name and opt_name != optimizer_name:
                    continue
                    
                all_diversity = []
                for result in results:
                    if result.diversity_history and len(result.diversity_history) > 0:
                        all_diversity.extend(result.diversity_history)
                
                if all_diversity:
                    if opt_name not in diversity_by_opt:
                        diversity_by_opt[opt_name] = []
                    diversity_by_opt[opt_name].append(np.mean(all_diversity))
        
        if diversity_by_opt:
            ax3.bar(range(len(diversity_by_opt)), 
                  [np.mean(v) for v in diversity_by_opt.values()],
                  tick_label=list(diversity_by_opt.keys()))
            ax3.set_title('Average Diversity by Optimizer')
            ax3.set_xticks(range(len(diversity_by_opt)))
            ax3.set_xticklabels(list(diversity_by_opt.keys()), rotation=45)
            ax3.set_ylabel('Average Diversity')
            ax3.grid(True)
        
        # Plot 4: Diversity Trend Correlation
        ax4 = plt.subplot(2, 2, 4)
        has_trend_data = False
        
        for func_name, func_results in self.results.items():
            if function_name and func_name != function_name:
                continue
                
            for opt_name, results in func_results.items():
                if optimizer_name and opt_name != optimizer_name:
                    continue
                    
                for result in results:
                    if result.diversity_history and len(result.diversity_history) > 2:
                        diversity = np.array(result.diversity_history)
                        iterations = np.arange(len(diversity))
                        
                        try:
                            # Simple linear regression to show trend
                            z = np.polyfit(iterations, diversity, 1)
                            p = np.poly1d(z)
                            
                            ax4.plot(iterations, p(iterations), 
                                   label=f"{opt_name}-{func_name}", 
                                   linewidth=2, alpha=0.8)
                            has_trend_data = True
                        except np.linalg.LinAlgError:
                            # Skip if polyfit fails
                            continue
        
        ax4.set_title('Diversity Trend Over Iterations')
        ax4.set_xlabel('Iteration')
        ax4.set_ylabel('Diversity')
        if has_trend_data:  # Only add legend if we have data
            ax4.legend()
        ax4.grid(True)
        
        try:
            plt.tight_layout()
        except Exception as e:
            print(f"Warning: Could not apply tight_layout: {e}")
            
        if save_path:
            plt.savefig(save_path)
            
        # Close the figure to avoid memory leaks if not returning it
        if 'return_fig' not in locals() or not return_fig:
            plt.close(fig)
            
        return fig
    
    def plot_convergence_comparison(self):
        """Plot convergence comparison for all optimizers and functions"""
        if not self.results:
            raise ValueError("No results available. Run comparison first.")
            
        fig, axes = plt.subplots(len(self.results), 1, figsize=(10, 6 * len(self.results)))
        
        # Close previously opened figures to avoid warnings
        for fig_num in plt.get_fignums():
            if fig_num != fig.number:
                plt.close(fig_num)
        
        if len(self.results) == 1:  # Make sure axes is always a list
            axes = [axes]
            
        for i, (func_name, optimizer_results) in enumerate(self.results.items()):
            ax = axes[i]
            has_plot_data = False
            
            for opt_name, results in optimizer_results.items():
                if not results:
                    continue
                    
                try:
                    # Extract convergence data
                    fitness_histories = [result.fitness_history for result in results 
                                         if result.fitness_history and len(result.fitness_history) > 0]
                    
                    if not fitness_histories:
                        continue
                        
                    # Compute mean and std
                    min_length = min(len(history) for history in fitness_histories)
                    aligned_histories = [history[:min_length] for history in fitness_histories]
                    mean_fitness = np.mean(aligned_histories, axis=0)
                    std_fitness = np.std(aligned_histories, axis=0)
                    
                    # Plot mean with std as shaded region
                    iterations = range(min_length)
                    ax.plot(iterations, mean_fitness, label=opt_name)
                    ax.fill_between(iterations, mean_fitness - std_fitness, 
                                   mean_fitness + std_fitness, alpha=0.3)
                    has_plot_data = True
                except (ValueError, TypeError, ZeroDivisionError) as e:
                    print(f"Error plotting convergence for {opt_name} on {func_name}: {e}")
                    continue
            
            ax.set_title(f'Convergence Comparison for {func_name}')
            ax.set_xlabel('Iteration')
            ax.set_ylabel('Fitness Value')
            if has_plot_data:
                ax.legend()
            
        try:
            plt.tight_layout()
        except Exception as e:
            print(f"Warning: Could not apply tight_layout: {e}")
            
        # We return the figure object for potential further use
        return fig
    
    def plot_performance_heatmap(self, metric='final_fitness', save_path=None):
        """
        Plot a heatmap comparing different optimizers across functions.
        
        Args:
            metric (str): The metric to use for comparison ('final_fitness', 'time', etc.)
            save_path (str, optional): Path to save the figure
        """
        if not self.results:
            raise ValueError("No results available. Run comparison first.")
        
        functions = list(self.results.keys())
        optimizers = set()
        for func_results in self.results.values():
            optimizers.update(func_results.keys())
        optimizers = list(optimizers)
        
        # Create a matrix for the heatmap
        matrix = np.zeros((len(optimizers), len(functions)))
        
        # Fill the matrix with the chosen metric
        for i, optimizer in enumerate(optimizers):
            for j, function in enumerate(functions):
                if optimizer in self.results[function]:
                    results = self.results[function][optimizer]
                    if not results:
                        matrix[i, j] = np.nan
                        continue
                        
                    try:
                        if metric == 'final_fitness':
                            values = [result.best_fitness for result in results if hasattr(result, 'best_fitness')]
                        elif metric == 'time':
                            values = [result.total_time for result in results if hasattr(result, 'total_time')]
                        elif metric == 'iterations':
                            values = [result.iterations for result in results if hasattr(result, 'iterations')]
                        elif metric == 'evaluations':
                            values = [result.function_evaluations for result in results if hasattr(result, 'function_evaluations')]
                        else:
                            values = []
                            
                        if values:
                            matrix[i, j] = np.mean(values)
                        else:
                            matrix[i, j] = np.nan
                    except Exception as e:
                        print(f"Error calculating {metric} for {optimizer} on {function}: {e}")
                        matrix[i, j] = np.nan
                else:
                    matrix[i, j] = np.nan
        
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Close previously opened figures to avoid warnings
        for fig_num in plt.get_fignums():
            if fig_num != fig.number:
                plt.close(fig_num)
        
        # Create the heatmap
        has_valid_data = not np.isnan(matrix).all()
        if has_valid_data:
            im = ax.imshow(matrix, cmap='viridis')
            
            # Add colorbar
            cbar = ax.figure.colorbar(im, ax=ax)
            cbar.ax.set_ylabel(metric.replace('_', ' ').title(), rotation=-90, va="bottom")
            
            # Show ticks and label them
            ax.set_xticks(np.arange(len(functions)))
            ax.set_yticks(np.arange(len(optimizers)))
            ax.set_xticklabels(functions)
            ax.set_yticklabels(optimizers)
            
            # Rotate the tick labels and set alignment
            plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
            
            # Loop over data dimensions and create text annotations
            for i in range(len(optimizers)):
                for j in range(len(functions)):
                    if not np.isnan(matrix[i, j]):
                        text = ax.text(j, i, f"{matrix[i, j]:.2f}",
                                      ha="center", va="center", color="white" if matrix[i, j] > np.nanmax(matrix)/2 else "black")
            
            ax.set_title(f"Performance Comparison ({metric.replace('_', ' ').title()})")
        else:
            ax.text(0.5, 0.5, "No valid data available for heatmap", 
                   ha="center", va="center", fontsize=14, transform=ax.transAxes)
        
        try:
            plt.tight_layout()
        except Exception as e:
            print(f"Warning: Could not apply tight_layout: {e}")
        
        if save_path:
            plt.savefig(save_path)
            
        if save_path is None:  # Only close if not saving to file
            plt.close()
            
        return fig
    
    def clean_name(self, name: str) -> str:
        """Clean name for filenames"""
        return name.lower().replace(' ', '_').replace('(', '').replace(')', '')
    
    def create_html_report(
            self,
            statistical_results: pd.DataFrame,
            sota_results: pd.DataFrame
        ):
        """Create interactive HTML report with all results"""
        # Create basic template
        html_content = """
        <html>
        <head>
            <title>Optimization Results Report</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 20px; }
                .section { margin: 20px 0; }
                table { border-collapse: collapse; width: 100%; }
                th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
                th { background-color: #f2f2f2; }
                img { max-width: 100%; height: auto; }
            </style>
        </head>
        <body>
            <h1>Optimization Results Report</h1>
        """
        
        # Add sections
        sections = {
            'Convergence Analysis': 'convergence_*.png',
            'Performance Heatmap': 'performance_heatmap.png',
            'Parameter Adaptation': 'param_adaptation_*.png',
            'Diversity Analysis': 'diversity_*.png'
        }
        
        for title, pattern in sections.items():
            html_content += f"<div class='section'><h2>{title}</h2>"
            
            # Add images
            for img_path in Path('results/plots').glob(pattern):
                html_content += f"<img src='{img_path.relative_to('results')}' /><br/>"
            
            html_content += "</div>"
        
        # Add statistical results
        html_content += """
            <div class='section'>
                <h2>Statistical Analysis</h2>
                {statistical_table}
            </div>
        """.format(statistical_table=statistical_results.to_html())
        
        # Add SOTA comparison
        html_content += """
            <div class='section'>
                <h2>Comparison with State-of-the-Art</h2>
                {sota_table}
            </div>
        """.format(sota_table=sota_results.to_html())
        
        html_content += "</body></html>"
        
        # Save report
        with open('results/optimization_report.html', 'w') as f:
            f.write(html_content)
