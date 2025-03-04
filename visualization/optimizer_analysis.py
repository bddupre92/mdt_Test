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
        
        # Plot 1: Ruggedness Analysis
        plt.subplot(2, 2, 1)
        for func_name in self.results:
            for opt_name, results in self.results[func_name].items():
                metrics = [r.landscape_metrics for r in results if r.landscape_metrics]
                if metrics:
                    ruggedness = [m.get('ruggedness', 0) for m in metrics]
                    plt.boxplot(ruggedness, positions=[len(plt.gca().get_xticks())],
                              labels=[f"{opt_name}\n{func_name}"])
        plt.title('Landscape Ruggedness')
        plt.xticks(rotation=45)
        
        # Plot 2: Local Optima Analysis
        plt.subplot(2, 2, 2)
        for func_name in self.results:
            for opt_name, results in self.results[func_name].items():
                metrics = [r.landscape_metrics for r in results if r.landscape_metrics]
                if metrics:
                    optima = [m.get('local_optima_count', 0) for m in metrics]
                    plt.boxplot(optima, positions=[len(plt.gca().get_xticks())],
                              labels=[f"{opt_name}\n{func_name}"])
        plt.title('Local Optima Count')
        plt.xticks(rotation=45)
        
        # Plot 3: Fitness-Distance Correlation
        plt.subplot(2, 2, 3)
        for func_name in self.results:
            for opt_name, results in self.results[func_name].items():
                metrics = [r.landscape_metrics for r in results if r.landscape_metrics]
                if metrics:
                    fdc = [m.get('fitness_distance_correlation', 0) for m in metrics]
                    plt.boxplot(fdc, positions=[len(plt.gca().get_xticks())],
                              labels=[f"{opt_name}\n{func_name}"])
        plt.title('Fitness-Distance Correlation')
        plt.xticks(rotation=45)
        
        # Plot 4: Landscape Smoothness
        plt.subplot(2, 2, 4)
        for func_name in self.results:
            for opt_name, results in self.results[func_name].items():
                metrics = [r.landscape_metrics for r in results if r.landscape_metrics]
                if metrics:
                    smoothness = [m.get('smoothness', 0) for m in metrics]
                    plt.boxplot(smoothness, positions=[len(plt.gca().get_xticks())],
                              labels=[f"{opt_name}\n{func_name}"])
        plt.title('Landscape Smoothness')
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path)
        plt.close()
    
    def plot_gradient_analysis(self, save_path: Optional[str] = None):
        """Plot gradient-based analysis"""
        if not self.results:
            raise ValueError("No results available. Run comparison first.")
            
        fig = plt.figure(figsize=(15, 10))
        
        # Plot 1: Gradient Magnitude Evolution
        plt.subplot(2, 2, 1)
        for func_name in self.results:
            for opt_name, results in self.results[func_name].items():
                for result in results:
                    if result.gradient_history:
                        magnitudes = [np.linalg.norm(g) for g in result.gradient_history]
                        plt.plot(magnitudes, label=f"{opt_name}-{func_name}")
        plt.title('Gradient Magnitude Evolution')
        plt.xlabel('Iteration')
        plt.ylabel('Gradient Magnitude')
        plt.legend()
        
        # Plot 2: Gradient Direction Change
        plt.subplot(2, 2, 2)
        for func_name in self.results:
            for opt_name, results in self.results[func_name].items():
                for result in results:
                    if result.gradient_history and len(result.gradient_history) > 1:
                        angles = []
                        for i in range(1, len(result.gradient_history)):
                            g1 = result.gradient_history[i-1]
                            g2 = result.gradient_history[i]
                            cos_angle = np.dot(g1, g2) / (np.linalg.norm(g1) * np.linalg.norm(g2))
                            angle = np.arccos(np.clip(cos_angle, -1.0, 1.0))
                            angles.append(angle)
                        plt.plot(angles, label=f"{opt_name}-{func_name}")
        plt.title('Gradient Direction Change')
        plt.xlabel('Iteration')
        plt.ylabel('Angle (radians)')
        plt.legend()
        
        # Plot 3: Gradient Component Analysis
        plt.subplot(2, 2, 3)
        for func_name in self.results:
            for opt_name, results in self.results[func_name].items():
                for result in results:
                    if result.gradient_history:
                        components = np.array(result.gradient_history)
                        plt.boxplot(components, labels=[f"Dim{i+1}" for i in range(components.shape[1])])
        plt.title('Gradient Components Distribution')
        plt.xlabel('Dimension')
        plt.ylabel('Gradient Value')
        
        # Plot 4: Gradient Stability
        plt.subplot(2, 2, 4)
        for func_name in self.results:
            for opt_name, results in self.results[func_name].items():
                for result in results:
                    if result.gradient_history:
                        stability = np.std(result.gradient_history, axis=0)
                        plt.bar(range(len(stability)), stability, 
                               label=f"{opt_name}-{func_name}", alpha=0.5)
        plt.title('Gradient Stability (Standard Deviation)')
        plt.xlabel('Dimension')
        plt.ylabel('Standard Deviation')
        plt.legend()
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path)
        plt.close()
    
    def plot_parameter_adaptation(self, save_path: Optional[str] = None):
        """Plot parameter adaptation history"""
        if not self.results:
            raise ValueError("No results available. Run comparison first.")
            
        # Collect all unique parameter names
        param_names = set()
        for func_results in self.results.values():
            for opt_results in func_results.values():
                for result in opt_results:
                    if result.param_history:
                        param_names.update(result.param_history.keys())
        
        if not param_names:
            return
            
        n_params = len(param_names)
        fig = plt.figure(figsize=(15, 3*n_params))
        
        for i, param_name in enumerate(sorted(param_names), 1):
            plt.subplot(n_params, 1, i)
            
            for func_name in self.results:
                for opt_name, results in self.results[func_name].items():
                    for result in results:
                        if result.param_history and param_name in result.param_history:
                            history = result.param_history[param_name]
                            plt.plot(history, label=f"{opt_name}-{func_name}", alpha=0.7)
            
            plt.title(f'{param_name} Adaptation')
            plt.xlabel('Iteration')
            plt.ylabel('Parameter Value')
            plt.legend()
            plt.grid(True)
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path)
        plt.close()
    
    def clean_name(self, name: str) -> str:
        """Clean name for filenames"""
        return name.lower().replace(' ', '_').replace('(', '').replace(')', '')
    
    def plot_convergence_comparison(self):
        """Plot convergence comparison for all functions and optimizers"""
        if not self.results:
            raise ValueError("No results available. Run comparison first.")
            
        for func_name in self.results:
            plt.figure(figsize=(12, 8))
            
            # Find max length of convergence curves
            max_len = 0
            for opt_name, results in self.results[func_name].items():
                for r in results:
                    max_len = max(max_len, len(r.convergence_curve))
            
            for opt_name, results in self.results[func_name].items():
                # Pad curves to same length
                padded_curves = []
                for r in results:
                    curve = np.array(r.convergence_curve)
                    if len(curve) == 0:
                        # Handle empty curves by creating an array of zeros
                        curve = np.full(max_len, np.inf)
                    elif len(curve) < max_len:
                        # Pad with the last value
                        padding = np.full(max_len - len(curve), curve[-1])
                        curve = np.concatenate([curve, padding])
                    padded_curves.append(curve)
                
                # Average convergence curves across runs
                curves = np.array(padded_curves)
                mean_curve = np.mean(curves, axis=0)
                std_curve = np.std(curves, axis=0)
                x = np.arange(len(mean_curve))
                
                plt.plot(x, mean_curve, label=opt_name)
                plt.fill_between(x, mean_curve - std_curve, mean_curve + std_curve, alpha=0.2)
            
            plt.title(f'Convergence Comparison - {func_name}')
            plt.xlabel('Function Evaluations')
            plt.ylabel('Objective Value')
            plt.yscale('log')
            plt.legend()
            plt.grid(True)
            
            # Save plot
            fig = plt.gcf()
            filename = f'convergence_{self.clean_name(func_name)}.png'
            save_plot(fig, filename, plot_type='benchmarks')
            plt.close()
    
    def plot_performance_heatmap(self):
        """Plot performance heatmap comparing optimizers across functions"""
        if not self.results:
            raise ValueError("No results available. Run comparison first.")
            
        # Prepare data for heatmap
        data = []
        for func_name in self.results:
            for opt_name in self.results[func_name]:
                results = self.results[func_name][opt_name]
                mean_score = np.mean([r.best_score for r in results])
                data.append({
                    'Function': func_name,
                    'Optimizer': opt_name,
                    'Score': mean_score
                })
        
        df = pd.DataFrame(data)
        pivot = df.pivot(index='Function', columns='Optimizer', values='Score')
        
        # Create heatmap
        plt.figure(figsize=(12, 8))
        sns.heatmap(pivot, annot=True, fmt='.2e', cmap='viridis')
        plt.title('Performance Comparison')
        plt.tight_layout()
        
        # Save plot
        fig = plt.gcf()
        filename = 'performance_heatmap.png'
        save_plot(fig, filename, plot_type='benchmarks')
        plt.close()
    
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
