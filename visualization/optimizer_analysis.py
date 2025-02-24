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

class OptimizerAnalyzer:
    def __init__(self, optimizers: Dict[str, Any]):
        """
        Initialize optimizer analyzer.
        
        Args:
            optimizers: Dictionary mapping optimizer names to optimizer instances
        """
        self.optimizers = optimizers
        self.results = {}
        
    def run_comparison(
            self,
            test_functions: Dict[str, Any],
            n_runs: int = 30,
            record_convergence: bool = True
        ) -> Dict[str, Dict[str, List[OptimizationResult]]]:
        """
        Run optimization comparison.
        
        Args:
            test_functions: Dictionary mapping function names to test functions
            n_runs: Number of independent runs per optimizer
            record_convergence: Whether to record convergence history
            
        Returns:
            Dictionary mapping function names to dictionaries mapping optimizer names to lists of results
        """
        print(f"Running comparison with {n_runs} independent runs per configuration")
        
        for func_name, func in test_functions.items():
            print(f"\nOptimizing {func_name}")
            self.results[func_name] = {}
            
            for opt_name, optimizer in self.optimizers.items():
                print(f"  Using {opt_name}")
                results = []
                
                for run in range(n_runs):
                    start_time = time.time()
                    optimizer.reset()  # Reset optimizer state
                    
                    # Run optimization
                    best_solution = optimizer.optimize(
                        func,
                        max_evals=10000,
                        record_history=record_convergence
                    )
                    
                    # Store results
                    results.append(OptimizationResult(
                        optimizer_name=opt_name,
                        function_name=func_name,
                        best_solution=best_solution,
                        best_score=optimizer.best_score,
                        convergence_curve=optimizer.convergence_curve if record_convergence else [],
                        execution_time=time.time() - start_time,
                        hyperparameters=optimizer.get_parameters(),
                        success_rate=optimizer.success_rate if hasattr(optimizer, 'success_rate') else None,
                        diversity_history=optimizer.diversity_history if hasattr(optimizer, 'diversity_history') else None,
                        param_history=optimizer.param_history if hasattr(optimizer, 'param_history') else None
                    ))
                
                self.results[func_name][opt_name] = results
                
        return self.results
    
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
                    if len(curve) < max_len:
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
            plt.savefig(f'results/plots/convergence_{self.clean_name(func_name)}.png')
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
        plt.savefig('results/plots/performance_heatmap.png')
        plt.close()
    
    def plot_parameter_adaptation(self, optimizer_name: str, function_name: str):
        """Plot parameter adaptation history for a specific optimizer and function"""
        if not self.results or function_name not in self.results:
            raise ValueError("No results available for the specified function.")
            
        if optimizer_name not in self.results[function_name]:
            raise ValueError("No results available for the specified optimizer.")
            
        results = self.results[function_name][optimizer_name]
        param_histories = [r.param_history for r in results if r.param_history]
        
        if not param_histories:
            return  # Skip if no parameter histories available
            
        plt.figure(figsize=(12, 8))
        n_params = len(param_histories[0])
        n_rows = (n_params + 1) // 2
        
        for i, param_name in enumerate(param_histories[0].keys(), 1):
            plt.subplot(n_rows, 2, i)
            
            # Plot parameter trajectories
            for history in param_histories:
                if param_name in history:
                    plt.plot(history[param_name], alpha=0.3)
            
            # Plot mean trajectory
            mean_trajectory = np.mean([h[param_name] for h in param_histories], axis=0)
            plt.plot(mean_trajectory, 'k-', linewidth=2, label='Mean')
            
            plt.title(f'{param_name} Adaptation')
            plt.xlabel('Iterations')
            plt.ylabel('Value')
            plt.grid(True)
        
        plt.suptitle(f'Parameter Adaptation - {optimizer_name} on {function_name}')
        plt.tight_layout()
        
        # Save plot
        clean_opt_name = self.clean_name(optimizer_name)
        clean_func_name = self.clean_name(function_name)
        plt.savefig(f'results/plots/param_adaptation_{clean_opt_name}_{clean_func_name}.png')
        plt.close()
    
    def plot_diversity_analysis(self, optimizer_name: str, function_name: str):
        """Plot diversity analysis for a specific optimizer and function"""
        if not self.results or function_name not in self.results:
            raise ValueError("No results available for the specified function.")
            
        if optimizer_name not in self.results[function_name]:
            raise ValueError("No results available for the specified optimizer.")
            
        results = self.results[function_name][optimizer_name]
        diversity_histories = [r.diversity_history for r in results if r.diversity_history]
        
        if not diversity_histories:
            return  # Skip if no diversity histories available
            
        plt.figure(figsize=(12, 6))
        
        # Find max length of diversity histories
        max_len = max(len(h) for h in diversity_histories)
        
        # Pad shorter histories with their last value
        padded_histories = []
        for history in diversity_histories:
            if len(history) < max_len:
                padding = [history[-1]] * (max_len - len(history))
                history = list(history) + padding
            padded_histories.append(history)
        
        # Plot individual trajectories
        for history in padded_histories:
            plt.plot(history, alpha=0.3)
        
        # Plot mean trajectory
        mean_diversity = np.mean(padded_histories, axis=0)
        plt.plot(mean_diversity, 'k-', linewidth=2, label='Mean')
        
        plt.title(f'Population Diversity - {optimizer_name} on {function_name}')
        plt.xlabel('Iterations')
        plt.ylabel('Diversity')
        plt.grid(True)
        plt.legend()
        
        # Save plot
        clean_opt_name = self.clean_name(optimizer_name)
        clean_func_name = self.clean_name(function_name)
        plt.savefig(f'results/plots/diversity_{clean_opt_name}_{clean_func_name}.png')
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
