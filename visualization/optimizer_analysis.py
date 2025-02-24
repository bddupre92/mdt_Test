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
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from concurrent.futures import ThreadPoolExecutor
import time

from optimizers.base_optimizer import BaseOptimizer
from meta.meta_optimizer import MetaOptimizer
from benchmarking.test_functions import TestFunction

@dataclass
class OptimizationResult:
    """Store optimization run results"""
    optimizer_name: str
    best_solution: np.ndarray
    best_score: float
    convergence_curve: List[float]
    execution_time: float
    hyperparameters: Dict[str, Any]
    success_rate: Optional[float] = None
    diversity_history: Optional[List[float]] = None

class OptimizerAnalyzer:
    def __init__(self, 
                 optimizers: List[BaseOptimizer], 
                 test_functions: List[TestFunction],
                 meta_optimizer: Optional[MetaOptimizer] = None):
        """
        Initialize optimizer analyzer.
        
        Args:
            optimizers: List of optimizers to analyze
            test_functions: List of test functions
            meta_optimizer: Optional meta-optimizer
        """
        self.optimizers = {
            opt.__class__.__name__: opt
            for opt in optimizers
        }
        self.test_functions = test_functions
        self.meta_optimizer = meta_optimizer
        self.results = {}
        
    def run_comparison(self, n_runs: int = 30, record_convergence: bool = True) -> Dict[str, Dict[str, List[OptimizationResult]]]:
        """
        Run comprehensive comparison of optimizers.
        
        Args:
            n_runs: Number of runs per optimizer/function combination
            record_convergence: Whether to record convergence curves
        Returns:
            Dictionary mapping function names to dictionaries mapping optimizer names to lists of results
        """
        results = {}
        
        for func in self.test_functions:
            print(f"\nOptimizing {func.name}...")
            results[func.name] = {}
            
            for name, optimizer in self.optimizers.items():
                opt_results = []
                
                for _ in range(n_runs):
                    # Reset optimizer state
                    optimizer.reset()
                    
                    # Run optimization
                    solution = optimizer.optimize(func.func)
                    state = optimizer.get_state()
                    
                    # Record results
                    result = OptimizationResult(
                        optimizer_name=name,
                        best_solution=solution,
                        best_score=func.func(solution),
                        convergence_curve=optimizer.get_convergence_curve(),
                        execution_time=state.runtime,
                        hyperparameters=optimizer.get_parameter_history(),
                        success_rate=state.success_rate,
                        diversity_history=state.diversity_history
                    )
                    opt_results.append(result)
                
                results[func.name][name] = opt_results
        
        self.results = results
        return results
    
    def plot_convergence_curves(self, func_name: str, save_path: str = None):
        """Plot convergence curves for all optimizers on given function"""
        plt.figure(figsize=(12, 8))
        
        for optimizer in set(r.optimizer_name for r in self.results[func_name].values()):
            # Get all runs for this optimizer
            curves = [
                r.convergence_curve 
                for results in self.results[func_name].values() 
                for r in results 
                if r.optimizer_name == optimizer
            ]
            
            # Calculate mean and std
            max_len = max(len(curve) for curve in curves)
            aligned_curves = np.array([
                np.pad(curve, (0, max_len - len(curve)), 'edge')
                for curve in curves
            ])
            
            mean_curve = np.mean(aligned_curves, axis=0)
            std_curve = np.std(aligned_curves, axis=0)
            generations = range(1, len(mean_curve) + 1)
            
            # Plot with confidence interval
            plt.plot(generations, mean_curve, label=optimizer)
            plt.fill_between(
                generations,
                mean_curve - std_curve,
                mean_curve + std_curve,
                alpha=0.2
            )
        
        plt.xlabel('Generation')
        plt.ylabel('Objective Value')
        plt.title(f'Convergence Curves on {func_name}')
        plt.legend()
        plt.yscale('log')
        plt.grid(True)
        
        if save_path:
            plt.savefig(save_path)
        plt.show()
    
    def plot_performance_heatmap(self, metric: str = 'best_score', save_path: str = None):
        """Create heatmap comparing optimizers across functions"""
        # Prepare data
        data = []
        for func_name, results in self.results.items():
            for optimizer, opt_results in results.items():
                values = [
                    getattr(r, metric)
                    for r in opt_results
                ]
                data.append({
                    'Function': func_name,
                    'Optimizer': optimizer,
                    'Mean': np.mean(values),
                    'Std': np.std(values)
                })
        
        df = pd.DataFrame(data)
        pivot = df.pivot(index='Function', columns='Optimizer', values='Mean')
        
        # Create heatmap
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            pivot,
            annot=True,
            fmt='.2e',
            cmap='RdYlBu_r',
            center=0
        )
        plt.title(f'Performance Comparison ({metric})')
        
        if save_path:
            plt.savefig(save_path)
        plt.show()
    
    def create_performance_table(self) -> pd.DataFrame:
        """Create detailed performance comparison table"""
        data = []
        metrics = ['best_score', 'execution_time']
        
        for func_name, results in self.results.items():
            for optimizer, opt_results in results.items():
                # Filter results for this optimizer/function
                row = {
                    'Function': func_name,
                    'Optimizer': optimizer
                }
                
                # Calculate statistics for each metric
                for metric in metrics:
                    values = [getattr(r, metric) for r in opt_results]
                    row.update({
                        f'{metric}_mean': np.mean(values),
                        f'{metric}_std': np.std(values),
                        f'{metric}_min': np.min(values),
                        f'{metric}_max': np.max(values)
                    })
                
                data.append(row)
        
        return pd.DataFrame(data)
    
    def plot_parameter_sensitivity(self, param_name: str, param_range: List[Any],
                                 func: TestFunction, save_path: str = None):
        """Analyze sensitivity to a specific parameter"""
        results = []
        
        for optimizer in self.optimizers.values():
            scores = []
            for value in param_range:
                # Set parameter
                setattr(optimizer, param_name, value)
                
                # Run optimization
                best_sol, best_score = optimizer.optimize(func)
                scores.append(best_score)
            
            results.append({
                'optimizer': optimizer.__class__.__name__,
                'param_values': param_range,
                'scores': scores
            })
        
        # Plot results
        plt.figure(figsize=(10, 6))
        for result in results:
            plt.plot(
                result['param_values'],
                result['scores'],
                'o-',
                label=result['optimizer']
            )
        
        plt.xlabel(param_name)
        plt.ylabel('Best Score')
        plt.title(f'Parameter Sensitivity Analysis: {param_name}')
        plt.legend()
        plt.grid(True)
        
        if save_path:
            plt.savefig(save_path)
        plt.show()
    
    def plot_3d_fitness_landscape(self, func: TestFunction, 
                                n_points: int = 50, save_path: str = None):
        """Plot 3D fitness landscape for 2D functions"""
        if func.dim != 2:
            raise ValueError("Can only plot landscape for 2D functions")
        
        # Create grid of points
        x = np.linspace(func.bounds[0][0], func.bounds[0][1], n_points)
        y = np.linspace(func.bounds[1][0], func.bounds[1][1], n_points)
        X, Y = np.meshgrid(x, y)
        
        # Calculate function values
        Z = np.zeros_like(X)
        for i in range(n_points):
            for j in range(n_points):
                Z[i,j] = func(np.array([X[i,j], Y[i,j]]))
        
        # Create 3D plot
        fig = go.Figure(data=[go.Surface(x=X, y=Y, z=Z)])
        fig.update_layout(
            title=f'Fitness Landscape: {func.name}',
            scene = dict(
                xaxis_title='x',
                yaxis_title='y',
                zaxis_title='f(x,y)'
            )
        )
        
        if save_path:
            fig.write_html(save_path)
        fig.show()
    
    def plot_hyperparameter_correlation(self, save_path: str = None):
        """Plot correlation between hyperparameters and performance"""
        data = []
        for func_name, results in self.results.items():
            for optimizer, opt_results in results.items():
                for result in opt_results:
                    row = {
                        'Function': func_name,
                        'Optimizer': optimizer,
                        'Score': result.best_score,
                        'Time': result.execution_time,
                        **result.hyperparameters
                    }
                    data.append(row)
        
        df = pd.DataFrame(data)
        
        # Select only numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        # Create correlation plot
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            df[numeric_cols].corr(),
            annot=True,
            cmap='RdYlBu_r',
            center=0,
            fmt='.2f'
        )
        plt.title('Hyperparameter Correlation Matrix')
        
        if save_path:
            plt.savefig(save_path)
        plt.show()
        
    def plot_convergence_comparison(self, function_name: str, log_scale: bool = True):
        """Plot convergence curves for all optimizers on given function."""
        if function_name not in self.results:
            print(f"No results found for function {function_name}")
            return
            
        plt.figure(figsize=(12, 8))
        
        for optimizer_name, opt_results in self.results[function_name].items():
            # Get all convergence curves for this optimizer
            curves = [r.convergence_curve for r in opt_results if r.convergence_curve]
            if not curves:
                print(f"No convergence data for {optimizer_name}")
                continue
            
            # Calculate mean curve
            max_len = max(len(curve) for curve in curves)
            aligned_curves = np.array([
                np.pad(curve, (0, max_len - len(curve)), 'edge')
                for curve in curves
            ])
            mean_curve = np.mean(aligned_curves, axis=0)
            
            # Plot mean curve
            plt.plot(mean_curve, label=optimizer_name)
        
        plt.title(f'Convergence on {function_name}')
        plt.xlabel('Function Evaluations')
        plt.ylabel('Best Score')
        if log_scale:
            plt.yscale('log')
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        
        # Save plot
        plt.savefig(f'results/plots/{function_name.lower()}_convergence.png')
        plt.close()
    
    def plot_performance_heatmap(self, metric: str = 'best_score'):
        """
        Create heatmap of optimizer performance across test functions.
        
        Args:
            metric: Performance metric to visualize ('best_score', 'time', 'success_rate')
        """
        data = []
        optimizers = []
        functions = []
        
        for func_name, results in self.results.items():
            functions.append(func_name)
            for optimizer, opt_results in results.items():
                if optimizer not in optimizers:
                    optimizers.append(optimizer)
                    
                if metric == 'best_score':
                    value = np.mean([r.best_score for r in opt_results])
                elif metric == 'time':
                    value = np.mean([r.execution_time for r in opt_results])
                else:  # success_rate
                    value = np.mean([r.success_rate for r in opt_results if r.success_rate is not None])
                    
                data.append({
                    'function': func_name,
                    'optimizer': optimizer,
                    'value': value
                })
                
        df = pd.DataFrame(data)
        pivot = df.pivot(index='optimizer', columns='function', values='value')
        
        plt.figure(figsize=(12, 8))
        sns.heatmap(pivot, annot=True, fmt='.2e', cmap='viridis')
        plt.title(f'Optimizer Performance ({metric})')
        plt.show()
        
    def plot_parameter_adaptation(self, optimizer_name: str, function_name: str):
        """Plot parameter adaptation for a specific optimizer."""
        if function_name not in self.results:
            print(f"No results found for function {function_name}")
            return
            
        if optimizer_name not in self.results[function_name]:
            print(f"No results found for optimizer {optimizer_name}")
            return
            
        opt_results = self.results[function_name][optimizer_name]
        if not opt_results:
            print(f"No results available for {optimizer_name} on {function_name}")
            return
            
        # Get parameter history from first run
        param_history = opt_results[0].hyperparameters
        if not param_history:
            print(f"No parameter history available for {optimizer_name}")
            return
            
        plt.figure(figsize=(12, 6))
        
        # Plot each parameter
        for param_name, values in param_history.items():
            if param_name != 'diversity':  # Skip diversity, it's plotted separately
                plt.plot(values, label=param_name)
        
        plt.title(f'Parameter Adaptation - {optimizer_name} on {function_name}')
        plt.xlabel('Iteration')
        plt.ylabel('Parameter Value')
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        
        # Save plot
        plt.savefig(f'results/plots/params_{optimizer_name.lower()}_{function_name.lower()}.png')
        plt.close()
    
    def plot_diversity_analysis(self, optimizer_name: str, function_name: str):
        """Plot diversity analysis for a specific optimizer."""
        if function_name not in self.results:
            print(f"No results found for function {function_name}")
            return
            
        if optimizer_name not in self.results[function_name]:
            print(f"No results found for optimizer {optimizer_name}")
            return
            
        opt_results = self.results[function_name][optimizer_name]
        if not opt_results:
            print(f"No results available for {optimizer_name} on {function_name}")
            return
            
        # Get diversity history from first run
        param_history = opt_results[0].hyperparameters
        if not param_history or 'diversity' not in param_history:
            print(f"No diversity history available for {optimizer_name}")
            return
            
        plt.figure(figsize=(10, 6))
        
        # Plot diversity for each run
        for result in opt_results:
            diversity = result.hyperparameters['diversity']
            plt.plot(diversity, alpha=0.3, color='blue', label='Individual Run' if result == opt_results[0] else None)
            
        # Plot mean diversity
        mean_diversity = np.mean([r.hyperparameters['diversity'] for r in opt_results], axis=0)
        plt.plot(mean_diversity, color='red', linewidth=2, label='Mean Diversity')
        
        plt.title(f'Population Diversity - {optimizer_name} on {function_name}')
        plt.xlabel('Iteration')
        plt.ylabel('Diversity')
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        
        # Save plot
        plt.savefig(f'results/plots/diversity_{optimizer_name.lower()}_{function_name.lower()}.png')
        plt.close()
        
    def plot_meta_optimizer_analysis(self):
        """Analyze meta-optimizer performance and decisions."""
        if not self.meta_optimizer:
            print("No meta-optimizer available")
            return
            
        # Get meta-optimizer state
        state = self.meta_optimizer.get_state()
        history = state['performance_history']
        
        if history.empty:
            print("No meta-optimizer history available")
            return
            
        # Create figure with subplots
        fig = plt.figure(figsize=(15, 10))
        gs = plt.GridSpec(2, 2)
        
        # 1. Optimizer selection frequency
        ax1 = fig.add_subplot(gs[0, 0])
        selection_counts = history['optimizer'].value_counts()
        selection_counts.plot(kind='bar', ax=ax1)
        ax1.set_title('Optimizer Selection Frequency')
        ax1.set_xlabel('Optimizer')
        ax1.set_ylabel('Times Selected')
        plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        # 2. Performance over time
        ax2 = fig.add_subplot(gs[0, 1])
        history.plot(x='timestamp', y='score', style='o-', ax=ax2)
        ax2.set_title('Performance Over Time')
        ax2.set_xlabel('Time')
        ax2.set_ylabel('Score')
        
        # 3. Runtime distribution
        ax3 = fig.add_subplot(gs[1, 0])
        history['runtime'].hist(bins=20, ax=ax3)
        ax3.set_title('Runtime Distribution')
        ax3.set_xlabel('Runtime (seconds)')
        ax3.set_ylabel('Frequency')
        
        # 4. Performance by problem dimension
        ax4 = fig.add_subplot(gs[1, 1])
        sns.boxplot(data=history, x='problem_dim', y='score', ax=ax4)
        ax4.set_title('Performance by Problem Dimension')
        ax4.set_xlabel('Problem Dimension')
        ax4.set_ylabel('Score')
        
        plt.tight_layout()
        plt.savefig('results/plots/meta_optimizer_analysis.png')
        plt.close()
        
    def create_summary_report(self, output_file: str = 'optimization_report.html'):
        """
        Create an interactive HTML report with all visualizations.
        
        Args:
            output_file: Path to save the HTML report
        """
        # Create subplot figure
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=(
                'Convergence Comparison',
                'Performance Heatmap',
                'Parameter Adaptation',
                'Population Diversity',
                'Meta-Optimizer Analysis',
                'Selection Frequency'
            )
        )
        
        # Add traces for each visualization
        # (Implementation details for adding Plotly traces)
        
        # Update layout
        fig.update_layout(height=1200, width=1000, title_text="Optimization Analysis Report")
        
        # Save to HTML
        fig.write_html(output_file)
        print(f"Report saved to {output_file}")
