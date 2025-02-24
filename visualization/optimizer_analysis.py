"""
optimizer_analysis.py
-------------------
Visualization and analysis tools for optimization algorithms.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Any, Tuple, Callable
from dataclasses import dataclass
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from concurrent.futures import ThreadPoolExecutor
import time

from optimizers.base_optimizer import BaseOptimizer
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

class OptimizerAnalyzer:
    def __init__(self, optimizers: List[BaseOptimizer], test_functions: List[TestFunction]):
        """
        Initialize analyzer.
        
        :param optimizers: List of optimizer instances
        :param test_functions: List of test functions
        """
        self.optimizers = optimizers
        self.test_functions = test_functions
        self.results = {}
        
    def run_comparison(self, n_runs: int = 30, record_convergence: bool = True) -> Dict[str, List[OptimizationResult]]:
        """
        Run comprehensive comparison of optimizers.
        
        :param n_runs: Number of runs per optimizer/function combination
        :param record_convergence: Whether to record convergence curves
        :return: Dictionary of results
        """
        results = {}
        
        for func in self.test_functions:
            results[func.name] = []
            
            for optimizer in self.optimizers:
                for run in range(n_runs):
                    start_time = time.time()
                    convergence = []
                    
                    # Wrap objective function to record convergence
                    if record_convergence:
                        def wrapped_func(x):
                            score = func(x)
                            convergence.append(score)
                            return score
                        objective = wrapped_func
                    else:
                        objective = func
                    
                    # Run optimization
                    best_sol, best_score = optimizer.optimize(objective)
                    execution_time = time.time() - start_time
                    
                    # Get hyperparameters
                    hyperparameters = {
                        key: getattr(optimizer, key)
                        for key in ['dim', 'population_size', 'num_generations']
                        if hasattr(optimizer, key)
                    }
                    
                    results[func.name].append(
                        OptimizationResult(
                            optimizer_name=optimizer.__class__.__name__,
                            best_solution=best_sol,
                            best_score=best_score,
                            convergence_curve=convergence if record_convergence else [],
                            execution_time=execution_time,
                            hyperparameters=hyperparameters
                        )
                    )
        
        self.results = results
        return results
    
    def plot_convergence_curves(self, func_name: str, save_path: str = None):
        """Plot convergence curves for all optimizers on given function"""
        plt.figure(figsize=(12, 8))
        
        for optimizer in set(r.optimizer_name for r in self.results[func_name]):
            # Get all runs for this optimizer
            curves = [
                r.convergence_curve 
                for r in self.results[func_name] 
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
            for optimizer in set(r.optimizer_name for r in results):
                values = [
                    getattr(r, metric)
                    for r in results
                    if r.optimizer_name == optimizer
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
            for optimizer in set(r.optimizer_name for r in results):
                # Filter results for this optimizer/function
                opt_results = [
                    r for r in results
                    if r.optimizer_name == optimizer
                ]
                
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
        
        for optimizer in self.optimizers:
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
            for result in results:
                row = {
                    'Function': func_name,
                    'Optimizer': result.optimizer_name,
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
