"""
optimizer_explainer.py
---------------------
Explainability tools for optimization algorithms
"""

from typing import Dict, Any, List, Optional, Union, Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
from .base_explainer import BaseExplainer
from optimizers.base_optimizer import BaseOptimizer, OptimizerState
import os
import logging
import argparse
import time
import warnings

from utils.json_utils import save_json
from utils.plotting import save_plot, setup_plot_style


class OptimizerExplainer(BaseExplainer):
    """Explainer for optimization algorithms"""
    
    def __init__(self, optimizer=None, feature_names: Optional[List[str]] = None, **kwargs):
        """
        Initialize optimizer explainer
        
        Args:
            optimizer: Optimizer instance to explain
            feature_names: List of feature names (dimensions)
            **kwargs: Additional parameters
        """
        super().__init__("optimizer", optimizer, feature_names)
        self.optimizer = optimizer
        self.supported_plot_types = [
            'convergence', 
            'parameter_adaptation', 
            'diversity', 
            'landscape_analysis',
            'decision_process',
            'exploration_exploitation',
            'gradient_estimation',
            'performance_profile'
        ]
        
    def explain(self, X: Optional[Union[np.ndarray, pd.DataFrame]] = None, 
               y: Optional[np.ndarray] = None, 
               **kwargs) -> Dict[str, Any]:
        """
        Generate explanation for the optimizer
        
        Args:
            X: Optional input data (not used for optimizer explanation)
            y: Optional target values (not used for optimizer explanation)
            **kwargs: Additional parameters
            
        Returns:
            Dictionary containing explanation data
        """
        if not isinstance(self.model, BaseOptimizer):
            raise ValueError("Model must be an instance of BaseOptimizer")
        
        # Get optimizer state
        optimizer_state = self.model.get_state()
        
        # Extract key metrics
        explanation = {
            'optimizer_type': self.model.__class__.__name__,
            'dimensions': self.model.dim,
            'population_size': self.model.population_size,
            'evaluations': optimizer_state.evaluations,
            'iterations': optimizer_state.iteration,
            'best_score': float(optimizer_state.best_score),
            'execution_time': optimizer_state.end_time - optimizer_state.start_time if optimizer_state.end_time > 0 else 0,
            'convergence_curve': [float(x) for x in optimizer_state.convergence_curve],
            'diversity_history': [float(x) for x in optimizer_state.diversity_history] if optimizer_state.diversity_history else [],
            'parameter_history': {k: [float(x) for x in v] for k, v in optimizer_state.parameter_history.items()},
            'success_history': optimizer_state.success_history,
            'gradient_estimates': [float(x) for x in optimizer_state.gradient_estimates] if optimizer_state.gradient_estimates else [],
            'local_optima_count': optimizer_state.local_optima_count,
            'landscape_ruggedness': float(optimizer_state.landscape_ruggedness) if optimizer_state.landscape_ruggedness is not None else None,
            'selection_pressure': [float(x) for x in optimizer_state.selection_pressure] if optimizer_state.selection_pressure else [],
            'stagnation_count': optimizer_state.stagnation_count,
            'time_per_iteration': [float(x) for x in optimizer_state.time_per_iteration] if optimizer_state.time_per_iteration else []
        }
        
        # Calculate feature importance (parameter sensitivity)
        feature_importance = self._calculate_parameter_sensitivity()
        explanation['feature_importance'] = feature_importance
        
        # Store explanation
        self.last_explanation = explanation
        
        return explanation
    
    def _calculate_parameter_sensitivity(self) -> Dict[str, float]:
        """
        Calculate parameter sensitivity as a measure of feature importance
        
        Returns:
            Dictionary mapping parameter names to sensitivity values
        """
        if not isinstance(self.model, BaseOptimizer):
            return {}
        
        # Get parameter history
        parameter_history = self.model.parameter_history
        
        # If no parameter history, return empty dict
        if not parameter_history:
            return {}
        
        # Calculate sensitivity as coefficient of variation (std/mean)
        sensitivity = {}
        for param_name, param_values in parameter_history.items():
            if not param_values or len(param_values) < 2:
                continue
                
            values = np.array(param_values)
            if np.mean(values) != 0:
                sensitivity[param_name] = float(np.std(values) / np.mean(values))
            else:
                sensitivity[param_name] = float(np.std(values))
        
        return sensitivity
    
    def plot(self, plot_type: str = 'convergence', **kwargs) -> plt.Figure:
        """
        Create visualization of the optimizer behavior
        
        Args:
            plot_type: Type of plot to generate
            **kwargs: Additional parameters for specific plot types
            
        Returns:
            Matplotlib figure object
        """
        if not self.last_explanation:
            raise ValueError("No explanation available. Run explain() first.")
        
        if plot_type == 'convergence':
            return self._plot_convergence(**kwargs)
        elif plot_type == 'parameter_adaptation':
            return self._plot_parameter_adaptation(**kwargs)
        elif plot_type == 'diversity':
            return self._plot_diversity(**kwargs)
        elif plot_type == 'landscape_analysis':
            return self._plot_landscape_analysis(**kwargs)
        elif plot_type == 'decision_process':
            return self._plot_decision_process(**kwargs)
        elif plot_type == 'exploration_exploitation':
            return self._plot_exploration_exploitation(**kwargs)
        elif plot_type == 'gradient_estimation':
            return self._plot_gradient_estimation(**kwargs)
        elif plot_type == 'performance_profile':
            return self._plot_performance_profile(**kwargs)
        else:
            raise ValueError(f"Unsupported plot type: {plot_type}")
    
    def _plot_convergence(self, **kwargs) -> plt.Figure:
        """Plot convergence curve"""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        convergence_curve = self.last_explanation['convergence_curve']
        iterations = range(1, len(convergence_curve) + 1)
        
        ax.plot(iterations, convergence_curve, marker='o', linestyle='-', alpha=0.7)
        ax.set_title(f"Convergence Curve - {self.last_explanation['optimizer_type']}")
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Objective Value")
        ax.grid(True, linestyle='--', alpha=0.7)
        
        # Add annotations for key events
        if len(convergence_curve) > 1:
            # Annotate best value
            best_idx = np.argmin(convergence_curve)
            ax.scatter(best_idx + 1, convergence_curve[best_idx], color='red', s=100, zorder=5)
            ax.annotate(f"Best: {convergence_curve[best_idx]:.4f}", 
                       (best_idx + 1, convergence_curve[best_idx]),
                       xytext=(10, -20), textcoords='offset points',
                       arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=.2'))
        
        plt.tight_layout()
        return fig
    
    def _plot_parameter_adaptation(self, **kwargs) -> plt.Figure:
        """Plot parameter adaptation over iterations"""
        parameter_history = self.last_explanation['parameter_history']
        
        if not parameter_history:
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.text(0.5, 0.5, "No parameter adaptation data available", 
                   ha='center', va='center', fontsize=12)
            return fig
        
        # Determine number of parameters to plot
        n_params = len(parameter_history)
        if n_params == 0:
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.text(0.5, 0.5, "No parameter adaptation data available", 
                   ha='center', va='center', fontsize=12)
            return fig
        
        # Create subplots
        fig, axes = plt.subplots(n_params, 1, figsize=(10, 3 * n_params), sharex=True)
        if n_params == 1:
            axes = [axes]
        
        # Plot each parameter
        for i, (param_name, param_values) in enumerate(parameter_history.items()):
            if param_name in ['success_rate', 'population_diversity']:
                continue  # Skip these as they're not algorithm parameters
                
            iterations = range(1, len(param_values) + 1)
            axes[i].plot(iterations, param_values, marker='o', linestyle='-', alpha=0.7)
            axes[i].set_title(f"Parameter Adaptation: {param_name}")
            axes[i].set_ylabel("Value")
            axes[i].grid(True, linestyle='--', alpha=0.7)
        
        axes[-1].set_xlabel("Iteration")
        plt.tight_layout()
        return fig
    
    def _plot_diversity(self, **kwargs) -> plt.Figure:
        """Plot population diversity over iterations"""
        diversity_history = self.last_explanation['diversity_history']
        
        if not diversity_history:
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.text(0.5, 0.5, "No diversity data available", 
                   ha='center', va='center', fontsize=12)
            return fig
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        iterations = range(1, len(diversity_history) + 1)
        ax.plot(iterations, diversity_history, marker='o', linestyle='-', alpha=0.7)
        ax.set_title(f"Population Diversity - {self.last_explanation['optimizer_type']}")
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Diversity")
        ax.grid(True, linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        return fig
    
    def _plot_landscape_analysis(self, **kwargs) -> plt.Figure:
        """Plot landscape analysis metrics"""
        gradient_estimates = self.last_explanation['gradient_estimates']
        local_optima_count = self.last_explanation['local_optima_count']
        landscape_ruggedness = self.last_explanation['landscape_ruggedness']
        
        if not gradient_estimates:
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.text(0.5, 0.5, "No landscape analysis data available", 
                   ha='center', va='center', fontsize=12)
            return fig
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Plot gradient estimates
        iterations = range(1, len(gradient_estimates) + 1)
        ax.plot(iterations, gradient_estimates, marker='o', linestyle='-', alpha=0.7)
        ax.set_title(f"Landscape Analysis - {self.last_explanation['optimizer_type']}")
        ax.set_xlabel("Sample")
        ax.set_ylabel("Gradient Magnitude")
        ax.grid(True, linestyle='--', alpha=0.7)
        
        # Add annotations for landscape metrics
        text = f"Estimated Local Optima: {local_optima_count}\n"
        if landscape_ruggedness is not None:
            text += f"Landscape Ruggedness: {landscape_ruggedness:.4f}"
        
        ax.text(0.02, 0.98, text, transform=ax.transAxes, 
               verticalalignment='top', horizontalalignment='left',
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        return fig
    
    def _plot_decision_process(self, **kwargs) -> plt.Figure:
        """Plot decision process metrics"""
        success_history = self.last_explanation['success_history']
        
        if not success_history:
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.text(0.5, 0.5, "No decision process data available", 
                   ha='center', va='center', fontsize=12)
            return fig
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Convert boolean success history to 0/1
        success_values = [1 if x else 0 for x in success_history]
        
        # Calculate moving average
        window_size = min(10, len(success_values))
        moving_avg = []
        for i in range(len(success_values)):
            if i < window_size - 1:
                moving_avg.append(sum(success_values[:i+1]) / (i+1))
            else:
                moving_avg.append(sum(success_values[i-window_size+1:i+1]) / window_size)
        
        iterations = range(1, len(success_values) + 1)
        
        # Plot individual success/failure
        ax.scatter(iterations, success_values, marker='o', alpha=0.5, label='Success/Failure')
        
        # Plot moving average
        ax.plot(iterations, moving_avg, 'r-', linewidth=2, label=f'{window_size}-point Moving Avg')
        
        ax.set_title(f"Decision Process - {self.last_explanation['optimizer_type']}")
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Success Rate")
        ax.set_ylim(-0.1, 1.1)
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.legend()
        
        # Add overall success rate
        overall_rate = sum(success_values) / len(success_values)
        ax.text(0.02, 0.02, f"Overall Success Rate: {overall_rate:.2f}", 
               transform=ax.transAxes, verticalalignment='bottom', 
               horizontalalignment='left',
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        return fig
    
    def _plot_exploration_exploitation(self, **kwargs) -> plt.Figure:
        """Plot exploration/exploitation balance"""
        selection_pressure = self.last_explanation['selection_pressure']
        
        if not selection_pressure:
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.text(0.5, 0.5, "No exploration/exploitation data available", 
                   ha='center', va='center', fontsize=12)
            return fig
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        iterations = range(1, len(selection_pressure) + 1)
        ax.plot(iterations, selection_pressure, marker='o', linestyle='-', alpha=0.7)
        ax.set_title(f"Exploration/Exploitation Balance - {self.last_explanation['optimizer_type']}")
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Selection Pressure")
        ax.grid(True, linestyle='--', alpha=0.7)
        
        # Add threshold line
        ax.axhline(y=0.5, color='r', linestyle='--', alpha=0.7)
        ax.text(len(selection_pressure) * 0.02, 0.52, "Exploration", color='blue')
        ax.text(len(selection_pressure) * 0.02, 0.48, "Exploitation", color='green')
        
        plt.tight_layout()
        return fig
    
    def _plot_gradient_estimation(self, **kwargs) -> plt.Figure:
        """Plot gradient estimation over iterations"""
        gradient_estimates = self.last_explanation['gradient_estimates']
        
        if not gradient_estimates:
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.text(0.5, 0.5, "No gradient estimation data available", 
                   ha='center', va='center', fontsize=12)
            return fig
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        iterations = range(1, len(gradient_estimates) + 1)
        ax.plot(iterations, gradient_estimates, marker='o', linestyle='-', alpha=0.7)
        ax.set_title(f"Gradient Estimation - {self.last_explanation['optimizer_type']}")
        ax.set_xlabel("Sample")
        ax.set_ylabel("Gradient Magnitude")
        ax.grid(True, linestyle='--', alpha=0.7)
        
        # Add trend line
        if len(gradient_estimates) > 1:
            z = np.polyfit(iterations, gradient_estimates, 1)
            p = np.poly1d(z)
            ax.plot(iterations, p(iterations), "r--", alpha=0.7, 
                   label=f"Trend: {z[0]:.4f}x + {z[1]:.4f}")
            ax.legend()
        
        plt.tight_layout()
        return fig
    
    def _plot_performance_profile(self, **kwargs) -> plt.Figure:
        """Plot performance profile"""
        time_per_iteration = self.last_explanation['time_per_iteration']
        
        if not time_per_iteration:
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.text(0.5, 0.5, "No performance data available", 
                   ha='center', va='center', fontsize=12)
            return fig
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        iterations = range(1, len(time_per_iteration) + 1)
        ax.plot(iterations, time_per_iteration, marker='o', linestyle='-', alpha=0.7)
        ax.set_title(f"Performance Profile - {self.last_explanation['optimizer_type']}")
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Time (s)")
        ax.grid(True, linestyle='--', alpha=0.7)
        
        # Add average line
        avg_time = sum(time_per_iteration) / len(time_per_iteration)
        ax.axhline(y=avg_time, color='r', linestyle='--', alpha=0.7)
        ax.text(len(time_per_iteration) * 0.02, avg_time * 1.05, 
               f"Average: {avg_time:.4f}s", color='r')
        
        plt.tight_layout()
        return fig

def run_optimizer_explainability(optimizer, plot_types=None, generate_plots=True, **kwargs):
    """
    Run explainability analysis on an optimizer
    
    Parameters:
    -----------
    optimizer : object
        Optimizer instance to explain
    plot_types : List[str], optional
        List of plot types to generate
    generate_plots : bool, optional
        Whether to generate plots
    **kwargs : dict
        Additional parameters for the explainer
        
    Returns:
    --------
    Dict[str, Any]
        Dictionary containing explanation data and plot paths
    """
    logging.info("Running optimizer explainability analysis")
    
    # Placeholder implementation
    # Will be replaced with full implementation
    
    return {
        "message": "Optimizer explainability placeholder",
        "optimizer_type": type(optimizer).__name__
    }
