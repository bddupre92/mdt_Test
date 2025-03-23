"""
Benchmark Visualization Module

This module provides visualization tools for benchmark results, including:
- Convergence curves for optimizer runs
- Performance comparisons between different optimizers
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Any, Optional, Tuple
from collections import Counter

class BenchmarkVisualizer:
    """Class for visualizing benchmark results."""
    
    def __init__(self, results_dir: str = "benchmark_results"):
        """Initialize visualizer.
        
        Args:
            results_dir: Directory containing benchmark results
        """
        self.results_dir = results_dir
        # Create results directory if it doesn't exist
        os.makedirs(results_dir, exist_ok=True)
        
        # Set default style
        sns.set_theme(style="whitegrid")
        plt.rcParams.update({
            'figure.figsize': (10, 6),
            'figure.dpi': 100,
            'axes.titlesize': 16,
            'axes.labelsize': 14
        })
    
    def load_results(self, filename: str) -> Dict[str, Any]:
        """Load benchmark results from a JSON file.
        
        Args:
            filename: Name of the file (without path)
            
        Returns:
            Dictionary containing the loaded results
            
        Raises:
            FileNotFoundError: If file does not exist
        """
        filepath = os.path.join(self.results_dir, filename)
        with open(filepath, 'r') as f:
            return json.load(f)
    
    def plot_convergence(self, 
                        results: Dict[str, Any],
                        title: Optional[str] = None,
                        log_scale: bool = True) -> plt.Figure:
        """Plot convergence curves from benchmark results.
        
        Args:
            results: Dictionary of benchmark results
            title: Title for the plot
            log_scale: Whether to use log scale for y-axis
            
        Returns:
            Matplotlib figure object
        """
        plt.close('all')  # Close any existing figures
        fig, ax = plt.subplots()
        
        # Extract convergence data
        if "optimizers" in results:
            # Multiple optimizer results
            for opt_name, opt_data in results["optimizers"].items():
                if "convergence" in opt_data and opt_data["convergence"]:
                    iterations = list(range(len(opt_data["convergence"])))
                    ax.plot(iterations, opt_data["convergence"], label=opt_name)
        elif "convergence" in results and results["convergence"]:
            # Single optimizer results
            iterations = list(range(len(results["convergence"])))
            ax.plot(iterations, results["convergence"], label=results.get("optimizer_id", "Unknown"))
        
        ax.set_xlabel("Iterations")
        ax.set_ylabel("Best Fitness")
        
        if log_scale and ax.get_ylim()[0] > 0:
            ax.set_yscale('log')
        
        ax.set_title(title or "Optimization Convergence")
        ax.legend()
        ax.grid(True)
        
        plt.tight_layout()
        return fig
    
    def plot_optimizer_comparison(self, 
                                results: Dict[str, Any],
                                metric: str = "best_fitness",
                                title: Optional[str] = None) -> plt.Figure:
        """Plot comparison of different optimizers based on a metric.
        
        Args:
            results: Dictionary of benchmark results
            metric: Metric to compare (best_fitness, evaluations, runtime)
            title: Title for the plot
            
        Returns:
            Matplotlib figure object
        """
        if "optimizers" not in results:
            print("No optimizer comparison data found in results")
            return None
        
        plt.close('all')  # Close any existing figures
        fig, ax = plt.subplots()
        
        # Extract data for comparison
        optimizer_names = []
        metric_values = []
        
        for opt_name, opt_data in results["optimizers"].items():
            if metric in opt_data:
                optimizer_names.append(opt_name)
                metric_values.append(opt_data[metric])
        
        if not optimizer_names:
            print(f"No data found for metric: {metric}")
            return None
        
        # Create barplot
        bars = ax.bar(optimizer_names, metric_values)
        
        # Add value labels on top of bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.4g}',
                    ha='center', va='bottom')
        
        ax.set_xlabel("Optimizer")
        
        # Format y-label based on metric
        metric_labels = {
            "best_fitness": "Best Fitness",
            "evaluations": "Function Evaluations",
            "runtime": "Runtime (seconds)"
        }
        ax.set_ylabel(metric_labels.get(metric, metric))
        
        ax.set_title(title or f"Optimizer Comparison - {metric_labels.get(metric, metric)}")
        ax.grid(True, axis='y')
        
        plt.tight_layout()
        return fig
    
    def plot_meta_optimizer_selections(self, 
                                     results: Dict[str, Any],
                                     title: Optional[str] = None) -> plt.Figure:
        """Plot optimizer selection patterns from meta-optimizer results.
        
        Args:
            results: Dictionary of benchmark results
            title: Title for the plot
            
        Returns:
            Matplotlib figure object
        """
        if "selected_optimizers" not in results:
            print("No optimizer selection data found in results")
            return None
        
        plt.close('all')  # Close any existing figures
        fig, ax = plt.subplots()
        
        # Count selections
        selections = Counter(results["selected_optimizers"])
        
        # Convert to percentages
        total = sum(selections.values())
        percentages = {opt: (count / total) * 100 for opt, count in selections.items()}
        
        # Sort by selection frequency
        sorted_items = sorted(percentages.items(), key=lambda x: x[1], reverse=True)
        optimizer_names = [item[0] for item in sorted_items]
        selection_percentages = [item[1] for item in sorted_items]
        
        # Create barplot
        bars = ax.bar(optimizer_names, selection_percentages)
        
        # Add percentage labels on top of bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.1f}%',
                    ha='center', va='bottom')
        
        ax.set_xlabel("Optimizer")
        ax.set_ylabel("Selection Percentage (%)")
        ax.set_title(title or "Meta-Optimizer Selection Patterns")
        ax.grid(True, axis='y')
        
        plt.tight_layout()
        return fig 