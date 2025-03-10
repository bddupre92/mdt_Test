"""
Visualization tools for baseline comparison

This module provides visualization tools for comparing baseline algorithm
selectors with the Meta Optimizer.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Any, Optional, Tuple
from matplotlib.figure import Figure
from scipy import stats
from matplotlib.axes import Axes

class ComparisonVisualizer:
    """
    Comprehensive visualization tool for comparing algorithm performance
    
    This class provides methods for creating various visualizations to compare
    the performance of baseline algorithm selectors and the Meta Optimizer.
    """
    
    def __init__(self, results: Dict[str, Dict], export_dir: str = 'results/baseline_comparison'):
        """
        Initialize the visualization tool
        
        Args:
            results: Dictionary with results from BaselineComparison.run_comparison
            export_dir: Directory to export visualizations
        """
        self.results = results
        self.export_dir = export_dir
        self.figure_counter = 0
        
        # Create export directory if it doesn't exist
        os.makedirs(export_dir, exist_ok=True)
        
        # Function names from results
        self.function_names = list(results.keys())
        
        # Extract algorithm names
        self.algorithms = set()
        for func_name, func_results in results.items():
            self.algorithms.update(func_results["baseline_selected_algorithms"])
            self.algorithms.update(func_results["meta_selected_algorithms"])
        self.algorithms = sorted(list(self.algorithms))
        
        # Available metrics
        self.metrics = [
            "best_fitness_avg",
            "evaluations_avg",
            "time_avg"
        ]
    
    def performance_comparison(self, metric: str = "best_fitness_avg", 
                             save: bool = True) -> Figure:
        """
        Create a performance comparison plot
        
        Args:
            metric: Which metric to plot
            save: Whether to save the plot
            
        Returns:
            The figure
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Get data
        baseline_values = [self.results[f]["baseline_{0}".format(metric)] for f in self.function_names]
        meta_values = [self.results[f]["meta_{0}".format(metric)] for f in self.function_names]
        
        # Create bar chart
        x = np.arange(len(self.function_names))
        width = 0.35
        
        ax.bar(x - width/2, baseline_values, width, label='Baseline')
        ax.bar(x + width/2, meta_values, width, label='Meta Optimizer')
        
        # Labels and title
        ax.set_ylabel(metric.replace("_", " ").title())
        ax.set_title(f'Performance Comparison ({metric})')
        ax.set_xticks(x)
        ax.set_xticklabels(self.function_names, rotation=45, ha='right')
        ax.legend()
        
        fig.tight_layout()
        
        # Save figure
        if save:
            fig_path = f"{self.export_dir}/performance_comparison_{metric}.png"
            fig.savefig(fig_path, dpi=300, bbox_inches='tight')
            print(f"Saved figure to {fig_path}")
        
        return fig
    
    def algorithm_selection_frequency(self, save: bool = True) -> Figure:
        """
        Create an algorithm selection frequency plot
        
        Args:
            save: Whether to save the plot
            
        Returns:
            The figure
        """
        # Collect all selected algorithms
        baseline_algorithms = []
        meta_algorithms = []
        
        for func_name, func_results in self.results.items():
            baseline_algorithms.extend(func_results["baseline_selected_algorithms"])
            meta_algorithms.extend(func_results["meta_selected_algorithms"])
        
        # Count frequencies
        baseline_counts = {}
        for alg in baseline_algorithms:
            baseline_counts[alg] = baseline_counts.get(alg, 0) + 1
            
        meta_counts = {}
        for alg in meta_algorithms:
            meta_counts[alg] = meta_counts.get(alg, 0) + 1
        
        # Create bar chart
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Baseline
        ax1.bar(baseline_counts.keys(), baseline_counts.values())
        ax1.set_title('Baseline Algorithm Selection')
        ax1.set_ylabel('Frequency')
        ax1.set_xticklabels(baseline_counts.keys(), rotation=45, ha='right')
        
        # Meta Optimizer
        ax2.bar(meta_counts.keys(), meta_counts.values())
        ax2.set_title('Meta Optimizer Algorithm Selection')
        ax2.set_ylabel('Frequency')
        ax2.set_xticklabels(meta_counts.keys(), rotation=45, ha='right')
        
        fig.tight_layout()
        
        # Save figure
        if save:
            fig_path = f"{self.export_dir}/algorithm_selection_frequency.png"
            fig.savefig(fig_path, dpi=300, bbox_inches='tight')
            print(f"Saved figure to {fig_path}")
        
        return fig
    
    def create_all_visualizations(self) -> None:
        """
        Create all available visualizations
        """
        # Create performance comparison for each metric
        for metric in self.metrics:
            self.performance_comparison(metric)
        
        # Create algorithm selection frequency plot
        self.algorithm_selection_frequency()
        
        print(f"All visualizations saved to {self.export_dir}")

def plot_performance_comparison(
    results: Dict[str, Dict],
    metric: str = "best_fitness_avg",
    title: Optional[str] = None,
    save_path: Optional[str] = None
) -> Tuple[Figure, Axes]:
    """
    Plot a comparison of performance between baseline and Meta Optimizer
    
    Args:
        results: Dictionary with results from run_comparison
        metric: Which metric to plot (default: best_fitness_avg)
        title: Plot title
        save_path: Path to save the figure
        
    Returns:
        Figure and axes
    """
    func_names = list(results.keys())
    baseline_values = [results[f]["baseline_{0}".format(metric)] for f in func_names]
    meta_values = [results[f]["meta_{0}".format(metric)] for f in func_names]
    
    # Create a bar chart
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(func_names))
    width = 0.35
    
    ax.bar(x - width/2, baseline_values, width, label='Baseline')
    ax.bar(x + width/2, meta_values, width, label='Meta Optimizer')
    
    ax.set_ylabel(metric.replace("_", " ").title())
    ax.set_title(title or f'Performance Comparison ({metric})')
    ax.set_xticks(x)
    ax.set_xticklabels(func_names, rotation=45, ha='right')
    ax.legend()
    
    fig.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig, ax

def plot_algorithm_selection_frequency(
    results: Dict[str, Dict],
    title: Optional[str] = None,
    save_path: Optional[str] = None
) -> Tuple[Figure, Axes]:
    """
    Plot the frequency of algorithm selection
    
    Args:
        results: Dictionary with results from run_comparison
        title: Plot title
        save_path: Path to save the figure
        
    Returns:
        Figure and axes
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
    
    # Baseline
    ax1.bar(baseline_counts.keys(), baseline_counts.values())
    ax1.set_title('Baseline Algorithm Selection')
    ax1.set_ylabel('Frequency')
    ax1.set_xticklabels(baseline_counts.keys(), rotation=45, ha='right')
    
    # Meta Optimizer
    ax2.bar(meta_counts.keys(), meta_counts.values())
    ax2.set_title('Meta Optimizer Algorithm Selection')
    ax2.set_ylabel('Frequency')
    ax2.set_xticklabels(meta_counts.keys(), rotation=45, ha='right')
    
    fig.suptitle(title or 'Algorithm Selection Frequency')
    fig.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig, (ax1, ax2)

def plot_convergence_curves(
    results: Dict[str, Dict],
    title: Optional[str] = None,
    save_path: Optional[str] = None
) -> Tuple[Figure, Axes]:
    """
    Plot convergence curves for baseline and Meta Optimizer
    
    Args:
        results: Dictionary with results from run_comparison
        title: Plot title
        save_path: Path to save the figure
        
    Returns:
        Figure and axes
    """
    # This function assumes results contains convergence data
    # If not available, it should be adapted to your specific data structure
    
    if "convergence" not in list(results.values())[0].get("baseline_best_fitness", {}):
        # Placeholder if convergence data is not available
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.text(0.5, 0.5, "Convergence data not available", 
                ha='center', va='center', fontsize=14)
        
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            
        return fig, ax
    
    # Create subplots for each function
    n_funcs = len(results)
    fig, axes = plt.subplots(n_funcs, 1, figsize=(10, 5 * n_funcs), sharex=True)
    
    if n_funcs == 1:
        axes = [axes]
    
    for i, (func_name, func_results) in enumerate(results.items()):
        ax = axes[i]
        
        # Extract convergence data
        baseline_convergence = func_results.get("baseline_convergence", {})
        meta_convergence = func_results.get("meta_convergence", {})
        
        if not baseline_convergence or not meta_convergence:
            ax.text(0.5, 0.5, "Convergence data not available", 
                    ha='center', va='center', fontsize=12)
            continue
        
        # Plot convergence curves
        ax.plot(baseline_convergence["evaluations"], baseline_convergence["fitness"],
                label='Baseline', color='blue')
        ax.plot(meta_convergence["evaluations"], meta_convergence["fitness"],
                label='Meta Optimizer', color='orange')
        
        # Add standard deviation bands if available
        if "fitness_std" in baseline_convergence:
            ax.fill_between(
                baseline_convergence["evaluations"],
                baseline_convergence["fitness"] - baseline_convergence["fitness_std"],
                baseline_convergence["fitness"] + baseline_convergence["fitness_std"],
                alpha=0.2, color='blue'
            )
            
        if "fitness_std" in meta_convergence:
            ax.fill_between(
                meta_convergence["evaluations"],
                meta_convergence["fitness"] - meta_convergence["fitness_std"],
                meta_convergence["fitness"] + meta_convergence["fitness_std"],
                alpha=0.2, color='orange'
            )
        
        ax.set_title(f"Convergence for {func_name}")
        ax.set_ylabel("Fitness")
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.legend()
        
        if i == n_funcs - 1:
            ax.set_xlabel("Function Evaluations")
    
    fig.suptitle(title or "Convergence Curves")
    fig.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig, axes

def plot_radar_chart(
    results: Dict[str, Dict],
    metrics: List[str] = None,
    title: Optional[str] = None,
    save_path: Optional[str] = None
) -> Tuple[Figure, Axes]:
    """
    Plot a radar chart comparing baseline and Meta Optimizer across metrics
    
    Args:
        results: Dictionary with results from run_comparison
        metrics: List of metrics to include in the radar chart
        title: Plot title
        save_path: Path to save the figure
        
    Returns:
        Figure and axes
    """
    # Default metrics if not provided
    if metrics is None:
        metrics = [
            "best_fitness_avg",
            "evaluations_avg",
            "time_avg"
        ]
    
    # Number of metrics
    n_metrics = len(metrics)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))
    
    # Angles for each metric
    angles = np.linspace(0, 2 * np.pi, n_metrics, endpoint=False).tolist()
    angles += angles[:1]  # Close the loop
    
    # Labels for each metric
    labels = [m.replace("_", " ").title() for m in metrics]
    labels += labels[:1]  # Close the loop
    
    # Calculate average values across all functions
    baseline_values = []
    meta_values = []
    
    for metric in metrics:
        # Ensure metrics exist in the results
        if f"baseline_{metric}" not in results[list(results.keys())[0]]:
            continue
            
        # Extract values for this metric
        baseline_metric_values = [results[f][f"baseline_{metric}"] for f in results]
        meta_metric_values = [results[f][f"meta_{metric}"] for f in results]
        
        # Calculate averages
        baseline_avg = np.mean(baseline_metric_values)
        meta_avg = np.mean(meta_metric_values)
        
        # Normalize so that higher is always better
        if "fitness" in metric or "time" in metric or "evaluations" in metric:
            # Lower is better, so invert
            baseline_avg = 1.0 / (baseline_avg + 1e-10)
            meta_avg = 1.0 / (meta_avg + 1e-10)
        
        baseline_values.append(baseline_avg)
        meta_values.append(meta_avg)
    
    # Normalize to [0, 1]
    for i in range(len(metrics)):
        max_val = max(baseline_values[i], meta_values[i])
        if max_val > 0:
            baseline_values[i] /= max_val
            meta_values[i] /= max_val
    
    # Close the loop
    baseline_values += baseline_values[:1]
    meta_values += meta_values[:1]
    
    # Plot data
    ax.plot(angles, baseline_values, 'o-', linewidth=2, label='Baseline')
    ax.plot(angles, meta_values, 'o-', linewidth=2, label='Meta Optimizer')
    ax.fill(angles, baseline_values, alpha=0.1)
    ax.fill(angles, meta_values, alpha=0.1)
    
    # Set labels
    ax.set_thetagrids(np.degrees(angles[:-1]), labels[:-1])
    
    ax.set_title(title or "Performance Comparison")
    ax.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
    
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig, ax

def plot_boxplots(
    results: Dict[str, Dict],
    metric: str = "best_fitness",
    title: Optional[str] = None,
    save_path: Optional[str] = None
) -> Tuple[Figure, Axes]:
    """
    Plot boxplots comparing baseline and Meta Optimizer
    
    Args:
        results: Dictionary with results from run_comparison
        metric: Which metric to plot
        title: Plot title
        save_path: Path to save the figure
        
    Returns:
        Figure and axes
    """
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Prepare data
    data = []
    labels = []
    
    for func_name, func_results in results.items():
        # Extract baseline and meta values
        baseline_values = func_results[f"baseline_{metric}"]
        meta_values = func_results[f"meta_{metric}"]
        
        # Add to data
        data.append(baseline_values)
        data.append(meta_values)
        
        # Add labels
        labels.append(f"{func_name} (Baseline)")
        labels.append(f"{func_name} (Meta)")
    
    # Plot boxplots
    box = ax.boxplot(data, patch_artist=True, labels=labels)
    
    # Color the boxes
    colors = []
    for i in range(len(data)):
        if i % 2 == 0:
            colors.append('lightblue')
        else:
            colors.append('lightcoral')
    
    for patch, color in zip(box['boxes'], colors):
        patch.set_facecolor(color)
    
    # Set labels and title
    ax.set_ylabel(metric.replace("_", " ").title())
    ax.set_title(title or f"{metric.replace('_', ' ').title()} Comparison")
    ax.set_xticklabels(labels, rotation=45, ha='right')
    
    # Add a legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='lightblue', label='Baseline'),
        Patch(facecolor='lightcoral', label='Meta Optimizer')
    ]
    ax.legend(handles=legend_elements, loc='upper right')
    
    fig.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig, ax

def plot_heatmap(
    results: Dict[str, Dict],
    metric: str = "best_fitness_avg",
    title: Optional[str] = None,
    save_path: Optional[str] = None
) -> Tuple[Figure, Axes]:
    """
    Plot a heatmap of performance improvement
    
    Args:
        results: Dictionary with results from run_comparison
        metric: Which metric to plot
        title: Plot title
        save_path: Path to save the figure
        
    Returns:
        Figure and axes
    """
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Extract function names
    func_names = list(results.keys())
    
    # Create a list of available algorithms
    all_algorithms = set()
    for func_name, func_results in results.items():
        all_algorithms.update(func_results["baseline_selected_algorithms"])
        all_algorithms.update(func_results["meta_selected_algorithms"])
    all_algorithms = sorted(list(all_algorithms))
    
    # Create the data matrix
    data = np.zeros((len(func_names), len(all_algorithms)))
    
    for i, func_name in enumerate(func_names):
        func_results = results[func_name]
        
        # Count selections for each algorithm
        for j, alg in enumerate(all_algorithms):
            baseline_count = func_results["baseline_selected_algorithms"].count(alg)
            meta_count = func_results["meta_selected_algorithms"].count(alg)
            
            # Calculate selection difference
            data[i, j] = meta_count - baseline_count
    
    # Plot the heatmap
    cmap = plt.cm.RdBu_r
    im = ax.imshow(data, cmap=cmap, aspect='auto')
    
    # Add colorbar
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Selection Difference (Meta - Baseline)")
    
    # Add labels
    ax.set_xticks(np.arange(len(all_algorithms)))
    ax.set_yticks(np.arange(len(func_names)))
    ax.set_xticklabels(all_algorithms, rotation=45, ha='right')
    ax.set_yticklabels(func_names)
    
    # Add a grid
    ax.set_xticks(np.arange(-.5, len(all_algorithms), 1), minor=True)
    ax.set_yticks(np.arange(-.5, len(func_names), 1), minor=True)
    ax.grid(which='minor', color='w', linestyle='-', linewidth=2)
    
    # Add title
    ax.set_title(title or "Algorithm Selection Difference (Meta - Baseline)")
    
    fig.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig, ax 