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
import logging
from typing import Dict, List, Any, Optional, Tuple
from matplotlib.figure import Figure
from scipy import stats
from matplotlib.axes import Axes

logger = logging.getLogger(__name__)

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
            if "simple_baseline" in func_results:
                self.algorithms.update(func_results["simple_baseline"]["selections"])
            if "meta_learner" in func_results:
                self.algorithms.update(func_results["meta_learner"]["selections"])
            if "enhanced_meta" in func_results:
                self.algorithms.update(func_results["enhanced_meta"]["selections"])
            if "satzilla" in func_results:
                self.algorithms.update(func_results["satzilla"]["selections"])
        self.algorithms = sorted(list(self.algorithms))
        
        # Available metrics
        self.metrics = [
            "best_fitness",
            "evaluations",
            "time"
        ]
    
    def performance_comparison(self, metric: str = "best_fitness", 
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
        simple_values = []
        meta_values = []
        enhanced_values = []
        satzilla_values = []
        
        for f in self.function_names:
            # Get average values for each approach
            if "simple_baseline" in self.results[f]:
                simple_values.append(np.mean(self.results[f]["simple_baseline"][metric]))
            if "meta_learner" in self.results[f]:
                meta_values.append(np.mean(self.results[f]["meta_learner"][metric]))
            if "enhanced_meta" in self.results[f]:
                enhanced_values.append(np.mean(self.results[f]["enhanced_meta"][metric]))
            if "satzilla" in self.results[f]:
                satzilla_values.append(np.mean(self.results[f]["satzilla"][metric]))
        
        # Create bar chart
        x = np.arange(len(self.function_names))
        width = 0.2  # Narrower bars to fit all four
        
        if simple_values:
            ax.bar(x - 1.5*width, simple_values, width, label='Simple Baseline')
        if meta_values:
            ax.bar(x - 0.5*width, meta_values, width, label='Meta Learner')
        if enhanced_values:
            ax.bar(x + 0.5*width, enhanced_values, width, label='Enhanced Meta')
        if satzilla_values:
            ax.bar(x + 1.5*width, satzilla_values, width, label='SATzilla')
        
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
        simple_algorithms = []
        meta_algorithms = []
        enhanced_algorithms = []
        satzilla_algorithms = []
        
        for func_name, func_results in self.results.items():
            if "simple_baseline" in func_results:
                simple_algorithms.extend(func_results["simple_baseline"]["selections"])
            if "meta_learner" in func_results:
                meta_algorithms.extend(func_results["meta_learner"]["selections"])
            if "enhanced_meta" in func_results:
                enhanced_algorithms.extend(func_results["enhanced_meta"]["selections"])
            if "satzilla" in func_results:
                satzilla_algorithms.extend(func_results["satzilla"]["selections"])
        
        # Count frequencies
        simple_counts = {}
        for alg in simple_algorithms:
            simple_counts[alg] = simple_counts.get(alg, 0) + 1
            
        meta_counts = {}
        for alg in meta_algorithms:
            meta_counts[alg] = meta_counts.get(alg, 0) + 1
            
        enhanced_counts = {}
        for alg in enhanced_algorithms:
            enhanced_counts[alg] = enhanced_counts.get(alg, 0) + 1
            
        satzilla_counts = {}
        for alg in satzilla_algorithms:
            satzilla_counts[alg] = satzilla_counts.get(alg, 0) + 1
        
        # Create bar chart
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # Simple Baseline
        if simple_counts:
            ax1.bar(simple_counts.keys(), simple_counts.values())
            ax1.set_title('Simple Baseline Selection')
            ax1.set_ylabel('Frequency')
            ax1.set_xticklabels(simple_counts.keys(), rotation=45, ha='right')
        else:
            ax1.text(0.5, 0.5, 'No data', ha='center', va='center')
        
        # Meta Learner
        if meta_counts:
            ax2.bar(meta_counts.keys(), meta_counts.values())
            ax2.set_title('Meta Learner Selection')
            ax2.set_ylabel('Frequency')
            ax2.set_xticklabels(meta_counts.keys(), rotation=45, ha='right')
        else:
            ax2.text(0.5, 0.5, 'No data', ha='center', va='center')
        
        # Enhanced Meta
        if enhanced_counts:
            ax3.bar(enhanced_counts.keys(), enhanced_counts.values())
            ax3.set_title('Enhanced Meta Selection')
            ax3.set_ylabel('Frequency')
            ax3.set_xticklabels(enhanced_counts.keys(), rotation=45, ha='right')
        else:
            ax3.text(0.5, 0.5, 'No data', ha='center', va='center')
        
        # SATzilla
        if satzilla_counts:
            ax4.bar(satzilla_counts.keys(), satzilla_counts.values())
            ax4.set_title('SATzilla Selection')
            ax4.set_ylabel('Frequency')
            ax4.set_xticklabels(satzilla_counts.keys(), rotation=45, ha='right')
        else:
            ax4.text(0.5, 0.5, 'No data', ha='center', va='center')
        
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
        # Basic performance comparisons
        for metric in self.metrics:
            self.performance_comparison(metric)
            self.plot_boxplots(metric)
            self.plot_violin(metric)
        
        # Algorithm selection analysis
        self.algorithm_selection_frequency()
        self.plot_heatmap()
        
        # Convergence analysis
        self.plot_convergence_curves()
        
        # Statistical analysis
        self.plot_critical_difference_diagram()
        self.plot_performance_profiles()
        
        # Radar plot for overall comparison
        self.plot_radar_chart()
        
        print(f"All visualizations saved to {self.export_dir}")

    def plot_violin(self, metric: str = "best_fitness_avg", save: bool = True) -> Figure:
        """Create violin plots showing the distribution of results"""
        fig, ax = plt.subplots(figsize=(12, 6))
        
        data = []
        labels = []
        for func_name in self.function_names:
            if f"baseline_{metric}_all" in self.results[func_name]:
                baseline_data = self.results[func_name][f"baseline_{metric}_all"]
                meta_data = self.results[func_name][f"meta_{metric}_all"]
                data.extend([baseline_data, meta_data])
                labels.extend([f"{func_name}\nBaseline", f"{func_name}\nMeta"])
        
        if data:
            parts = ax.violinplot(data, showmeans=True, showmedians=True)
            
            # Customize colors
            for i, pc in enumerate(parts['bodies']):
                if i % 2 == 0:
                    pc.set_facecolor('lightblue')
                else:
                    pc.set_facecolor('lightcoral')
            
            ax.set_xticks(range(1, len(labels) + 1))
            ax.set_xticklabels(labels, rotation=45, ha='right')
            ax.set_ylabel(metric.replace("_", " ").title())
            ax.set_title(f"Distribution of {metric.replace('_', ' ').title()}")
            
            # Add legend
            from matplotlib.patches import Patch
            legend_elements = [
                Patch(facecolor='lightblue', label='Baseline'),
                Patch(facecolor='lightcoral', label='Meta Optimizer')
            ]
            ax.legend(handles=legend_elements)
            
            fig.tight_layout()
            
            if save:
                fig.savefig(f"{self.export_dir}/violin_{metric}.png", dpi=300, bbox_inches='tight')
                print(f"Saved violin plot to {self.export_dir}/violin_{metric}.png")
        
        return fig

    def plot_convergence_curves(self, save: bool = True) -> Figure:
        """Plot convergence curves for all approaches."""
        fig, ax = plt.subplots(figsize=(12, 8))
        
        for func_name in self.function_names:
            func_results = self.results[func_name]
            
            # Plot convergence for each approach
            for approach in ["simple", "meta", "enhanced", "satzilla"]:
                if approach in func_results:
                    convergence_data = func_results[approach]["convergence"]
                    if convergence_data and any(len(data) > 0 for data in convergence_data):
                        try:
                            # Find the minimum length to truncate all arrays to the same length
                            non_empty_data = [data for data in convergence_data if len(data) > 0]
                            if not non_empty_data:
                                continue
                                
                            min_length = min(len(data) for data in non_empty_data)
                            
                            # Truncate all arrays to the minimum length
                            truncated_data = [data[:min_length] for data in non_empty_data]
                            
                            # Now we can safely create a numpy array
                            convergence_array = np.array(truncated_data)
                            
                            if len(convergence_array) > 0:
                                mean_curve = np.mean(convergence_array, axis=0)
                                std_curve = np.std(convergence_array, axis=0)
                                iterations = range(len(mean_curve))
                                
                                # Plot mean with std shading
                                label = f"{approach.capitalize()} - {func_name}"
                                ax.plot(iterations, mean_curve, label=label)
                                ax.fill_between(iterations, mean_curve - std_curve, mean_curve + std_curve, alpha=0.2)
                        except Exception as e:
                            logger.warning(f"Error plotting convergence for {approach} on {func_name}: {e}")
                            continue
        
        ax.set_xlabel('Iterations')
        ax.set_ylabel('Fitness Value')
        ax.set_title('Convergence Comparison')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        if save:
            plt.tight_layout()
            save_path = os.path.join(self.export_dir, "convergence_curves.png")
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved convergence curves to {save_path}")
        
        return fig

    def plot_critical_difference_diagram(self, save: bool = True) -> Figure:
        """Create a critical difference diagram using statistical tests"""
        from scipy import stats
        
        # Collect rankings for each function
        rankings = []
        for func_name in self.function_names:
            func_results = self.results[func_name]
            if "averages" in func_results:
                simple_perf = func_results["averages"]["simple"]["avg_fitness"]
                meta_perf = func_results["averages"]["meta"]["avg_fitness"]
                enhanced_perf = func_results["averages"]["enhanced"]["avg_fitness"]
                satzilla_perf = func_results["averages"]["satzilla"]["avg_fitness"]
                
                # Get rankings (1 is best, 4 is worst) - Lower is better for fitness
                perfs = [simple_perf, meta_perf, enhanced_perf, satzilla_perf]
                sorted_idx = np.argsort(perfs)
                trial_ranks = np.zeros(4)
                for i, idx in enumerate(sorted_idx):
                    trial_ranks[idx] = i + 1
                rankings.append(trial_ranks)
            else:
                logger.warning(f"No averages found for {func_name}, skipping in critical difference diagram")
        
        if not rankings:
            logger.warning("No valid rankings found for critical difference diagram")
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.text(0.5, 0.5, "Insufficient data for critical difference diagram", 
                   ha='center', va='center', fontsize=12)
            
            if save:
                fig.savefig(f"{self.export_dir}/critical_difference.png", dpi=300, bbox_inches='tight')
                print(f"Saved critical difference diagram to {self.export_dir}/critical_difference.png")
            
            return fig
        
        rankings = np.array(rankings)
        avg_ranks = np.mean(rankings, axis=0)
        
        # Perform Friedman test if we have enough data
        if len(rankings) >= 2:
            try:
                stat, pvalue = stats.friedmanchisquare(*[rankings[:, i] for i in range(rankings.shape[1])])
            except:
                pvalue = float('nan')
        else:
            pvalue = float('nan')
        
        # Create the visualization
        fig, ax = plt.subplots(figsize=(10, 6))
        
        y_pos = [1, 2, 3, 4]
        ax.barh(y_pos, avg_ranks, height=0.5)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(['Simple', 'Meta', 'Enhanced', 'SATzilla'])
        ax.set_xlabel('Average Rank (lower is better)')
        ax.set_title(f'Critical Difference Diagram\n(p-value: {pvalue:.4f})')
        
        if save:
            fig.savefig(f"{self.export_dir}/critical_difference.png", dpi=300, bbox_inches='tight')
            print(f"Saved critical difference diagram to {self.export_dir}/critical_difference.png")
        
        return fig

    def plot_performance_profiles(self, save: bool = True) -> Figure:
        """Create performance profiles showing the cumulative distribution of performance ratios"""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # DEBUG: Print the structure of results for one function
        for func_name in self.function_names:
            print(f"DEBUG - Performance Profiles - Function: {func_name}")
            print(f"Structure of results[{func_name}]:")
            for key1, value1 in self.results[func_name].items():
                print(f"  {key1}: {type(value1)}")
                if isinstance(value1, dict):
                    for key2, value2 in value1.items():
                        print(f"    {key2}: {type(value2)}")
                        if isinstance(value2, dict):
                            for key3, value3 in value2.items():
                                print(f"      {key3}: {type(value3)} = {value3}")
            break  # Only print for one function
        
        # Collect performance ratios
        ratios = []
        for func_name in self.function_names:
            try:
                # Get performance metrics from results using the correct keys
                if "averages" in self.results[func_name]:
                    averages = self.results[func_name]["averages"]
                    
                    # Extract fitness values from the nested dictionaries
                    simple_perf = float(averages["simple"]["avg_fitness"]) if "simple" in averages and "avg_fitness" in averages["simple"] else float('inf')
                    meta_perf = float(averages["meta"]["avg_fitness"]) if "meta" in averages and "avg_fitness" in averages["meta"] else float('inf')
                    enhanced_perf = float(averages["enhanced"]["avg_fitness"]) if "enhanced" in averages and "avg_fitness" in averages["enhanced"] else float('inf')
                    satzilla_perf = float(averages["satzilla"]["avg_fitness"]) if "satzilla" in averages and "avg_fitness" in averages["satzilla"] else float('inf')
                    
                    # Find the best performance across all approaches (excluding infinity and zero)
                    valid_perfs = [p for p in [simple_perf, meta_perf, enhanced_perf, satzilla_perf] if p != float('inf') and p > 0]
                    if valid_perfs:
                        best_perf = min(valid_perfs)
                        
                        # Add performance ratios for all approaches (for finite values only)
                        # Avoid division by zero and handle zero values separately
                        ratios.append([
                            1.0 if simple_perf == 0 else (simple_perf/best_perf if simple_perf != float('inf') else float('inf')), 
                            1.0 if meta_perf == 0 else (meta_perf/best_perf if meta_perf != float('inf') else float('inf')),
                            1.0 if enhanced_perf == 0 else (enhanced_perf/best_perf if enhanced_perf != float('inf') else float('inf')),
                            1.0 if satzilla_perf == 0 else (satzilla_perf/best_perf if satzilla_perf != float('inf') else float('inf'))
                        ])
                    else:
                        # If all valid performances are zero or inf, assign ratio of 1.0 for zero values and inf for others
                        ratios.append([
                            1.0 if simple_perf == 0 else float('inf'),
                            1.0 if meta_perf == 0 else float('inf'),
                            1.0 if enhanced_perf == 0 else float('inf'),
                            1.0 if satzilla_perf == 0 else float('inf')
                        ])
                else:
                    logger.warning(f"No averages data found for function {func_name}")
            except KeyError as e:
                logger.warning(f"Missing key in results for function {func_name}: {e}")
                continue
            except Exception as e:
                logger.warning(f"Error processing performance data for {func_name}: {e}")
                continue
        
        if not ratios:
            ax.text(0.5, 0.5, "No performance data available for profiles", 
                   ha='center', va='center', fontsize=12)
            return fig
        
        ratios = np.array(ratios)
        
        # Create cumulative distribution
        labels = ['Simple', 'Meta', 'Enhanced', 'SATzilla']
        for i, label in enumerate(labels):
            if i < ratios.shape[1]:  # Check if we have data for this approach
                # Filter out inf values
                valid_ratios = ratios[:, i][~np.isinf(ratios[:, i])]
                if len(valid_ratios) > 0:
                    sorted_ratios = np.sort(valid_ratios)
                    cumulative = np.arange(1, len(sorted_ratios) + 1) / len(sorted_ratios)
                    ax.step(sorted_ratios, cumulative, label=label, where='post')
        
        ax.set_xlabel('Performance Ratio τ')
        ax.set_ylabel('Probability P(ratio ≤ τ)')
        ax.set_title('Performance Profiles')
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        if save:
            fig.savefig(f"{self.export_dir}/performance_profiles.png", dpi=300, bbox_inches='tight')
            print(f"Saved performance profiles to {self.export_dir}/performance_profiles.png")
        
        return fig

    def plot_heatmap(self, save: bool = True) -> Figure:
        """
        Create a heatmap showing algorithm selection patterns across different functions
        
        Args:
            save: Whether to save the plot
            
        Returns:
            The figure
        """
        # Collect all algorithms used
        all_algorithms = set()
        for func_results in self.results.values():
            if "simple_baseline" in func_results:
                all_algorithms.update(func_results["simple_baseline"]["selections"])
            if "meta_learner" in func_results:
                all_algorithms.update(func_results["meta_learner"]["selections"])
            if "enhanced_meta" in func_results:
                all_algorithms.update(func_results["enhanced_meta"]["selections"])
            if "satzilla" in func_results:
                all_algorithms.update(func_results["satzilla"]["selections"])
        
        all_algorithms = sorted(list(all_algorithms))
        
        # Initialize data arrays
        simple_data = np.zeros((len(self.results), len(all_algorithms)))
        meta_data = np.zeros((len(self.results), len(all_algorithms)))
        enhanced_data = np.zeros((len(self.results), len(all_algorithms)))
        satzilla_data = np.zeros((len(self.results), len(all_algorithms)))
        
        # Fill data arrays
        for i, (func_name, func_results) in enumerate(self.results.items()):
            if "simple_baseline" in func_results:
                for alg in func_results["simple_baseline"]["selections"]:
                    j = all_algorithms.index(alg)
                    simple_data[i, j] += 1
            if "meta_learner" in func_results:
                for alg in func_results["meta_learner"]["selections"]:
                    j = all_algorithms.index(alg)
                    meta_data[i, j] += 1
            if "enhanced_meta" in func_results:
                for alg in func_results["enhanced_meta"]["selections"]:
                    j = all_algorithms.index(alg)
                    enhanced_data[i, j] += 1
            if "satzilla" in func_results:
                for alg in func_results["satzilla"]["selections"]:
                    j = all_algorithms.index(alg)
                    satzilla_data[i, j] += 1
        
        # Normalize data
        simple_data = simple_data / np.sum(simple_data, axis=1, keepdims=True)
        meta_data = meta_data / np.sum(meta_data, axis=1, keepdims=True)
        enhanced_data = enhanced_data / np.sum(enhanced_data, axis=1, keepdims=True)
        satzilla_data = satzilla_data / np.sum(satzilla_data, axis=1, keepdims=True)
        
        # Create figure
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # Plot heatmaps
        sns.heatmap(simple_data, ax=ax1, xticklabels=all_algorithms, yticklabels=list(self.results.keys()),
                   cmap='YlOrRd', annot=True, fmt='.2f')
        ax1.set_title('Simple Baseline Selection Frequency')
        ax1.set_xlabel('Algorithm')
        ax1.set_ylabel('Function')
        plt.setp(ax1.get_xticklabels(), rotation=45, ha='right')
        
        sns.heatmap(meta_data, ax=ax2, xticklabels=all_algorithms, yticklabels=list(self.results.keys()),
                   cmap='YlOrRd', annot=True, fmt='.2f')
        ax2.set_title('Meta Learner Selection Frequency')
        ax2.set_xlabel('Algorithm')
        ax2.set_ylabel('Function')
        plt.setp(ax2.get_xticklabels(), rotation=45, ha='right')
        
        sns.heatmap(enhanced_data, ax=ax3, xticklabels=all_algorithms, yticklabels=list(self.results.keys()),
                   cmap='YlOrRd', annot=True, fmt='.2f')
        ax3.set_title('Enhanced Meta Selection Frequency')
        ax3.set_xlabel('Algorithm')
        ax3.set_ylabel('Function')
        plt.setp(ax3.get_xticklabels(), rotation=45, ha='right')
        
        sns.heatmap(satzilla_data, ax=ax4, xticklabels=all_algorithms, yticklabels=list(self.results.keys()),
                   cmap='YlOrRd', annot=True, fmt='.2f')
        ax4.set_title('SATzilla Selection Frequency')
        ax4.set_xlabel('Algorithm')
        ax4.set_ylabel('Function')
        plt.setp(ax4.get_xticklabels(), rotation=45, ha='right')
        
        fig.tight_layout()
        
        # Save figure
        if save:
            fig_path = f"{self.export_dir}/algorithm_selection_heatmap.png"
            fig.savefig(fig_path, dpi=300, bbox_inches='tight')
            print(f"Saved figure to {fig_path}")
        
        return fig

    def plot_radar_chart(self, save: bool = True) -> Figure:
        """Create a radar chart comparing baseline and meta optimizer performance"""
        # DEBUG: Print the structure of results for one function
        for func_name in self.function_names:
            print(f"DEBUG - Radar Chart - Function: {func_name}")
            print(f"Structure of results[{func_name}]:")
            for key1, value1 in self.results[func_name].items():
                print(f"  {key1}: {type(value1)}")
                if isinstance(value1, dict):
                    for key2, value2 in value1.items():
                        print(f"    {key2}: {type(value2)}")
                        if isinstance(value2, dict):
                            for key3, value3 in value2.items():
                                print(f"      {key3}: {type(value3)} = {value3}")
            break  # Only print for one function
            
        # Prepare the data
        metrics = ["Best Fitness", "Convergence Speed", "Success Rate"]
        num_metrics = len(metrics)
        
        # Calculate normalized scores for each metric
        simple_scores = []
        meta_scores = []
        enhanced_scores = []
        satzilla_scores = []
        
        for func_name in self.function_names:
            try:
                if "averages" in self.results[func_name]:
                    averages = self.results[func_name]["averages"]
                    
                    # Extract fitness values from the nested dictionaries
                    simple_fit = float(averages["simple"]["avg_fitness"]) if "simple" in averages and "avg_fitness" in averages["simple"] else float('inf')
                    meta_fit = float(averages["meta"]["avg_fitness"]) if "meta" in averages and "avg_fitness" in averages["meta"] else float('inf')
                    enhanced_fit = float(averages["enhanced"]["avg_fitness"]) if "enhanced" in averages and "avg_fitness" in averages["enhanced"] else float('inf')
                    satzilla_fit = float(averages["satzilla"]["avg_fitness"]) if "satzilla" in averages and "avg_fitness" in averages["satzilla"] else float('inf')
                    
                    # Use valid fitness values only (exclude infinity)
                    valid_fits = [f for f in [simple_fit, meta_fit, enhanced_fit, satzilla_fit] if f != float('inf')]
                    if valid_fits:
                        max_fit = max(valid_fits)
                        
                        # For fitness, lower is better, so normalize accordingly (1 is best)
                        simple_scores.append(1 - (simple_fit / max_fit) if simple_fit != float('inf') and max_fit > 0 else 0)
                        meta_scores.append(1 - (meta_fit / max_fit) if meta_fit != float('inf') and max_fit > 0 else 0)
                        enhanced_scores.append(1 - (enhanced_fit / max_fit) if enhanced_fit != float('inf') and max_fit > 0 else 0)
                        satzilla_scores.append(1 - (satzilla_fit / max_fit) if satzilla_fit != float('inf') and max_fit > 0 else 0)
                    
                    # Convergence speed (based on evaluations, lower is better)
                    simple_evals = float(averages["simple"]["avg_evaluations"]) if "simple" in averages and "avg_evaluations" in averages["simple"] else 0
                    meta_evals = float(averages["meta"]["avg_evaluations"]) if "meta" in averages and "avg_evaluations" in averages["meta"] else 0
                    enhanced_evals = float(averages["enhanced"]["avg_evaluations"]) if "enhanced" in averages and "avg_evaluations" in averages["enhanced"] else 0
                    satzilla_evals = float(averages["satzilla"]["avg_evaluations"]) if "satzilla" in averages and "avg_evaluations" in averages["satzilla"] else 0
                    
                    max_evals = max(simple_evals, meta_evals, enhanced_evals, satzilla_evals)
                    
                    # For evaluations, lower is better, so normalize accordingly
                    simple_scores.append(1 - (simple_evals / max_evals) if max_evals > 0 else 0)
                    meta_scores.append(1 - (meta_evals / max_evals) if max_evals > 0 else 0)
                    enhanced_scores.append(1 - (enhanced_evals / max_evals) if max_evals > 0 else 0)
                    satzilla_scores.append(1 - (satzilla_evals / max_evals) if max_evals > 0 else 0)
                    
                    # Success rate (higher is better)
                    # Use a default of 0 if not available
                    simple_success = 0
                    meta_success = 0
                    enhanced_success = 0
                    satzilla_success = 0
                    
                    simple_scores.append(simple_success)
                    meta_scores.append(meta_success)
                    enhanced_scores.append(enhanced_success)
                    satzilla_scores.append(satzilla_success)
            except KeyError as e:
                logger.warning(f"Missing key in results for function {func_name}: {e}")
                continue
            except Exception as e:
                logger.warning(f"Error processing data for radar chart for {func_name}: {e}")
                continue
        
        if not simple_scores:
            fig, ax = plt.subplots(figsize=(8, 8))
            ax.text(0.5, 0.5, "No data available for radar chart", 
                   ha='center', va='center', fontsize=12)
            return fig
        
        # Average scores across functions
        simple_avg = np.mean(np.array(simple_scores).reshape(-1, 3), axis=0)
        meta_avg = np.mean(np.array(meta_scores).reshape(-1, 3), axis=0)
        enhanced_avg = np.mean(np.array(enhanced_scores).reshape(-1, 3), axis=0)
        satzilla_avg = np.mean(np.array(satzilla_scores).reshape(-1, 3), axis=0)
        
        # Set up the angles for the radar chart
        angles = np.linspace(0, 2*np.pi, num_metrics, endpoint=False)
        
        # Close the plot by appending the first value
        simple_avg = np.concatenate((simple_avg, [simple_avg[0]]))
        meta_avg = np.concatenate((meta_avg, [meta_avg[0]]))
        enhanced_avg = np.concatenate((enhanced_avg, [enhanced_avg[0]]))
        satzilla_avg = np.concatenate((satzilla_avg, [satzilla_avg[0]]))
        angles = np.concatenate((angles, [angles[0]]))
        
        # Create the plot
        fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(projection='polar'))
        
        # Plot data
        ax.plot(angles, simple_avg, 'o-', linewidth=2, label='Simple')
        ax.fill(angles, simple_avg, alpha=0.1)
        ax.plot(angles, meta_avg, 'o-', linewidth=2, label='Meta')
        ax.fill(angles, meta_avg, alpha=0.1)
        ax.plot(angles, enhanced_avg, 'o-', linewidth=2, label='Enhanced')
        ax.fill(angles, enhanced_avg, alpha=0.1)
        ax.plot(angles, satzilla_avg, 'o-', linewidth=2, label='SATzilla')
        ax.fill(angles, satzilla_avg, alpha=0.1)
        
        # Set labels
        ax.set_thetagrids(np.degrees(angles[:-1]), metrics)
        
        ax.set_title("Overall Performance Comparison")
        ax.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
        
        if save:
            fig.savefig(f"{self.export_dir}/radar_chart.png", dpi=300, bbox_inches='tight')
            print(f"Saved radar chart to {self.export_dir}/radar_chart.png")
        
        return fig

    def plot_boxplots(self, metric: str = "best_fitness_avg", save: bool = True) -> Figure:
        """Create box plots showing the distribution of results"""
        fig, ax = plt.subplots(figsize=(12, 6))
        
        data = []
        labels = []
        for func_name in self.function_names:
            if f"baseline_{metric}_all" in self.results[func_name]:
                baseline_data = self.results[func_name][f"baseline_{metric}_all"]
                meta_data = self.results[func_name][f"meta_{metric}_all"]
                data.extend([baseline_data, meta_data])
                labels.extend([f"{func_name}\nBaseline", f"{func_name}\nMeta"])
        
        if data:
            bp = ax.boxplot(data, labels=labels, patch_artist=True)
            
            # Customize colors
            colors = ['lightblue', 'lightcoral'] * len(self.function_names)
            for patch, color in zip(bp['boxes'], colors):
                patch.set_facecolor(color)
            
            plt.xticks(rotation=45, ha='right')
            plt.ylabel(metric.replace("_", " ").title())
            plt.title(f"Distribution of {metric.replace('_', ' ').title()}")
            
            # Add legend
            from matplotlib.patches import Patch
            legend_elements = [
                Patch(facecolor='lightblue', label='Baseline'),
                Patch(facecolor='lightcoral', label='Meta Optimizer')
            ]
            ax.legend(handles=legend_elements)
            
            fig.tight_layout()
            
            if save:
                fig.savefig(f"{self.export_dir}/boxplot_{metric}.png", dpi=300, bbox_inches='tight')
                print(f"Saved box plot to {self.export_dir}/boxplot_{metric}.png")
        
        return fig

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