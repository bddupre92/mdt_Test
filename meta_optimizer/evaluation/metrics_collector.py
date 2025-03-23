"""
Metrics collection and analysis for optimizer comparison.

This module provides tools for collecting, storing, and analyzing
performance metrics from optimization runs to facilitate comparison
of different optimization algorithms.
"""

from collections import defaultdict
import numpy as np
import os
import json
import logging
from typing import Dict, List, Any, Optional, Tuple, Union
import matplotlib.pyplot as plt
import seaborn as sns

class MetricsCollector:
    """Collects, stores, and computes metrics for optimizer comparisons."""
    
    def __init__(self):
        """Initialize an empty metrics collector."""
        self.metrics = defaultdict(lambda: defaultdict(list))
        self.logger = logging.getLogger(self.__class__.__name__)
        
    def add_run_result(self, optimizer_name: str, problem_name: str, 
                       best_score: float, convergence_time: float,
                       evaluations: int, success: bool = False,
                       convergence_curve: Optional[List[float]] = None):
        """
        Add a single optimization run result.
        
        Parameters
        ----------
        optimizer_name : str
            Name of the optimizer used
        problem_name : str
            Name of the test problem
        best_score : float
            Best score achieved by the optimizer
        convergence_time : float
            Time taken to converge (in seconds)
        evaluations : int
            Number of function evaluations used
        success : bool
            Whether the run was successful (reached target value)
        convergence_curve : Optional[List[float]]
            List of best scores throughout the optimization process
        """
        self.metrics[optimizer_name][problem_name].append({
            'best_score': best_score,
            'convergence_time': convergence_time,
            'evaluations': evaluations,
            'success': success,
            'convergence_curve': convergence_curve
        })
        
    def calculate_statistics(self) -> Dict[str, Dict[str, Dict[str, float]]]:
        """
        Calculate summary statistics for all collected metrics.
        
        Returns
        -------
        Dict[str, Dict[str, Dict[str, float]]]
            Nested dictionary with statistics organized by optimizer and problem
        """
        stats = {}
        for optimizer, problems in self.metrics.items():
            stats[optimizer] = {}
            for problem, runs in problems.items():
                if not runs:
                    continue
                    
                scores = [run['best_score'] for run in runs]
                times = [run['convergence_time'] for run in runs]
                evals = [run['evaluations'] for run in runs]
                success_rate = sum(run['success'] for run in runs) / len(runs) if runs else 0
                
                stats[optimizer][problem] = {
                    'mean_score': float(np.mean(scores)),
                    'median_score': float(np.median(scores)),
                    'std_score': float(np.std(scores)),
                    'min_score': float(np.min(scores)),
                    'max_score': float(np.max(scores)),
                    'mean_time': float(np.mean(times)),
                    'mean_evals': float(np.mean(evals)),
                    'success_rate': float(success_rate),
                    'run_count': len(runs)
                }
        return stats
    
    def calculate_rankings(self) -> Dict[str, Dict[str, int]]:
        """
        Calculate rankings of optimizers for each problem.
        
        Returns
        -------
        Dict[str, Dict[str, int]]
            Dictionary with rankings by problem and metric
        """
        stats = self.calculate_statistics()
        
        # Get all optimizers and problems
        optimizers = list(stats.keys())
        problems = set()
        for opt in stats:
            problems.update(stats[opt].keys())
        
        rankings = {problem: {} for problem in problems}
        
        # Calculate rankings for each problem and metric
        for problem in problems:
            # Extract values for each optimizer
            scores = {}
            times = {}
            evals = {}
            
            for optimizer in optimizers:
                if problem in stats[optimizer]:
                    scores[optimizer] = stats[optimizer][problem]['mean_score']
                    times[optimizer] = stats[optimizer][problem]['mean_time']
                    evals[optimizer] = stats[optimizer][problem]['mean_evals']
            
            # Rank by score (lower is better)
            score_ranking = sorted([(opt, val) for opt, val in scores.items()], key=lambda x: x[1])
            for i, (opt, _) in enumerate(score_ranking):
                rankings[problem][f"{opt}_score_rank"] = i + 1
                
            # Rank by time (lower is better)
            time_ranking = sorted([(opt, val) for opt, val in times.items()], key=lambda x: x[1])
            for i, (opt, _) in enumerate(time_ranking):
                rankings[problem][f"{opt}_time_rank"] = i + 1
                
            # Rank by evaluations (lower is better)
            eval_ranking = sorted([(opt, val) for opt, val in evals.items()], key=lambda x: x[1])
            for i, (opt, _) in enumerate(eval_ranking):
                rankings[problem][f"{opt}_evals_rank"] = i + 1
        
        return rankings
    
    def to_dataframe(self):
        """
        Convert metrics to a pandas DataFrame for easier analysis.
        
        Returns
        -------
        pd.DataFrame
            DataFrame with metrics data
        """
        try:
            import pandas as pd
            
            data = []
            for optimizer, problems in self.metrics.items():
                for problem, runs in problems.items():
                    for run_idx, run in enumerate(runs):
                        entry = {
                            'optimizer': optimizer,
                            'problem': problem,
                            'run': run_idx,
                            'best_score': run['best_score'],
                            'convergence_time': run['convergence_time'],
                            'evaluations': run['evaluations'],
                            'success': run['success']
                        }
                        data.append(entry)
            
            return pd.DataFrame(data)
        except ImportError:
            self.logger.warning("Pandas not available, cannot convert to DataFrame")
            return None
    
    def save_to_csv(self, filename: str) -> None:
        """
        Save metrics to CSV file.
        
        Parameters
        ----------
        filename : str
            Path to save the CSV file
        """
        df = self.to_dataframe()
        if df is not None:
            df.to_csv(filename, index=False)
            self.logger.info(f"Metrics saved to {filename}")
        else:
            self.logger.error("Could not save metrics to CSV: pandas not available")
    
    def save_to_json(self, filename: str) -> None:
        """
        Save metrics to JSON file.
        
        Parameters
        ----------
        filename : str
            Path to save the JSON file
        """
        # Convert defaultdict to regular dict for JSON serialization
        metrics_dict = {}
        for optimizer, problems in self.metrics.items():
            metrics_dict[optimizer] = {}
            for problem, runs in problems.items():
                # Convert any numpy values to Python native types
                serializable_runs = []
                for run in runs:
                    serializable_run = {
                        'best_score': float(run['best_score']),
                        'convergence_time': float(run['convergence_time']),
                        'evaluations': int(run['evaluations']),
                        'success': bool(run['success'])
                    }
                    # Only include convergence curve if it exists
                    if run.get('convergence_curve') is not None:
                        serializable_run['convergence_curve'] = [float(x) for x in run['convergence_curve']]
                    
                    serializable_runs.append(serializable_run)
                
                metrics_dict[optimizer][problem] = serializable_runs
        
        # Calculate and include statistics
        stats = self.calculate_statistics()
        
        # Create output structure
        output = {
            'raw_metrics': metrics_dict,
            'statistics': stats
        }
        
        # Save to file
        with open(filename, 'w') as f:
            json.dump(output, f, indent=2)
        
        self.logger.info(f"Metrics saved to {filename}")
    
    def plot_convergence_comparison(self, 
                                    problem_name: Optional[str] = None,
                                    optimizers: Optional[List[str]] = None,
                                    max_iterations: Optional[int] = None,
                                    log_scale: bool = False,
                                    save_path: Optional[str] = None,
                                    title: Optional[str] = None,
                                    figsize: Tuple[int, int] = (10, 6)) -> plt.Figure:
        """
        Plot convergence comparison for selected optimizers on a specific problem.
        
        Parameters
        ----------
        problem_name : Optional[str]
            Name of the problem to plot. If None, plots for all problems.
        optimizers : Optional[List[str]]
            List of optimizers to include. If None, includes all.
        max_iterations : Optional[int]
            Maximum number of iterations to show. If None, shows all.
        log_scale : bool
            Whether to use log scale for y-axis.
        save_path : Optional[str]
            Path to save the plot. If None, doesn't save.
        title : Optional[str]
            Custom title for the plot.
        figsize : Tuple[int, int]
            Figure size in inches.
            
        Returns
        -------
        plt.Figure
            The generated figure
        """
        # Set up plot
        fig, ax = plt.subplots(figsize=figsize)
        
        # Determine which problem to plot
        if problem_name is None:
            # Get the first problem that has convergence curves
            for opt, problems in self.metrics.items():
                for prob, runs in problems.items():
                    if runs and any('convergence_curve' in run and run['convergence_curve'] for run in runs):
                        problem_name = prob
                        break
                if problem_name:
                    break
            
            if problem_name is None:
                self.logger.warning("No convergence curves found in metrics")
                return fig
        
        # Determine which optimizers to include
        if optimizers is None:
            optimizers = list(self.metrics.keys())
        
        # Prepare color cycle
        colors = plt.cm.tab10.colors
        color_idx = 0
        
        # Track y limits to ensure all data is visible
        y_min, y_max = float('inf'), float('-inf')
        
        # Plot convergence curves for each optimizer
        for optimizer in optimizers:
            if optimizer not in self.metrics or problem_name not in self.metrics[optimizer]:
                continue
                
            runs = self.metrics[optimizer][problem_name]
            curves = [run.get('convergence_curve') for run in runs if run.get('convergence_curve')]
            
            if not curves or all(not curve for curve in curves):
                continue
            
            # Clean up curves - ensure they're all valid numbers
            valid_curves = []
            for curve in curves:
                if curve:
                    # Replace any NaN or inf values with the last valid value
                    clean_curve = []
                    last_valid = None
                    for value in curve:
                        if np.isfinite(value):
                            clean_curve.append(value)
                            last_valid = value
                        elif last_valid is not None:
                            clean_curve.append(last_valid)
                        else:
                            clean_curve.append(1e10)  # Some large number as fallback
                    valid_curves.append(clean_curve)
            
            if not valid_curves:
                continue
                
            # Determine the maximum length of convergence curves
            max_len = max(len(curve) for curve in valid_curves)
            if max_iterations is not None:
                max_len = min(max_len, max_iterations)
            
            if max_len == 0:
                continue
            
            # Calculate mean and std of convergence curves
            mean_curve = np.zeros(max_len)
            std_curve = np.zeros(max_len)
            
            for i in range(max_len):
                values = [curve[i] if i < len(curve) else curve[-1] for curve in valid_curves]
                mean_curve[i] = np.mean(values)
                std_curve[i] = np.std(values)
            
            # Update y-axis limits
            y_min = min(y_min, np.min(mean_curve - std_curve))
            y_max = max(y_max, np.max(mean_curve + std_curve))
            
            # Plot mean convergence curve with std shading
            x = np.arange(max_len)
            color = colors[color_idx % len(colors)]
            ax.plot(x, mean_curve, label=optimizer, color=color)
            ax.fill_between(x, mean_curve - std_curve, mean_curve + std_curve, alpha=0.2, color=color)
            
            color_idx += 1
        
        # Ensure y limits are set properly
        if np.isfinite(y_min) and np.isfinite(y_max):
            if y_min == y_max:  # Edge case: all values are the same
                y_min *= 0.9
                y_max *= 1.1
            margin = (y_max - y_min) * 0.1
            ax.set_ylim(y_min - margin, y_max + margin)
        
        # Configure plot
        ax.set_xlabel('Iterations')
        ax.set_ylabel('Objective Value')
        if log_scale:
            if y_min <= 0:
                self.logger.warning("Cannot use log scale with non-positive values, switching to linear scale")
            else:
                ax.set_yscale('log')
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        # Set title
        if title:
            ax.set_title(title)
        else:
            ax.set_title(f'Convergence Comparison - {problem_name}')
        
        # Save if requested
        if save_path:
            plt.tight_layout()
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            self.logger.info(f"Convergence plot saved to {save_path}")
        
        return fig
    
    def plot_performance_boxplot(self,
                                 metric: str = 'best_score',
                                 save_path: Optional[str] = None,
                                 title: Optional[str] = None,
                                 figsize: Tuple[int, int] = (12, 8)) -> plt.Figure:
        """
        Create boxplot of performance metrics across optimizers and problems.
        
        Parameters
        ----------
        metric : str
            Metric to plot ('best_score', 'convergence_time', or 'evaluations')
        save_path : Optional[str]
            Path to save the plot. If None, doesn't save.
        title : Optional[str]
            Custom title for the plot.
        figsize : Tuple[int, int]
            Figure size in inches.
            
        Returns
        -------
        plt.Figure
            The generated figure
        """
        # Convert to DataFrame for easier plotting
        df = self.to_dataframe()
        if df is None:
            self.logger.warning("Pandas not available, cannot create boxplot")
            return None
        
        # Set up plot
        fig, ax = plt.subplots(figsize=figsize)
        
        # Create boxplot
        sns.boxplot(x='problem', y=metric, hue='optimizer', data=df, ax=ax)
        
        # Configure plot
        ax.set_xlabel('Problem')
        
        # Set appropriate y-label based on metric
        if metric == 'best_score':
            ax.set_ylabel('Best Score')
        elif metric == 'convergence_time':
            ax.set_ylabel('Convergence Time (s)')
        elif metric == 'evaluations':
            ax.set_ylabel('Function Evaluations')
        else:
            ax.set_ylabel(metric.replace('_', ' ').title())
        
        # Set title
        if title:
            ax.set_title(title)
        else:
            ax.set_title(f'Performance Comparison - {metric.replace("_", " ").title()}')
        
        # Rotate x-tick labels for better readability
        plt.xticks(rotation=45)
        
        # Add grid
        ax.grid(True, axis='y', alpha=0.3)
        
        # Adjust layout
        plt.tight_layout()
        
        # Save if requested
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            self.logger.info(f"Boxplot saved to {save_path}")
        
        return fig
    
    def plot_radar_chart(self,
                         problem_name: Optional[str] = None,
                         metrics: Optional[List[str]] = None,
                         save_path: Optional[str] = None,
                         title: Optional[str] = None,
                         figsize: Tuple[int, int] = (10, 10)) -> plt.Figure:
        """
        Create a radar chart comparing optimizers across different metrics.
        
        Parameters
        ----------
        problem_name : Optional[str]
            Name of the problem to plot. If None, compares across all problems.
        metrics : Optional[List[str]]
            List of metrics to include. If None, uses default metrics.
        save_path : Optional[str]
            Path to save the plot. If None, doesn't save.
        title : Optional[str]
            Custom title for the plot.
        figsize : Tuple[int, int]
            Figure size in inches.
            
        Returns
        -------
        plt.Figure
            The generated figure
        """
        # Gather statistics
        stats = self.calculate_statistics()
        
        # Determine which metrics to include
        if metrics is None:
            metrics = ['mean_score', 'mean_time', 'success_rate', 'mean_evals']
        
        # Determine which optimizers to include
        optimizers = list(stats.keys())
        
        # Set up color cycle
        colors = plt.cm.tab10.colors
        
        # Handle all problems if no specific problem is provided
        if problem_name is None:
            fig = plt.figure(figsize=figsize)
            ax = fig.add_subplot(111, polar=True)
            
            # Compute average metrics across all problems
            all_metrics = {opt: {} for opt in optimizers}
            
            for opt in optimizers:
                problems = list(stats[opt].keys())
                
                for metric in metrics:
                    all_metrics[opt][metric] = 0
                    count = 0
                    
                    for prob in problems:
                        if metric in stats[opt][prob]:
                            # Skip non-finite values
                            value = stats[opt][prob][metric]
                            if np.isfinite(value):
                                all_metrics[opt][metric] += value
                                count += 1
                    
                    if count > 0:
                        all_metrics[opt][metric] /= count
                    else:
                        all_metrics[opt][metric] = 0  # Default if no valid data
            
            # Normalize metrics to 0-1 range for radar chart
            normalized_metrics = self._normalize_metrics_for_radar(all_metrics, metrics)
            
            # Plot radar chart
            angles = np.linspace(0, 2*np.pi, len(metrics), endpoint=False).tolist()
            angles += angles[:1]  # Close the loop
            
            # Set up axis
            ax.set_theta_offset(np.pi / 2)
            ax.set_theta_direction(-1)
            ax.set_rlabel_position(0)
            
            # Add labels
            plt.xticks(angles[:-1], metrics)
            
            # Plot each optimizer
            for i, opt in enumerate(optimizers):
                color = colors[i % len(colors)]
                values = [normalized_metrics[opt][metric] for metric in metrics]
                values += values[:1]  # Close the loop
                
                # Handle case where values are all zero
                if all(v == 0 for v in values[:-1]):
                    continue
                
                ax.plot(angles, values, color=color, linewidth=2, label=opt)
                ax.fill(angles, values, color=color, alpha=0.25)
            
            # Add legend and title
            ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
            if title:
                ax.set_title(title)
            else:
                ax.set_title('Optimizer Comparison - All Problems')
                
            # Add rings
            ax.set_rticks([0.2, 0.4, 0.6, 0.8, 1.0])
            ax.grid(True)
            
            # Save if requested
            if save_path:
                plt.tight_layout()
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                self.logger.info(f"Radar chart saved to {save_path}")
                
            return fig
            
        # Handle specific problem
        else:
            if problem_name not in next(iter(stats.values())):
                self.logger.warning(f"Problem {problem_name} not found in metrics")
                fig = plt.figure(figsize=figsize)
                plt.text(0.5, 0.5, f"No data for problem: {problem_name}", 
                         ha='center', va='center')
                plt.axis('off')
                return fig
            
            fig = plt.figure(figsize=figsize)
            ax = fig.add_subplot(111, polar=True)
            
            # Extract metrics for this problem
            problem_metrics = {opt: {} for opt in optimizers}
            
            for opt in optimizers:
                if problem_name not in stats[opt]:
                    # Skip optimizers without data for this problem
                    continue
                    
                for metric in metrics:
                    if metric in stats[opt][problem_name]:
                        problem_metrics[opt][metric] = stats[opt][problem_name][metric]
                    else:
                        problem_metrics[opt][metric] = 0  # Default if metric not found
            
            # Normalize metrics to 0-1 range for radar chart
            normalized_metrics = self._normalize_metrics_for_radar(problem_metrics, metrics)
            
            # Plot radar chart
            angles = np.linspace(0, 2*np.pi, len(metrics), endpoint=False).tolist()
            angles += angles[:1]  # Close the loop
            
            # Set up axis
            ax.set_theta_offset(np.pi / 2)
            ax.set_theta_direction(-1)
            ax.set_rlabel_position(0)
            
            # Add labels
            plt.xticks(angles[:-1], metrics)
            
            # Plot each optimizer
            for i, opt in enumerate(optimizers):
                if problem_name not in stats[opt]:
                    continue
                    
                color = colors[i % len(colors)]
                
                if opt not in normalized_metrics:
                    continue
                    
                # Skip if we don't have all metrics
                if not all(metric in normalized_metrics[opt] for metric in metrics):
                    continue
                
                values = [normalized_metrics[opt][metric] for metric in metrics]
                values += values[:1]  # Close the loop
                
                # Handle case where values are all zero
                if all(v == 0 for v in values[:-1]):
                    continue
                
                ax.plot(angles, values, color=color, linewidth=2, label=opt)
                ax.fill(angles, values, color=color, alpha=0.25)
            
            # Add legend and title
            ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
            if title:
                ax.set_title(title)
            else:
                ax.set_title(f'Optimizer Comparison - {problem_name}')
                
            # Add rings
            ax.set_rticks([0.2, 0.4, 0.6, 0.8, 1.0])
            ax.grid(True)
            
            # Save if requested
            if save_path:
                plt.tight_layout()
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                self.logger.info(f"Radar chart saved to {save_path}")
                
            return fig
    
    def _normalize_metrics_for_radar(self, metrics_dict, metric_names):
        """
        Normalize metrics to 0-1 range for radar chart.
        For metrics where lower is better (time, evaluations), the values are inverted.
        
        Parameters
        ----------
        metrics_dict : Dict[str, Dict[str, float]]
            Dictionary of metrics by optimizer
        metric_names : List[str]
            List of metric names to normalize
            
        Returns
        -------
        Dict[str, Dict[str, float]]
            Dictionary of normalized metrics
        """
        # Initialize normalized dict
        normalized = {opt: {} for opt in metrics_dict.keys()}
        
        # Determine min/max for each metric
        min_vals = {metric: float('inf') for metric in metric_names}
        max_vals = {metric: float('-inf') for metric in metric_names}
        
        for opt, metrics in metrics_dict.items():
            for metric in metric_names:
                if metric in metrics and np.isfinite(metrics[metric]):
                    min_vals[metric] = min(min_vals[metric], metrics[metric])
                    max_vals[metric] = max(max_vals[metric], metrics[metric])
        
        # Normalize each metric
        for opt, metrics in metrics_dict.items():
            for metric in metric_names:
                if metric in metrics and np.isfinite(metrics[metric]):
                    value = metrics[metric]
                    
                    # Handle case where min == max
                    if min_vals[metric] == max_vals[metric]:
                        normalized[opt][metric] = 1.0 if value > 0 else 0.0
                    else:
                        # For 'best_score', 'mean_time', 'mean_evals' - lower is better
                        if metric in ['best_score', 'mean_score', 'mean_time', 'mean_evals']:
                            # Invert so that lower values -> higher on radar chart
                            normalized[opt][metric] = 1.0 - (value - min_vals[metric]) / (max_vals[metric] - min_vals[metric])
                        # For 'success_rate' - higher is better
                        else:
                            normalized[opt][metric] = (value - min_vals[metric]) / (max_vals[metric] - min_vals[metric])
                else:
                    normalized[opt][metric] = 0.0
        
        return normalized
    
    def generate_performance_report(self, output_dir: str) -> None:
        """
        Generate comprehensive performance report with visualizations.
        
        Parameters
        ----------
        output_dir : str
            Directory to save the report files
        """
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Save raw metrics
        self.save_to_json(os.path.join(output_dir, 'metrics.json'))
        
        try:
            self.save_to_csv(os.path.join(output_dir, 'metrics.csv'))
        except Exception as e:
            self.logger.warning(f"Could not save CSV: {e}")
        
        # Generate plots
        try:
            # 1. Convergence plots for each problem
            problems = set()
            for optimizer, problem_dict in self.metrics.items():
                problems.update(problem_dict.keys())
            
            for problem in problems:
                self.plot_convergence_comparison(
                    problem_name=problem,
                    save_path=os.path.join(output_dir, f'convergence_{problem}.png')
                )
                
            # 2. Box plots for different metrics
            for metric in ['best_score', 'convergence_time', 'evaluations']:
                self.plot_performance_boxplot(
                    metric=metric,
                    save_path=os.path.join(output_dir, f'boxplot_{metric}.png')
                )
            
            # 3. Radar charts
            self.plot_radar_chart(
                save_path=os.path.join(output_dir, 'radar_all_problems.png')
            )
            
            for problem in problems:
                self.plot_radar_chart(
                    problem_name=problem,
                    save_path=os.path.join(output_dir, f'radar_{problem}.png')
                )
                
        except Exception as e:
            self.logger.error(f"Error generating visualization: {e}")
        
        # Generate summary text report
        try:
            stats = self.calculate_statistics()
            with open(os.path.join(output_dir, 'performance_summary.txt'), 'w') as f:
                f.write("Performance Evaluation Summary\n")
                f.write("=============================\n\n")
                
                # Overall best performer
                f.write("OVERALL PERFORMANCE\n")
                f.write("-----------------\n")
                
                overall_scores = {}
                for optimizer in stats:
                    problem_scores = []
                    for problem in stats[optimizer]:
                        problem_scores.append(stats[optimizer][problem]['mean_score'])
                    
                    if problem_scores:
                        overall_scores[optimizer] = np.mean(problem_scores)
                
                if overall_scores:
                    best_optimizer = min(overall_scores.items(), key=lambda x: x[1])[0]
                    f.write(f"Best overall optimizer: {best_optimizer}\n\n")
                
                # Problem-specific results
                f.write("PROBLEM-SPECIFIC RESULTS\n")
                f.write("------------------------\n")
                
                problems = set()
                for optimizer in stats:
                    problems.update(stats[optimizer].keys())
                
                for problem in sorted(problems):
                    f.write(f"\nProblem: {problem}\n")
                    
                    # Sort optimizers by score
                    problem_results = []
                    for optimizer in stats:
                        if problem in stats[optimizer]:
                            problem_results.append((
                                optimizer,
                                stats[optimizer][problem]['mean_score'],
                                stats[optimizer][problem]['mean_time'],
                                stats[optimizer][problem]['mean_evals']
                            ))
                    
                    problem_results.sort(key=lambda x: x[1])  # Sort by score
                    
                    for i, (opt, score, time, evals) in enumerate(problem_results):
                        f.write(f"  {i+1}. {opt}: Score = {score:.6f}, Time = {time:.2f}s, Evals = {int(evals)}\n")
                
                # Generate detailed statistics
                f.write("\n\nDETAILED STATISTICS\n")
                f.write("------------------\n")
                
                for optimizer in sorted(stats.keys()):
                    f.write(f"\n{optimizer}:\n")
                    for problem in sorted(stats[optimizer].keys()):
                        f.write(f"  {problem}:\n")
                        for metric, value in sorted(stats[optimizer][problem].items()):
                            f.write(f"    {metric}: {value}\n")
                            
                f.write("\n\nGenerated visualizations saved to:\n")
                f.write(f"{os.path.abspath(output_dir)}\n")
                
            self.logger.info(f"Performance summary saved to {os.path.join(output_dir, 'performance_summary.txt')}")
                
        except Exception as e:
            self.logger.error(f"Error generating summary report: {e}") 