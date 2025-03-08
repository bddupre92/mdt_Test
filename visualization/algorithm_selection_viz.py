"""
algorithm_selection_viz.py
----------------------
Visualization tools specifically for algorithm selection tracking and analysis.
"""

import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for saving only
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
import seaborn as sns
from pathlib import Path
import os
import sys
from typing import Dict, List, Optional, Any
from collections import defaultdict
import logging
import time

# Import save_plot function
sys.path.append(str(Path(__file__).parent.parent))
from utils.plot_utils import save_plot

class AlgorithmSelectionVisualizer:
    """
    Visualizes the algorithm selection process in meta-optimization.
    """
    
    def __init__(self, save_dir: Optional[str] = None):
        """
        Initialize visualizer.
        
        Args:
            save_dir: Directory to save visualization images
        """
        self.save_dir = save_dir
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
        
        self.logger = logging.getLogger(__name__)
        self.reset()
    
    def reset(self):
        """Reset tracking data."""
        self.selection_history = []
        self.performance_history = {}
        self.problem_types = set()
    
    def record_selection(self, iteration: int, optimizer: str, problem_type: str, 
                         score: float, context: Optional[Dict[str, Any]] = None):
        """
        Record an algorithm selection event.
        
        Args:
            iteration: Current iteration number
            optimizer: Name of selected optimizer
            problem_type: Type of problem being solved
            score: Current best score
            context: Additional context information (phase, features, etc.)
        """
        entry = {
            'iteration': iteration,
            'optimizer': optimizer,
            'problem_type': problem_type,
            'score': score,
            'timestamp': time.time(),  # Add timestamp
            'context': context or {}
        }
        self.selection_history.append(entry)
        self.problem_types.add(problem_type)
        
        # Track optimizer performance
        if optimizer not in self.performance_history:
            self.performance_history[optimizer] = []
        
        # Calculate improvement if there are previous entries
        improvement = 0.0
        relative_improvement = 0.0
        previous_entries = [e for e in self.performance_history[optimizer] 
                          if e['problem_type'] == problem_type]
        
        if previous_entries:
            prev_score = previous_entries[-1]['score']
            improvement = prev_score - score  # Positive means improvement
            
            # Calculate relative improvement safely (avoid division by zero)
            if abs(prev_score) > 1e-10:  # Use small epsilon instead of exact zero
                relative_improvement = improvement / abs(prev_score)
            else:
                # If previous score was effectively zero, use absolute improvement
                relative_improvement = improvement
        
        self.performance_history[optimizer].append({
            'iteration': iteration,
            'score': score,
            'problem_type': problem_type,
            'improvement': improvement,
            'relative_improvement': relative_improvement,
            'timestamp': time.time()
        })
    
    def plot_selection_frequency(self, title: str = "Algorithm Selection Frequency", save: bool = True, filename: str = "algorithm_selection_frequency.png") -> None:
        """
        Generate a bar chart showing the frequency of algorithm selections.
        
        Args:
            title: Title for the plot
            save: Whether to save the plot to a file
            filename: Filename to save the plot (ignored if save is False)
        """
        # Check if there's any data to plot
        if not self.selection_history:
            self.logger.warning("No selection history to visualize")
            # Create an empty plot with a message
            plt.figure(figsize=(10, 6))
            plt.text(0.5, 0.5, "No algorithm selection data available", 
                    horizontalalignment='center', verticalalignment='center',
                    transform=plt.gca().transAxes, fontsize=14)
            plt.title(title)
            
            if save and self.save_dir:
                save_path = Path(self.save_dir) / filename
                save_path.parent.mkdir(parents=True, exist_ok=True)
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                print(f"Saved empty selection frequency plot to {save_path}")
            
            plt.close()
            return
        
        # Count the frequency of each algorithm
        optimizer_counts = {}
        for record in self.selection_history:
            optimizer = record['optimizer']
            optimizer_counts[optimizer] = optimizer_counts.get(optimizer, 0) + 1
        
        # Sort by frequency (descending)
        sorted_optimizers = sorted(optimizer_counts.items(), key=lambda x: x[1], reverse=True)
        
        # Create the plot
        plt.figure(figsize=(10, 6))
        
        # Plot the bars
        bars = plt.bar(
            [opt[0] for opt in sorted_optimizers], 
            [opt[1] for opt in sorted_optimizers],
            color='skyblue'
        )
        
        # Add values on top of bars
        for bar in bars:
            height = bar.get_height()
            plt.text(
                bar.get_x() + bar.get_width()/2.,
                height + 0.1,
                str(int(height)),
                ha='center', 
                va='bottom'
            )
        
        plt.title(title)
        plt.xlabel('Algorithm')
        plt.ylabel('Selection Frequency')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        # Save the plot if requested
        if save and self.save_dir:
            save_path = Path(self.save_dir) / filename
            save_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved selection frequency plot to {save_path}")
        
        plt.close()
    
    def plot_selection_timeline(self, title: str = "Algorithm Selection Timeline", save: bool = True, filename: str = "algorithm_selection_timeline.png"):
        """
        Plot the timeline of algorithm selections.
        
        Args:
            title: Plot title
            save: Whether to save the plot
            filename: Filename to save the plot
            
        Returns:
            Matplotlib figure
        """
        if not self.selection_history:
            self.logger.warning("No selection history to visualize")
            
            # Create an empty plot with a message
            plt.figure(figsize=(12, 6))
            plt.text(0.5, 0.5, "No algorithm selection data available for timeline", 
                    horizontalalignment='center', verticalalignment='center',
                    transform=plt.gca().transAxes, fontsize=14)
            plt.title(title)
            
            if save and self.save_dir:
                save_path = Path(self.save_dir) / filename
                save_path.parent.mkdir(parents=True, exist_ok=True)
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                print(f"Saved empty timeline plot to {save_path}")
            
            plt.close()
            return None
        
        # Prepare data
        df = pd.DataFrame(self.selection_history)
        
        # Create figure
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Plot selection points
        optimizers = df['optimizer'].unique()
        colors = plt.cm.tab10.colors[:len(optimizers)]
        optimizer_colors = {opt: color for opt, color in zip(optimizers, colors)}
        
        # Get all iterations
        iterations = df['iteration'].values
        
        # Get y-positions (jittered by optimizer)
        y_pos = {opt: i for i, opt in enumerate(optimizers)}
        
        # Plot points
        for opt in optimizers:
            opt_data = df[df['optimizer'] == opt]
            ax.scatter(opt_data['iteration'], [y_pos[opt]] * len(opt_data), 
                      label=opt, color=optimizer_colors[opt], s=50, alpha=0.7)
        
        # Add chart labels
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Optimizer')
        ax.set_title(title)
        ax.set_yticks(list(y_pos.values()))
        ax.set_yticklabels(list(y_pos.keys()))
        ax.grid(True, linestyle='--', alpha=0.7)
        
        # Add legend
        ax.legend(title="Optimizers", bbox_to_anchor=(1.05, 1), loc='upper left')
        
        plt.tight_layout()
        
        # Save if requested
        if save and self.save_dir:
            save_path = os.path.join(self.save_dir, filename)
            fig.savefig(save_path)
            self.logger.info(f"Saved selection timeline plot to {save_path}")
        
        return fig
    
    def plot_problem_distribution(self, title: str = "Algorithm Selection by Problem Type", save: bool = True, filename: str = "algorithm_selection_by_problem.png"):
        """
        Plot the distribution of algorithm selections by problem type.
        
        Args:
            title: Plot title
            save: Whether to save the plot
            filename: Filename to save the plot
            
        Returns:
            Matplotlib figure
        """
        # Check if there's any data to plot
        if not self.selection_history:
            self.logger.warning("No selection history to visualize")
            return None
        
        # Create figure
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Prepare data
        df = pd.DataFrame(self.selection_history)
        
        # Check if more than one problem type
        if len(self.problem_types) > 1:
            try:
                # Create a pivot table for problem types vs. optimizers
                pivot = pd.crosstab(df['problem_type'], df['optimizer'])
                
                # Create a heatmap
                sns.heatmap(pivot, annot=True, fmt='d', cmap='YlGnBu', ax=ax, cbar_kws={'label': 'Count'})
                
                ax.set_title(title)
                ax.set_xlabel('Optimizer')
                ax.set_ylabel('Problem Type')
            except Exception as e:
                self.logger.warning(f"Error creating problem distribution heatmap: {e}")
                # Fallback to simple text display
                ax.text(0.5, 0.5, f"Error creating heatmap: {str(e)}",
                       horizontalalignment='center', verticalalignment='center',
                       transform=ax.transAxes, fontsize=12)
                ax.set_title(title)
                ax.axis('off')
        else:
            # Only one problem type, show count per optimizer
            problem_type = list(self.problem_types)[0]
            
            # Count by optimizer
            optimizer_counts = df.groupby('optimizer').size().reset_index(name='count')
            
            # Sort by count
            optimizer_counts = optimizer_counts.sort_values('count', ascending=False)
            
            # Create bar chart
            sns.barplot(x='optimizer', y='count', data=optimizer_counts, ax=ax)
            
            # Add the count above each bar
            for i, (_, row) in enumerate(optimizer_counts.iterrows()):
                ax.text(i, row['count'] + 0.1, str(row['count']), 
                       ha='center', fontsize=10)
            
            ax.set_title(f"{title}\nProblem Type: {problem_type}")
            ax.set_xlabel('Optimizer')
            ax.set_ylabel('Selection Count')
            
            # Rotate labels if there are many optimizers
            if len(optimizer_counts) > 4:
                plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
        
        plt.tight_layout()
        
        # Save if requested
        if save:
            os.makedirs(os.path.dirname(os.path.abspath(filename)), exist_ok=True)
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            print(f"Saved problem distribution plot to {filename}")
        
        return fig
    
    def plot_performance_comparison(self, title: str = "Optimizer Performance Comparison", save: bool = True, filename: str = "optimizer_performance_comparison.png"):
        """
        Plot performance comparison of different optimizers.
        
        Args:
            title: Plot title
            save: Whether to save the plot
            filename: Filename to save the plot
            
        Returns:
            Matplotlib figure
        """
        if not self.performance_history:
            self.logger.warning("No performance history to visualize")
            return None
        
        # Prepare data
        data = []
        for optimizer, history in self.performance_history.items():
            for entry in history:
                data.append({
                    'optimizer': optimizer,
                    'iteration': entry['iteration'],
                    'score': entry['score'],
                    'problem_type': entry['problem_type'],
                    'improvement': entry.get('improvement', 0),
                    'relative_improvement': entry.get('relative_improvement', 0)
                })
        
        df = pd.DataFrame(data)
        
        # Create figure with multiple subplots
        fig, axs = plt.subplots(2, 2, figsize=(16, 12))
        
        # 1. Boxplot of final scores by optimizer and problem type
        sns.boxplot(x='optimizer', y='score', hue='problem_type', data=df, ax=axs[0, 0])
        axs[0, 0].set_title('Score Distribution by Optimizer')
        axs[0, 0].set_xlabel('Optimizer')
        axs[0, 0].set_ylabel('Score (lower is better)')
        
        # If scores are very small, use log scale
        if df['score'].median() < 0.01 and df['score'].min() > 0:
            axs[0, 0].set_yscale('log')
        
        # 2. Line plot showing score progression by iteration
        # Group by optimizer and iteration, get the min score at each iteration
        progression_df = df.groupby(['optimizer', 'iteration']).agg({'score': 'min'}).reset_index()
        
        # Plot the progression lines for each optimizer
        for optimizer in progression_df['optimizer'].unique():
            subset = progression_df[progression_df['optimizer'] == optimizer]
            axs[0, 1].plot(subset['iteration'], subset['score'], 'o-', label=optimizer)
        
        axs[0, 1].set_title('Score Progression by Iteration')
        axs[0, 1].set_xlabel('Iteration')
        axs[0, 1].set_ylabel('Best Score')
        axs[0, 1].legend()
        
        # 3. Bar chart of improvement rates
        improvement_df = df.groupby('optimizer').agg({
            'improvement': 'mean',
            'relative_improvement': 'mean'
        }).reset_index()
        
        improvement_df = improvement_df.sort_values('relative_improvement', ascending=False)
        sns.barplot(x='optimizer', y='relative_improvement', data=improvement_df, ax=axs[1, 0])
        axs[1, 0].set_title('Average Relative Improvement by Optimizer')
        axs[1, 0].set_xlabel('Optimizer')
        axs[1, 0].set_ylabel('Relative Improvement')
        
        # 4. Success rate (improvement > 0)
        success_df = df.groupby(['optimizer', 'iteration']).agg({
            'improvement': lambda x: (np.array(x) > 0).mean() * 100  # Calculate % of positive improvements
        }).reset_index()
        
        success_summary = success_df.groupby('optimizer').agg({
            'improvement': 'mean'  # Average success rate across iterations
        }).reset_index()
        
        success_summary = success_summary.sort_values('improvement', ascending=False)
        sns.barplot(x='optimizer', y='improvement', data=success_summary, ax=axs[1, 1])
        axs[1, 1].set_title('Success Rate by Optimizer (%)')
        axs[1, 1].set_xlabel('Optimizer')
        axs[1, 1].set_ylabel('Success Rate (%)')
        
        plt.tight_layout()
        plt.suptitle(title, fontsize=16)
        plt.subplots_adjust(top=0.95)
        
        if save:
            os.makedirs(os.path.dirname(os.path.abspath(filename)), exist_ok=True)
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            print(f"Saved performance comparison plot to {filename}")
        
        return fig
    
    def plot_phase_selection(self, title: str = "Algorithm Selection by Phase", save: bool = True, filename: str = "algorithm_selection_by_phase.png"):
        """
        Plot the algorithm selection by optimization phase.
        
        Args:
            title: Plot title
            save: Whether to save the plot
            filename: Filename to save the plot
            
        Returns:
            Matplotlib figure
        """
        # Check if we have phase information
        phase_data = []
        for entry in self.selection_history:
            if 'context' in entry and entry['context'] and 'phase' in entry['context']:
                phase_data.append({
                    'optimizer': entry['optimizer'],
                    'phase': entry['context']['phase'],
                    'score': entry['score']
                })
        
        if not phase_data:
            self.logger.warning("No phase information available in selection history")
            return None
        
        # Create dataframe
        df = pd.DataFrame(phase_data)
        
        # Create figure
        fig = plt.figure(figsize=(15, 10))
        gs = gridspec.GridSpec(2, 2, figure=fig)
        
        # Plot 1: Selection frequency by phase (heatmap)
        ax1 = fig.add_subplot(gs[0, 0])
        phase_counts = df.groupby(['phase', 'optimizer']).size().unstack(fill_value=0)
        sns.heatmap(phase_counts, annot=True, fmt='d', cmap='YlGnBu', ax=ax1)
        ax1.set_title('Algorithm Selection Frequency by Phase')
        ax1.set_xlabel('Optimizer')
        ax1.set_ylabel('Phase')
        
        # Plot 2: Performance by phase and optimizer (boxplot)
        ax2 = fig.add_subplot(gs[0, 1])
        sns.boxplot(x='phase', y='score', hue='optimizer', data=df, ax=ax2)
        ax2.set_title('Performance by Phase and Optimizer')
        ax2.set_xlabel('Phase')
        ax2.set_ylabel('Score')
        
        # Plot 3: Selection proportion by phase (stacked bar)
        ax3 = fig.add_subplot(gs[1, 0])
        phase_props = phase_counts.div(phase_counts.sum(axis=1), axis=0)
        phase_props.plot(kind='bar', stacked=True, ax=ax3)
        ax3.set_title('Selection Proportion by Phase')
        ax3.set_xlabel('Phase')
        ax3.set_ylabel('Proportion')
        
        # Plot 4: Phase transition patterns (line plot)
        ax4 = fig.add_subplot(gs[1, 1])
        # Sort by iteration
        df_sorted = df.sort_values('phase')
        for optimizer in df_sorted['optimizer'].unique():
            opt_data = df_sorted[df_sorted['optimizer'] == optimizer]
            ax4.plot(opt_data['phase'], opt_data['score'], marker='o', label=optimizer)
        ax4.set_title('Score Trajectory Across Phases')
        ax4.set_xlabel('Phase')
        ax4.set_ylabel('Score')
        ax4.legend()
        
        plt.suptitle(title, fontsize=16)
        plt.tight_layout()
        
        # Save if requested
        if save and self.save_dir:
            save_path = os.path.join(self.save_dir, filename)
            fig.savefig(save_path)
            self.logger.info(f"Saved phase selection plot to {save_path}")
        
        return fig
    
    def create_summary_dashboard(self, title: str = "Algorithm Selection Summary", save: bool = True, filename: str = "algorithm_selection_dashboard.png"):
        """
        Create a comprehensive dashboard of algorithm selection data.
        
        Args:
            title: Plot title
            save: Whether to save the plot
            filename: Filename to save the plot
            
        Returns:
            Matplotlib figure
        """
        # Check if there's any data to plot
        if not self.selection_history:
            self.logger.warning("No selection history to visualize in dashboard")
            return None
        
        # Create a larger figure
        fig = plt.figure(figsize=(18, 14))
        
        # Set up a complex grid layout for better organization
        gs = gridspec.GridSpec(3, 3, figure=fig, height_ratios=[1, 1, 1.2])
        
        # ====================== TOP ROW ======================
        # 1. Algorithm Selection Frequency (Bar chart)
        ax1 = fig.add_subplot(gs[0, 0])
        # Count selections by optimizer
        optimizer_counts = {}
        for entry in self.selection_history:
            optimizer = entry['optimizer']
            optimizer_counts[optimizer] = optimizer_counts.get(optimizer, 0) + 1
        
        # Sort by frequency
        sorted_optimizers = sorted(optimizer_counts.items(), key=lambda x: x[1], reverse=True)
        optimizers, counts = zip(*sorted_optimizers) if sorted_optimizers else ([], [])
        
        sns.barplot(x=list(optimizers), y=list(counts), ax=ax1)
        ax1.set_title("Algorithm Selection Frequency")
        ax1.set_xlabel("Optimizer")
        ax1.set_ylabel("Frequency")
        
        # Rotate labels if there are many optimizers
        if len(optimizers) > 4:
            plt.setp(ax1.get_xticklabels(), rotation=45, ha='right')
        
        # 2. Problem Type Distribution (Pie chart or bar chart depending on number of problem types)
        ax2 = fig.add_subplot(gs[0, 1])
        problem_counts = {}
        for entry in self.selection_history:
            problem = entry['problem_type']
            problem_counts[problem] = problem_counts.get(problem, 0) + 1
        
        if len(problem_counts) > 1:
            # Multiple problem types - use pie chart
            problems = list(problem_counts.keys())
            counts = list(problem_counts.values())
            ax2.pie(counts, labels=problems, autopct='%1.1f%%', startangle=90,
                    wedgeprops={'linewidth': 1, 'edgecolor': 'white'})
            ax2.set_title("Problem Type Distribution")
        else:
            # Only one problem type - use text display
            ax2.text(0.5, 0.5, f"Only one problem type available\n{list(problem_counts.keys())[0]}",
                    horizontalalignment='center', verticalalignment='center', 
                    transform=ax2.transAxes, fontsize=12)
            ax2.set_title("Problem Type Distribution")
            ax2.axis('off')
        
        # 3. Algorithm Selection Timeline (Line plot with dots)
        ax3 = fig.add_subplot(gs[0, 2])
        # Group by iteration
        timeline_data = []
        for entry in self.selection_history:
            timeline_data.append({
                'Iteration': entry['iteration'],
                'Optimizer': entry['optimizer']
            })
        
        df_timeline = pd.DataFrame(timeline_data)
        
        # Get unique optimizers and assign colors
        unique_optimizers = df_timeline['Optimizer'].unique()
        colors = plt.cm.tab10(np.linspace(0, 1, len(unique_optimizers)))
        color_map = dict(zip(unique_optimizers, colors))
        
        # Plot each optimizer as a separate line
        for optimizer in unique_optimizers:
            subset = df_timeline[df_timeline['Optimizer'] == optimizer]
            ax3.plot(subset['Iteration'], [unique_optimizers.tolist().index(optimizer)] * len(subset),
                    'o-', label=optimizer)
        
        ax3.set_title('Algorithm Selection Timeline')
        ax3.set_xlabel('Iteration')
        ax3.set_ylabel('Optimizer')
        ax3.set_yticks(range(len(unique_optimizers)))
        ax3.set_yticklabels(unique_optimizers)
        ax3.grid(True, linestyle='--', alpha=0.7)
        
        # ====================== MIDDLE ROW ======================
        # 4. Performance Comparison (Box plot or violin plot)
        ax4 = fig.add_subplot(gs[1, :2])  # Span two columns
        
        # Prepare performance data
        perf_data = []
        for optimizer, history in self.performance_history.items():
            for entry in history:
                perf_data.append({
                    'Optimizer': optimizer,
                    'Score': entry['score'],
                    'Problem Type': entry['problem_type']
                })
        
        df_perf = pd.DataFrame(perf_data)
        
        # Create a more informative boxplot
        sns.boxplot(x='Optimizer', y='Score', hue='Problem Type', data=df_perf, ax=ax4)
        ax4.set_title('Optimizer Performance Comparison')
        ax4.set_xlabel('Optimizer')
        ax4.set_ylabel('Score (lower is better)')
        
        # If scores are very small, use log scale
        if df_perf['Score'].median() < 0.01 and df_perf['Score'].min() > 0:
            ax4.set_yscale('log')
        
        # Rotate labels if there are many optimizers
        if len(unique_optimizers) > 4:
            plt.setp(ax4.get_xticklabels(), rotation=45, ha='right')
        
        # 5. Improvement Rate (Bar chart)
        ax5 = fig.add_subplot(gs[1, 2])
        
        # Calculate improvement for each optimizer
        improvement_data = {}
        
        for optimizer, history in self.performance_history.items():
            # Sort by iteration
            sorted_history = sorted(history, key=lambda x: x['iteration'])
            
            # Calculate improvements between iterations
            improvements = []
            for i in range(1, len(sorted_history)):
                prev_score = sorted_history[i-1]['score']
                curr_score = sorted_history[i]['score']
                # Positive improvement means score got better (lower)
                imp = prev_score - curr_score
                # Calculate relative improvement
                if abs(prev_score) > 1e-10:  # Avoid division by zero
                    relative_imp = imp / abs(prev_score)
                else:
                    relative_imp = imp  # Use absolute if previous was zero
                improvements.append(relative_imp)
            
            # Store average improvement
            if improvements:
                improvement_data[optimizer] = np.mean(improvements)
            else:
                improvement_data[optimizer] = 0
        
        # Sort by improvement rate
        sorted_improvements = sorted(improvement_data.items(), key=lambda x: x[1], reverse=True)
        imp_optimizers, imp_values = zip(*sorted_improvements) if sorted_improvements else ([], [])
        
        # Create color-coded bar chart (green for positive, red for negative)
        colors = ['green' if imp >= 0 else 'red' for imp in imp_values]
        ax5.bar(imp_optimizers, imp_values, color=colors)
        ax5.set_title('Algorithm Improvement Rate')
        ax5.set_xlabel('Optimizer')
        ax5.set_ylabel('Relative Improvement')
        ax5.axhline(y=0, color='black', linestyle='-', alpha=0.3)  # Add zero line
        
        # Rotate labels if needed
        if len(imp_optimizers) > 4:
            plt.setp(ax5.get_xticklabels(), rotation=45, ha='right')
        
        # ====================== BOTTOM ROW ======================
        # 6. Detailed Statistics Table
        ax6 = fig.add_subplot(gs[2, :])  # Span all columns
        ax6.axis('off')  # Turn off axis
        
        # Compute detailed statistics
        stats_data = []
        for optimizer in unique_optimizers:
            # Filter entries for this optimizer
            entries = [e for e in self.selection_history if e['optimizer'] == optimizer]
            
            # Get performance history
            perf_entries = self.performance_history.get(optimizer, [])
            
            # Calculate statistics
            selection_count = len(entries)
            selection_percentage = (selection_count / len(self.selection_history)) * 100 if self.selection_history else 0
            
            best_score = min([e['score'] for e in perf_entries]) if perf_entries else float('nan')
            avg_score = np.mean([e['score'] for e in perf_entries]) if perf_entries else float('nan')
            
            # Calculate success rate (how often this optimizer improved the solution)
            improvements = []
            for i in range(1, len(perf_entries)):
                prev = perf_entries[i-1]['score'] 
                curr = perf_entries[i]['score']
                improvements.append(1 if curr < prev else 0)  # 1 if improved, 0 otherwise
            
            success_rate = (np.mean(improvements) * 100) if improvements else 0
            
            # Calculate average improvement
            avg_improvement = np.mean([prev - curr for prev, curr in zip(
                [e['score'] for e in perf_entries[:-1]], 
                [e['score'] for e in perf_entries[1:]]
            )]) if len(perf_entries) > 1 else 0
            
            # Store statistics
            stats_data.append({
                'Optimizer': optimizer,
                'Selections': selection_count,
                'Selection %': f"{selection_percentage:.1f}%",
                'Best Score': f"{best_score:.6f}",
                'Avg Score': f"{avg_score:.6f}",
                'Success Rate': f"{success_rate:.1f}%",
                'Avg Improvement': f"{avg_improvement:.6f}"
            })
        
        # Create a visually appealing table
        table_data = [[d[col] for col in d.keys()] for d in stats_data]
        table_columns = list(stats_data[0].keys()) if stats_data else []
        
        table = ax6.table(
            cellText=table_data,
            colLabels=table_columns,
            loc='center',
            cellLoc='center',
            colColours=['#c9daf8'] * len(table_columns)
        )
        
        # Style the table
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 1.5)  # Adjust row height
        
        # Add table title
        ax6.set_title('Optimizer Performance Statistics', pad=20, fontsize=14, fontweight='bold')
        
        # Add mega title
        plt.suptitle(title, fontsize=16, y=0.98)
        
        # Adjust layout
        plt.tight_layout()
        plt.subplots_adjust(top=0.92)
        
        # Save if requested
        if save:
            os.makedirs(os.path.dirname(os.path.abspath(filename)), exist_ok=True)
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            print(f"Saved algorithm selection dashboard to {filename}")
        
        return fig

    def interactive_selection_timeline(self, title: str = "Interactive Algorithm Selection Timeline", 
                                      save: bool = True, 
                                      filename: str = "interactive_algorithm_timeline.html") -> None:
        """
        Generate an interactive timeline visualization of algorithm selections using Plotly.
        
        Args:
            title: Title for the plot
            save: Whether to save the plot to an HTML file
            filename: Filename to save the plot (ignored if save is False)
        """
        import plotly.graph_objects as go
        import plotly.express as px
        from plotly.subplots import make_subplots
        
        # Check if there's any data to plot
        if not self.selection_history:
            # Create empty plot with message
            fig = go.Figure()
            fig.add_annotation(
                text="No optimizer selections recorded",
                xref="paper", yref="paper",
                x=0.5, y=0.5,
                showarrow=False,
                font=dict(size=20)
            )
            fig.update_layout(
                title=title,
                xaxis_title="Iteration",
                yaxis_title="Optimizer"
            )
            
            if save and self.save_dir:
                filepath = os.path.join(self.save_dir, filename)
                fig.write_html(filepath)
                self.logger.info(f"Empty selection timeline saved to {filepath}")
            return
        
        # Group data by problem type
        problem_types = list(self.problem_types)
        n_problems = len(problem_types)
        
        # Create subplots - one for each problem type
        fig = make_subplots(
            rows=n_problems, 
            cols=1,
            subplot_titles=[f"Problem: {p_type}" for p_type in problem_types],
            shared_xaxes=True,
            vertical_spacing=0.05
        )
        
        colors = px.colors.qualitative.Plotly  # Colorscale
        
        # Add data for each problem type
        for i, problem_type in enumerate(problem_types):
            problem_data = [entry for entry in self.selection_history if entry['problem_type'] == problem_type]
            
            # Get unique optimizers for this problem
            optimizers = set(entry['optimizer'] for entry in problem_data)
            optimizer_to_idx = {opt: idx for idx, opt in enumerate(sorted(optimizers))}
            
            # Create data for hover info
            hover_text = []
            for entry in problem_data:
                context = entry.get('context', {})
                function_name = context.get('function_name', 'Unknown')
                phase = context.get('phase', 'Unknown')
                score = entry.get('score', 'N/A')
                
                hover_info = f"Optimizer: {entry['optimizer']}<br>" + \
                             f"Function: {function_name}<br>" + \
                             f"Phase: {phase}<br>" + \
                             f"Score: {score:.4e}<br>" + \
                             f"Iteration: {entry['iteration']}"
                hover_text.append(hover_info)
            
            # Add scatter plot for this problem type
            fig.add_trace(
                go.Scatter(
                    x=[entry['iteration'] for entry in problem_data],
                    y=[entry['optimizer'] for entry in problem_data],
                    mode='markers',
                    marker=dict(
                        size=10,
                        color=[colors[optimizer_to_idx[entry['optimizer']] % len(colors)] for entry in problem_data],
                        symbol='circle',
                        line=dict(width=1, color='DarkSlateGrey')
                    ),
                    text=hover_text,
                    hoverinfo='text',
                    name=problem_type
                ),
                row=i+1, col=1
            )
            
            # Add performance trend line
            for optimizer in optimizers:
                opt_data = [entry for entry in problem_data if entry['optimizer'] == optimizer]
                if opt_data:
                    iterations = [entry['iteration'] for entry in opt_data]
                    scores = [entry['score'] for entry in opt_data]
                    
                    # Only add trend if multiple data points exist
                    if len(iterations) > 1:
                        fig.add_trace(
                            go.Scatter(
                                x=iterations,
                                y=[optimizer] * len(iterations),
                                mode='lines',
                                line=dict(
                                    width=1, 
                                    color=colors[optimizer_to_idx[optimizer] % len(colors)],
                                    dash='dot'
                                ),
                                showlegend=False
                            ),
                            row=i+1, col=1
                        )
            
        # Update layout
        fig.update_layout(
            title=dict(
                text=title,
                x=0.5
            ),
            height=300 * n_problems,
            width=900,
            hovermode='closest',
            showlegend=False
        )
        
        # Update axes
        for i in range(n_problems):
            fig.update_yaxes(title_text="Optimizer", row=i+1, col=1)
            if i == n_problems - 1:  # Only add x-axis title to the bottom plot
                fig.update_xaxes(title_text="Iteration", row=i+1, col=1)
        
        # Save figure if requested
        if save and self.save_dir:
            filepath = os.path.join(self.save_dir, filename)
            fig.write_html(filepath)
            self.logger.info(f"Interactive selection timeline saved to {filepath}")

    def interactive_dashboard(self, title: str = "Interactive Algorithm Selection Dashboard",
                             save: bool = True, 
                             filename: str = "interactive_dashboard.html") -> None:
        """
        Create an interactive summary dashboard for algorithm selection using Plotly.
        
        Args:
            title: Title for the dashboard
            save: Whether to save the dashboard
            filename: Filename to save the dashboard (ignored if save is False)
        """
        import plotly.graph_objects as go
        import plotly.express as px
        from plotly.subplots import make_subplots
        
        # Check if there's any data to plot
        if not self.selection_history:
            # Create empty plot with message
            fig = go.Figure()
            fig.add_annotation(
                text="No optimizer selections recorded",
                xref="paper", yref="paper",
                x=0.5, y=0.5,
                showarrow=False,
                font=dict(size=20)
            )
            fig.update_layout(
                title=title,
                height=600,
                width=900
            )
            
            if save and self.save_dir:
                filepath = os.path.join(self.save_dir, filename)
                fig.write_html(filepath)
                self.logger.info(f"Empty dashboard saved to {filepath}")
            return
            
        # Create grid of plots
        fig = make_subplots(
            rows=2, cols=2,
            specs=[[{"type": "bar"}, {"type": "pie"}],
                   [{"type": "scatter"}, {"type": "heatmap"}]],
            subplot_titles=["Selection Frequency", "Problem Type Distribution", 
                           "Performance Over Time", "Optimizer Success Heat Map"],
            vertical_spacing=0.15,
            horizontal_spacing=0.1
        )
        
        # 1. Selection Frequency (Bar chart)
        selection_counts = {}
        for entry in self.selection_history:
            opt = entry['optimizer']
            selection_counts[opt] = selection_counts.get(opt, 0) + 1
        
        # Sort by frequency
        sorted_selections = sorted(selection_counts.items(), key=lambda x: x[1], reverse=True)
        optimizers = [item[0] for item in sorted_selections]
        counts = [item[1] for item in sorted_selections]
        
        fig.add_trace(
            go.Bar(
                x=optimizers,
                y=counts,
                text=counts,
                textposition='auto',
                marker_color=px.colors.qualitative.Plotly[:len(optimizers)],
                hovertemplate='<b>%{x}</b><br>Selected %{y} times<extra></extra>'
            ),
            row=1, col=1
        )
        
        # 2. Problem Type Distribution (Pie chart)
        problem_counts = {}
        for entry in self.selection_history:
            p_type = entry['problem_type']
            problem_counts[p_type] = problem_counts.get(p_type, 0) + 1
            
        fig.add_trace(
            go.Pie(
                labels=list(problem_counts.keys()),
                values=list(problem_counts.values()),
                hole=0.4,
                hoverinfo="label+percent",
                marker=dict(
                    colors=px.colors.sequential.Plasma[:len(problem_counts)]
                )
            ),
            row=1, col=2
        )
        
        # 3. Performance Over Time (Scatter plot)
        optimizers_set = set(entry['optimizer'] for entry in self.selection_history)
        opt_colors = {opt: px.colors.qualitative.Plotly[i % len(px.colors.qualitative.Plotly)] 
                     for i, opt in enumerate(optimizers_set)}
        
        for opt in optimizers_set:
            opt_data = [entry for entry in self.selection_history if entry['optimizer'] == opt]
            iterations = [entry['iteration'] for entry in opt_data]
            scores = [entry['score'] for entry in opt_data]
            
            if iterations and scores:
                fig.add_trace(
                    go.Scatter(
                        x=iterations,
                        y=scores,
                        mode='markers+lines',
                        name=opt,
                        marker=dict(color=opt_colors[opt]),
                        hovertemplate='<b>%{fullData.name}</b><br>Iteration: %{x}<br>Score: %{y:.4e}<extra></extra>'
                    ),
                    row=2, col=1
                )
        
        # 4. Performance Heatmap
        # Create a matrix of problem types vs optimizers
        problem_types = sorted(list(self.problem_types))
        
        # Calculate average performance (score) for each optimizer on each problem
        heatmap_data = []
        for p_type in problem_types:
            row_data = []
            for opt in optimizers:
                # Filter data for this optimizer and problem type
                filtered_data = [
                    entry['score'] for entry in self.selection_history 
                    if entry['optimizer'] == opt and entry['problem_type'] == p_type
                ]
                
                # Calculate average score (or 0 if no data)
                avg_score = np.mean(filtered_data) if filtered_data else np.nan
                row_data.append(avg_score)
            heatmap_data.append(row_data)
            
        # Convert to log scale for better visualization
        heatmap_data = -np.log10(np.array(heatmap_data) + 1e-10)  # Add small epsilon to avoid log(0)
        
        fig.add_trace(
            go.Heatmap(
                z=heatmap_data,
                x=optimizers,
                y=problem_types,
                colorscale='Viridis',
                colorbar=dict(title='Performance<br>(-log10(score))'),
                hovertemplate='<b>Problem: %{y}</b><br>Optimizer: %{x}<br>Performance: %{z:.2f}<extra></extra>'
            ),
            row=2, col=2
        )
            
        # Update layout
        fig.update_layout(
            title=dict(
                text=title,
                x=0.5
            ),
            height=800,
            width=1000,
            showlegend=False,
            template='plotly_white'
        )
        
        # Update axes
        fig.update_xaxes(title_text="Optimizer", row=1, col=1)
        fig.update_yaxes(title_text="Selection Count", row=1, col=1)
        
        fig.update_xaxes(title_text="Iteration", row=2, col=1)
        fig.update_yaxes(title_text="Score (lower is better)", row=2, col=1, type="log")
        
        # Save interactive dashboard
        if save and self.save_dir:
            filepath = os.path.join(self.save_dir, filename)
            fig.write_html(filepath)
            self.logger.info(f"Interactive dashboard saved to {filepath}")
