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
            'context': context or {}
        }
        self.selection_history.append(entry)
        self.problem_types.add(problem_type)
        
        # Track optimizer performance
        if optimizer not in self.performance_history:
            self.performance_history[optimizer] = []
        
        self.performance_history[optimizer].append({
            'iteration': iteration,
            'score': score,
            'problem_type': problem_type
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
        Plot the distribution of algorithms selected for each problem type.
        
        Args:
            title: Plot title
            save: Whether to save the plot
            filename: Filename to save the plot
            
        Returns:
            Matplotlib figure
        """
        if not self.selection_history:
            self.logger.warning("No selection history to visualize")
            return None
        
        # Prepare data
        df = pd.DataFrame(self.selection_history)
        
        # Count selections per optimizer and problem type
        problem_optimizer_counts = df.groupby(['problem_type', 'optimizer']).size().unstack(fill_value=0)
        
        # Create figure
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Plot heatmap
        sns.heatmap(problem_optimizer_counts, annot=True, fmt='d', cmap='YlGnBu', ax=ax)
        
        # Add chart labels
        ax.set_title(title)
        ax.set_xlabel('Optimizer')
        ax.set_ylabel('Problem Type')
        
        plt.tight_layout()
        
        # Save if requested
        if save and self.save_dir:
            save_path = os.path.join(self.save_dir, filename)
            fig.savefig(save_path)
            self.logger.info(f"Saved problem distribution plot to {save_path}")
        
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
                    'problem_type': entry['problem_type']
                })
        
        df = pd.DataFrame(data)
        
        # Create figure
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Plot boxplot
        sns.boxplot(x='optimizer', y='score', hue='problem_type', data=df, ax=ax)
        
        # If scores are very small, use log scale
        if df['score'].median() < 0.01:
            ax.set_yscale('log')
        
        # Add chart labels
        ax.set_title(title)
        ax.set_xlabel('Optimizer')
        ax.set_ylabel('Score (lower is better)')
        
        # Rotate x-labels if many optimizers
        if len(df['optimizer'].unique()) > 4:
            plt.xticks(rotation=45, ha='right')
        
        plt.tight_layout()
        
        # Save if requested
        if save and self.save_dir:
            save_path = os.path.join(self.save_dir, filename)
            fig.savefig(save_path)
            self.logger.info(f"Saved performance comparison plot to {save_path}")
        
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
        Create a comprehensive dashboard with multiple visualization panels.
        
        Args:
            title: Title for the dashboard
            save: Whether to save the dashboard
            filename: Filename to save the dashboard
            
        Returns:
            Matplotlib figure
        """
        if not self.selection_history:
            self.logger.warning("No selection history to visualize")
            
            # Create an empty plot with a message
            plt.figure(figsize=(16, 10))
            plt.text(0.5, 0.5, "No algorithm selection data available for dashboard", 
                    horizontalalignment='center', verticalalignment='center',
                    transform=plt.gca().transAxes, fontsize=16)
            plt.title(title)
            
            if save and self.save_dir:
                save_path = Path(self.save_dir) / filename
                save_path.parent.mkdir(parents=True, exist_ok=True)
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                print(f"Saved empty dashboard to {save_path}")
            
            plt.close()
            return None
        
        # Create figure
        fig = plt.figure(figsize=(20, 16))
        gs = gridspec.GridSpec(3, 2, figure=fig)
        
        # Panel 1: Selection frequency
        ax1 = fig.add_subplot(gs[0, 0])
        optimizer_counts = defaultdict(int)
        for entry in self.selection_history:
            optimizer_counts[entry['optimizer']] += 1
        
        optimizers = sorted(optimizer_counts.keys(), key=lambda x: optimizer_counts[x], reverse=True)
        frequencies = [optimizer_counts[opt] for opt in optimizers]
        
        bars = ax1.bar(optimizers, frequencies)
        for bar in bars:
            height = bar.get_height()
            ax1.annotate(f'{height}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom')
        
        ax1.set_title('Algorithm Selection Frequency')
        ax1.set_xlabel('Optimizer')
        ax1.set_ylabel('Frequency')
        plt.setp(ax1.get_xticklabels(), rotation=45, ha='right')
        
        # Panel 2: Problem type distribution
        ax2 = fig.add_subplot(gs[0, 1])
        df = pd.DataFrame(self.selection_history)
        if len(self.problem_types) > 1:
            problem_optimizer_counts = df.groupby(['problem_type', 'optimizer']).size().unstack(fill_value=0)
            sns.heatmap(problem_optimizer_counts, annot=True, fmt='d', cmap='YlGnBu', ax=ax2)
            ax2.set_title('Algorithm Selection by Problem Type')
            ax2.set_xlabel('Optimizer')
            ax2.set_ylabel('Problem Type')
        else:
            ax2.text(0.5, 0.5, "Only one problem type available", ha='center', va='center', fontsize=14)
            ax2.set_title('Problem Type Distribution')
            ax2.axis('off')
        
        # Panel 3: Selection timeline
        ax3 = fig.add_subplot(gs[1, :])
        optimizers = df['optimizer'].unique()
        colors = plt.cm.tab10.colors[:len(optimizers)]
        optimizer_colors = {opt: color for opt, color in zip(optimizers, colors)}
        
        y_pos = {opt: i for i, opt in enumerate(optimizers)}
        
        for opt in optimizers:
            opt_data = df[df['optimizer'] == opt]
            ax3.scatter(opt_data['iteration'], [y_pos[opt]] * len(opt_data), 
                      label=opt, color=optimizer_colors[opt], s=50, alpha=0.7)
        
        ax3.set_title('Algorithm Selection Timeline')
        ax3.set_xlabel('Iteration')
        ax3.set_ylabel('Optimizer')
        ax3.set_yticks(list(y_pos.values()))
        ax3.set_yticklabels(list(y_pos.keys()))
        ax3.grid(True, linestyle='--', alpha=0.7)
        ax3.legend(title="Optimizers", bbox_to_anchor=(1.05, 1), loc='upper left')
        
        # Panel 4: Performance comparison
        ax4 = fig.add_subplot(gs[2, 0])
        data = []
        for optimizer, history in self.performance_history.items():
            for entry in history:
                data.append({
                    'optimizer': optimizer,
                    'iteration': entry['iteration'],
                    'score': entry['score'],
                    'problem_type': entry['problem_type']
                })
        
        perf_df = pd.DataFrame(data)
        sns.boxplot(x='optimizer', y='score', data=perf_df, ax=ax4)
        
        if perf_df['score'].median() < 0.01:
            ax4.set_yscale('log')
        
        ax4.set_title('Performance Comparison')
        ax4.set_xlabel('Optimizer')
        ax4.set_ylabel('Score (lower is better)')
        plt.setp(ax4.get_xticklabels(), rotation=45, ha='right')
        
        # Panel 5: Algorithm success rate
        ax5 = fig.add_subplot(gs[2, 1])
        success_data = {}
        for optimizer, history in self.performance_history.items():
            # Calculate improvement rate
            if len(history) > 1:
                start_score = history[0]['score']
                end_score = history[-1]['score']
                improvement = start_score - end_score
                success_data[optimizer] = improvement / start_score if start_score != 0 else 0
            else:
                success_data[optimizer] = 0
        
        # Sort by success rate
        sorted_optimizers = sorted(success_data.keys(), key=lambda x: success_data[x], reverse=True)
        success_rates = [success_data[opt] for opt in sorted_optimizers]
        
        success_bars = ax5.bar(sorted_optimizers, success_rates)
        for bar in success_bars:
            height = bar.get_height()
            ax5.annotate(f'{height:.2f}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom')
        
        ax5.set_title('Algorithm Improvement Rate')
        ax5.set_xlabel('Optimizer')
        ax5.set_ylabel('Relative Improvement')
        plt.setp(ax5.get_xticklabels(), rotation=45, ha='right')
        
        plt.suptitle(title, fontsize=16)
        plt.tight_layout()
        
        # Save if requested
        if save and self.save_dir:
            save_path = os.path.join(self.save_dir, filename)
            fig.savefig(save_path)
            self.logger.info(f"Saved summary dashboard to {save_path}")
        
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
