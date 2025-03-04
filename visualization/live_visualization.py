"""
live_visualization.py
--------------------
Real-time visualization tools for optimization algorithms.
"""

# Set matplotlib backend for non-interactive use
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for saving only
import matplotlib.pyplot as plt

import numpy as np
from matplotlib.animation import FuncAnimation
import threading
import queue
import time
from typing import Dict, List, Any, Optional
import logging
import os
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent))
from utils.plot_utils import save_plot

class LiveOptimizationMonitor:
    """Real-time visualization for optimization progress."""
    
    def __init__(self, max_data_points: int = 1000, auto_show: bool = False, headless: bool = True):
        """
        Initialize the live monitor.
        
        Args:
            max_data_points: Maximum number of data points to store per optimizer
            auto_show: Whether to automatically show the plot when starting monitoring
            headless: Whether to run in headless mode (no display, save only)
        """
        self.data_queue = queue.Queue()
        self.running = False
        self.fig, self.axes = plt.subplots(2, 2, figsize=(14, 10))
        self.optimizer_data = {}  # Store data for each optimizer
        self.animation = None
        self.logger = logging.getLogger(__name__)
        self.max_data_points = max_data_points
        self.auto_show = auto_show
        self.headless = headless
        
    def start_monitoring(self):
        """Start the visualization thread."""
        if self.running:
            return
            
        self.running = True
        self.logger.info("Starting optimization monitoring")
        
        # Set up the plots
        self.axes[0, 0].set_title('Optimization Progress')
        self.axes[0, 0].set_xlabel('Evaluations')
        self.axes[0, 0].set_ylabel('Best Score (log scale)')
        self.axes[0, 0].set_yscale('log')
        self.axes[0, 0].grid(True)
        
        self.axes[0, 1].set_title('Improvement Rate')
        self.axes[0, 1].set_xlabel('Evaluations')
        self.axes[0, 1].set_ylabel('Score Improvement')
        self.axes[0, 1].grid(True)
        
        self.axes[1, 0].set_title('Convergence Speed')
        self.axes[1, 0].set_xlabel('Time (s)')
        self.axes[1, 0].set_ylabel('Best Score (log scale)')
        self.axes[1, 0].set_yscale('log')
        self.axes[1, 0].grid(True)
        
        self.axes[1, 1].set_title('Optimization Statistics')
        self.axes[1, 1].axis('off')  # No axes for the text area
        
        # Create legend handles
        self.lines = {}
        self.improvement_lines = {}
        self.time_lines = {}
        
        # Start time for relative timing
        self.start_time = time.time()
        
        # Only set up animation if not in headless mode
        if not self.headless:
            # Enable interactive mode if showing plots
            if self.auto_show:
                plt.ion()
                
            self.animation = FuncAnimation(
                self.fig, self._update_plot, interval=500, 
                cache_frame_data=False, blit=False
            )
            
            plt.tight_layout()
            
            # Show the plot if auto_show is enabled
            if self.auto_show:
                plt.show(block=False)
                plt.pause(0.1)  # Add pause to ensure the plot is displayed
        
    def stop_monitoring(self):
        """Stop the visualization."""
        self.running = False
        if self.animation:
            self.animation.event_source.stop()
        
        # Update the plot one last time to ensure all data is visualized
        self._update_final_plot()
        
        if not self.headless:
            plt.close(self.fig)
        self.logger.info("Stopped optimization monitoring")
        
    def update_data(self, optimizer: str, iteration: int, score: float, evaluations: int):
        """
        Update the data for a specific optimizer.
        
        Args:
            optimizer: Name of the optimizer
            iteration: Current iteration
            score: Current best score
            evaluations: Current number of function evaluations
        """
        if not self.running:
            return
            
        # Add data to the queue
        self.data_queue.put({
            'optimizer': optimizer,
            'iteration': iteration,
            'score': score,
            'evaluations': evaluations,
            'timestamp': time.time()
        })
        
        # If in headless mode, process the queue immediately
        if self.headless:
            self._process_data_queue()
            
    def _process_data_queue(self):
        """Process all data in the queue."""
        while not self.data_queue.empty():
            try:
                data = self.data_queue.get_nowait()
                optimizer = data['optimizer']
                
                # Initialize data for new optimizer
                if optimizer not in self.optimizer_data:
                    self.optimizer_data[optimizer] = {
                        'iterations': [],
                        'scores': [],
                        'evaluations': [],
                        'timestamps': [],
                        'improvements': []
                    }
                
                # Store the data
                opt_data = self.optimizer_data[optimizer]
                opt_data['iterations'].append(data['iteration'])
                opt_data['scores'].append(data['score'])
                opt_data['evaluations'].append(data['evaluations'])
                opt_data['timestamps'].append(data['timestamp'])
                
                # Calculate improvement
                if len(opt_data['scores']) > 1:
                    improvement = opt_data['scores'][-2] - opt_data['scores'][-1]
                    opt_data['improvements'].append(improvement)
                else:
                    opt_data['improvements'].append(0)
                
                # Limit the amount of data stored
                if len(opt_data['iterations']) > self.max_data_points:
                    # Keep first few points, most recent points, and downsample the middle
                    keep_first = min(100, self.max_data_points // 10)
                    keep_last = min(900, self.max_data_points - keep_first)
                    
                    # Downsample the middle part if needed
                    if len(opt_data['iterations']) > keep_first + keep_last:
                        for key in opt_data:
                            opt_data[key] = (
                                opt_data[key][:keep_first] + 
                                opt_data[key][-keep_last:]
                            )
                
            except queue.Empty:
                break
                
    def _update_plot(self, frame):
        """Update the plot with new data for animation."""
        self._process_data_queue()
        self._update_plot_data()
        
        # Force a redraw of the figure if not in headless mode
        if not self.headless:
            self.fig.canvas.draw()
            self.fig.canvas.flush_events()
            plt.pause(0.01)  # Add a small pause to allow the GUI to update
        
        return list(self.lines.values()) + list(self.improvement_lines.values()) + list(self.time_lines.values())
        
    def _update_final_plot(self):
        """Update the plot with all data for final visualization."""
        self._process_data_queue()
        
        # Create lines for each optimizer if they don't exist
        for optimizer in self.optimizer_data:
            if optimizer not in self.lines:
                line, = self.axes[0, 0].plot([], [], label=optimizer)
                self.lines[optimizer] = line
                
                imp_line, = self.axes[0, 1].plot([], [], label=f"{optimizer} improvement")
                self.improvement_lines[optimizer] = imp_line
                
                time_line, = self.axes[1, 0].plot([], [], label=optimizer)
                self.time_lines[optimizer] = time_line
        
        self._update_plot_data()
        
    def _update_plot_data(self):
        """Update the plot data for all optimizers."""
        # Update the line data for each optimizer
        for optimizer, opt_data in self.optimizer_data.items():
            if optimizer not in self.lines:
                line, = self.axes[0, 0].plot([], [], label=optimizer)
                self.lines[optimizer] = line
                
                imp_line, = self.axes[0, 1].plot([], [], label=f"{optimizer} improvement")
                self.improvement_lines[optimizer] = imp_line
                
                time_line, = self.axes[1, 0].plot([], [], label=optimizer)
                self.time_lines[optimizer] = time_line
            
            # Update the line data
            self.lines[optimizer].set_data(
                opt_data['evaluations'], 
                opt_data['scores']
            )
            
            # Update improvement line
            if len(opt_data['improvements']) > 1:
                self.improvement_lines[optimizer].set_data(
                    opt_data['evaluations'][1:],
                    opt_data['improvements']
                )
            
            # Update time line
            relative_time = [t - self.start_time for t in opt_data['timestamps']]
            self.time_lines[optimizer].set_data(
                relative_time, 
                opt_data['scores']
            )
        
        # Adjust axes limits
        if self.optimizer_data:
            all_evals = []
            all_scores = []
            all_improvements = []
            all_times = []
            
            for opt_data in self.optimizer_data.values():
                all_evals.extend(opt_data['evaluations'])
                all_scores.extend(opt_data['scores'])
                all_improvements.extend(opt_data['improvements'])
                all_times.extend([t - self.start_time for t in opt_data['timestamps']])
            
            if all_evals:
                self.axes[0, 0].set_xlim(0, max(all_evals) * 1.1)
                
            if all_scores:
                min_score = max(1e-10, min([s for s in all_scores if s > 0]))
                max_score = max(all_scores)
                self.axes[0, 0].set_ylim(min_score * 0.1, max_score * 2)
                
            if all_improvements:
                max_imp = max(all_improvements) if all_improvements else 1
                self.axes[0, 1].set_ylim(0, max_imp * 1.1)
                self.axes[0, 1].set_xlim(0, max(all_evals) * 1.1 if all_evals else 100)
            
            if all_times:
                max_time = max(all_times)
                self.axes[1, 0].set_xlim(0, max_time * 1.1)
                self.axes[1, 0].set_ylim(min_score * 0.1, max_score * 2)
        
        # Update legend
        self.axes[0, 0].legend(loc='upper right')
        self.axes[0, 1].legend(loc='upper right')
        self.axes[1, 0].legend(loc='upper right')
        
        # Update statistics text in the fourth panel
        self.axes[1, 1].clear()
        self.axes[1, 1].axis('off')
        
        # Prepare statistics text
        stats_text = "Optimization Statistics:\n\n"
        
        if self.optimizer_data:
            # Find best optimizer so far
            best_opt = None
            best_score = float('inf')
            
            for opt_name, opt_data in self.optimizer_data.items():
                if opt_data['scores'] and min(opt_data['scores']) < best_score:
                    best_score = min(opt_data['scores'])
                    best_opt = opt_name
            
            stats_text += f"Best optimizer: {best_opt}\n"
            stats_text += f"Best score: {best_score:.3e}\n\n"
            
            # Add per-optimizer statistics
            stats_text += "Per-optimizer statistics:\n"
            for opt_name, opt_data in self.optimizer_data.items():
                if not opt_data['scores']:
                    continue
                    
                current_score = opt_data['scores'][-1]
                total_evals = opt_data['evaluations'][-1] if opt_data['evaluations'] else 0
                
                # Calculate improvement rate (per evaluation)
                if len(opt_data['scores']) > 10:
                    recent_improvement = opt_data['scores'][-10] - opt_data['scores'][-1]
                    recent_evals = opt_data['evaluations'][-1] - opt_data['evaluations'][-10]
                    imp_rate = recent_improvement / recent_evals if recent_evals > 0 else 0
                    stats_text += f"\n{opt_name}:\n"
                    stats_text += f"  Current score: {current_score:.3e}\n"
                    stats_text += f"  Evaluations: {total_evals}\n"
                    stats_text += f"  Recent improvement rate: {imp_rate:.3e}/eval\n"
        
        # Display the statistics
        self.axes[1, 1].text(0.05, 0.95, stats_text, 
                          transform=self.axes[1, 1].transAxes,
                          fontsize=10, verticalalignment='top')
        
        plt.tight_layout()
        
    def save_results(self, filename: str):
        """Save the current plot to a file."""
        if self.fig:
            # Make sure the plot is up to date
            self._update_final_plot()
            
            # Save the figure using save_plot
            save_plot(self.fig, filename, plot_type='benchmarks')
            self.logger.info(f"Saved visualization to {filename}")
            
    def save_data(self, filename: str):
        """Save the collected data to a CSV file."""
        if not self.optimizer_data:
            self.logger.warning("No data to save")
            return
            
        # Convert data to DataFrame
        all_data = []
        
        for optimizer, data in self.optimizer_data.items():
            for i in range(len(data['iterations'])):
                all_data.append({
                    'optimizer': optimizer,
                    'iteration': data['iterations'][i],
                    'score': data['scores'][i],
                    'evaluations': data['evaluations'][i],
                    'timestamp': data['timestamps'][i],
                    'improvement': data['improvements'][i] if i < len(data['improvements']) else 0
                })
                
        # Create DataFrame and save to CSV
        import pandas as pd
        df = pd.DataFrame(all_data)
        df.to_csv(filename, index=False)
        self.logger.info(f"Saved optimization data to {filename}")
