"""
live_visualization.py
--------------------
Real-time visualization tools for optimization algorithms.
"""

# Force the use of TkAgg backend for interactive plotting
import matplotlib
print(f"Current backend before setting: {matplotlib.get_backend()}")
matplotlib.use('TkAgg', force=True)  # Force TkAgg backend
print(f"Backend after forcing to TkAgg: {matplotlib.get_backend()}")

import platform
import sys
import os
import matplotlib.pyplot as plt

# Configure matplotlib for interactivity
plt.ion()  # Turn on interactive mode
print(f"Interactive mode after plt.ion(): {plt.isinteractive()}")

import numpy as np
from matplotlib.animation import FuncAnimation
import threading
import queue
import time
import logging
from typing import Dict, List, Any, Optional
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent))
from utils.plot_utils import save_plot

# Log the selected backend
logger = logging.getLogger(__name__)
logger.info(f"Using matplotlib backend: {matplotlib.get_backend()}, interactive: {plt.isinteractive()}")

class LiveOptimizationMonitor:
    """Real-time visualization for optimization progress."""
    
    def __init__(self, max_data_points: int = 1000, auto_show: bool = False, headless: bool = None):
        """
        Initialize the live monitor.
        
        Args:
            max_data_points: Maximum number of data points to store per optimizer
            auto_show: Whether to automatically show the plot when starting monitoring
            headless: Whether to run in headless mode (no display, save only).
                     If None, determine automatically based on matplotlib backend.
        """
        self.data_queue = queue.Queue()
        self.running = False
        self.fig = None
        self.axes = None
        self.optimizer_data = {}  # Store data for each optimizer
        self.animation = None
        self.logger = logging.getLogger(__name__)
        self.max_data_points = max_data_points
        self.auto_show = auto_show
        
        # Determine if we're in headless mode if not specified
        if headless is None:
            self.headless = matplotlib.get_backend() == 'Agg'
        else:
            self.headless = headless
            
        self.lines = {}
        self.improvement_lines = {}
        self.time_lines = {}
        self.start_time = None
        
    def start_monitoring(self):
        """Start the visualization thread."""
        if self.running:
            return
            
        self.running = True
        self.logger.info("Starting optimization monitoring")
        
        # Debug output
        print(f"Starting visualization with backend: {matplotlib.get_backend()}")
        print(f"Interactive mode: {plt.isinteractive()}")
        print(f"Headless mode: {self.headless}")
        
        # Initialize figure and axes
        if self.fig is None:
            self.fig, self.axes = plt.subplots(2, 2, figsize=(14, 10))
            
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
            
            # Reset line collections
            self.lines = {}
            self.improvement_lines = {}
            self.time_lines = {}
        
        # Start time for relative timing
        self.start_time = time.time()
        
        # Only set up animation if not in headless mode
        if not self.headless and plt.isinteractive():
            # Apply tight layout before showing
            plt.tight_layout()
            
            # Show the plot if auto_show is enabled
            if self.auto_show:
                try:
                    plt.ion()  # Ensure interactive mode is on
                    plt.show(block=False)
                    plt.pause(0.1)  # Add pause to ensure the plot is displayed
                except Exception as e:
                    self.logger.warning(f"Could not show interactive plot: {e}")
                    
            # Set up animation only in interactive mode
            try:
                self.animation = FuncAnimation(
                    self.fig, self._update_plot, interval=500, 
                    cache_frame_data=False, blit=False
                )
            except Exception as e:
                self.logger.warning(f"Could not set up animation: {e}")
        
    def stop_monitoring(self):
        """Stop the visualization."""
        if not self.running:
            return
            
        self.running = False
        
        # Stop animation if it exists
        if hasattr(self, 'animation') and self.animation:
            self.animation.event_source.stop()
        
        # Update the plot one last time to ensure all data is visualized
        if hasattr(self, '_update_final_plot'):
            self._update_final_plot()
        
        # Close figure if it exists and not in headless mode
        if not self.headless and hasattr(self, 'fig') and self.fig:
            plt.close(self.fig)
            self.fig = None
            self.axes = None
        
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
            
        try:
            # Convert data to appropriate types
            iteration = int(iteration)
            score = float(score)
            evaluations = int(evaluations)
            
            # Add data to the queue
            self.data_queue.put({
                'optimizer': str(optimizer),
                'iteration': iteration,
                'score': score,
                'evaluations': evaluations,
                'timestamp': time.time()
            })
            
            # If in headless mode, process the queue immediately
            if self.headless:
                self._process_data_queue()
        except Exception as e:
            self.logger.warning(f"Error updating data for {optimizer}: {e}")
            
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
        if not self.running or self.fig is None or self.axes is None:
            return []
            
        try:
            self._process_data_queue()
            self._update_plot_data()
            
            # Force a redraw of the figure if not in headless mode
            if not self.headless:
                try:
                    # Make this figure active
                    plt.figure(self.fig.number)
                    self.fig.canvas.draw_idle()
                    
                    # Only flush events if the backend supports it
                    backend = matplotlib.get_backend().lower()
                    if plt.isinteractive() and any(x in backend for x in ['qt', 'tk', 'wx', 'gtk', 'macos']):
                        try:
                            self.fig.canvas.flush_events()
                        except Exception as e:
                            pass
                            
                    # Add a small pause to allow the GUI to update
                    try:
                        plt.pause(0.01)
                    except Exception as e:
                        pass
                except Exception as e:
                    self.logger.warning(f"Error updating canvas: {e}")
            
            return list(self.lines.values()) + list(self.improvement_lines.values()) + list(self.time_lines.values())
        except Exception as e:
            self.logger.warning(f"Error updating plot: {e}")
            return []
        
    def _update_final_plot(self):
        """Update the final plot with all accumulated data."""
        if self.fig is None or self.axes is None:
            return
        
        try:
            # Clear the axes
            for ax in self.axes.flat:
                ax.clear()
            
            # Set up the subplots with better titles and labels
            self.axes[0, 0].set_title('Optimization Progress', fontweight='bold')
            self.axes[0, 0].set_xlabel('Evaluations')
            self.axes[0, 0].set_ylabel('Best Score (log scale)')
            self.axes[0, 0].set_yscale('log')
            self.axes[0, 0].grid(True, which='both', linestyle='--', alpha=0.7)
            
            self.axes[0, 1].set_title('Improvement Rate', fontweight='bold')
            self.axes[0, 1].set_xlabel('Evaluations')
            self.axes[0, 1].set_ylabel('Score Improvement')
            self.axes[0, 1].grid(True, alpha=0.7)
            
            self.axes[1, 0].set_title('Convergence Speed', fontweight='bold')
            self.axes[1, 0].set_xlabel('Time (s)')
            self.axes[1, 0].set_ylabel('Best Score (log scale)')
            self.axes[1, 0].set_yscale('log')
            self.axes[1, 0].grid(True, which='both', linestyle='--', alpha=0.7)
            
            # Add a text box for statistics in the fourth panel
            self.axes[1, 1].set_title('Optimization Statistics:', fontweight='bold')
            self.axes[1, 1].axis('off')
            
            # Compute statistics for each optimizer
            stats_text = ""
            best_optimizer = None
            best_score = float('inf')
            
            for optimizer, data in self.optimizer_data.items():
                if not data['scores']:
                    continue
                    
                min_score = min(data['scores'])
                if min_score < best_score:
                    best_score = min_score
                    best_optimizer = optimizer
                    
                evals = len(data['scores'])
                avg_improvement = 0
                if len(data['improvements']) > 0:
                    avg_improvement = sum(data['improvements']) / len(data['improvements'])
                    
                success_rate = 0
                if len(data['improvements']) > 0:
                    success_rate = sum(1 for imp in data['improvements'] if imp > 0) / len(data['improvements']) * 100
                    
                stats_text += f"{optimizer}:\n"
                stats_text += f"  - Best Score: {min_score:.6f}\n"
                stats_text += f"  - Evaluations: {evals}\n"
                stats_text += f"  - Avg Improvement: {avg_improvement:.6f}\n"
                stats_text += f"  - Success Rate: {success_rate:.2f}%\n\n"
                
            if best_optimizer:
                stats_text = f"Best Overall:\n  {best_optimizer} ({best_score:.6f})\n\n" + stats_text
                
            # Display optimizer statistics
            self.axes[1, 1].text(0.05, 0.95, stats_text, 
                                transform=self.axes[1, 1].transAxes,
                                fontsize=10, verticalalignment='top',
                                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
                                
            # Plot data for all optimizers
            for optimizer, opt_data in self.optimizer_data.items():
                if not opt_data['scores'] or not opt_data['evaluations']:
                    continue
                    
                # Create better plots with markers for key points
                evals = opt_data['evaluations']
                scores = opt_data['scores']
                
                # Plot scores vs evaluations (optimization progress)
                self.axes[0, 0].plot(evals, scores, '-o', label=optimizer, 
                                     markevery=[0, len(scores)-1] if len(scores) > 1 else None)
                
                # Plot improvement rate
                if len(opt_data['improvements']) > 0 and len(evals) > 1:
                    # Calculate moving average of improvements for smoother line
                    window = min(5, len(opt_data['improvements']))
                    imp_smooth = np.convolve(opt_data['improvements'], 
                                             np.ones(window)/window, mode='valid')
                    evals_smooth = evals[window-1:] if window > 1 else evals
                    
                    self.axes[0, 1].plot(evals_smooth, imp_smooth, '-', label=optimizer)
                    
                    # Highlight points with significant improvement
                    threshold = np.mean(opt_data['improvements']) * 1.5
                    significant_idx = [i for i, imp in enumerate(opt_data['improvements']) 
                                     if imp > threshold]
                    
                    if significant_idx:
                        self.axes[0, 1].plot([evals[i+1] for i in significant_idx],
                                           [opt_data['improvements'][i] for i in significant_idx],
                                           'o', markersize=8)
                
                # Plot scores vs time (convergence speed)
                if opt_data['timestamps']:
                    times = [(t - self.start_time) for t in opt_data['timestamps']]
                    self.axes[1, 0].plot(times, scores, '-o', label=optimizer,
                                        markevery=[0, len(scores)-1] if len(scores) > 1 else None)
            
            # Add legends to the plots
            self.axes[0, 0].legend(loc='upper right')
            self.axes[0, 1].legend(loc='upper right')
            self.axes[1, 0].legend(loc='upper right')
            
            # Adjust the layout
            plt.tight_layout()
            
            # Add overall title
            plt.suptitle('Live Optimization Monitoring', fontsize=16, y=0.98)
            
            # Force redraw
            self.fig.canvas.draw_idle()
            
        except Exception as e:
            self.logger.warning(f"Error updating final plot: {e}")
            import traceback
            traceback.print_exc()
        
    def _update_plot_data(self):
        """Update the plot data for all optimizers."""
        if self.fig is None or self.axes is None:
            return
            
        try:
            # Create lines for each optimizer if they don't exist
            for optimizer in self.optimizer_data:
                if optimizer not in self.lines:
                    line, = self.axes[0, 0].plot([], [], label=optimizer)
                    self.lines[optimizer] = line
                    
                    imp_line, = self.axes[0, 1].plot([], [], label=f"{optimizer} improvement")
                    self.improvement_lines[optimizer] = imp_line
                    
                    time_line, = self.axes[1, 0].plot([], [], label=optimizer)
                    self.time_lines[optimizer] = time_line
        
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
            
        except Exception as e:
            print(f"An error occurred while updating plot data: {e}")
            # Optionally, you can log the error or handle it in another way
        
    def save_results(self, filename: str):
        """
        Save the current visualization to a file.
        
        Args:
            filename: Path to save the visualization
        """
        try:
            # Process any remaining data in the queue
            self._process_data_queue()
            
            # Update the final plot with all data
            self._update_final_plot()
            
            # Ensure the directory exists
            os.makedirs(os.path.dirname(os.path.abspath(filename)), exist_ok=True)
            
            # Save the figure with high resolution
            self.fig.savefig(filename, dpi=300, bbox_inches='tight')
            print(f"Saved plot to {filename}")
            
            # Generate additional plots with more detailed information if there's enough data
            if any(len(data['scores']) > 2 for _, data in self.optimizer_data.items()):
                # Create detailed optimizer comparison figure
                self._save_detailed_comparison(os.path.splitext(filename)[0] + "_detailed.png")
            
        except Exception as e:
            self.logger.warning(f"Error saving results: {e}")
            import traceback
            traceback.print_exc()

    def _save_detailed_comparison(self, filename: str):
        """
        Save a detailed comparison of all optimizers.
        
        Args:
            filename: Path to save the comparison
        """
        try:
            # Create a new figure for detailed comparison
            plt.figure(figsize=(16, 12))
            
            # Set up the subplot grid - more space for detailed views
            gs = plt.GridSpec(3, 3)
            
            # 1. Convergence plot (log scale) - main plot
            ax1 = plt.subplot(gs[0, :])
            ax1.set_title('Convergence Comparison (log scale)', fontweight='bold')
            ax1.set_xlabel('Evaluations')
            ax1.set_ylabel('Best Score')
            ax1.set_yscale('log')
            ax1.grid(True, which='both', linestyle='--', alpha=0.7)
            
            # 2. Improvement rate over time
            ax2 = plt.subplot(gs[1, 0])
            ax2.set_title('Improvement Rate', fontweight='bold')
            ax2.set_xlabel('Iteration')
            ax2.set_ylabel('Score Improvement')
            ax2.grid(True, alpha=0.7)
            
            # 3. Convergence speed comparison
            ax3 = plt.subplot(gs[1, 1])
            ax3.set_title('Convergence Speed', fontweight='bold')
            ax3.set_xlabel('Time (s)')
            ax3.set_ylabel('Best Score')
            ax3.grid(True, alpha=0.7)
            
            # 4. Success rate by optimizer
            ax4 = plt.subplot(gs[1, 2])
            ax4.set_title('Success Rate by Optimizer', fontweight='bold')
            ax4.set_xlabel('Optimizer')
            ax4.set_ylabel('Success Rate (%)')
            ax4.grid(True, alpha=0.7)
            
            # 5. Detailed progress view - linear scale
            ax5 = plt.subplot(gs[2, :])
            ax5.set_title('Optimization Progress (linear scale)', fontweight='bold')
            ax5.set_xlabel('Evaluations')
            ax5.set_ylabel('Best Score')
            ax5.grid(True, alpha=0.7)
            
            # Calculate success rates and other metrics
            success_rates = []
            
            # Plot data for each optimizer
            for optimizer, data in self.optimizer_data.items():
                if not data['scores'] or not data['evaluations']:
                    continue
                    
                # 1. Plot convergence (log scale)
                ax1.plot(data['evaluations'], data['scores'], '-o', label=optimizer)
                
                # 2. Plot improvement rate
                if len(data['improvements']) > 1:
                    # Use moving average for smoother lines
                    window = min(3, len(data['improvements']))
                    imp_smooth = np.convolve(data['improvements'], np.ones(window)/window, mode='valid')
                    iterations = list(range(len(imp_smooth)))
                    ax2.plot(iterations, imp_smooth, '-', label=optimizer)
                
                # 3. Plot convergence speed
                if data['timestamps']:
                    times = [(t - data['timestamps'][0]) for t in data['timestamps']]
                    ax3.plot(times, data['scores'], '-', label=optimizer)
                
                # 5. Plot detailed progress (linear scale)
                ax5.plot(data['evaluations'], data['scores'], '-o', label=optimizer)
                
                # Calculate success rate (% of improvements > 0)
                if len(data['improvements']) > 0:
                    success_rate = sum(1 for imp in data['improvements'] if imp > 0) / len(data['improvements']) * 100
                    success_rates.append((optimizer, success_rate))
            
            # 4. Plot success rates
            if success_rates:
                success_rates.sort(key=lambda x: x[1], reverse=True)  # Sort by success rate
                optimizers, rates = zip(*success_rates)
                ax4.bar(optimizers, rates)
                ax4.set_xticklabels(optimizers, rotation=45, ha='right')
                
            # Add legends
            ax1.legend(loc='upper right')
            ax2.legend(loc='upper right')
            ax3.legend(loc='upper right')
            ax5.legend(loc='upper right')
            
            # Add overall title
            plt.suptitle('Optimizer Performance Comparison', fontsize=16, y=0.98)
            
            # Adjust layout
            plt.tight_layout()
            plt.subplots_adjust(top=0.93)
            
            # Save the figure
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            print(f"Saved detailed comparison to {filename}")
            
            plt.close()
            
        except Exception as e:
            self.logger.warning(f"Error saving detailed comparison: {e}")
            import traceback
            traceback.print_exc()

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

    def flush(self):
        """
        Process all pending data and update the visualization.
        This method ensures all queued data is processed and the plot is updated.
        """
        if not self.running:
            return
            
        try:
            self._process_data_queue()
            if self.fig is not None and self.axes is not None:
                self._update_plot_data()
                
                # Force a redraw if possible
                try:
                    plt.figure(self.fig.number)  # Make sure this figure is active
                    self.fig.canvas.draw_idle()
                    
                    # Only flush events if the backend supports it
                    backend = matplotlib.get_backend().lower()
                    if any(x in backend for x in ['qt', 'tk', 'wx', 'gtk', 'macos']):
                        try:
                            self.fig.canvas.flush_events()
                        except Exception as e:
                            self.logger.warning(f"Could not flush events: {e}")
                    
                    # Give a small pause to allow GUI updates
                    try:
                        plt.pause(0.01)
                    except Exception as e:
                        self.logger.warning(f"Error in plt.pause: {e}")
                        
                except Exception as e:
                    self.logger.warning(f"Error during canvas update: {e}")
        except Exception as e:
            self.logger.warning(f"Error in flush method: {e}")
