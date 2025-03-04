"""
Utility functions for plotting and saving plots.
"""
import os
import matplotlib.pyplot as plt
from datetime import datetime
from pathlib import Path

def save_plot(fig, filename, plot_type='general'):
    """
    Save a plot to the appropriate directory based on plot type.
    
    Args:
        fig: The matplotlib figure to save
        filename: The filename to save the plot as
        plot_type: The type of plot (drift, performance, explainability, benchmarks)
    """
    # Create results directory if it doesn't exist
    results_dir = Path('results')
    results_dir.mkdir(exist_ok=True)
    
    # Create subdirectory based on plot type
    if plot_type == 'drift':
        subdir = results_dir / 'drift'
    elif plot_type == 'performance':
        subdir = results_dir / 'performance'
    elif plot_type == 'explainability':
        subdir = results_dir / 'explainability'
    elif plot_type == 'benchmarks':
        subdir = results_dir / 'benchmarks'
    else:
        subdir = results_dir / 'general'
    
    subdir.mkdir(exist_ok=True)
    
    # Add timestamp to filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_name, ext = os.path.splitext(filename)
    if not ext:
        ext = '.png'  # Default extension
    
    timestamped_filename = f"{base_name}_{timestamp}{ext}"
    
    # Save the figure
    fig.savefig(subdir / timestamped_filename, dpi=300, bbox_inches='tight')
    print(f"Saved plot to {subdir / timestamped_filename}")
