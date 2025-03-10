import os
import logging
from pathlib import Path
import matplotlib.pyplot as plt

def save_plot(fig, filename, plot_type='general', formats=None):
    """
    Save a plot to the appropriate directory based on its type
    
    Parameters:
    -----------
    fig : matplotlib.figure.Figure
        Matplotlib figure to save
    filename : str
        Name of the file (without extension)
    plot_type : str, optional
        Type of plot (drift, performance, explainability, benchmarks, meta_learning, etc.)
    formats : list, optional
        List of file formats to save (e.g., ['png', 'pdf', 'svg'])
        If None, defaults to ['png']
    
    Returns:
    --------
    Path
        Path to the saved figure
    """
    # Default formats
    if formats is None:
        formats = ['png']
    
    # Ensure filename has no extension
    filename = os.path.splitext(filename)[0]
    
    # Create base results directory if it doesn't exist
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
    elif plot_type == 'meta_learning':
        subdir = results_dir / 'meta_learning'
    elif plot_type == 'enhanced_meta':
        subdir = results_dir / 'enhanced_meta' / 'visualizations'
    else:
        subdir = results_dir
    
    # Create subdirectory if it doesn't exist
    subdir.mkdir(exist_ok=True, parents=True)
    
    # Save the figure in all requested formats
    saved_paths = []
    for fmt in formats:
        fig_path = subdir / f"{filename}.{fmt}"
        fig.savefig(fig_path, bbox_inches='tight', dpi=300)
        saved_paths.append(fig_path)
        logging.info(f"Saved plot to {fig_path}")
    
    # Close the figure to free memory
    plt.close(fig)
    
    return saved_paths[0]  # Return the path to the first format

def setup_plot_style():
    """
    Set up a consistent plot style for all visualizations
    """
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams['figure.figsize'] = (10, 6)
    plt.rcParams['font.size'] = 12
    plt.rcParams['axes.labelsize'] = 14
    plt.rcParams['axes.titlesize'] = 16
    plt.rcParams['xtick.labelsize'] = 12
    plt.rcParams['ytick.labelsize'] = 12
    plt.rcParams['legend.fontsize'] = 12
    plt.rcParams['figure.titlesize'] = 20

def create_comparison_plot(data, x_label, y_label, title, filename, plot_type='general', 
                           kind='bar', **kwargs):
    """
    Create and save a comparison plot
    
    Parameters:
    -----------
    data : pandas.DataFrame or dict
        Data to plot
    x_label : str
        Label for x-axis
    y_label : str
        Label for y-axis
    title : str
        Plot title
    filename : str
        Output filename
    plot_type : str, optional
        Type of plot
    kind : str, optional
        Kind of plot ('bar', 'line', etc.)
    **kwargs : 
        Additional arguments to pass to the plotting function
    
    Returns:
    --------
    Path
        Path to the saved figure
    """
    import pandas as pd
    
    setup_plot_style()
    
    # Convert dict to DataFrame if needed
    if isinstance(data, dict):
        data = pd.DataFrame(data)
    
    fig, ax = plt.subplots()
    
    if kind == 'bar':
        data.plot(kind='bar', ax=ax, **kwargs)
    elif kind == 'line':
        data.plot(kind='line', ax=ax, **kwargs)
    elif kind == 'boxplot':
        data.boxplot(ax=ax, **kwargs)
    else:
        data.plot(kind=kind, ax=ax, **kwargs)
    
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title)
    
    if 'legend_title' in kwargs:
        ax.legend(title=kwargs['legend_title'])
    
    plt.tight_layout()
    
    return save_plot(fig, filename, plot_type=plot_type)
