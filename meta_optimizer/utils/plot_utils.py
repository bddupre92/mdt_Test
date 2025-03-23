"""
Plot utilities for the meta_optimizer package.
"""
import os
import matplotlib.pyplot as plt


def save_plot(fig, filename, output_dir="plots", formats=None):
    """
    Save the given figure to the specified output directory with the given filename.
    
    Parameters
    ----------
    fig : matplotlib.figure.Figure
        The figure to save
    filename : str
        The base filename (without extension)
    output_dir : str, optional
        The directory to save the figure to, by default "plots"
    formats : list, optional
        The formats to save the figure in, by default ["png", "pdf"]
    """
    if formats is None:
        formats = ["png", "pdf"]
    
    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Save the figure in each format
    for fmt in formats:
        output_path = os.path.join(output_dir, f"{filename}.{fmt}")
        fig.savefig(output_path, bbox_inches="tight", dpi=300)
        print(f"Saved figure to {output_path}")
