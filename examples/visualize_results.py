"""
visualize_results.py
-------------------
Load and visualize optimization results.
"""

import os
import sys
from pathlib import Path

# Add project root to Python path
project_root = str(Path(__file__).parent.parent)
if project_root not in sys.path:
    sys.path.append(project_root)

import numpy as np
from typing import Dict, Any, Optional, List

from visualization.optimizer_analysis import OptimizerAnalyzer
from examples.run_benchmarks import load_results

def load_all_results(results_dir: str) -> Dict[str, Dict[str, List[Any]]]:
    """
    Load all results from directory
    
    Args:
        results_dir: Directory containing benchmark results
        
    Returns:
        Dictionary mapping function names to results
    """
    results = {}
    results_dir = Path(results_dir)
    
    for result_file in results_dir.glob('*.pkl'):
        func_name = result_file.stem
        results[func_name] = load_results(str(result_file))
        
    return results

def clean_optimizer_name(name: str) -> str:
    """Clean optimizer name for filenames"""
    return name.lower().replace(' ', '_').replace('(', '').replace(')', '')

def main(results_dir: str = 'results/benchmarks',
         plots_dir: str = 'results/plots',
         functions: Optional[List[str]] = None,
         optimizers: Optional[List[str]] = None,
         test_mode: bool = False):
    """
    Generate visualizations from saved results
    
    Args:
        results_dir: Directory containing benchmark results
        plots_dir: Directory to save plots
        functions: List of functions to visualize (default: all)
        optimizers: List of optimizers to visualize (default: all)
        test_mode: If True, only visualize sphere function with DE optimizers
    """
    # Create output directory
    plots_dir = Path(plots_dir)
    plots_dir.mkdir(parents=True, exist_ok=True)
    
    # Clean old plots if they exist
    for plot_file in plots_dir.glob('*.png'):
        plot_file.unlink()
    
    # Set test mode parameters
    if test_mode:
        functions = ['sphere']
        optimizers = ['DE (Standard)', 'DE (Adaptive)']
    
    # Load results
    print("Loading results...")
    results = load_all_results(results_dir)
    
    # Filter functions if specified
    if functions:
        results = {k: v for k, v in results.items() if k in functions}
    
    # Filter optimizers if specified
    if optimizers:
        for func_name in results:
            results[func_name] = {k: v for k, v in results[func_name].items() 
                                if k in optimizers}
    
    # Create analyzer with dummy optimizers (not needed for visualization)
    analyzer = OptimizerAnalyzer({})
    analyzer.results = results
    
    print("\nGenerating visualizations...")
    
    print("\n1. Plotting convergence comparisons...")
    analyzer.plot_convergence_comparison()
    
    print("\n2. Creating performance heatmaps...")
    analyzer.plot_performance_heatmap()
    
    print("\n3. Analyzing parameter adaptation...")
    for optimizer_name in next(iter(results.values())).keys():
        if 'Adaptive' not in optimizer_name:
            continue
        for func_name in results:
            clean_name = clean_optimizer_name(optimizer_name)
            plot_path = plots_dir / f"param_adaptation_{clean_name}_{func_name}.png"
            analyzer.plot_parameter_adaptation(optimizer_name, func_name)
    
    print("\n4. Analyzing population diversity...")
    for optimizer_name in next(iter(results.values())).keys():
        for func_name in results:
            clean_name = clean_optimizer_name(optimizer_name)
            plot_path = plots_dir / f"diversity_{clean_name}_{func_name}.png"
            analyzer.plot_diversity_analysis(optimizer_name, func_name)
    
    print(f"\nAll visualizations have been saved to {plots_dir}")

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--test', action='store_true', help='Run in test mode')
    args = parser.parse_args()
    
    main(test_mode=args.test)
