"""
run_benchmarks.py
----------------
Run optimization benchmarks and save results for later visualization.
"""

import os
import sys
from pathlib import Path
import argparse

# Add project root to Python path
project_root = str(Path(__file__).parent.parent)
if project_root not in sys.path:
    sys.path.append(project_root)

import numpy as np
import pickle
from typing import Dict, Any, Optional

from benchmarking.test_functions import TEST_FUNCTIONS
from benchmarking.cec_functions import create_cec_suite
from visualization.optimizer_analysis import OptimizerAnalyzer
from optimizers.optimizer_factory import create_optimizers

def save_results(results: Dict[str, Any], filename: str):
    """Save optimization results to file"""
    with open(filename, 'wb') as f:
        pickle.dump(results, f)

def load_results(filename: str) -> Dict[str, Any]:
    """Load optimization results from file"""
    with open(filename, 'rb') as f:
        return pickle.load(f)

def main(save_dir: str = 'results/benchmarks',
         n_runs: int = 30,
         dim: int = 30,
         bounds: Optional[list] = None,
         test_mode: bool = False):
    """
    Run optimization benchmarks
    
    Args:
        save_dir: Directory to save results
        n_runs: Number of independent runs per optimizer
        dim: Problem dimensionality
        bounds: Parameter bounds (default: [-100, 100])
        test_mode: If True, only run one function with two optimizers
    """
    # Create directories
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    
    # Set default bounds if not provided
    if bounds is None:
        bounds = [(-100, 100)] * dim
    
    # Create optimizers
    optimizers = create_optimizers(dim, bounds)
    
    # If in test mode, only use two optimizers
    if test_mode:
        optimizers = {k: optimizers[k] for k in ['DE (Standard)', 'DE (Adaptive)']}
    
    # Create analyzer
    analyzer = OptimizerAnalyzer(optimizers)
    
    # Get test functions
    test_functions = {}
    
    # Add classical test functions
    for name, func_factory in TEST_FUNCTIONS.items():
        test_functions[name] = func_factory(dim, bounds)
        if test_mode and name == 'sphere':  # Only use sphere function in test mode
            break
            
    # Add CEC test functions if not in test mode
    if not test_mode:
        cec_funcs = create_cec_suite(dim, bounds)
        test_functions.update(cec_funcs)
    
    print("Running optimization comparison...\n")
    
    # Run comparison
    results = analyzer.run_comparison(
        test_functions=test_functions,
        n_runs=2 if test_mode else n_runs,  # Use fewer runs in test mode
        record_convergence=True
    )
    
    # Save results for each function
    for func_name, func_results in results.items():
        save_path = Path(save_dir) / f"{func_name}.pkl"
        save_results(func_results, str(save_path))
        print(f"Saved results for {func_name} to {save_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--test', action='store_true', help='Run in test mode')
    args = parser.parse_args()
    
    main(test_mode=args.test)
