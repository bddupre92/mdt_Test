#!/usr/bin/env python3
"""
Utility script for importing optimization data.
"""

import os
import sys
import argparse
import logging
import traceback
from pathlib import Path

# Add the project root to the python path to ensure imports work correctly
sys.path.append(str(Path(__file__).parent.parent))

# Configure matplotlib backend
import matplotlib
try:
    current_backend = matplotlib.get_backend()
    print(f"Current backend before setting: {current_backend}")
    matplotlib.use('TkAgg')
    print(f"Backend after forcing to TkAgg: {matplotlib.get_backend()}")
    import matplotlib.pyplot as plt
    plt.ion()  # Enable interactive mode
    print(f"Interactive mode after plt.ion(): {plt.isinteractive()}")
except Exception as e:
    print(f"Warning: Could not set matplotlib backend: {e}")

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Import optimization data')
    parser.add_argument('--input', type=str, required=True, help='Input file path')
    parser.add_argument('--restore-state', action='store_true', help='Restore optimizer state')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose output')
    return parser.parse_args()

def import_optimization_data(args):
    """Import optimization data from file."""
    import_file = args.input
    
    if not os.path.exists(import_file):
        logging.error(f"Import file not found: {import_file}")
        return None
    
    try:
        # Create a MetaOptimizer instance
        from meta_optimizer.meta.meta_optimizer import MetaOptimizer
        from meta_optimizer.optimizers.optimizer_factory import create_optimizers
        
        # Default parameters for demonstration
        dim = 3
        bounds = [(-5, 5) for _ in range(dim)]
        
        # Create optimizers
        optimizers = create_optimizers(dim=dim, bounds=bounds)
        
        # Create MetaOptimizer
        meta_optimizer = MetaOptimizer(
            dim=dim,
            bounds=bounds,
            optimizers=optimizers,
            verbose=args.verbose
        )
        
        # Import data
        meta_optimizer.import_data(import_file, restore_state=args.restore_state)
        
        # Print summary of imported data
        print("Import test successful!")
        print(f"Dimensions: {meta_optimizer.dim}")
        print(f"Problem type: {meta_optimizer.current_problem_type}")
        print(f"Best score: {meta_optimizer.best_score}")
        
        # Check if history was imported
        if hasattr(meta_optimizer, 'optimization_history'):
            print(f"History entries: {len(meta_optimizer.optimization_history)}")
        
        # Check if algorithm selections were imported
        if hasattr(meta_optimizer, 'selection_tracker') and meta_optimizer.selection_tracker:
            selections = meta_optimizer.selection_tracker.get_history()
            print(f"Algorithm selections: {len(selections)}")
        
        # Check available optimizers
        print(f"Optimizers: {list(meta_optimizer.optimizers.keys())}")
        
        return meta_optimizer
    except Exception as e:
        logging.error(f"Failed to import data: {str(e)}")
        traceback.print_exc()
        return None

def main():
    """Main function."""
    args = parse_args()
    
    # Configure logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=log_level, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # Import data
    meta_optimizer = import_optimization_data(args)
    
    if meta_optimizer is None:
        sys.exit(1)

if __name__ == "__main__":
    main() 