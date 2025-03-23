#!/usr/bin/env python3
"""
export_results.py
----------------
Utility for exporting optimization results from MetaOptimizer runs.
"""

import os
import sys
import argparse
from pathlib import Path
import json
import time

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from meta_optimizer.meta.meta_optimizer import MetaOptimizer
from meta_optimizer.meta.optimization_history import OptimizationHistory
from meta_optimizer.meta.selection_tracker import SelectionTracker

def find_history_file(data_dir: str = "results/history", pattern: str = "*_history.json") -> str:
    """
    Find the most recent history file in the given directory.
    
    Args:
        data_dir: Directory to search for history files
        pattern: Glob pattern for history files
        
    Returns:
        Path to the most recent history file
    """
    data_dir = Path(data_dir)
    if not data_dir.exists():
        return None
        
    history_files = list(data_dir.glob(pattern))
    if not history_files:
        return None
        
    # Sort by modification time (most recent first)
    return str(sorted(history_files, key=lambda x: x.stat().st_mtime, reverse=True)[0])

def load_meta_optimizer(history_file: str, selection_file: str = None) -> MetaOptimizer:
    """
    Load a MetaOptimizer from history files.
    
    Args:
        history_file: Path to optimization history file
        selection_file: Path to algorithm selection file
        
    Returns:
        MetaOptimizer instance loaded with data
    """
    # Create a basic MetaOptimizer
    meta_opt = MetaOptimizer(
        dim=2,  # Will be overridden by history
        bounds=[(-5, 5), (-5, 5)],  # Will be overridden by history
        optimizers={},  # Empty because we're just using it for data export
        history_file=history_file,
        selection_file=selection_file
    )
    
    # Load history and selection data
    if history_file and os.path.exists(history_file):
        meta_opt.optimization_history = OptimizationHistory.load(history_file)
        
    if selection_file and os.path.exists(selection_file):
        meta_opt.selection_tracker = SelectionTracker.load(selection_file)
        
    return meta_opt

def main():
    """Main function for the CLI."""
    parser = argparse.ArgumentParser(description="Export optimization results to various formats")
    
    # File paths
    parser.add_argument("--history-file", type=str, 
                      help="Path to optimization history file (if not provided, most recent will be used)")
    parser.add_argument("--selection-file", type=str,
                      help="Path to algorithm selection file (if not provided, most recent will be used)")
    
    # Export options
    parser.add_argument("--format", type=str, choices=["json", "csv", "both"], default="both",
                      help="Export format(s) to use")
    parser.add_argument("--output-dir", type=str, default="results/exports",
                      help="Directory to store exported data")
    parser.add_argument("--filename", type=str, 
                      help="Base filename for exports (defaults to timestamp-based)")
    
    # Content options
    parser.add_argument("--include-history", action="store_true", default=True,
                      help="Include full optimization history")
    parser.add_argument("--include-selections", action="store_true", default=True,
                      help="Include algorithm selections")
    parser.add_argument("--include-parameters", action="store_true", default=True,
                      help="Include parameter adaptation history")
    
    # Parse arguments
    args = parser.parse_args()
    
    # Find history file if not provided
    history_file = args.history_file
    if not history_file:
        history_file = find_history_file()
        if not history_file:
            print("Error: No history file found. Please provide a path with --history-file.")
            return 1
    
    # Find selection file if not provided
    selection_file = args.selection_file
    if not selection_file:
        # Try to find a selection file with similar name
        history_path = Path(history_file)
        potential_selection_file = history_path.parent / history_path.name.replace("_history", "_selections")
        if potential_selection_file.exists():
            selection_file = str(potential_selection_file)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Create filename if not provided
    if not args.filename:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        args.filename = f"optimization_export_{timestamp}"
    
    # Complete output path
    output_path = os.path.join(args.output_dir, args.filename)
    
    print(f"Loading optimization data from {history_file}")
    if selection_file:
        print(f"Loading selection data from {selection_file}")
    
    # Load meta optimizer
    meta_opt = load_meta_optimizer(history_file, selection_file)
    
    # Export data
    export_path = meta_opt.export_data(
        filename=output_path,
        format=args.format,
        include_history=args.include_history,
        include_selections=args.include_selections,
        include_parameters=args.include_parameters
    )
    
    print(f"Exported data to {export_path}")
    return 0

if __name__ == "__main__":
    sys.exit(main()) 