#!/usr/bin/env python
"""
Copy Visualization Files

This script copies the visualization files from the most recent benchmark results
to the Next.js public directory for easy access.
"""

import os
import shutil
import glob
import sys
import argparse
from datetime import datetime

def find_most_recent_results(base_dir):
    """Find the most recent results directory based on timestamp."""
    if not os.path.exists(base_dir):
        print(f"Base directory does not exist: {base_dir}")
        return None
        
    # List all directories in the base directory
    directories = [d for d in os.listdir(base_dir) 
                  if os.path.isdir(os.path.join(base_dir, d))]
    
    # Filter for directories that look like timestamps (YYYYMMDD_HHMMSS)
    # or any directory containing a 'visualizations' subdirectory
    result_dirs = []
    for d in directories:
        full_path = os.path.join(base_dir, d)
        # Check for timestamp format
        if len(d) == 15 and d[8] == '_' and d[:8].isdigit() and d[9:].isdigit():
            result_dirs.append(full_path)
        # Or check if it contains visualizations directory
        elif os.path.exists(os.path.join(full_path, "visualizations")):
            result_dirs.append(full_path)
    
    if not result_dirs:
        return None
    
    # Sort by modification time (most recent first)
    result_dirs.sort(key=lambda x: os.path.getmtime(x), reverse=True)
    return result_dirs[0]

def copy_visualizations(source_dir, target_dir, verbose=False):
    """Copy visualization files from source to target directory."""
    if not os.path.exists(source_dir):
        print(f"Source directory does not exist: {source_dir}")
        return False
    
    # Ensure target directory exists
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
        if verbose:
            print(f"Created target directory: {target_dir}")
    
    # Find visualization files
    vis_dir = os.path.join(source_dir, "visualizations")
    if not os.path.exists(vis_dir):
        print(f"Visualizations directory not found: {vis_dir}")
        return False
    
    # Copy all visualization files
    count = 0
    for file in glob.glob(os.path.join(vis_dir, "*.png")) + glob.glob(os.path.join(vis_dir, "*.svg")):
        filename = os.path.basename(file)
        target_file = os.path.join(target_dir, filename)
        shutil.copy2(file, target_file)
        count += 1
        if verbose:
            print(f"Copied: {filename}")
    
    print(f"Copied {count} visualization files to {target_dir}")
    return True

def main():
    parser = argparse.ArgumentParser(description="Copy visualization files to Next.js public directory")
    parser.add_argument("--source", help="Source directory containing results (default: results/paper_visuals)", 
                        default="results/paper_visuals")
    parser.add_argument("--target", help="Target directory (default: v0test/public/visualizations)", 
                        default="v0test/public/visualizations")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    
    args = parser.parse_args()
    
    # Find most recent results directory
    source_dir = args.source
    recent_dir = find_most_recent_results(source_dir)
    
    if not recent_dir:
        print(f"No results directories found in {source_dir}")
        return 1
    
    if args.verbose:
        print(f"Found most recent results directory: {recent_dir}")
    
    # Copy visualizations
    success = copy_visualizations(recent_dir, args.target, args.verbose)
    
    if success:
        # Also create a timestamp file to track when visualizations were last updated
        timestamp_file = os.path.join(args.target, "last_updated.txt")
        with open(timestamp_file, "w") as f:
            f.write(f"Visualizations last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Source directory: {os.path.basename(recent_dir)}\n")
        
        if args.verbose:
            print(f"Created timestamp file: {timestamp_file}")
        
        return 0
    else:
        return 1

if __name__ == "__main__":
    sys.exit(main()) 