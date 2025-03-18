#!/usr/bin/env python3

import os
import json
import argparse
from pathlib import Path
from baseline_comparison.visualization import ComparisonVisualizer

def load_results(results_dir: str) -> dict:
    """Load results from the baseline comparison output directory"""
    results = {}
    results_path = Path(results_dir) / "results.json"
    
    if not results_path.exists():
        raise FileNotFoundError(f"Results file not found at {results_path}")
    
    with open(results_path, 'r') as f:
        results = json.load(f)
    
    return results

def main():
    parser = argparse.ArgumentParser(description="Generate visualizations for baseline comparison results")
    parser.add_argument("--results-dir", type=str, required=True,
                       help="Directory containing the baseline comparison results")
    parser.add_argument("--output-dir", type=str, default=None,
                       help="Directory to save visualizations (defaults to results-dir/visualizations)")
    args = parser.parse_args()
    
    # Load results
    results = load_results(args.results_dir)
    
    # Set output directory
    output_dir = args.output_dir or os.path.join(args.results_dir, "visualizations")
    os.makedirs(output_dir, exist_ok=True)
    
    # Create visualizer
    visualizer = ComparisonVisualizer(results, output_dir)
    
    # Generate all visualizations
    print(f"Generating visualizations in {output_dir}...")
    visualizer.create_all_visualizations()
    print("Done!")

if __name__ == "__main__":
    main() 