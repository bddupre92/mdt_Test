#!/usr/bin/env python3
"""
export_sample_data.py
--------------------
Utility for generating sample optimization data in different formats.
"""

import os
import json
import csv
import time
import random
import argparse
import numpy as np
from pathlib import Path

def generate_sample_optimization_history(dim: int = 2, 
                                       iterations: int = 10, 
                                       optimizers: list = None,
                                       problem_type: str = "multimodal"):
    """
    Generate sample optimization history.
    
    Args:
        dim: Dimensionality
        iterations: Number of iterations
        optimizers: List of optimizers
        problem_type: Problem type
        
    Returns:
        Dictionary with sample data
    """
    if optimizers is None:
        optimizers = ["DE", "ES", "ACO", "GWO", "DE-Adaptive", "ES-Adaptive"]
        
    # Generate sample data
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    
    # Basic information
    data = {
        "optimization_info": {
            "dimensions": dim,
            "bounds": [(-5, 5)] * dim,
            "total_evaluations": iterations * 100,
            "best_score": 1.234,
            "best_solution": [0.1, 0.2] if dim == 2 else [0.1] * dim,
            "runtime": 10.5,
            "iterations": iterations,
            "timestamp": timestamp,
            "problem_type": problem_type
        },
        "optimization_history": [],
        "algorithm_selections": [],
        "parameter_history": {}
    }
    
    # Generate problem features
    data["problem_features"] = {
        "dimension": float(dim),
        "range": 75.0,
        "std": 15.5,
        "gradient_variance": 8e8,
        "modality": 100.0,
        "convexity": 0.5,
        "ruggedness": 0.07,
        "separability": 0.05,
        "local_structure": 0.5,
        "global_structure": 0.1,
        "fitness_distance_correlation": 0.6,
        "information_content": 1.9,
        "basin_ratio": 0.6,
        "gradient_homogeneity": 0.49
    }
    
    # Generate optimization history
    start_score = 100.0
    current_score = start_score
    
    for i in range(iterations):
        # Select optimizers for this iteration
        selected_optimizers = random.sample(optimizers, min(3, len(optimizers)))
        
        for j, optimizer in enumerate(selected_optimizers):
            # Generate random improvement
            improvement = random.uniform(0.5, 5.0)
            new_score = max(0.1, current_score - improvement)
            
            # Record selection
            data["algorithm_selections"].append({
                "iteration": i + 1,
                "optimizer": optimizer,
                "problem_type": problem_type,
                "score": new_score,
                "timestamp": time.time(),
                "improvement": current_score - new_score,
                "relative_improvement": (current_score - new_score) / current_score if current_score > 0 else 0
            })
            
            # Record history entry
            history_entry = {
                "iteration": i + 1,
                "optimizer": optimizer,
                "score": new_score,
                "solution": [random.uniform(-1, 1) for _ in range(dim)],
                "evaluations": (i * len(selected_optimizers) + j + 1) * 30,
                "runtime": (i * len(selected_optimizers) + j + 1) * 0.1,
                "improvement": current_score - new_score,
                "success": True
            }
            
            data["optimization_history"].append(history_entry)
            
            current_score = new_score
    
    # Generate parameter history for each optimizer
    for optimizer in optimizers:
        data["parameter_history"][optimizer] = {
            "population_size": 30,
            "crossover_rate": [0.5 + random.uniform(-0.1, 0.1) for _ in range(iterations)] if "DE" in optimizer else [0.5],
            "mutation_rate": [0.1 + random.uniform(-0.05, 0.05) for _ in range(iterations)] if "DE" in optimizer else [0.1],
            "success_rate": [random.uniform(0.4, 0.9) for _ in range(iterations)],
            "diversity": [random.uniform(0.1, 0.8) for _ in range(iterations)]
        }
    
    return data

def export_to_json(data: dict, filename: str):
    """
    Export data to JSON.
    
    Args:
        data: Data to export
        filename: Output filename
    """
    with open(filename, 'w') as f:
        json.dump(data, f, indent=2)
    print(f"Exported JSON data to {filename}")

def export_to_csv(data: dict, base_filename: str):
    """
    Export data to CSV files.
    
    Args:
        data: Data to export
        base_filename: Base filename for CSV files
    """
    # Export optimization info
    with open(f"{base_filename}_info.csv", 'w', newline='') as f:
        writer = csv.writer(f)
        
        # Write header
        writer.writerow(data["optimization_info"].keys())
        
        # Write data
        writer.writerow(data["optimization_info"].values())
    
    # Export optimization history
    if data["optimization_history"]:
        with open(f"{base_filename}_history.csv", 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=data["optimization_history"][0].keys())
            
            # Write header
            writer.writeheader()
            
            # Write data
            writer.writerows(data["optimization_history"])
    
    # Export algorithm selections
    if data["algorithm_selections"]:
        with open(f"{base_filename}_selections.csv", 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=data["algorithm_selections"][0].keys())
            
            # Write header
            writer.writeheader()
            
            # Write data
            writer.writerows(data["algorithm_selections"])
    
    # Export parameter history
    for optimizer, params in data["parameter_history"].items():
        with open(f"{base_filename}_{optimizer}_params.csv", 'w', newline='') as f:
            # Get max parameter history length
            max_length = max(len(values) if isinstance(values, list) else 1 
                           for values in params.values())
            
            # Create list of dicts for DictWriter
            rows = []
            for i in range(max_length):
                row = {}
                for param, values in params.items():
                    if isinstance(values, list) and i < len(values):
                        row[param] = values[i]
                    elif isinstance(values, list):
                        row[param] = None
                    else:
                        row[param] = values
                rows.append(row)
            
            # Write CSV
            writer = csv.DictWriter(f, fieldnames=params.keys())
            writer.writeheader()
            writer.writerows(rows)
    
    # Export problem features
    with open(f"{base_filename}_features.csv", 'w', newline='') as f:
        writer = csv.writer(f)
        
        # Write header
        writer.writerow(data["problem_features"].keys())
        
        # Write data
        writer.writerow(data["problem_features"].values())
        
    print(f"Exported CSV data to {os.path.dirname(base_filename)}")

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Generate sample optimization data")
    
    # General options
    parser.add_argument("--dim", type=int, default=2, help="Dimensionality")
    parser.add_argument("--iterations", type=int, default=10, help="Number of iterations")
    parser.add_argument("--problem-type", type=str, default="multimodal", 
                      choices=["multimodal", "unimodal", "separable", "non-separable"],
                      help="Problem type")
    
    # Export options
    parser.add_argument("--format", type=str, choices=["json", "csv", "both"], default="both",
                      help="Export format")
    parser.add_argument("--output-dir", type=str, default="results/sample_data",
                      help="Output directory")
    parser.add_argument("--filename", type=str, default=None,
                      help="Base filename (defaults to timestamp-based)")
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Generate sample data
    data = generate_sample_optimization_history(
        dim=args.dim,
        iterations=args.iterations,
        problem_type=args.problem_type
    )
    
    # Create filename if not provided
    if not args.filename:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        args.filename = f"sample_optimization_data_{timestamp}"
    
    # Complete output path
    output_path = os.path.join(args.output_dir, args.filename)
    
    # Export data in selected format
    if args.format in ["json", "both"]:
        export_to_json(data, f"{output_path}.json")
    
    if args.format in ["csv", "both"]:
        export_to_csv(data, output_path)
    
    return 0

if __name__ == "__main__":
    main() 