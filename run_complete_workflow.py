#!/usr/bin/env python
"""
Run a complete MoE workflow from data generation to dashboard visualization.

This script:
1. Generates sample data
2. Runs the example workflow with tracking
3. Launches the dashboard to visualize the results
"""

import os
import sys
import subprocess
import argparse
import time

# Add the project root to the Python path
project_root = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, project_root)

def run_command(command, env=None):
    """Run a command with the correct Python path."""
    if env is None:
        env = os.environ.copy()
        env["PYTHONPATH"] = project_root + os.pathsep + env.get("PYTHONPATH", "")
    
    print(f"Running: {' '.join(command)}")
    subprocess.run(command, env=env)

def main():
    parser = argparse.ArgumentParser(description="Run complete MoE workflow")
    parser.add_argument("--samples", type=int, default=100, help="Number of samples to generate")
    parser.add_argument("--output-dir", type=str, default="./results", help="Output directory")
    parser.add_argument("--tracking-dir", type=str, default="./.workflow_tracking", help="Tracking directory")
    parser.add_argument("--skip-dashboard", action="store_true", help="Skip launching the dashboard")
    
    args = parser.parse_args()
    
    # Create directories
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.tracking_dir, exist_ok=True)
    
    # Step 1: Generate sample data
    print("\n=== Step 1: Generating Sample Data ===\n")
    sample_data_path = os.path.join(args.output_dir, "sample_data.csv")
    
    # Import and run the sample data generator
    from generate_sample_data import generate_sample_data
    generate_sample_data(n_samples=args.samples, output_path=sample_data_path)
    
    # Step 2: Run the example workflow
    print("\n=== Step 2: Running Example Workflow ===\n")
    
    # Import and run the example
    from moe_framework.event_tracking.example import run_demo
    run_demo()
    
    print("\n=== Workflow Completed Successfully! ===\n")
    print(f"Visualizations have been saved to the './visualizations' directory.")
    
    # Step 3: Launch the dashboard
    if not args.skip_dashboard:
        print("\n=== Step 3: Launching Dashboard ===\n")
        print("Starting Streamlit dashboard...")
        
        # Run the dashboard script
        run_command([sys.executable, "run_dashboard.py"])
    else:
        print("\nSkipping dashboard launch as requested.")
        print("To launch the dashboard later, run: python run_dashboard.py")

if __name__ == "__main__":
    main() 