#!/usr/bin/env python
"""
Run Script for Benchmark Dashboard

This script launches the Streamlit dashboard for visualizing benchmark results
and comparing optimizer performance.

Usage:
    python run_dashboard.py [--results_dir RESULTS_DIR] [--port PORT]
"""

import os
import sys
import argparse
import subprocess

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Launch benchmark dashboard")
    parser.add_argument(
        "--results_dir", 
        type=str, 
        default="benchmark_results",
        help="Directory containing benchmark results"
    )
    parser.add_argument(
        "--port", 
        type=int, 
        default=8501, 
        help="Port to run the dashboard on"
    )
    return parser.parse_args()

def main():
    """Main function to launch dashboard."""
    args = parse_args()
    
    # Ensure results directory exists
    os.makedirs(args.results_dir, exist_ok=True)
    
    # Get script path
    script_path = os.path.join(os.path.dirname(__file__), "ui/benchmark_dashboard.py")
    
    # Set environment variables for the subprocess
    env = os.environ.copy()
    env["STREAMLIT_SERVER_PORT"] = str(args.port)
    env["STREAMLIT_SERVER_HEADLESS"] = "true"
    env["STREAMLIT_BROWSER_GATHER_USAGE_STATS"] = "false"
    env["RESULTS_DIR"] = args.results_dir
    
    # Run the Streamlit command as a subprocess
    cmd = [
        "streamlit", "run", script_path, 
        "--", f"--results_dir={args.results_dir}"
    ]
    
    print(f"Starting Streamlit dashboard on port {args.port}...")
    print(f"Dashboard URL: http://localhost:{args.port}")
    print(f"Results directory: {args.results_dir}")
    
    # Run the command
    subprocess.run(cmd, env=env)

if __name__ == "__main__":
    main() 