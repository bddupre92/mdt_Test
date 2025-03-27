#!/usr/bin/env python
"""
MoE Framework Runner

This script sets up the Python path correctly and runs the specified MoE component.
"""

import os
import sys
import subprocess

# Add the project root to the Python path
project_root = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, project_root)

def run_command(command):
    """Run a command with the correct Python path."""
    env = os.environ.copy()
    env["PYTHONPATH"] = project_root + os.pathsep + env.get("PYTHONPATH", "")
    
    print(f"Running: {' '.join(command)}")
    subprocess.run(command, env=env)

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run MoE Framework components")
    parser.add_argument("component", choices=["example", "dashboard", "execution", "workflow"],
                        help="Which component to run")
    parser.add_argument("--data", help="Path to data file (for workflow and execution)")
    parser.add_argument("--output-dir", default="./results", help="Output directory")
    parser.add_argument("--tracking-dir", default="./.workflow_tracking", help="Tracking directory")
    
    args = parser.parse_args()
    
    if args.component == "example":
        # Run the example script
        run_command([sys.executable, "-m", "moe_framework.event_tracking.example"])
    
    elif args.component == "dashboard":
        # Run the dashboard
        run_command([
            sys.executable, 
            "moe_tests/run_dashboard.py", 
            "--tracking-dir", args.tracking_dir
        ])
    
    elif args.component == "workflow":
        # Run the example workflow
        cmd = [
            sys.executable, 
            "moe_tests/run_example_workflow.py",
            "--output-dir", args.output_dir,
            "--tracking-dir", args.tracking_dir
        ]
        if args.data:
            cmd.extend(["--data", args.data])
        run_command(cmd)
    
    elif args.component == "execution":
        # Run the execution pipeline
        cmd = [
            sys.executable, 
            "moe_tests/run_execution_pipeline.py",
            "--output-dir", args.output_dir
        ]
        if args.data:
            cmd.extend(["--data", args.data])
        run_command(cmd) 