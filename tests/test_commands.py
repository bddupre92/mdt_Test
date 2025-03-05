#!/usr/bin/env python
"""
Test script for checking various command line options in main.py
"""

import subprocess
import sys
import os
from pathlib import Path
import time
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Define the base command
BASE_CMD = ["python", "main.py"]

# Create results directory if it doesn't exist
os.makedirs("command_test_results", exist_ok=True)

def run_command(cmd, description, expected_success=True, timeout=60):
    """Run a command and log the results"""
    cmd_str = " ".join(cmd)
    logging.info(f"Testing: {description}")
    logging.info(f"Command: {cmd_str}")
    
    try:
        # Run the command and capture output
        start_time = time.time()
        process = subprocess.run(
            cmd, 
            stdout=subprocess.PIPE, 
            stderr=subprocess.PIPE,
            text=True,
            timeout=timeout
        )
        end_time = time.time()
        elapsed = end_time - start_time
        
        # Log the results
        output_file = Path(f"command_test_results/{description.replace(' ', '_').lower()}.txt")
        with open(output_file, "w") as f:
            f.write(f"Command: {cmd_str}\n")
            f.write(f"Exit code: {process.returncode}\n")
            f.write(f"Elapsed time: {elapsed:.2f} seconds\n")
            f.write("\n=== STDOUT ===\n")
            f.write(process.stdout)
            f.write("\n=== STDERR ===\n")
            f.write(process.stderr)
        
        # Check if the command was successful
        if (process.returncode == 0) == expected_success:
            logging.info(f"✅ {description} - {'Success' if expected_success else 'Failed as expected'} ({elapsed:.2f}s)")
            return True
        else:
            logging.error(f"❌ {description} - {'Failed' if expected_success else 'Should have failed'} ({elapsed:.2f}s)")
            logging.error(f"Error: {process.stderr.strip()}")
            return False
            
    except subprocess.TimeoutExpired:
        logging.error(f"❌ {description} - Timed out after {timeout} seconds")
        return False
    except Exception as e:
        logging.error(f"❌ {description} - Exception: {str(e)}")
        return False

def main():
    """Run tests for each command line option"""
    
    # List of commands to test
    test_commands = [
        # Basic commands
        (BASE_CMD + ["--summary"], "Basic summary"),
        
        # Optimization commands
        (BASE_CMD + ["--optimize", "--summary"], "Optimize with summary"),
        
        # Meta-learning commands
        (BASE_CMD + ["--meta", "--summary"], "Meta-learning with summary"),
        (BASE_CMD + ["--meta", "--method", "random", "--summary"], "Meta-learning with random method"),
        (BASE_CMD + ["--meta", "--method", "bayesian", "--summary"], "Meta-learning with bayesian method"),
        
        # Drift detection commands
        (BASE_CMD + ["--drift", "--summary"], "Drift detection with summary"),
        (BASE_CMD + ["--drift", "--drift-window", "100", "--summary"], "Drift detection with custom window"),
        
        # Explainability commands
        (BASE_CMD + ["--explain", "--summary"], "Explainability with summary"),
        (BASE_CMD + ["--explain", "--explainer", "shap", "--summary"], "Explainability with SHAP"),
        (BASE_CMD + ["--explain", "--explainer", "lime", "--summary"], "Explainability with LIME"),
        
        # Optimizer explainability
        (BASE_CMD + ["--explain-optimizer", "--optimizer-type", "differential_evolution", "--summary"], 
         "Optimizer explainability for DE"),
        
        # Visualization
        (BASE_CMD + ["--optimize", "--visualize", "--summary"], "Optimization with visualization"),
        
        # Combined commands
        (BASE_CMD + ["--optimize", "--explain", "--summary"], "Optimize with explainability"),
        (BASE_CMD + ["--meta", "--drift", "--summary"], "Meta-learning with drift detection"),
    ]
    
    # Run each test
    results = []
    for cmd, description in test_commands:
        success = run_command(cmd, description)
        results.append((description, success))
    
    # Print summary
    print("\n=== TEST SUMMARY ===")
    passed = sum(1 for _, success in results if success)
    total = len(results)
    print(f"{passed}/{total} tests passed ({passed/total:.0%})")
    
    # Print failed tests
    if passed < total:
        print("\nFailed tests:")
        for description, success in results:
            if not success:
                print(f"- {description}")

if __name__ == "__main__":
    main()
