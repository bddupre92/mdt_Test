#!/usr/bin/env python3
"""
Example script demonstrating how to use the migraine data handling capabilities from the command line.
"""

import os
import sys
import subprocess
from pathlib import Path

def main():
    """
    Demonstrate the command-line interface for migraine data handling.
    This script simulates using the main.py CLI with various migraine data arguments.
    """
    # Get the project root directory
    proj_root = Path(__file__).resolve().parent.parent.parent
    
    # Create necessary directories
    data_dir = proj_root / "migraine_data"
    model_dir = proj_root / "migraine_models"
    data_dir.mkdir(exist_ok=True)
    model_dir.mkdir(exist_ok=True)
    
    # Generate some example data files
    generate_example_data(data_dir)
    
    print("=" * 80)
    print("MIGRAINE DATA CLI EXAMPLES")
    print("=" * 80)
    
    # Example 1: Import original data
    print("\n\nEXAMPLE 1: Importing original data and training baseline model")
    print("-" * 60)
    
    cmd = [
        "python", str(proj_root / "main.py"),
        "--import-migraine-data",
        "--data-path", str(data_dir / "original_data.csv"),
        "--data-dir", str(data_dir),
        "--model-dir", str(model_dir),
        "--train-model",
        "--model-name", "baseline_model",
        "--model-description", "Baseline model with original features",
        "--make-default",
        "--summary"
    ]
    
    print(f"Running command: {' '.join(cmd)}")
    run_command(cmd)
    
    # Example 2: Import new data with additional columns
    print("\n\nEXAMPLE 2: Importing new data with additional columns")
    print("-" * 60)
    
    cmd = [
        "python", str(proj_root / "main.py"),
        "--import-migraine-data",
        "--data-path", str(data_dir / "new_data.csv"),
        "--data-dir", str(data_dir),
        "--model-dir", str(model_dir),
        "--add-new-columns",
        "--train-model",
        "--model-name", "enhanced_model",
        "--model-description", "Enhanced model with new features",
        "--make-default",
        "--summary"
    ]
    
    print(f"Running command: {' '.join(cmd)}")
    run_command(cmd)
    
    # Example 3: Add derived features
    print("\n\nEXAMPLE 3: Adding derived features")
    print("-" * 60)
    
    cmd = [
        "python", str(proj_root / "main.py"),
        "--import-migraine-data",
        "--data-path", str(data_dir / "new_data.csv"),
        "--data-dir", str(data_dir),
        "--model-dir", str(model_dir),
        "--derived-features", 
        "stress_per_sleep:df['stress_level']/df['sleep_hours']",
        "hydration_per_kg:df['hydration_ml']/(df['weight_kg']+0.001)",
        "--train-model",
        "--model-name", "derived_features_model",
        "--model-description", "Model with derived features",
        "--make-default",
        "--summary"
    ]
    
    print(f"Running command: {' '.join(cmd)}")
    run_command(cmd)
    
    # Example 4: Make predictions with missing features
    print("\n\nEXAMPLE 4: Making predictions with missing features")
    print("-" * 60)
    
    cmd = [
        "python", str(proj_root / "main.py"),
        "--predict-migraine",
        "--prediction-data", str(data_dir / "prediction_data.csv"),
        "--data-dir", str(data_dir),
        "--model-dir", str(model_dir),
        "--save-predictions",
        "--summary"
    ]
    
    print(f"Running command: {' '.join(cmd)}")
    run_command(cmd)
    
    print("\n" + "=" * 80)
    print("CLI EXAMPLES COMPLETED")
    print("=" * 80)

def generate_example_data(data_dir):
    """Generate example data files for the examples"""
    import pandas as pd
    import numpy as np
    
    # Create original data
    np.random.seed(42)
    n_samples = 100
    
    original_data = pd.DataFrame({
        'sleep_hours': np.random.normal(7, 1, n_samples),
        'stress_level': np.random.randint(1, 11, n_samples),
        'weather_pressure': np.random.normal(1013, 10, n_samples),
        'heart_rate': np.random.normal(75, 8, n_samples),
        'hormonal_level': np.random.normal(5, 2, n_samples),
        'migraine_occurred': np.random.binomial(1, 0.3, n_samples),
    })
    
    original_data.to_csv(data_dir / "original_data.csv", index=False)
    
    # Create new data with additional columns
    np.random.seed(43)
    n_samples = 80
    
    new_data = pd.DataFrame({
        'sleep_hours': np.random.normal(7, 1.2, n_samples),
        'stress_level': np.random.randint(1, 11, n_samples),
        'weather_pressure': np.random.normal(1010, 12, n_samples),
        'heart_rate': np.random.normal(78, 10, n_samples),
        'hormonal_level': np.random.normal(5.2, 2.2, n_samples),
        'screen_time_hours': np.random.normal(4.5, 2, n_samples),
        'hydration_ml': np.random.normal(1500, 500, n_samples),
        'activity_minutes': np.random.gamma(4, 15, n_samples),
        'caffeine_mg': np.random.exponential(100, n_samples),
        'weight_kg': np.random.normal(70, 12, n_samples),
        'migraine_occurred': np.random.binomial(1, 0.35, n_samples),
    })
    
    new_data.to_csv(data_dir / "new_data.csv", index=False)
    
    # Create prediction data with missing values
    pred_data = pd.DataFrame({
        'sleep_hours': [6.2, 8.1, 7.5, 5.8],
        'stress_level': [9, 3, 6, 8],
        'weather_pressure': [1020, 1005, 1015, 1008],
        'heart_rate': [78, 65, 72, 88],  
        'hormonal_level': [6.2, 4.8, 5.1, 7.2],
        'screen_time_hours': [7.5, 2.0, 3.5, 6.0],
        'hydration_ml': [1200, 2000, 1500, 800],  
        'activity_minutes': [45, 120, 60, 30],
        'caffeine_mg': [120, 80, 150, 20],
        'weight_kg': [65, 72, 80, 68],  
    })
    
    pred_data.to_csv(data_dir / "prediction_data.csv", index=False)
    
    print(f"Generated example data files in {data_dir}")
    print(f"  - original_data.csv: {original_data.shape} records")
    print(f"  - new_data.csv: {new_data.shape} records")
    print(f"  - prediction_data.csv: {pred_data.shape} records")

def run_command(cmd_list):
    """Run a command and print its output"""
    try:
        result = subprocess.run(
            cmd_list, 
            capture_output=True, 
            text=True, 
            check=True
        )
        print(result.stdout)
        if result.stderr:
            print(f"STDERR: {result.stderr}")
    except subprocess.CalledProcessError as e:
        print(f"Error running command: {e}")
        print(f"STDOUT: {e.stdout}")
        print(f"STDERR: {e.stderr}")
    except Exception as e:
        print(f"Exception running command: {e}")

if __name__ == "__main__":
    main()
