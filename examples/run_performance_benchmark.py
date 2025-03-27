"""
Example script demonstrating the MoE performance benchmarking framework.

This script runs performance benchmarks on the MoE framework using the synthetic
medical dataset, comparing different integration strategies and gating networks.
"""

import os
import sys
import pandas as pd
import numpy as np
import logging
from pathlib import Path

# Add parent directory to path to allow imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from moe_framework.benchmark.performance_benchmarks import BenchmarkRunner

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

def main():
    """Run MoE performance benchmarks."""
    # Load synthetic data
    data_path = Path(__file__).parent.parent / 'data' / 'synthetic_medical_data.csv'
    if not data_path.exists():
        print(f"Error: Synthetic data file not found at {data_path}")
        print("Please run examples/synthetic_data.py first to generate the data.")
        return
    
    data = pd.read_csv(data_path)
    print(f"Loaded synthetic data with {len(data)} samples.")
    
    # Convert timestamp to datetime
    data['timestamp'] = pd.to_datetime(data['timestamp'])
    
    # Handle missing values (simple imputation for demonstration)
    for col in data.columns:
        if col not in ['patient_id', 'timestamp', 'migraine_intensity']:
            data[col] = data[col].fillna(data[col].mean())
    
    # Define features and target
    physiological_features = ['heart_rate', 'blood_pressure_sys', 'blood_pressure_dia', 'temperature']
    behavioral_features = ['sleep_hours', 'steps', 'exercise_minutes', 'stress_level']
    environmental_features = ['temperature_env', 'humidity', 'barometric_pressure', 'air_quality']
    medication_features = ['medication_a_dose', 'medication_b_dose', 'days_since_last_medication']
    
    # All features
    features = physiological_features + behavioral_features + environmental_features + medication_features
    target = 'migraine_intensity'
    
    # Create output directory for benchmark results
    output_dir = Path(__file__).parent.parent / 'benchmark_results'
    os.makedirs(output_dir, exist_ok=True)
    
    # Create benchmark runner
    runner = BenchmarkRunner(
        output_dir=str(output_dir),
        create_visualizations=True
    )
    
    # Define base configuration for the MoE pipeline
    base_config = {
        'experts': {
            'physiological_expert': {
                'type': 'physiological',
                'model': 'gradient_boosting',
                'features': physiological_features
            },
            'behavioral_expert': {
                'type': 'behavioral',
                'model': 'random_forest',
                'features': behavioral_features
            },
            'environmental_expert': {
                'type': 'environmental',
                'model': 'linear',
                'features': environmental_features
            },
            'medication_expert': {
                'type': 'medication_history',
                'model': 'gradient_boosting',
                'features': medication_features
            }
        },
        'gating_network': {
            'type': 'quality_aware',
            'quality_threshold': 0.7,
            'default_weights': {
                'physiological_expert': 0.4,
                'behavioral_expert': 0.2,
                'environmental_expert': 0.2,
                'medication_expert': 0.2
            }
        },
        'integration': {
            'strategy': 'weighted_average'
        }
    }
    
    print("Starting MoE performance benchmarking...")
    print(f"Results will be saved to {output_dir}")
    
    # Split data for benchmarking
    from sklearn.model_selection import train_test_split
    train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)
    
    # Run integration strategy benchmarks with a smaller dataset for quick demonstration
    sample_size = min(500, len(train_data))
    sample_train_data = train_data.sample(sample_size, random_state=42)
    sample_test_data = test_data.sample(min(100, len(test_data)), random_state=42)
    
    print(f"\nBenchmarking integration strategies using {sample_size} samples...")
    integration_results = runner.benchmark_integration_strategies(
        data=sample_train_data,
        features=features,
        target=target,
        base_config=base_config,
        test_data=sample_test_data
    )
    
    print("\nBenchmarking gating networks...")
    gating_results = runner.benchmark_gating_networks(
        data=sample_train_data,
        features=features,
        target=target,
        base_config=base_config,
        test_data=sample_test_data
    )
    
    # Compare all results
    print("\nGenerating comparison report...")
    comparison = runner.compare_all_results()
    
    print("\nBenchmarking complete!")
    print(f"Results have been saved to {output_dir}")
    print("Summary of findings:")
    
    # Print summary of best performing configurations
    best_rmse = float('inf')
    best_config = None
    
    for result in runner.results:
        rmse = result['performance']['average']['accuracy'].get('rmse', float('inf'))
        if rmse < best_rmse:
            best_rmse = rmse
            best_config = result['name']
    
    if best_config:
        print(f"Best performing configuration: {best_config}")
        print(f"Best RMSE: {best_rmse:.4f}")
    
    print("\nVisualization charts have been generated in the benchmark_results directory.")

if __name__ == "__main__":
    main()
