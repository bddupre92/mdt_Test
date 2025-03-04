"""Test script for optimization pipeline."""

import os
import sys
import numpy as np
import pandas as pd
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from main import run_optimization_and_evaluation

def generate_test_data(n_samples=1000, n_features=10):
    """Generate synthetic test data."""
    X, y = make_regression(
        n_samples=n_samples,
        n_features=n_features,
        noise=0.1,
        random_state=42
    )
    
    # Create a DataFrame
    X = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(n_features)])
    X['target'] = y
    
    # Save to CSV
    data_path = 'test_data.csv'
    X.to_csv(data_path, index=False)
    
    return data_path

def test_optimization_pipeline():
    """Test the complete optimization pipeline."""
    print("Starting optimization pipeline test...")
    
    # Generate test data
    data_path = generate_test_data()
    print("Generated test data...")
    
    # Run optimization and evaluation
    results = run_optimization_and_evaluation(
        data_path=data_path,
        save_dir='test_results',
        n_runs=5,  # Reduced for testing
        max_evals=100  # Reduced for testing
    )
    print("Completed optimization and evaluation...")
    
    # Verify results
    assert 'model' in results, "Model not found in results"
    assert 'evaluator' in results, "Evaluator not found in results"
    assert 'drift_detector' in results, "Drift detector not found in results"
    assert 'optimizer_analyzer' in results, "Optimizer analyzer not found in results"
    assert 'performance_metrics' in results, "Performance metrics not found in results"
    
    # Check if files were generated
    expected_files = [
        'test_results/plots/drift_detection_results.png',
        'test_results/plots/drift_analysis.png',
        'test_results/plots/framework_performance.png',
        'test_results/plots/pipeline_performance.png',
        'test_results/plots/performance_boxplot.png',
        'test_results/plots/model_evaluation.png',
        'test_results/plots/optimizer_landscape.png',
        'test_results/plots/optimizer_gradient.png',
        'test_results/plots/optimizer_parameters.png',
        'test_results/best_model.pkl',
        'test_results/optimization_report.txt'
    ]
    
    for file_path in expected_files:
        assert os.path.exists(file_path), f"Missing file: {file_path}"
        
    print("All tests passed!")
    print("\nGenerated files:")
    for file_path in expected_files:
        print(f"- {file_path}")
        
    # Clean up
    os.remove(data_path)

if __name__ == '__main__':
    test_optimization_pipeline()
