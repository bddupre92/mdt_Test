"""
Example demonstrating the Time Series Validation framework for medical data.

This script shows how to use the specialized cross-validation strategies for time-series 
medical data, which prevent data leakage and respect temporal dependencies.
"""

import sys
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Add parent directory to path to allow imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import the time series validation module
from moe_framework.validation.time_series_validation import TimeSeriesValidator

def main():
    """Run time series validation example."""
    # Load the synthetic medical data
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
    features = [
        'heart_rate', 'blood_pressure_sys', 'blood_pressure_dia', 'temperature',
        'sleep_hours', 'steps', 'exercise_minutes', 'stress_level',
        'temperature_env', 'humidity', 'barometric_pressure', 'air_quality',
        'medication_a_dose', 'medication_b_dose', 'days_since_last_medication'
    ]
    target = 'migraine_intensity'
    
    # Create TimeSeriesValidator
    validator = TimeSeriesValidator(
        time_column='timestamp',
        patient_column='patient_id',
        gap_size=1,  # Gap between train and test to prevent leakage
    )
    
    # Create a simple model for demonstration
    model = RandomForestRegressor(n_estimators=50, random_state=42)
    
    # Compare different validation strategies
    validation_methods = ['rolling_window', 'patient_aware', 'expanding_window']
    results = {}
    
    for method in validation_methods:
        print(f"\nRunning {method} validation...")
        validation_results = validator.get_validation_scores(
            data=data,
            model=model,
            features=features,
            target=target,
            method=method,
            n_splits=5
        )
        results[method] = validation_results
        
        # Print summary statistics
        rmse_mean = validation_results['scores']['rmse']['mean']
        rmse_std = validation_results['scores']['rmse']['std']
        r2_mean = validation_results['scores']['r2']['mean']
        r2_std = validation_results['scores']['r2']['std']
        
        print(f"  RMSE: {rmse_mean:.4f} ± {rmse_std:.4f}")
        print(f"  R²: {r2_mean:.4f} ± {r2_std:.4f}")
        
        # Print fold information
        print("  Fold details:")
        for fold in validation_results['folds']:
            print(f"    Fold {fold['fold']+1}: Train size={fold['train_size']}, "
                  f"Test size={fold['test_size']}, RMSE={fold['scores']['rmse']:.4f}")
    
    # Create visualizations directory
    viz_dir = Path(__file__).parent.parent / 'visualizations'
    os.makedirs(viz_dir, exist_ok=True)
    
    # Visualize results
    plt.figure(figsize=(12, 6))
    
    # Plot RMSE comparison
    methods = list(results.keys())
    rmse_means = [results[m]['scores']['rmse']['mean'] for m in methods]
    rmse_stds = [results[m]['scores']['rmse']['std'] for m in methods]
    
    plt.subplot(1, 2, 1)
    plt.bar(methods, rmse_means, yerr=rmse_stds, capsize=10)
    plt.title('RMSE by Validation Method (lower is better)')
    plt.ylabel('RMSE')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Plot R² comparison
    r2_means = [results[m]['scores']['r2']['mean'] for m in methods]
    r2_stds = [results[m]['scores']['r2']['std'] for m in methods]
    
    plt.subplot(1, 2, 2)
    plt.bar(methods, r2_means, yerr=r2_stds, capsize=10)
    plt.title('R² by Validation Method (higher is better)')
    plt.ylabel('R²')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig(viz_dir / 'validation_method_comparison.png')
    
    # Create another visualization showing prediction vs actual for one method
    best_method = methods[np.argmin(rmse_means)]
    print(f"\nCreating visualization for best method: {best_method}")
    
    # Train model on a specific train/test split
    validator = TimeSeriesValidator(
        time_column='timestamp',
        patient_column='patient_id',
        gap_size=1
    )
    
    # Get a robust split for the best method
    # Use n_splits=2 to ensure we have at least one split available
    split_func = getattr(validator, f"{best_method}_split")
    splits = list(split_func(data, n_splits=2))
    
    if not splits:
        print("Warning: No splits were generated. Using simple train/test split.")
        # Fallback to a simple temporal split
        data = data.sort_values('timestamp')
        split_point = int(len(data) * 0.8)
        train_idx = data.index[:split_point].tolist()
        test_idx = data.index[split_point:].tolist()
    else:
        train_idx, test_idx = splits[0]  # Use the first split
    
    X_train = data.iloc[train_idx][features]
    y_train = data.iloc[train_idx][target]
    X_test = data.iloc[test_idx][features]
    y_test = data.iloc[test_idx][target]
    
    # Train model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    
    # Plot predictions vs actual
    plt.figure(figsize=(10, 6))
    
    # Sort by timestamp for proper temporal visualization
    test_data = data.iloc[test_idx].sort_values('timestamp')
    actual = test_data[target].values
    
    # Re-get predictions in time order
    temporal_preds = model.predict(test_data[features])
    
    plt.plot(actual, label='Actual', marker='o')
    plt.plot(temporal_preds, label='Predicted', marker='x')
    plt.title(f'Time Series Predictions using {best_method.replace("_", " ").title()} Validation\nRMSE: {rmse:.4f}, R²: {r2:.4f}')
    plt.xlabel('Time Step (sorted by timestamp)')
    plt.ylabel('Migraine Intensity')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig(viz_dir / 'time_series_predictions.png')
    
    print(f"\nValidation comparison complete.")
    print(f"Visualizations saved to {viz_dir}")
    print(f"Best validation method: {best_method} (RMSE: {min(rmse_means):.4f})")

if __name__ == "__main__":
    main()
