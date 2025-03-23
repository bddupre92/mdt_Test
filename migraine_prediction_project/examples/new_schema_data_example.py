#!/usr/bin/env python3
"""
Example script demonstrating how to import new migraine data with additional columns
using the new MigrainePredictorV2 class.
"""

import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path

# Add the project root to the Python path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Import the migraine predictor
from src.migraine_model.new_data_migraine_predictor import MigrainePredictorV2

def create_sample_data(n_samples=100, seed=42):
    """Generate sample migraine data with core and optional features"""
    np.random.seed(seed)
    
    # Core features
    data = {
        'sleep_hours': np.random.normal(7, 1, n_samples),
        'stress_level': np.random.randint(1, 11, n_samples),
        'weather_pressure': np.random.normal(1013, 10, n_samples),
        'heart_rate': np.random.normal(75, 8, n_samples),
        'hormonal_level': np.random.normal(5, 2, n_samples),
        'migraine_occurred': np.random.binomial(1, 0.3, n_samples),  # 30% migraine rate
    }
    
    return pd.DataFrame(data)

def create_new_data_with_columns(n_samples=80, seed=43):
    """Generate new data with additional columns"""
    np.random.seed(seed)
    
    # Start with core features
    data = {
        'sleep_hours': np.random.normal(7, 1.2, n_samples),
        'stress_level': np.random.randint(1, 11, n_samples),
        'weather_pressure': np.random.normal(1010, 12, n_samples),
        'heart_rate': np.random.normal(78, 10, n_samples),
        'hormonal_level': np.random.normal(5.2, 2.2, n_samples),
        'migraine_occurred': np.random.binomial(1, 0.35, n_samples),  # 35% migraine rate
    }
    
    # Add new features
    data.update({
        'screen_time_hours': np.random.normal(4.5, 2, n_samples),
        'hydration_ml': np.random.normal(1500, 500, n_samples),
        'activity_minutes': np.random.gamma(4, 15, n_samples),  # Gamma distribution for skewed activity
        'caffeine_mg': np.random.exponential(100, n_samples),  # Exponential for caffeine consumption
    })
    
    return pd.DataFrame(data)

def main():
    """Demonstrate handling new data with different schema"""
    print("Migraine Prediction Example - Handling New Data Schema")
    print("=" * 50)
    
    # Create directories for data and models
    data_dir = Path("example_data")
    model_dir = Path("example_models")
    data_dir.mkdir(exist_ok=True)
    model_dir.mkdir(exist_ok=True)
    
    # Step 1: Create original data and save it
    print("\nStep 1: Creating original training data...")
    original_data = create_sample_data(n_samples=100)
    original_data_path = data_dir / "original_migraine_data.csv"
    original_data.to_csv(original_data_path, index=False)
    print(f"Original data shape: {original_data.shape}")
    print(f"Original data columns: {original_data.columns.tolist()}")
    
    # Step 2: Create new data with additional columns and save it
    print("\nStep 2: Creating new data with additional columns...")
    new_data = create_new_data_with_columns(n_samples=80)
    new_data_path = data_dir / "new_migraine_data.csv"
    new_data.to_csv(new_data_path, index=False)
    print(f"New data shape: {new_data.shape}")
    print(f"New data columns: {new_data.columns.tolist()}")
    print(f"New columns added: {set(new_data.columns) - set(original_data.columns)}")
    
    # Step 3: Initialize the predictor
    print("\nStep 3: Initializing migraine predictor...")
    predictor = MigrainePredictorV2(
        model_dir=str(model_dir),
        data_dir=str(data_dir)
    )
    print(f"Initial feature columns: {predictor.feature_columns}")
    
    # Step 4: Train with original data
    print("\nStep 4: Training model with original data...")
    model_id = predictor.train_with_new_data(
        data_path=str(original_data_path),
        model_name="baseline_model",
        description="Baseline model with original features",
        make_default=True,
        add_new_columns=False  # Not adding new columns in this step
    )
    print(f"Trained model ID: {model_id}")
    print(f"Feature columns after training: {predictor.feature_columns}")
    
    # Step 5: Import new data with additional columns
    print("\nStep 5: Importing new data with additional columns...")
    imported_data = predictor.import_data(
        data_path=str(new_data_path),
        add_new_columns=True  # Add new columns to schema
    )
    print(f"Imported data shape: {imported_data.shape}")
    print(f"Updated feature columns: {predictor.feature_columns}")
    
    # Step 6: Train an enhanced model with the new features
    print("\nStep 6: Training enhanced model with new features...")
    enhanced_model_id = predictor.train(
        data=imported_data,
        model_name="enhanced_model",
        description="Enhanced model with additional features",
        make_default=True
    )
    print(f"Enhanced model ID: {enhanced_model_id}")
    
    # Step 7: Create a derived feature
    print("\nStep 7: Adding a derived feature...")
    predictor.add_derived_feature(
        name="stress_per_sleep",
        formula="df['stress_level'] / df['sleep_hours']"
    )
    print(f"Feature columns after adding derived feature: {predictor.feature_columns}")
    
    # Step 8: Train a model with the derived feature
    print("\nStep 8: Training model with derived feature...")
    derived_model_id = predictor.train(
        data=imported_data,  # The derived feature will be calculated during training
        model_name="derived_features_model",
        description="Model with derived stress_per_sleep feature",
        make_default=True
    )
    print(f"Model with derived features ID: {derived_model_id}")
    
    # Step 9: Make predictions with missing features
    print("\nStep 9: Making predictions with incomplete data...")
    test_data = pd.DataFrame({
        'sleep_hours': [6.2, 8.1],
        'stress_level': [9, 3],
        'weather_pressure': [1020, 1005],
        # Missing 'heart_rate'
        'hormonal_level': [6.2, 4.8],
        'screen_time_hours': [7.5, 2.0],
        # Missing 'hydration_ml'
        'activity_minutes': [45, 120],
        # Missing 'caffeine_mg'
    })
    
    print("Test data for prediction:")
    print(test_data)
    
    try:
        predictions = predictor.predict_with_missing_features(test_data)
        print("\nPredictions with missing features:")
        for i, pred in enumerate(predictions):
            print(f"Sample {i+1}:")
            print(f"  Prediction: {'Migraine' if pred['prediction'] == 1 else 'No Migraine'}")
            print(f"  Probability: {pred['probability']:.2f}")
            print(f"  Top features by importance:")
            # Sort feature importances
            sorted_features = sorted(
                pred['feature_importances'].items(), 
                key=lambda x: x[1], 
                reverse=True
            )[:3]  # Top 3 features
            for feature, importance in sorted_features:
                print(f"    {feature}: {importance:.4f}")
    except Exception as e:
        print(f"Error making predictions: {e}")
        
    # Step 10: Show schema information
    print("\nStep 10: Schema information:")
    schema_info = predictor.get_schema_info()
    print(f"Schema version: {schema_info['version']}")
    print(f"Core features: {schema_info['core_features']}")
    print(f"Optional features: {schema_info['optional_features']}")
    print(f"Derived features: {schema_info['derived_features']}")
    
    print("\nExample completed successfully!")

if __name__ == "__main__":
    main()
