#!/usr/bin/env python3
"""
Example script demonstrating how to import new migraine data with additional columns.
"""

import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path

# Add the project root to the Python path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Import the enhanced migraine predictor
from src.migraine_model.migraine_predictor_extension import EnhancedMigrainePredictor

def create_sample_data(output_path, has_new_columns=False):
    """Create sample data with or without new columns."""
    # Create a dataframe with the core features
    np.random.seed(42)
    n_samples = 100
    
    data = {
        'sleep_hours': np.random.normal(7, 1, n_samples),
        'stress_level': np.random.randint(1, 11, n_samples),
        'weather_pressure': np.random.normal(1013, 10, n_samples),
        'heart_rate': np.random.normal(75, 8, n_samples),
        'hormonal_level': np.random.normal(5, 2, n_samples),
        'migraine_occurred': np.random.randint(0, 2, n_samples)
    }
    
    # Add new columns if specified
    if has_new_columns:
        data['activity_minutes'] = np.random.randint(0, 120, n_samples)
        data['hydration_level'] = np.random.normal(70, 15, n_samples)
        data['screen_time_hours'] = np.random.normal(4, 2, n_samples)
    
    # Create the dataframe
    df = pd.DataFrame(data)
    
    # Save to CSV
    df.to_csv(output_path, index=False)
    print(f"Sample data saved to {output_path}")
    
    return df

def main():
    """Main function demonstrating the use of the enhanced migraine predictor."""
    # Create directories for data and models
    data_dir = Path("example_data")
    model_dir = Path("example_models")
    
    data_dir.mkdir(exist_ok=True)
    model_dir.mkdir(exist_ok=True)
    
    # Create sample data files
    original_data_path = data_dir / "original_migraine_data.csv"
    new_data_path = data_dir / "new_migraine_data_with_columns.csv"
    
    create_sample_data(original_data_path, has_new_columns=False)
    create_sample_data(new_data_path, has_new_columns=True)
    
    # Initialize the enhanced migraine predictor
    predictor = EnhancedMigrainePredictor(
        model_dir=str(model_dir),
        data_dir=str(data_dir)
    )
    
    # Demo 1: Train a model with original data
    print("\n=== Step 1: Training with original data ===")
    model_id = predictor.train_with_new_data(
        data_path=str(original_data_path),
        model_name="original_model",
        description="Model trained with original migraine data",
        make_default=True
    )
    print(f"Model trained with ID: {model_id}")
    
    # Print feature columns used
    print(f"Features used: {predictor.feature_columns}")
    
    # Demo 2: Import new data with additional columns
    print("\n=== Step 2: Importing new data with additional columns ===")
    new_data = predictor.import_data(
        data_path=str(new_data_path),
        add_new_columns=True  # This will add the new columns to the schema
    )
    print(f"New data imported with columns: {new_data.columns.tolist()}")
    
    # Print updated feature columns
    print(f"Updated features: {predictor.feature_columns}")
    
    # Demo 3: Train a new model with the new data and columns
    print("\n=== Step 3: Training with new data including additional columns ===")
    new_model_id = predictor.train(
        data=new_data,
        model_name="enhanced_model",
        description="Model trained with enhanced migraine data including new features",
        make_default=True
    )
    print(f"Enhanced model trained with ID: {new_model_id}")
    
    # Demo 4: Add a derived feature
    print("\n=== Step 4: Adding a derived feature ===")
    predictor.add_derived_feature(
        name="stress_sleep_ratio",
        formula="df['stress_level'] / df['sleep_hours']"
    )
    print(f"Derived feature added. Updated features: {predictor.feature_columns}")
    
    # Demo 5: Train with the derived feature
    print("\n=== Step 5: Training with derived features ===")
    # Process data to include derived features
    new_data_with_derived = predictor.data_handler.process_data(new_data)
    
    derived_model_id = predictor.train(
        data=new_data_with_derived,
        model_name="derived_model",
        description="Model trained with derived features",
        make_default=True
    )
    print(f"Model with derived features trained with ID: {derived_model_id}")
    
    # Demo 6: Display schema information
    print("\n=== Step 6: Schema information ===")
    schema_info = predictor.get_schema_info()
    for key, value in schema_info.items():
        print(f"{key}: {value}")
    
    # Demo 7: Make predictions with missing columns
    print("\n=== Step 7: Making predictions with missing columns ===")
    
    # Store feature defaults in model metadata
    predictor.model_metadata["feature_defaults"] = {
        "heart_rate": new_data["heart_rate"].mean(),
        "hydration_level": new_data["hydration_level"].mean() if "hydration_level" in new_data else 70.0,
        "activity_minutes": new_data["activity_minutes"].mean() if "activity_minutes" in new_data else 60.0
    }
    
    # Create a sample with missing columns
    test_data = pd.DataFrame({
        'sleep_hours': [6.5],
        'stress_level': [8],
        'weather_pressure': [1015],
        # 'heart_rate' is missing
        'hormonal_level': [4.5],
        'screen_time_hours': [6],  # This is one of the new columns
    })
    
    try:
        # With defaults in model_metadata, this should now work
        predictions = predictor.predict_with_missing_features(test_data)
        print(f"Predictions with missing features: {predictions}")
    except Exception as e:
        print(f"Error making predictions with missing features: {e}")
    
    print("\nExample completed successfully!")

if __name__ == "__main__":
    main()
