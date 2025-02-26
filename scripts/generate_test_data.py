"""
Generate synthetic test data for development and testing.
"""
import os
import json
import pandas as pd
from pathlib import Path
from datetime import datetime

from app.core.data.test_data_generator import TestDataGenerator

def main():
    """Generate test datasets."""
    # Create data directory if it doesn't exist
    data_dir = Path(__file__).parent.parent / 'data'
    data_dir.mkdir(exist_ok=True)
    
    # Initialize generator with seed for reproducibility
    generator = TestDataGenerator(seed=42)
    
    # Generate datasets
    datasets = {
        'development': {
            'n_patients': 10,
            'n_days': 90,
            'include_drift': True
        },
        'testing': {
            'n_patients': 5,
            'n_days': 30,
            'include_drift': False
        },
        'validation': {
            'n_patients': 3,
            'n_days': 60,
            'include_drift': True
        }
    }
    
    for dataset_name, config in datasets.items():
        print(f"Generating {dataset_name} dataset...")
        
        # Create dataset directory
        dataset_dir = data_dir / dataset_name
        dataset_dir.mkdir(exist_ok=True)
        
        # Generate patient data
        patient_data = generator.generate_test_dataset(
            n_patients=config['n_patients'],
            n_days=config['n_days'],
            include_drift=config['include_drift']
        )
        
        # Save individual patient files
        for patient_id, data in patient_data.items():
            file_path = dataset_dir / f"patient_{patient_id}.csv"
            data.to_csv(file_path, index=False)
        
        # Save dataset metadata
        metadata = {
            'generated_at': datetime.now().isoformat(),
            'config': config,
            'n_patients': len(patient_data),
            'features': list(generator.feature_configs.keys()),
            'patient_files': [f"patient_{i}.csv" for i in range(1, len(patient_data) + 1)]
        }
        
        with open(dataset_dir / 'metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)
    
    print("\nGenerated test datasets:")
    for dataset_name in datasets:
        dataset_dir = data_dir / dataset_name
        n_files = len(list(dataset_dir.glob('*.csv')))
        print(f"- {dataset_name}: {n_files} patient files")
    
    # Generate validation set
    print("\nGenerating validation set...")
    stable_data, drift_data = generator.generate_validation_set()
    
    validation_dir = data_dir / 'validation'
    validation_dir.mkdir(exist_ok=True)
    
    stable_data.to_csv(validation_dir / 'stable_period.csv', index=False)
    drift_data.to_csv(validation_dir / 'drift_period.csv', index=False)
    
    print("Done! Test data has been generated in the data/ directory.")

if __name__ == "__main__":
    main()
