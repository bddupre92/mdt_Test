"""
generate_synthetic.py
---------------------
Generates a synthetic dataset simulating migraine events
alongside triggers like sleep hours, weather, etc.
"""

import numpy as np
import pandas as pd
from app.core.data.test_data_generator import TestDataGenerator

def generate_synthetic_data(num_days=180, random_seed=42):
    """
    Generate a synthetic dataset for migraine prediction.
    
    :param num_days: Number of days of data to generate
    :param random_seed: Random seed for reproducibility
    :return: A pandas DataFrame with columns:
             ['sleep_hours', 'stress_level', 'weather_pressure', 'heart_rate',
              'hormonal_level', 'migraine_occurred']
    """
    # Use our test data generator
    generator = TestDataGenerator(seed=random_seed)
    df = generator.generate_time_series(n_days=num_days)
    
    # Introduce missing values (5% random)
    rng = np.random.default_rng(random_seed)
    for col in ['sleep_hours', 'weather_pressure', 'heart_rate', 'stress_level', 'hormonal_level']:
        mask = rng.random(num_days) < 0.05
        df.loc[mask, col] = np.nan
        
    return df

def main():
    """Generate sample dataset and print summary."""
    df = generate_synthetic_data()
    print("Generated synthetic dataset:")
    print(df.describe())
    print("\nMissing values:")
    print(df.isna().sum())
    print("\nMigraine frequency:", df['migraine_occurred'].mean())

if __name__ == "__main__":
    main()
