"""
test_data_generation.py
-----------------------
Unit tests for data generation and preprocessing.
"""

import unittest
from data.generate_synthetic import generate_synthetic_data
from data.preprocessing import preprocess_data

class TestDataGeneration(unittest.TestCase):
    def test_generate_synthetic(self):
        df = generate_synthetic_data(num_days=30, random_seed=0)
        self.assertFalse(df.empty)
        self.assertIn('migraine_occurred', df.columns)
    
    def test_preprocessing(self):
        df = generate_synthetic_data(num_days=30)
        df_clean = preprocess_data(df, strategy_numeric='mean', scale_method='minmax')
        self.assertFalse(df_clean.isna().any().any(), "Should have no missing after imputation")
        # Check scaling range
        for col in df_clean.select_dtypes(include=['float','int']):
            min_val = df_clean[col].min()
            max_val = df_clean[col].max()
            self.assertTrue(min_val >= 0 and max_val <= 1 or col in ['migraine_occurred','severity'],
                            f"{col} not scaled or is a label")

"""
Tests for synthetic data generation.
"""
import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

from app.core.data.test_data_generator import TestDataGenerator

def test_generator_initialization():
    """Test generator initialization with seed."""
    generator = TestDataGenerator(seed=42)
    assert generator.rng.rand() == pytest.approx(0.37454011884736246)

def test_single_record_generation():
    """Test generation of a single record."""
    generator = TestDataGenerator(seed=42)
    record = generator.generate_single_record()
    
    # Check all features are present
    expected_features = [
        'sleep_hours', 'stress_level', 'weather_pressure',
        'heart_rate', 'hormonal_level'
    ]
    assert all(feature in record for feature in expected_features)
    
    # Check values are within expected ranges
    for feature, value in record.items():
        config = generator.feature_configs[feature]
        assert config['min'] <= value <= config['max']

def test_migraine_probability_calculation():
    """Test migraine probability calculation."""
    generator = TestDataGenerator(seed=42)
    
    # Test optimal values (should give low probability)
    optimal_record = {
        'sleep_hours': 7.5,
        'stress_level': 5.0,
        'weather_pressure': 1013.0,
        'heart_rate': 70.0,
        'hormonal_level': 50.0
    }
    prob_optimal = generator.calculate_migraine_probability(optimal_record)
    assert prob_optimal < 0.3
    
    # Test extreme values (should give high probability)
    extreme_record = {
        'sleep_hours': 4.0,
        'stress_level': 9.0,
        'weather_pressure': 1029.0,
        'heart_rate': 95.0,
        'hormonal_level': 90.0
    }
    prob_extreme = generator.calculate_migraine_probability(extreme_record)
    assert prob_extreme > 0.7

def test_time_series_generation():
    """Test time series data generation."""
    generator = TestDataGenerator(seed=42)
    n_days = 60
    
    # Generate time series without drift
    data_no_drift = generator.generate_time_series(n_days=n_days)
    assert len(data_no_drift) == n_days
    assert all(col in data_no_drift.columns for col in ['date', 'migraine_probability', 'migraine_occurred'])
    
    # Generate time series with drift
    data_with_drift = generator.generate_time_series(n_days=n_days, drift_start=30)
    
    # Check if drift is present by comparing means
    early_mean = data_with_drift.iloc[:30]['stress_level'].mean()
    late_mean = data_with_drift.iloc[30:]['stress_level'].mean()
    assert abs(late_mean - early_mean) > 0.5

def test_multi_patient_generation():
    """Test generation of multiple patient datasets."""
    generator = TestDataGenerator(seed=42)
    n_patients = 3
    n_days = 30
    
    datasets = generator.generate_test_dataset(n_patients, n_days)
    
    assert len(datasets) == n_patients
    for patient_id, data in datasets.items():
        assert len(data) == n_days
        assert isinstance(data, pd.DataFrame)

def test_validation_set_generation():
    """Test validation set generation."""
    generator = TestDataGenerator(seed=42)
    
    stable_data, drift_data = generator.generate_validation_set()
    
    # Check data shapes
    assert len(stable_data) == 30
    assert len(drift_data) == 30
    
    # Verify drift presence
    for feature in generator.feature_configs.keys():
        stable_std = stable_data[feature].std()
        drift_std = drift_data[feature].std()
        assert drift_std > stable_std

def test_reproducibility():
    """Test reproducibility with same seed."""
    gen1 = TestDataGenerator(seed=42)
    gen2 = TestDataGenerator(seed=42)
    
    data1 = gen1.generate_single_record()
    data2 = gen2.generate_single_record()
    
    for feature in data1.keys():
        assert data1[feature] == data2[feature]

def test_drift_magnitude():
    """Test drift magnitude effects."""
    generator = TestDataGenerator(seed=42)
    
    # Generate data with different drift magnitudes
    small_drift = generator.generate_time_series(n_days=30, drift_start=0, drift_magnitude=0.2)
    large_drift = generator.generate_time_series(n_days=30, drift_start=0, drift_magnitude=1.0)
    
    # Compare drift effects using standard deviation
    for feature in generator.feature_configs.keys():
        small_std = small_drift[feature].std()
        large_std = large_drift[feature].std()
        assert large_std > small_std, f"Large drift should have higher variance for {feature}"
