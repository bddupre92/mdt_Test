"""
Test data generator for development and testing.
"""
from typing import Dict, Optional, Tuple
import pandas as pd
from datetime import datetime, timedelta
import numpy as np

from .base import BaseDataGenerator

class TestDataGenerator(BaseDataGenerator):
    """Simplified data generator for testing purposes."""
    
    def generate_single_record(self, include_drift: bool = False,
                             drift_factor: float = 0.0) -> Dict[str, float]:
        """Generate a single test record."""
        record = {}
        
        for feature, config in self.feature_configs.items():
            # Apply drift if specified
            if include_drift and config.get('drift_susceptible', True):
                mean = config['mean'] + (config['std'] * drift_factor * 4)  # Increased from 3 to 4
                std = config['std'] * (1 + abs(drift_factor) * 2)  # Increased variance impact
            else:
                mean = config['mean']
                std = config['std']
            
            # Generate value with occasional extreme values
            if include_drift and self.rng.random() < 0.2:  # 20% chance of extreme value
                value = mean + (self.rng.choice([-1, 1]) * config['std'] * 3)
            else:
                value = self.rng.normal(mean, std)
            record[feature] = self._clip_to_range(value, feature)
        
        return record
    
    def generate_time_series(self, n_days: int, 
                           drift_start: Optional[int] = None,
                           drift_magnitude: float = 1.0) -> pd.DataFrame:
        """Generate time series data with optional concept drift.
        
        Args:
            n_days: Number of days to generate
            drift_start: Day to start drift, or None for no drift
            drift_magnitude: Magnitude of drift (default: 1.0)
        """
        records = []
        start_date = datetime.now() - timedelta(days=n_days)
        
        for day in range(n_days):
            # Determine if drift should be applied
            include_drift = False
            drift_factor = 0.0
            if drift_start is not None and day >= drift_start:
                include_drift = True
                # More aggressive drift progression
                progress = (day - drift_start) / (n_days - drift_start)
                drift_factor = drift_magnitude * (1 - np.exp(-3 * progress))  # Changed from tanh to exponential
            
            # Generate record
            record = self.generate_single_record(include_drift, drift_factor)
            
            # Add date and calculate migraine probability
            record['date'] = start_date + timedelta(days=day)
            record['migraine_probability'] = self.calculate_migraine_probability(record)
            record['migraine_occurred'] = self.rng.random() < record['migraine_probability']
            
            records.append(record)
        
        return pd.DataFrame(records)
    
    def generate_test_dataset(self, n_patients: int, n_days: int,
                            include_drift: bool = True) -> Dict[int, pd.DataFrame]:
        """Generate test dataset for multiple patients."""
        datasets = {}
        
        for patient_id in range(1, n_patients + 1):
            # Randomly determine drift start day for each patient
            drift_start = self.rng.randint(n_days // 2, n_days) if include_drift else None
            
            # Generate patient data
            data = self.generate_time_series(
                n_days=n_days,
                drift_start=drift_start,
                drift_magnitude=1.0
            )
            
            datasets[patient_id] = data
            
        return datasets
    
    def generate_validation_set(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Generate validation dataset with known drift patterns."""
        # Generate stable period
        stable_data = self.generate_time_series(n_days=30, drift_start=None)
        
        # Generate drift period with strong drift
        drift_data = self.generate_time_series(n_days=30, drift_start=0, drift_magnitude=2.0)
        
        return stable_data, drift_data
