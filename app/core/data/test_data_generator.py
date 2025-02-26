"""
Test data generator for migraine prediction application.
"""
import numpy as np
import pandas as pd
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Tuple

class TestDataGenerator:
    """Generates synthetic data for testing and development."""
    
    def __init__(self, seed: Optional[int] = None):
        """Initialize generator with optional seed."""
        self.rng = np.random.RandomState(seed)
        
        # Define feature ranges and distributions
        self.feature_configs = {
            'sleep_hours': {
                'mean': 7.0,
                'std': 1.0,
                'min': 4.0,
                'max': 10.0,
                'weight': 2.0  # High importance
            },
            'stress_level': {
                'mean': 5.0,
                'std': 2.0,
                'min': 1.0,
                'max': 10.0,
                'weight': 2.5  # Highest importance
            },
            'weather_pressure': {
                'mean': 1013.0,
                'std': 5.0,
                'min': 990.0,
                'max': 1030.0,
                'weight': 1.0  # Lower importance
            },
            'heart_rate': {
                'mean': 75.0,
                'std': 8.0,
                'min': 50.0,
                'max': 100.0,
                'weight': 1.5  # Medium importance
            },
            'hormonal_level': {
                'mean': 50.0,
                'std': 15.0,
                'min': 0.0,
                'max': 100.0,
                'weight': 2.0  # High importance
            }
        }
        
        # Define migraine trigger thresholds
        self.trigger_thresholds = {
            'sleep_hours': {'low': 6.0, 'high': 9.0},
            'stress_level': {'low': 3.0, 'high': 7.0},
            'weather_pressure': {'low': 1000.0, 'high': 1025.0},
            'heart_rate': {'low': 60.0, 'high': 85.0},
            'hormonal_level': {'low': 30.0, 'high': 70.0}
        }
    
    def generate_single_record(self, include_drift: bool = False,
                             drift_factor: float = 0.0, time_index: int = 0) -> Dict[str, float]:
        """Generate a single data record."""
        record = {}
        
        for feature, config in self.feature_configs.items():
            # Apply drift if specified
            mean = config['mean']
            std = config['std']
            
            if include_drift:
                # Calculate drift effects
                range_width = config['max'] - config['min']
                
                # Use time_index to create consistent drift direction
                drift_phase = (time_index / 30) * 2 * np.pi  # Complete cycle every 30 days
                base_drift = np.sin(drift_phase)
                
                # Scale effects by drift_factor
                mean_shift = range_width * 0.3 * drift_factor  # Removed base_drift to make shift more consistent
                std_multiplier = 1 + 3 * abs(drift_factor)  # Up to 4x variance
                
                # Apply drift effects
                mean = mean + mean_shift
                std = std * std_multiplier
            
            # Generate value from normal distribution
            value = self.rng.normal(mean, std)
            
            # Clip to valid range
            value = np.clip(value, config['min'], config['max'])
            
            # Convert to appropriate type based on feature
            if feature in ['stress_level', 'heart_rate']:
                value = int(round(value))
            else:
                value = float(value)
            
            record[feature] = value
            
        return record
    
    def calculate_migraine_probability(self, record: Dict[str, float]) -> float:
        """Calculate migraine probability based on feature values."""
        risk_score = 0.0
        total_weight = 0.0
        
        # Check each feature against its thresholds
        for feature, value in record.items():
            if feature not in self.trigger_thresholds:
                continue
                
            thresholds = self.trigger_thresholds[feature]
            weight = self.feature_configs[feature]['weight']
            total_weight += weight
            
            # Calculate normalized deviation from optimal range
            if value < thresholds['low']:
                deviation = (thresholds['low'] - value) / (thresholds['low'] - self.feature_configs[feature]['min'])
            elif value > thresholds['high']:
                deviation = (value - thresholds['high']) / (self.feature_configs[feature]['max'] - thresholds['high'])
            else:
                deviation = 0.0
            
            risk_score += weight * deviation
        
        # Normalize risk score by total weight
        if total_weight > 0:
            risk_score = risk_score / total_weight
        
        # Apply sigmoid function with steeper slope
        probability = 1 / (1 + np.exp(-6 * (risk_score - 0.5)))
        return float(probability)
    
    def generate_time_series(self, n_days: int = 30, drift_start: int = None,
                        drift_magnitude: float = 0.5) -> pd.DataFrame:
        """Generate time series data."""
        records = []
        dates = []
        
        for i in range(n_days):
            # Determine if drift should be applied
            include_drift = drift_start is not None and i >= drift_start
            
            # Calculate drift factor with progressive increase
            if include_drift:
                progress = (i - drift_start) / (n_days - drift_start)
                base_drift = drift_magnitude * (1 + progress)  # Base drift increases over time
                
                # Add oscillation for more realistic patterns
                oscillation = 0.2 * np.sin(2 * np.pi * i / 7)  # Weekly cycles
                drift_factor = base_drift + oscillation
            else:
                drift_factor = 0.0
            
            # Generate record with time index for consistent drift
            record = self.generate_single_record(include_drift, drift_factor, i)
            
            # Calculate migraine probability
            prob = self.calculate_migraine_probability(record)
            record['migraine_probability'] = prob
            
            # Generate migraine occurrence based on probability
            record['migraine_occurred'] = 1 if self.rng.random() < prob else 0
            
            # Add date
            date = datetime.now(timezone.utc) + timedelta(days=i)
            dates.append(date)
            records.append(record)
        
        # Create DataFrame
        df = pd.DataFrame(records)
        df['date'] = dates
        
        return df
    
    def generate_test_dataset(self, n_patients: int, n_days: int,
                            include_drift: bool = True) -> Dict[int, pd.DataFrame]:
        """Generate test dataset for multiple patients."""
        datasets = {}
        
        for patient_id in range(1, n_patients + 1):
            # Randomly determine drift start day for each patient
            drift_start = self.rng.randint(n_days // 2, n_days) if include_drift else None
            
            # Generate patient data with increased drift magnitude
            data = self.generate_time_series(
                n_days=n_days,
                drift_start=drift_start,
                drift_magnitude=1.0 if include_drift else 0.0
            )
            
            # Add patient_id column
            data['patient_id'] = patient_id
            datasets[patient_id] = data
            
        return datasets
    
    def generate_validation_set(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Generate validation dataset with known drift patterns."""
        # Generate stable period with no drift
        stable_data = self.generate_time_series(n_days=30, drift_start=None)
        
        # Generate drift period with strong drift from the start
        drift_data = self.generate_time_series(n_days=30, drift_start=0, drift_magnitude=3.0)
        
        return stable_data, drift_data
