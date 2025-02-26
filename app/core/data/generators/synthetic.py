"""
Synthetic data generator for production use.
"""
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

from .base import BaseDataGenerator

@dataclass
class SyntheticConfig:
    """Configuration for synthetic data generation."""
    n_patients: int
    time_range_days: int
    missing_rate: float
    feature_ranges: Optional[Dict[str, Tuple[float, float]]] = None
    drift_points: Optional[List[int]] = None
    random_seed: Optional[int] = None

class SyntheticDataGenerator(BaseDataGenerator):
    """Production synthetic data generator with advanced features."""
    
    def __init__(self, config: SyntheticConfig):
        """Initialize generator with configuration."""
        super().__init__(seed=config.random_seed)
        self.config = config
        
        if config.feature_ranges:
            self._update_feature_ranges(config.feature_ranges)
    
    def _update_feature_ranges(self, ranges: Dict[str, Tuple[float, float]]):
        """Update feature ranges from configuration."""
        for feature, (min_val, max_val) in ranges.items():
            if feature in self.feature_configs:
                self.feature_configs[feature]['min'] = min_val
                self.feature_configs[feature]['max'] = max_val
    
    def generate_single_record(self, include_drift: bool = False,
                             drift_magnitude: float = 0.0) -> Dict[str, float]:
        """Generate a single data record with optional drift."""
        record = {}
        
        for feature, config in self.feature_configs.items():
            # Apply drift if specified
            mean = config['mean']
            if include_drift:
                mean += config['std'] * drift_magnitude
            
            # Generate value
            value = self.rng.normal(mean, config['std'])
            record[feature] = self._clip_to_range(value, feature)
        
        # Randomly introduce missing values
        if self.config.missing_rate > 0:
            for feature in list(record.keys()):
                if self.rng.random() < self.config.missing_rate:
                    record[feature] = np.nan
        
        return record
    
    def generate_time_series(self, patient_id: int) -> pd.DataFrame:
        """Generate time series data for a patient."""
        records = []
        start_date = datetime.now() - timedelta(days=self.config.time_range_days)
        
        for day in range(self.config.time_range_days):
            # Check if we're at a drift point
            include_drift = False
            drift_magnitude = 0.0
            if self.config.drift_points:
                for drift_point in self.config.drift_points:
                    if day >= drift_point:
                        include_drift = True
                        drift_magnitude = 0.5 * (day - drift_point) / (self.config.time_range_days - drift_point)
            
            # Generate record
            record = self.generate_single_record(include_drift, drift_magnitude)
            
            # Add metadata
            record['patient_id'] = patient_id
            record['date'] = start_date + timedelta(days=day)
            record['migraine_probability'] = self.calculate_migraine_probability(record)
            record['migraine_occurred'] = self.rng.random() < record['migraine_probability']
            
            records.append(record)
        
        return pd.DataFrame(records)
    
    def generate_dataset(self) -> Dict[int, pd.DataFrame]:
        """Generate complete dataset for all patients."""
        datasets = {}
        
        for patient_id in range(1, self.config.n_patients + 1):
            datasets[patient_id] = self.generate_time_series(patient_id)
        
        return datasets
