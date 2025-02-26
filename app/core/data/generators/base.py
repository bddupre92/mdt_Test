"""
Base class for data generation.
"""
from abc import ABC, abstractmethod
from typing import Dict, Optional
import numpy as np
import pandas as pd

class BaseDataGenerator(ABC):
    """Base class for all data generators."""
    
    def __init__(self, seed: Optional[int] = None):
        """Initialize generator with optional seed."""
        self.rng = np.random.RandomState(seed)
        
        # Default feature configurations
        self.feature_configs = {
            'sleep_hours': {
                'mean': 7.0,
                'std': 1.0,
                'min': 4.0,
                'max': 10.0
            },
            'stress_level': {
                'mean': 5.0,
                'std': 2.0,
                'min': 1.0,
                'max': 10.0
            },
            'weather_pressure': {
                'mean': 1013.0,
                'std': 5.0,
                'min': 990.0,
                'max': 1030.0
            },
            'heart_rate': {
                'mean': 75.0,
                'std': 8.0,
                'min': 50.0,
                'max': 100.0
            },
            'hormonal_level': {
                'mean': 50.0,
                'std': 15.0,
                'min': 0.0,
                'max': 100.0
            }
        }
        
        # Default trigger thresholds
        self.trigger_thresholds = {
            'sleep_hours': {'low': 6.0, 'high': 9.0},
            'stress_level': {'low': 3.0, 'high': 7.0},
            'weather_pressure': {'low': 1000.0, 'high': 1025.0},
            'heart_rate': {'low': 60.0, 'high': 85.0},
            'hormonal_level': {'low': 30.0, 'high': 70.0}
        }
    
    @abstractmethod
    def generate_single_record(self, **kwargs) -> Dict[str, float]:
        """Generate a single data record."""
        pass
    
    @abstractmethod
    def generate_time_series(self, **kwargs) -> pd.DataFrame:
        """Generate time series data."""
        pass
    
    def _clip_to_range(self, value: float, feature: str) -> float:
        """Clip value to feature's valid range."""
        config = self.feature_configs[feature]
        return float(np.clip(value, config['min'], config['max']))
    
    def calculate_migraine_probability(self, record: Dict[str, float]) -> float:
        """Calculate migraine probability based on feature values."""
        risk_score = 0.0
        
        for feature, value in record.items():
            if feature not in self.trigger_thresholds:
                continue
                
            thresholds = self.trigger_thresholds[feature]
            
            if value < thresholds['low']:
                risk_score += (thresholds['low'] - value) / thresholds['low']
            elif value > thresholds['high']:
                risk_score += (value - thresholds['high']) / thresholds['high']
        
        return float(1 / (1 + np.exp(-risk_score + 2)))
