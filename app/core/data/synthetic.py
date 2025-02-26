"""
Enhanced synthetic data generation for migraine prediction.
"""
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

@dataclass
class MigraineDataConfig:
    n_patients: int
    time_range_days: int
    missing_rate: float
    feature_ranges: Dict[str, Tuple[float, float]]
    drift_points: Optional[List[int]] = None
    random_seed: Optional[int] = None

class SyntheticDataGenerator:
    def __init__(self, config: MigraineDataConfig):
        """
        Initialize synthetic data generator.
        
        Args:
            config: Configuration for data generation
        """
        self.config = config
        self.rng = np.random.default_rng(config.random_seed)
        
    def generate_dataset(self) -> pd.DataFrame:
        """Generate complete dataset for all patients."""
        all_data = []
        for patient_id in range(self.config.n_patients):
            patient_data = self.generate_patient_data(patient_id)
            all_data.append(patient_data)
        
        return pd.concat(all_data, ignore_index=True)
    
    def generate_patient_data(self, patient_id: int) -> pd.DataFrame:
        """Generate synthetic data for one patient."""
        # Generate base timeline
        dates = pd.date_range(
            start=datetime.now(),
            periods=self.config.time_range_days,
            freq='D'
        )
        
        # Generate features
        data = {
            'patient_id': patient_id,
            'date': dates,
            'sleep_hours': self._generate_sleep_pattern(),
            'weather_pressure': self._generate_weather_pattern(),
            'stress_level': self._generate_stress_pattern(),
            'heart_rate': self._generate_heart_rate(),
            'hormonal_level': self._generate_hormonal_cycle()
        }
        
        # Add concept drift if specified
        if self.config.drift_points:
            self._inject_concept_drift(data)
        
        # Calculate migraine probability and occurrence
        data['migraine_prob'] = self._calculate_migraine_probability(data)
        data['migraine_occurred'] = self.rng.binomial(1, data['migraine_prob'])
        
        # Inject missing values
        data = self._inject_missing_values(data)
        
        return pd.DataFrame(data)
    
    def _generate_sleep_pattern(self) -> np.ndarray:
        """Generate realistic sleep patterns with weekly cycles."""
        base_sleep = np.random.normal(
            loc=7.0,
            scale=1.0,
            size=self.config.time_range_days
        )
        
        # Add weekly pattern (less sleep on weekends)
        weekly_pattern = np.tile(
            [-0.5, -0.5, 0, 0, 0, 1, 1],
            self.config.time_range_days // 7 + 1
        )[:self.config.time_range_days]
        
        return np.clip(
            base_sleep + weekly_pattern,
            self.config.feature_ranges['sleep_hours'][0],
            self.config.feature_ranges['sleep_hours'][1]
        )
    
    def _generate_weather_pattern(self) -> np.ndarray:
        """Generate weather pressure with seasonal variations."""
        # Base pressure with seasonal cycle
        t = np.linspace(0, 2*np.pi, self.config.time_range_days)
        base_pressure = 1013 + 10 * np.sin(t)
        
        # Add random fluctuations
        noise = self.rng.normal(0, 2, size=self.config.time_range_days)
        
        return np.clip(
            base_pressure + noise,
            self.config.feature_ranges['weather_pressure'][0],
            self.config.feature_ranges['weather_pressure'][1]
        )
    
    def _generate_stress_pattern(self) -> np.ndarray:
        """Generate stress levels with work-week pattern."""
        base_stress = self.rng.normal(
            loc=5.0,
            scale=1.5,
            size=self.config.time_range_days
        )
        
        # Add work-week pattern
        weekly_pattern = np.tile(
            [1, 1, 1, 1, 1, -0.5, -0.5],
            self.config.time_range_days // 7 + 1
        )[:self.config.time_range_days]
        
        return np.clip(
            base_stress + weekly_pattern,
            self.config.feature_ranges['stress_level'][0],
            self.config.feature_ranges['stress_level'][1]
        )
    
    def _generate_heart_rate(self) -> np.ndarray:
        """Generate heart rate data with activity patterns."""
        base_hr = self.rng.normal(
            loc=70,
            scale=5,
            size=self.config.time_range_days
        )
        
        return np.clip(
            base_hr,
            self.config.feature_ranges['heart_rate'][0],
            self.config.feature_ranges['heart_rate'][1]
        )
    
    def _generate_hormonal_cycle(self) -> np.ndarray:
        """Generate hormonal cycle data with ~28 day periodicity."""
        t = np.linspace(0, 2*np.pi * (self.config.time_range_days/28),
                       self.config.time_range_days)
        cycle = 50 + 45 * np.sin(t)
        noise = self.rng.normal(0, 5, size=self.config.time_range_days)
        
        return np.clip(
            cycle + noise,
            self.config.feature_ranges['hormonal_level'][0],
            self.config.feature_ranges['hormonal_level'][1]
        )
    
    def _calculate_migraine_probability(self, data: Dict) -> np.ndarray:
        """Calculate migraine probability based on features."""
        prob = np.zeros(self.config.time_range_days)
        
        # Base probability from sleep deprivation
        sleep_effect = 0.1 * np.maximum(0, 7 - data['sleep_hours'])
        
        # Stress effect
        stress_effect = 0.05 * (data['stress_level'] - 5).clip(0)
        
        # Weather effect (pressure changes)
        pressure_change = np.gradient(data['weather_pressure'])
        weather_effect = 0.1 * (np.abs(pressure_change) > 5)
        
        # Hormonal effect (highest near peak)
        hormonal_effect = 0.1 * (data['hormonal_level'] > 80)
        
        # Combine effects
        prob = 0.05 + sleep_effect + stress_effect + weather_effect + hormonal_effect
        
        return np.clip(prob, 0, 0.9)
    
    def _inject_concept_drift(self, data: Dict):
        """Inject concept drift at specified points."""
        if not self.config.drift_points:
            return
            
        for drift_point in self.config.drift_points:
            if drift_point >= self.config.time_range_days:
                continue
                
            # Change feature relationships after drift point
            data['stress_level'][drift_point:] *= 1.5
            data['weather_pressure'][drift_point:] += 10
    
    def _inject_missing_values(self, data: Dict) -> Dict:
        """Inject missing values randomly."""
        for key in data.keys():
            if key in ['patient_id', 'date', 'migraine_occurred']:
                continue
                
            mask = self.rng.random(self.config.time_range_days) < self.config.missing_rate
            data[key] = np.where(mask, np.nan, data[key])
        
        return data