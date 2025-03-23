"""
Trigger-Symptom Relationship Generators for Migraine Digital Twin Validation.

This module provides generators for creating synthetic trigger-symptom relationships,
including temporal patterns, intensities, and individual sensitivity profiles.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from datetime import datetime, timedelta

class TriggerProfile:
    """Individual trigger sensitivity profile."""
    def __init__(
        self,
        trigger_types: List[str],
        base_sensitivities: Optional[Dict[str, float]] = None,
        temporal_variation: bool = True
    ):
        self.trigger_types = trigger_types
        self.base_sensitivities = base_sensitivities or {
            t: np.random.uniform(0.3, 0.8) for t in trigger_types
        }
        self.temporal_variation = temporal_variation
        
        # Generate random phase shifts for temporal variations
        self.phase_shifts = {
            t: np.random.uniform(0, 2*np.pi) for t in trigger_types
        }
    
    def get_sensitivity(
        self,
        trigger_type: str,
        timestamp: datetime
    ) -> float:
        """
        Get sensitivity to a specific trigger at a given time.
        
        Args:
            trigger_type: Type of trigger
            timestamp: Time point
            
        Returns:
            Sensitivity value (0-1)
        """
        base = self.base_sensitivities.get(trigger_type, 0.0)
        
        if self.temporal_variation:
            # Add circadian and weekly variations
            hour_of_day = timestamp.hour + timestamp.minute/60
            day_of_week = timestamp.weekday()
            
            circadian = 0.1 * np.sin(2*np.pi*hour_of_day/24 + self.phase_shifts[trigger_type])
            weekly = 0.05 * np.sin(2*np.pi*day_of_week/7)
            
            return np.clip(base + circadian + weekly, 0, 1)
        
        return base

class TriggerGenerator:
    """Generate synthetic trigger events and intensities."""
    
    def __init__(
        self,
        trigger_types: List[str],
        base_frequencies: Optional[Dict[str, float]] = None,
        correlation_matrix: Optional[np.ndarray] = None
    ):
        self.trigger_types = trigger_types
        self.base_frequencies = base_frequencies or {
            t: np.random.uniform(0.1, 1.0) for t in trigger_types
        }
        
        n_triggers = len(trigger_types)
        if correlation_matrix is None:
            # Generate random correlation matrix
            random_matrix = np.random.uniform(-0.3, 0.7, (n_triggers, n_triggers))
            correlation_matrix = (random_matrix + random_matrix.T) / 2
            np.fill_diagonal(correlation_matrix, 1.0)
        
        self.correlation_matrix = correlation_matrix
    
    def generate(
        self,
        duration_days: int,
        start_date: Optional[datetime] = None,
        hourly: bool = True
    ) -> Dict[str, np.ndarray]:
        """
        Generate synthetic trigger occurrences.
        
        Args:
            duration_days: Number of days to generate
            start_date: Starting date (defaults to today)
            hourly: Whether to generate hourly (True) or daily (False) data
            
        Returns:
            Dictionary containing trigger intensities over time
        """
        if start_date is None:
            start_date = datetime.now()
        
        points_per_day = 24 if hourly else 1
        num_points = duration_days * points_per_day
        timestamps = [start_date + timedelta(hours=i/points_per_day*24) 
                     for i in range(num_points)]
        
        # Generate correlated random processes for triggers
        n_triggers = len(self.trigger_types)
        random_base = np.random.multivariate_normal(
            mean=np.zeros(n_triggers),
            cov=self.correlation_matrix,
            size=num_points
        )
        
        # Convert to trigger intensities
        trigger_data = {}
        for i, trigger_type in enumerate(self.trigger_types):
            # Add daily and weekly patterns
            time_array = np.arange(num_points)
            daily_pattern = np.sin(2*np.pi*time_array/points_per_day)
            weekly_pattern = np.sin(2*np.pi*time_array/(7*points_per_day))
            
            # Combine patterns with random variations
            intensity = (
                self.base_frequencies[trigger_type] * 
                (0.7 + 0.3 * daily_pattern + 0.2 * weekly_pattern) +
                0.2 * random_base[:, i]
            )
            
            # Ensure valid range
            trigger_data[trigger_type] = np.clip(intensity, 0, 1)
        
        return {
            'timestamps': timestamps,
            'intensities': trigger_data
        }

class SymptomGenerator:
    """Generate synthetic migraine symptoms based on triggers."""
    
    def __init__(
        self,
        trigger_profile: TriggerProfile,
        latency_range: Tuple[float, float] = (1, 24),  # hours
        symptom_types: List[str] = None
    ):
        self.trigger_profile = trigger_profile
        self.latency_range = latency_range
        self.symptom_types = symptom_types or [
            'headache', 'nausea', 'photophobia', 'phonophobia'
        ]
        
        # Generate random symptom correlations
        self.symptom_correlations = np.random.uniform(0.5, 0.9, len(self.symptom_types))
    
    def generate(
        self,
        trigger_data: Dict[str, np.ndarray],
        timestamps: List[datetime]
    ) -> Dict[str, np.ndarray]:
        """
        Generate synthetic symptoms based on trigger exposures.
        
        Args:
            trigger_data: Dictionary of trigger intensities over time
            timestamps: List of time points
            
        Returns:
            Dictionary containing symptom intensities over time
        """
        num_points = len(timestamps)
        base_probability = np.zeros(num_points)
        
        # Calculate trigger contributions
        for trigger_type, intensities in trigger_data.items():
            # Get time-varying sensitivities
            sensitivities = np.array([
                self.trigger_profile.get_sensitivity(trigger_type, t)
                for t in timestamps
            ])
            
            # Add trigger contribution
            base_probability += sensitivities * intensities
        
        # Generate migraine events
        migraine_events = base_probability > np.random.uniform(0.7, 0.9, num_points)
        
        # Generate symptoms for each migraine event
        symptoms = {}
        for i, symptom_type in enumerate(self.symptom_types):
            # Base symptom intensity from migraine events
            intensity = np.zeros(num_points)
            
            # Add latency and build-up/decay for each event
            for event_idx in np.where(migraine_events)[0]:
                # Random latency
                latency = np.random.uniform(*self.latency_range)
                latency_idx = int(latency * (num_points/len(timestamps)))
                
                if event_idx + latency_idx < num_points:
                    # Generate symptom profile
                    duration_idx = int(12 * (num_points/len(timestamps)))  # 12-hour typical duration
                    profile = self._generate_symptom_profile(duration_idx)
                    
                    # Add profile with correlation factor
                    end_idx = min(event_idx + latency_idx + duration_idx, num_points)
                    profile_length = end_idx - (event_idx + latency_idx)
                    intensity[event_idx + latency_idx:end_idx] += (
                        profile[:profile_length] * self.symptom_correlations[i]
                    )
            
            symptoms[symptom_type] = np.clip(intensity, 0, 1)
        
        return {
            'timestamps': timestamps,
            'migraine_events': migraine_events,
            'symptoms': symptoms
        }
    
    def _generate_symptom_profile(self, duration: int) -> np.ndarray:
        """Generate single symptom intensity profile."""
        t = np.linspace(0, 1, duration)
        # Asymmetric profile with faster onset than decay
        profile = (1 - np.exp(-5*t)) * np.exp(-2*t)
        return profile / np.max(profile)

def generate_patient_scenario(
    duration_days: int,
    trigger_types: List[str] = None,
    start_date: Optional[datetime] = None
) -> Dict[str, Dict[str, np.ndarray]]:
    """
    Generate complete patient scenario with triggers and symptoms.
    
    Args:
        duration_days: Number of days to generate
        trigger_types: List of trigger types to include
        start_date: Starting date
        
    Returns:
        Dictionary containing trigger exposures and resulting symptoms
    """
    if trigger_types is None:
        trigger_types = [
            'stress', 'sleep_disruption', 'weather_change', 
            'bright_light', 'noise', 'food_trigger'
        ]
    
    # Create patient profile
    profile = TriggerProfile(trigger_types, temporal_variation=True)
    
    # Generate triggers
    trigger_gen = TriggerGenerator(trigger_types)
    trigger_data = trigger_gen.generate(duration_days, start_date, hourly=True)
    
    # Generate symptoms
    symptom_gen = SymptomGenerator(profile)
    symptom_data = symptom_gen.generate(
        trigger_data['intensities'],
        trigger_data['timestamps']
    )
    
    return {
        'triggers': trigger_data,
        'symptoms': symptom_data
    } 