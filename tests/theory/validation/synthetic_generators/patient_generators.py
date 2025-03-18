"""
Patient Pattern Simulator for Migraine Digital Twin Validation.

This module provides generators for creating synthetic patient data,
including configurable profiles, migraine patterns, and longitudinal data.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from datetime import datetime, timedelta
from dataclasses import dataclass
from .trigger_generators import TriggerProfile, generate_patient_scenario
from .signal_generators import generate_multimodal_stress_response
from .environmental_generators import generate_environmental_scenario

@dataclass
class PatientProfile:
    """Patient characteristics and migraine patterns."""
    
    # Demographics
    age: int
    sex: str
    
    # Migraine characteristics
    migraine_frequency: float  # Average episodes per month
    typical_duration: float    # Hours
    aura_probability: float    # 0-1
    
    # Trigger sensitivities
    trigger_profile: TriggerProfile
    
    # Physiological baselines
    heart_rate_baseline: float = 70.0
    heart_rate_variability: float = 0.1
    eeg_baseline_weights: Optional[Dict[str, float]] = None
    skin_conductance_baseline: float = 2.0
    
    # Treatment response
    treatment_effectiveness: Dict[str, float] = None
    placebo_response: float = 0.2

class PatientGenerator:
    """Generate synthetic patient profiles and data."""
    
    def __init__(
        self,
        num_patients: int,
        age_range: Tuple[int, int] = (18, 65),
        sex_ratio: float = 0.75,  # Proportion female
        trigger_types: Optional[List[str]] = None
    ):
        self.num_patients = num_patients
        self.age_range = age_range
        self.sex_ratio = sex_ratio
        self.trigger_types = trigger_types or [
            'stress', 'sleep_disruption', 'weather_change',
            'bright_light', 'noise', 'food_trigger'
        ]
        
        # Treatment options
        self.treatments = {
            'acute_medication': {'effectiveness': (0.4, 0.8)},
            'preventive_medication': {'effectiveness': (0.3, 0.6)},
            'lifestyle_modification': {'effectiveness': (0.2, 0.5)},
            'stress_management': {'effectiveness': (0.3, 0.6)}
        }
    
    def generate_profile(self) -> PatientProfile:
        """Generate single patient profile."""
        # Generate demographics
        age = np.random.randint(*self.age_range)
        sex = 'F' if np.random.random() < self.sex_ratio else 'M'
        
        # Generate migraine characteristics
        migraine_frequency = np.random.lognormal(1.5, 0.5)  # ~4.5 per month
        typical_duration = np.random.normal(12, 4)  # ~12 hours
        aura_probability = np.random.beta(2, 5)  # ~0.29
        
        # Create trigger profile
        trigger_profile = TriggerProfile(
            self.trigger_types,
            temporal_variation=True
        )
        
        # Generate physiological baselines
        hr_baseline = np.random.normal(70, 5)
        hrv = np.random.uniform(0.05, 0.15)
        sc_baseline = np.random.normal(2.0, 0.3)
        
        # Generate EEG baseline
        eeg_baseline = {
            'delta': np.random.uniform(0.8, 1.2),
            'theta': np.random.uniform(0.4, 0.6),
            'alpha': np.random.uniform(0.3, 0.5),
            'beta': np.random.uniform(0.2, 0.4),
            'gamma': np.random.uniform(0.1, 0.3)
        }
        
        # Generate treatment responses
        treatment_effectiveness = {
            name: np.random.uniform(*params['effectiveness'])
            for name, params in self.treatments.items()
        }
        
        return PatientProfile(
            age=age,
            sex=sex,
            migraine_frequency=migraine_frequency,
            typical_duration=typical_duration,
            aura_probability=aura_probability,
            trigger_profile=trigger_profile,
            heart_rate_baseline=hr_baseline,
            heart_rate_variability=hrv,
            eeg_baseline_weights=eeg_baseline,
            skin_conductance_baseline=sc_baseline,
            treatment_effectiveness=treatment_effectiveness,
            placebo_response=np.random.beta(2, 8)
        )
    
    def generate_population(self) -> List[PatientProfile]:
        """Generate population of patient profiles."""
        return [self.generate_profile() for _ in range(self.num_patients)]

class LongitudinalDataGenerator:
    """Generate longitudinal patient data."""
    
    def __init__(
        self,
        patient_profile: PatientProfile,
        include_treatments: bool = True
    ):
        self.profile = patient_profile
        self.include_treatments = include_treatments
    
    def generate(
        self,
        duration_days: int,
        start_date: Optional[datetime] = None,
        include_physiological: bool = True,
        include_environmental: bool = True
    ) -> Dict[str, Dict]:
        """
        Generate longitudinal patient data.
        
        Args:
            duration_days: Number of days to generate
            start_date: Starting date (defaults to today)
            include_physiological: Whether to include physiological signals
            include_environmental: Whether to include environmental data
            
        Returns:
            Dictionary containing all patient data over time
        """
        if start_date is None:
            start_date = datetime.now()
        
        # Generate base scenario with triggers and symptoms
        scenario = generate_patient_scenario(
            duration_days,
            self.profile.trigger_profile.trigger_types,
            start_date
        )
        
        # Add physiological responses if requested
        if include_physiological:
            # Convert migraine events to stress events for physiological simulation
            stress_events = []
            for t, event in enumerate(scenario['symptoms']['migraine_events']):
                if event:
                    # Add prodrome and acute phase stress responses
                    stress_events.append((t-2, 0.5))  # Prodrome
                    stress_events.append((t, 0.9))    # Acute phase
            
            physio_data = generate_multimodal_stress_response(
                duration_days * 24 * 3600,  # Convert to seconds
                stress_events,
                sampling_rate=250.0
            )
            
            # Adjust baselines according to patient profile
            for signal_type, data in physio_data.items():
                if signal_type == 'ecg':
                    data['hr'] *= self.profile.heart_rate_baseline / 60.0
                elif signal_type == 'eeg':
                    for band, weight in self.profile.eeg_baseline_weights.items():
                        if band in data:
                            data[band] *= weight
                elif signal_type == 'sc':
                    data['sc'] *= self.profile.skin_conductance_baseline
            
            scenario['physiological'] = physio_data
        
        # Add environmental data if requested
        if include_environmental:
            env_data = generate_environmental_scenario(
                duration_days,
                include_events=True
            )
            scenario['environmental'] = env_data
        
        # Add treatments if enabled
        if self.include_treatments:
            treatments = self._generate_treatments(
                scenario['symptoms']['migraine_events'],
                scenario['triggers']['timestamps']
            )
            scenario['treatments'] = treatments
        
        return scenario
    
    def _generate_treatments(
        self,
        migraine_events: np.ndarray,
        timestamps: List[datetime]
    ) -> Dict[str, np.ndarray]:
        """Generate treatment applications and responses."""
        num_points = len(timestamps)
        treatments = {}
        
        # Generate acute medication use
        acute_med = np.zeros(num_points)
        for t, event in enumerate(migraine_events):
            if event and t > 0:  # Allow for treatment after onset
                # Simulate treatment decision
                if np.random.random() < 0.8:  # 80% treatment probability
                    acute_med[t] = 1
                    
                    # Calculate effectiveness
                    base_effect = self.profile.treatment_effectiveness['acute_medication']
                    placebo_effect = self.profile.placebo_response
                    total_effect = base_effect + (1 - base_effect) * placebo_effect
                    
                    # Apply treatment effect to subsequent hours
                    effect_duration = int(4 * (num_points/len(timestamps)))  # 4-hour effect
                    effect_profile = np.exp(-np.arange(effect_duration)/effect_duration)
                    end_idx = min(t + effect_duration, num_points)
                    acute_med[t:end_idx] = effect_profile[:end_idx-t] * total_effect
        
        treatments['acute_medication'] = acute_med
        
        # Generate preventive medication adherence
        if 'preventive_medication' in self.profile.treatment_effectiveness:
            adherence = np.random.binomial(
                1,
                0.85,  # 85% adherence rate
                num_points
            )
            treatments['preventive_medication'] = adherence
        
        return treatments

def generate_longitudinal_cohort(
    profiles: List[PatientProfile],
    duration_days: int,
    start_date: Optional[datetime] = None
) -> Dict[int, Dict]:
    """
    Generate longitudinal data for a cohort of patients.
    
    Args:
        profiles: List of patient profiles
        duration_days: Number of days to generate
        start_date: Starting date
        
    Returns:
        Dictionary mapping patient IDs to their longitudinal data
    """
    cohort_data = {}
    
    for i, profile in enumerate(profiles):
        generator = LongitudinalDataGenerator(profile)
        patient_data = generator.generate(
            duration_days,
            start_date,
            include_physiological=True,
            include_environmental=True
        )
        cohort_data[i] = patient_data
    
    return cohort_data 