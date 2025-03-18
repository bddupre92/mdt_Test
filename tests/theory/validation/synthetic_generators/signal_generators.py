"""
Synthetic Signal Generators for Migraine Digital Twin Validation.

This module provides generators for creating synthetic physiological signals
and environmental data for testing purposes.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from scipy import signal

class ECGGenerator:
    """Generate synthetic ECG signals with configurable patterns."""
    
    def __init__(
        self,
        sampling_rate: float = 250.0,
        heart_rate: float = 60.0,
        heart_rate_variability: float = 0.1,
        noise_level: float = 0.05
    ):
        self.sampling_rate = sampling_rate
        self.heart_rate = heart_rate
        self.hrv = heart_rate_variability
        self.noise_level = noise_level
    
    def generate(
        self,
        duration: float,
        stress_level: float = 0.0
    ) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        """
        Generate synthetic ECG signal.
        
        Args:
            duration: Signal duration in seconds
            stress_level: Simulated stress level (0-1)
            
        Returns:
            Tuple of (time_points, signal_components) where signal_components
            contains individual components (p_wave, qrs_complex, t_wave)
        """
        num_samples = int(duration * self.sampling_rate)
        time = np.linspace(0, duration, num_samples)
        
        # Adjust heart rate based on stress
        current_hr = self.heart_rate * (1 + stress_level * 0.5)
        
        # Generate heart rate variability
        hr_variation = np.random.normal(0, self.hrv * current_hr, num_samples)
        instantaneous_hr = current_hr + hr_variation
        
        # Generate ECG components
        p_wave = self._generate_p_wave(time, instantaneous_hr)
        qrs = self._generate_qrs_complex(time, instantaneous_hr)
        t_wave = self._generate_t_wave(time, instantaneous_hr)
        
        # Combine components
        ecg = p_wave + qrs + t_wave
        
        # Add noise
        noise = np.random.normal(0, self.noise_level, num_samples)
        ecg += noise
        
        return time, {
            'ecg': ecg,
            'p_wave': p_wave,
            'qrs': qrs,
            't_wave': t_wave,
            'hr': instantaneous_hr
        }
    
    def _generate_p_wave(self, time: np.ndarray, hr: np.ndarray) -> np.ndarray:
        """Generate P wave component."""
        p_wave = np.zeros_like(time)
        for t_idx, t in enumerate(time):
            cycle_position = (t * hr[t_idx] / 60) % 1
            if 0.0 <= cycle_position < 0.2:
                p_wave[t_idx] = 0.25 * np.sin(np.pi * cycle_position / 0.2)
        return p_wave
    
    def _generate_qrs_complex(self, time: np.ndarray, hr: np.ndarray) -> np.ndarray:
        """Generate QRS complex component."""
        qrs = np.zeros_like(time)
        for t_idx, t in enumerate(time):
            cycle_position = (t * hr[t_idx] / 60) % 1
            if 0.2 <= cycle_position < 0.3:
                qrs[t_idx] = -0.5
            elif 0.3 <= cycle_position < 0.4:
                qrs[t_idx] = 1.5
            elif 0.4 <= cycle_position < 0.5:
                qrs[t_idx] = -0.3
        return qrs
    
    def _generate_t_wave(self, time: np.ndarray, hr: np.ndarray) -> np.ndarray:
        """Generate T wave component."""
        t_wave = np.zeros_like(time)
        for t_idx, t in enumerate(time):
            cycle_position = (t * hr[t_idx] / 60) % 1
            if 0.5 <= cycle_position < 0.7:
                t_wave[t_idx] = 0.35 * np.sin(np.pi * (cycle_position - 0.5) / 0.2)
        return t_wave

class EEGGenerator:
    """Generate synthetic EEG signals with configurable frequency bands."""
    
    def __init__(
        self,
        sampling_rate: float = 250.0,
        noise_level: float = 0.1
    ):
        self.sampling_rate = sampling_rate
        self.noise_level = noise_level
        
        # Define frequency bands
        self.bands = {
            'delta': (0.5, 4),
            'theta': (4, 8),
            'alpha': (8, 13),
            'beta': (13, 30),
            'gamma': (30, 100)
        }
    
    def generate(
        self,
        duration: float,
        band_weights: Optional[Dict[str, float]] = None
    ) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        """
        Generate synthetic EEG signal with specified frequency band weights.
        
        Args:
            duration: Signal duration in seconds
            band_weights: Dictionary of relative weights for each frequency band
            
        Returns:
            Tuple of (time_points, signal_components) where signal_components
            contains individual frequency bands and combined signal
        """
        if band_weights is None:
            band_weights = {
                'delta': 1.0,
                'theta': 0.5,
                'alpha': 0.3,
                'beta': 0.2,
                'gamma': 0.1
            }
        
        num_samples = int(duration * self.sampling_rate)
        time = np.linspace(0, duration, num_samples)
        
        components = {}
        eeg = np.zeros_like(time)
        
        # Generate each frequency band
        for band, (low_freq, high_freq) in self.bands.items():
            # Generate band-limited noise
            nyquist = self.sampling_rate / 2
            b, a = signal.butter(4, [low_freq/nyquist, high_freq/nyquist], btype='band')
            white_noise = np.random.normal(0, 1, num_samples)
            band_signal = signal.filtfilt(b, a, white_noise)
            
            # Scale by weight
            weight = band_weights.get(band, 0.0)
            band_signal *= weight
            
            components[band] = band_signal
            eeg += band_signal
        
        # Add noise
        noise = np.random.normal(0, self.noise_level, num_samples)
        eeg += noise
        
        components['eeg'] = eeg
        return time, components

class SkinConductanceGenerator:
    """Generate synthetic skin conductance signals."""
    
    def __init__(
        self,
        sampling_rate: float = 128.0,
        base_level: float = 2.0,
        noise_level: float = 0.05
    ):
        self.sampling_rate = sampling_rate
        self.base_level = base_level
        self.noise_level = noise_level
    
    def generate(
        self,
        duration: float,
        stress_events: List[Tuple[float, float]] = None
    ) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        """
        Generate synthetic skin conductance signal with stress responses.
        
        Args:
            duration: Signal duration in seconds
            stress_events: List of (time, intensity) tuples for stress events
            
        Returns:
            Tuple of (time_points, signal_components) where signal_components
            contains tonic level, phasic responses, and combined signal
        """
        num_samples = int(duration * self.sampling_rate)
        time = np.linspace(0, duration, num_samples)
        
        # Generate tonic component (slow changes)
        tonic = self.base_level + 0.5 * np.sin(2 * np.pi * time / (duration/2))
        
        # Generate phasic responses to stress events
        phasic = np.zeros_like(time)
        if stress_events:
            for event_time, intensity in stress_events:
                event_idx = int(event_time * self.sampling_rate)
                response = self._generate_scr(num_samples - event_idx, intensity)
                phasic[event_idx:] += response[:num_samples-event_idx]
        
        # Combine components
        sc = tonic + phasic
        
        # Add noise
        noise = np.random.normal(0, self.noise_level, num_samples)
        sc += noise
        
        return time, {
            'sc': sc,
            'tonic': tonic,
            'phasic': phasic
        }
    
    def _generate_scr(self, length: int, intensity: float) -> np.ndarray:
        """Generate single skin conductance response."""
        t = np.arange(length) / self.sampling_rate
        response = intensity * (1 - np.exp(-t/0.75)) * np.exp(-t/2)
        return response

def generate_multimodal_stress_response(
    duration: float,
    stress_events: List[Tuple[float, float]],
    sampling_rate: float = 250.0
) -> Dict[str, Dict[str, np.ndarray]]:
    """
    Generate coordinated multimodal physiological response to stress events.
    
    Args:
        duration: Signal duration in seconds
        stress_events: List of (time, intensity) tuples for stress events
        sampling_rate: Sampling rate for all signals
        
    Returns:
        Dictionary containing synchronized physiological signals
    """
    # Initialize generators
    ecg_gen = ECGGenerator(sampling_rate=sampling_rate)
    eeg_gen = EEGGenerator(sampling_rate=sampling_rate)
    sc_gen = SkinConductanceGenerator(sampling_rate=sampling_rate)
    
    # Generate base signals
    time_ecg, ecg_data = ecg_gen.generate(duration)
    
    # Modulate EEG based on stress
    eeg_weights = {
        'delta': 1.0,
        'theta': 0.5,
        'alpha': 0.3,
        'beta': 0.8,  # Increased beta during stress
        'gamma': 0.4
    }
    _, eeg_data = eeg_gen.generate(duration, eeg_weights)
    
    # Generate skin conductance with stress events
    _, sc_data = sc_gen.generate(duration, stress_events)
    
    return {
        'time': time_ecg,
        'ecg': ecg_data,
        'eeg': eeg_data,
        'sc': sc_data
    } 