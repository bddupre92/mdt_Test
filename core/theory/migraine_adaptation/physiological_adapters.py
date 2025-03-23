"""
Physiological Signal Adapters
============================

This module provides specialized adapters for processing various physiological signals
in the context of migraine prediction. Each adapter implements signal-specific
preprocessing, feature extraction, and quality assessment methods.

Key Features:
- ECG/HRV processing with heart rate variability analysis
- EEG signal processing with frequency band analysis
- Skin conductance and temperature signal processing
- Respiratory signal analysis
- Mobile sensor data normalization
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from scipy import signal
from scipy.stats import kurtosis, skew
import pywt

from . import PhysiologicalSignalAdapter

def _validate_input(signal_data: np.ndarray, sampling_rate: float) -> None:
    """
    Validate input signal data and sampling rate.
    
    Parameters
    ----------
    signal_data : np.ndarray
        Signal data to validate
    sampling_rate : float
        Sampling rate to validate
        
    Raises
    ------
    ValueError
        If sampling_rate is not positive or signal_data contains NaN values
    """
    if sampling_rate <= 0:
        raise ValueError("Sampling rate must be positive")
        
    if np.any(np.isnan(signal_data)):
        raise ValueError("Signal contains NaN values")
        
    if len(signal_data) == 0:
        raise ValueError("Signal data is empty")

class ECGAdapter(PhysiologicalSignalAdapter):
    """Adapter for processing ECG signals and extracting HRV features."""
    
    def preprocess(self, signal_data: np.ndarray, sampling_rate: float, **kwargs) -> np.ndarray:
        """
        Preprocess ECG signal with filtering and R-peak detection.
        
        Parameters
        ----------
        signal_data : np.ndarray
            Raw ECG signal
        sampling_rate : float
            Sampling rate in Hz
        **kwargs : dict
            Optional parameters including:
            - lowcut : float (default: 0.5)
                Lower frequency cutoff for bandpass filter
            - highcut : float (default: 40.0)
                Upper frequency cutoff for bandpass filter
            
        Returns
        -------
        np.ndarray
            Preprocessed ECG signal
            
        Raises
        ------
        ValueError
            If sampling_rate is not positive or signal_data contains NaN values
        """
        _validate_input(signal_data, sampling_rate)
            
        # Get filter parameters
        lowcut = kwargs.get('lowcut', 0.5)
        highcut = kwargs.get('highcut', 40.0)
        
        # Apply bandpass filter
        nyquist = sampling_rate / 2
        b, a = signal.butter(4, [lowcut/nyquist, highcut/nyquist], btype='band')
        filtered_signal = signal.filtfilt(b, a, signal_data)
        
        # Baseline correction with odd kernel size
        kernel_size = int(sampling_rate)
        if kernel_size % 2 == 0:
            kernel_size += 1  # Make it odd
        baseline = signal.medfilt(filtered_signal, kernel_size=kernel_size)
        return filtered_signal - baseline
    
    def extract_features(self, preprocessed_data: np.ndarray, sampling_rate: float, **kwargs) -> Dict[str, np.ndarray]:
        """
        Extract HRV features from preprocessed ECG signal.
        
        Parameters
        ----------
        preprocessed_data : np.ndarray
            Preprocessed ECG signal
        sampling_rate : float
            Sampling rate in Hz
        **kwargs : dict
            Optional parameters
            
        Returns
        -------
        Dict[str, np.ndarray]
            Dictionary of HRV features including:
            - rr_intervals: Array of RR intervals
            - sdnn: Standard deviation of NN intervals
            - rmssd: Root mean square of successive differences
            - pnn50: Proportion of NN50
        """
        # Detect R-peaks
        r_peaks = self._detect_r_peaks(preprocessed_data, sampling_rate)
        
        # Calculate RR intervals
        rr_intervals = np.diff(r_peaks) / sampling_rate
        
        # Calculate HRV features
        features = {
            'rr_intervals': rr_intervals,
            'sdnn': np.std(rr_intervals),
            'rmssd': np.sqrt(np.mean(np.square(np.diff(rr_intervals)))),
            'pnn50': self._calculate_pnn50(rr_intervals)
        }
        
        return features
    
    def assess_quality(self, signal_data: np.ndarray, sampling_rate: float) -> float:
        """
        Assess ECG signal quality based on noise level and R-peak detectability.
        
        Parameters
        ----------
        signal_data : np.ndarray
            ECG signal data
        sampling_rate : float
            Sampling rate in Hz
            
        Returns
        -------
        float
            Quality score between 0 and 1
        """
        # Calculate signal-to-noise ratio
        signal_power = np.mean(np.square(signal_data))
        noise_est = np.std(np.diff(signal_data))
        snr = 10 * np.log10(signal_power / (noise_est ** 2))
        
        # Normalize SNR to 0-1 range
        quality_score = 1 / (1 + np.exp(-0.1 * (snr - 20)))
        return float(quality_score)
    
    def _detect_r_peaks(self, signal_data: np.ndarray, sampling_rate: float) -> np.ndarray:
        """Helper method to detect R-peaks in ECG signal."""
        # Pan-Tompkins algorithm simplified implementation
        diff = np.diff(signal_data)
        squared = diff ** 2
        window_size = int(0.1 * sampling_rate)
        integrated = np.convolve(squared, np.ones(window_size)/window_size, mode='same')
        peaks, _ = signal.find_peaks(integrated, distance=int(0.2*sampling_rate))
        return peaks
    
    def _calculate_pnn50(self, rr_intervals: np.ndarray) -> float:
        """Helper method to calculate pNN50."""
        differences = np.abs(np.diff(rr_intervals))
        nn50 = np.sum(differences > 0.05)
        return float(nn50 / len(differences)) if len(differences) > 0 else 0.0


class EEGAdapter(PhysiologicalSignalAdapter):
    """Adapter for processing EEG signals and extracting relevant features."""
    
    def preprocess(self, signal_data: np.ndarray, sampling_rate: float, **kwargs) -> np.ndarray:
        """
        Preprocess EEG signal with filtering and artifact removal.
        
        Parameters
        ----------
        signal_data : np.ndarray
            Raw EEG signal
        sampling_rate : float
            Sampling rate in Hz
        **kwargs : dict
            Optional parameters including:
            - notch_freq : float (default: 50.0)
                Frequency to remove with notch filter
            - q_factor : float (default: 30.0)
                Quality factor for notch filter
            - lowcut : float (default: 0.5)
                Lower frequency cutoff for bandpass filter
            - highcut : float (default: 50.0)
                Upper frequency cutoff for bandpass filter
            
        Returns
        -------
        np.ndarray
            Preprocessed EEG signal
            
        Raises
        ------
        ValueError
            If sampling_rate is not positive or signal_data contains NaN values
        """
        _validate_input(signal_data, sampling_rate)
            
        # Get filter parameters
        notch_freq = kwargs.get('notch_freq', 50.0)
        q_factor = kwargs.get('q_factor', 30.0)
        lowcut = kwargs.get('lowcut', 0.5)
        highcut = kwargs.get('highcut', 50.0)
        
        # Apply notch filter for line noise
        b_notch, a_notch = signal.iirnotch(notch_freq, q_factor, sampling_rate)
        notched_signal = signal.filtfilt(b_notch, a_notch, signal_data)
        
        # Apply bandpass filter
        nyquist = sampling_rate / 2
        b_band, a_band = signal.butter(4, [lowcut/nyquist, highcut/nyquist], btype='band')
        filtered_signal = signal.filtfilt(b_band, a_band, notched_signal)
        
        return filtered_signal
    
    def extract_features(self, preprocessed_data: np.ndarray, sampling_rate: float, **kwargs) -> Dict[str, np.ndarray]:
        """
        Extract frequency band features from preprocessed EEG signal.
        
        Parameters
        ----------
        preprocessed_data : np.ndarray
            Preprocessed EEG signal
        sampling_rate : float
            Sampling rate in Hz
        **kwargs : dict
            Optional parameters
            
        Returns
        -------
        Dict[str, np.ndarray]
            Dictionary of EEG features including power in different frequency bands
        """
        # Calculate power spectral density
        freqs, psd = signal.welch(preprocessed_data, fs=sampling_rate, nperseg=int(4*sampling_rate))
        
        # Extract band powers
        bands = {
            'delta': (0.5, 4),
            'theta': (4, 8),
            'alpha': (8, 13),
            'beta': (13, 30),
            'gamma': (30, 45)
        }
        
        features = {}
        for band_name, (low, high) in bands.items():
            mask = (freqs >= low) & (freqs <= high)
            features[f'{band_name}_power'] = np.mean(psd[mask])
        
        # Add ratio features
        features['theta_beta_ratio'] = features['theta_power'] / features['beta_power']
        features['alpha_beta_ratio'] = features['alpha_power'] / features['beta_power']
        
        return features
    
    def assess_quality(self, signal_data: np.ndarray, sampling_rate: float) -> float:
        """
        Assess EEG signal quality based on artifact presence and signal stability.
        
        Parameters
        ----------
        signal_data : np.ndarray
            EEG signal data
        sampling_rate : float
            Sampling rate in Hz
            
        Returns
        -------
        float
            Quality score between 0 and 1
        """
        # Calculate metrics for quality assessment
        amplitude_range = np.ptp(signal_data)
        line_noise = self._estimate_line_noise(signal_data, sampling_rate)
        movement_artifact = np.std(np.diff(signal_data))
        
        # Combine metrics into quality score
        quality_score = 1.0
        if amplitude_range > 200:  # Excessive amplitude
            quality_score *= 0.7
        if line_noise > 0.1:  # Significant line noise
            quality_score *= 0.8
        if movement_artifact > 20:  # Movement artifacts
            quality_score *= 0.6
            
        return float(quality_score)
    
    def _estimate_line_noise(self, signal_data: np.ndarray, sampling_rate: float) -> float:
        """Helper method to estimate power line noise."""
        freqs, psd = signal.welch(signal_data, fs=sampling_rate)
        line_freq_mask = (freqs >= 49) & (freqs <= 51)  # Around 50 Hz
        return float(np.mean(psd[line_freq_mask]))


class SkinConductanceAdapter(PhysiologicalSignalAdapter):
    """Adapter for processing skin conductance (EDA) signals."""
    
    def preprocess(self, signal_data: np.ndarray, sampling_rate: float, **kwargs) -> np.ndarray:
        """
        Preprocess skin conductance signal.
        
        Parameters
        ----------
        signal_data : np.ndarray
            Raw skin conductance signal
        sampling_rate : float
            Sampling rate in Hz
        **kwargs : dict
            Optional parameters including:
            - lowpass_cutoff : float (default: 5.0)
                Cutoff frequency for lowpass filter
            
        Returns
        -------
        np.ndarray
            Preprocessed skin conductance signal
            
        Raises
        ------
        ValueError
            If sampling_rate is not positive or signal_data contains NaN values
        """
        _validate_input(signal_data, sampling_rate)
            
        # Get filter parameters
        lowpass_cutoff = kwargs.get('lowpass_cutoff', 5.0)
        
        # Apply lowpass filter
        nyquist = sampling_rate / 2
        b, a = signal.butter(4, lowpass_cutoff/nyquist, btype='low')
        filtered_signal = signal.filtfilt(b, a, signal_data)
        
        return filtered_signal
    
    def extract_features(self, preprocessed_data: np.ndarray, sampling_rate: float, **kwargs) -> Dict[str, np.ndarray]:
        """
        Extract features from preprocessed skin conductance signal.
        
        Parameters
        ----------
        preprocessed_data : np.ndarray
            Preprocessed skin conductance signal
        sampling_rate : float
            Sampling rate in Hz
        **kwargs : dict
            Optional parameters
            
        Returns
        -------
        Dict[str, np.ndarray]
            Dictionary of skin conductance features
        """
        # Extract SCR peaks
        peaks, properties = signal.find_peaks(
            preprocessed_data,
            height=0.01,
            distance=int(sampling_rate),
            prominence=0.05
        )
        
        features = {
            'scr_rate': len(peaks) / (len(preprocessed_data) / sampling_rate),
            'mean_amplitude': np.mean(properties['peak_heights']) if len(peaks) > 0 else 0,
            'max_amplitude': np.max(properties['peak_heights']) if len(peaks) > 0 else 0,
            'tonic_level': np.median(preprocessed_data),
            'standard_deviation': np.std(preprocessed_data)
        }
        
        return features
    
    def assess_quality(self, signal_data: np.ndarray, sampling_rate: float) -> float:
        """
        Assess skin conductance signal quality.
        
        Parameters
        ----------
        signal_data : np.ndarray
            Skin conductance signal data
        sampling_rate : float
            Sampling rate in Hz
            
        Returns
        -------
        float
            Quality score between 0 and 1
        """
        # Check for physiologically plausible range (typically 0.5-20 µS)
        range_score = 1.0
        if np.min(signal_data) < 0 or np.max(signal_data) > 25:
            range_score = 0.7
            
        # Check for sudden jumps (motion artifacts)
        diff = np.diff(signal_data)
        artifact_score = 1.0
        if np.any(np.abs(diff) > 5):
            artifact_score = 0.8
            
        return float(range_score * artifact_score)


class RespiratoryAdapter(PhysiologicalSignalAdapter):
    """Adapter for processing respiratory signals."""
    
    def preprocess(self, signal_data: np.ndarray, sampling_rate: float, **kwargs) -> np.ndarray:
        """
        Preprocess respiratory signal.
        
        Parameters
        ----------
        signal_data : np.ndarray
            Raw respiratory signal
        sampling_rate : float
            Sampling rate in Hz
        **kwargs : dict
            Optional parameters including:
            - lowpass_cutoff : float (default: 1.0)
                Cutoff frequency for lowpass filter
            
        Returns
        -------
        np.ndarray
            Preprocessed respiratory signal
            
        Raises
        ------
        ValueError
            If sampling_rate is not positive or signal_data contains NaN values
        """
        _validate_input(signal_data, sampling_rate)
            
        # Get filter parameters
        lowpass_cutoff = kwargs.get('lowpass_cutoff', 1.0)
        
        # Apply lowpass filter
        nyquist = sampling_rate / 2
        b, a = signal.butter(4, lowpass_cutoff/nyquist, btype='low')
        filtered_signal = signal.filtfilt(b, a, signal_data)
        
        return filtered_signal
    
    def extract_features(self, preprocessed_data: np.ndarray, sampling_rate: float, **kwargs) -> Dict[str, np.ndarray]:
        """
        Extract features from preprocessed respiratory signal.
        
        Parameters
        ----------
        preprocessed_data : np.ndarray
            Preprocessed respiratory signal
        sampling_rate : float
            Sampling rate in Hz
        **kwargs : dict
            Optional parameters
            
        Returns
        -------
        Dict[str, np.ndarray]
            Dictionary of respiratory features
        """
        # Find breath cycles
        peaks, _ = signal.find_peaks(preprocessed_data, distance=int(sampling_rate * 2))
        breath_intervals = np.diff(peaks) / sampling_rate
        
        features = {
            'breathing_rate': 60 / np.mean(breath_intervals) if len(breath_intervals) > 0 else 0,
            'breath_interval_std': np.std(breath_intervals) if len(breath_intervals) > 0 else 0,
            'depth_variation': np.std(preprocessed_data[peaks]) if len(peaks) > 0 else 0,
            'irregularity': kurtosis(breath_intervals) if len(breath_intervals) > 0 else 0,
            'amplitude': np.mean(np.abs(signal.hilbert(preprocessed_data)))
        }
        
        return features
    
    def assess_quality(self, signal_data: np.ndarray, sampling_rate: float) -> float:
        """
        Assess respiratory signal quality.
        
        Parameters
        ----------
        signal_data : np.ndarray
            Respiratory signal data
        sampling_rate : float
            Sampling rate in Hz
            
        Returns
        -------
        float
            Quality score between 0 and 1
        """
        # Check breathing rate is physiologically plausible
        peaks, _ = signal.find_peaks(signal_data, distance=int(sampling_rate * 2))
        if len(peaks) >= 2:
            breathing_rate = 60 / np.mean(np.diff(peaks) / sampling_rate)
            if breathing_rate < 4 or breathing_rate > 40:
                return 0.6
        
        # Check signal variance
        if np.std(signal_data) < 0.01:
            return 0.5  # Very low variation suggests poor signal
            
        return 1.0


class TemperatureAdapter(PhysiologicalSignalAdapter):
    """Adapter for processing temperature signals."""
    
    def preprocess(self, signal_data: np.ndarray, sampling_rate: float, **kwargs) -> np.ndarray:
        """
        Preprocess temperature signal.
        
        Parameters
        ----------
        signal_data : np.ndarray
            Raw temperature signal
        sampling_rate : float
            Sampling rate in Hz
        **kwargs : dict
            Optional parameters including:
            - window_length : int (default: 5)
                Window length for moving average filter
            
        Returns
        -------
        np.ndarray
            Preprocessed temperature signal
            
        Raises
        ------
        ValueError
            If sampling_rate is not positive or signal_data contains NaN values
        """
        _validate_input(signal_data, sampling_rate)
            
        # Get filter parameters
        window_length = kwargs.get('window_length', 5)
        
        # Apply moving average filter
        window = np.ones(window_length) / window_length
        filtered_signal = np.convolve(signal_data, window, mode='same')
        
        return filtered_signal
    
    def extract_features(self, preprocessed_data: np.ndarray, sampling_rate: float, **kwargs) -> Dict[str, np.ndarray]:
        """
        Extract features from preprocessed temperature signal.
        
        Parameters
        ----------
        preprocessed_data : np.ndarray
            Preprocessed temperature signal
        sampling_rate : float
            Sampling rate in Hz
        **kwargs : dict
            Optional parameters
            
        Returns
        -------
        Dict[str, np.ndarray]
            Dictionary of temperature features
        """
        # Calculate basic statistics
        features = {
            'mean_temp': np.mean(preprocessed_data),
            'temp_std': np.std(preprocessed_data),
            'temp_range': np.ptp(preprocessed_data),
            'temp_gradient': np.mean(np.gradient(preprocessed_data, 1/sampling_rate)),
            'temp_variability': np.mean(np.abs(np.diff(preprocessed_data)))
        }
        
        return features
    
    def assess_quality(self, signal_data: np.ndarray, sampling_rate: float) -> float:
        """
        Assess temperature signal quality.
        
        Parameters
        ----------
        signal_data : np.ndarray
            Temperature signal data
        sampling_rate : float
            Sampling rate in Hz
            
        Returns
        -------
        float
            Quality score between 0 and 1
        """
        # Check for physiologically plausible range (typically 30-40°C for skin temp)
        if np.min(signal_data) < 25 or np.max(signal_data) > 42:
            return 0.6
            
        # Check for sudden changes that might indicate sensor issues
        temp_changes = np.abs(np.diff(signal_data))
        if np.any(temp_changes > 1.0):  # More than 1°C per sample
            return 0.7
            
        return 1.0 