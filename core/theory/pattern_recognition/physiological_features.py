"""Physiological feature extraction for biomedical signals.

This module implements specialized feature extraction methods for various physiological signals
including ECG, EEG, EMG, PPG, GSR, and respiratory signals.
"""

import numpy as np
from scipy import signal, stats
from typing import Dict, List, Optional, Union, Tuple
from .feature_extraction import FeatureExtractor

class PhysiologicalFeatures(FeatureExtractor):
    """Extracts physiological features from biomedical signals."""
    
    SUPPORTED_SIGNALS = ['ecg', 'eeg', 'emg', 'ppg', 'gsr', 'resp']
    
    def __init__(self, signal_type: str, sampling_rate: float,
                 window_size: int = 256, overlap: float = 0.5) -> None:
        """Initialize physiological feature extractor.
        
        Args:
            signal_type: Type of physiological signal ('ecg', 'eeg', 'emg', 'ppg', 'gsr', 'resp')
            sampling_rate: Sampling rate of the signal in Hz
            window_size: Number of samples in each analysis window
            overlap: Fraction of overlap between consecutive windows (0 to 1)
        """
        super().__init__()
        if signal_type.lower() not in self.SUPPORTED_SIGNALS:
            raise ValueError(f"Signal type must be one of {self.SUPPORTED_SIGNALS}")
            
        self.signal_type = signal_type.lower()
        self.sampling_rate = sampling_rate
        self.window_size = window_size
        self.overlap = overlap
        
        # Initialize signal-specific parameters
        self._init_signal_parameters()

    def _init_signal_parameters(self) -> None:
        """Initialize signal-specific parameters and thresholds."""
        if self.signal_type == 'ecg':
            self.features = [
                'heart_rate', 'rr_intervals', 'hrv_time_domain',
                'hrv_frequency_domain', 'qrs_duration', 'qt_interval'
            ]
            # ECG specific parameters
            self.qrs_threshold = 0.5
            
        elif self.signal_type == 'eeg':
            self.features = [
                'band_powers', 'spectral_edge', 'hjorth_parameters',
                'spectral_entropy', 'wavelet_coefficients'
            ]
            # EEG frequency bands
            self.freq_bands = {
                'delta': (0.5, 4),
                'theta': (4, 8),
                'alpha': (8, 13),
                'beta': (13, 30),
                'gamma': (30, 100)
            }
            
        elif self.signal_type == 'emg':
            self.features = [
                'rms', 'mav', 'wl', 'zc', 'ssc',
                'integrated_emg', 'frequency_median'
            ]
            # EMG specific parameters
            self.zc_threshold = 0.01
            
        elif self.signal_type == 'ppg':
            self.features = [
                'pulse_rate', 'peak_amplitude', 'peak_interval',
                'augmentation_index', 'reflection_index'
            ]
            # PPG specific parameters
            self.peak_threshold = 0.4
            
        elif self.signal_type == 'gsr':
            self.features = [
                'scr_amplitude', 'scr_rise_time', 'scr_recovery_time',
                'scl_mean', 'scl_slope'
            ]
            # GSR specific parameters
            self.scr_threshold = 0.05
            
        elif self.signal_type == 'resp':
            self.features = [
                'respiratory_rate', 'tidal_volume', 'minute_ventilation',
                'inspiration_time', 'expiration_time'
            ]
            # Respiratory specific parameters
            self.resp_threshold = 0.2

    def validate_input(self, signal: np.ndarray) -> bool:
        """Validate input signal.
        
        Args:
            signal: Input signal to validate
            
        Returns:
            True if valid
            
        Raises:
            ValueError: If signal is invalid
        """
        if not isinstance(signal, np.ndarray):
            raise ValueError("Signal must be a numpy array")
        if signal.ndim not in [1, 2]:
            raise ValueError("Signal must be 1D or 2D (for multi-channel)")
        if len(signal) < self.window_size:
            raise ValueError(f"Signal length ({len(signal)}) must be >= window_size ({self.window_size})")
        return True

    def extract(self, signal: np.ndarray, **kwargs) -> Dict[str, np.ndarray]:
        """Extract physiological features from the signal.
        
        Args:
            signal: Input signal array of shape (n_samples,) or (n_samples, n_channels)
            **kwargs: Additional parameters
            
        Returns:
            Dictionary containing computed features
        """
        # Validate input
        self.validate_input(signal)
        
        if signal.ndim == 1:
            signal = signal.reshape(-1, 1)
            
        n_samples, n_channels = signal.shape
        step_size = int(self.window_size * (1 - self.overlap))
        n_windows = (n_samples - self.window_size) // step_size + 1
        
        features = {}
        for start_idx in range(0, n_samples - self.window_size + 1, step_size):
            window = signal[start_idx:start_idx + self.window_size]
            window_features = self._extract_window_features(window)
            
            for feature_name, value in window_features.items():
                if feature_name not in features:
                    features[feature_name] = []
                features[feature_name].append(value)
        
        return {k: np.array(v) for k, v in features.items()}

    def _extract_window_features(self, window: np.ndarray) -> Dict[str, np.ndarray]:
        """Extract features from a single window based on signal type."""
        if self.signal_type == 'ecg':
            return self._extract_ecg_features(window)
        elif self.signal_type == 'eeg':
            return self._extract_eeg_features(window)
        elif self.signal_type == 'emg':
            return self._extract_emg_features(window)
        elif self.signal_type == 'ppg':
            return self._extract_ppg_features(window)
        elif self.signal_type == 'gsr':
            return self._extract_gsr_features(window)
        else:  # resp
            return self._extract_resp_features(window)

    def _extract_ecg_features(self, window: np.ndarray) -> Dict[str, np.ndarray]:
        """Extract ECG-specific features."""
        features = {}
        
        # Detect R-peaks
        peaks, _ = signal.find_peaks(window[:, 0], height=self.qrs_threshold)
        
        if len(peaks) > 1:
            # Heart rate
            rr_intervals = np.diff(peaks) / self.sampling_rate
            # Convert to beats per minute
            heart_rate = 60 / np.mean(rr_intervals) if np.mean(rr_intervals) > 0 else 0
            
            # Ensure heart rate is within physiological limits (40-200 bpm)
            if 40 <= heart_rate <= 200:
                features['heart_rate'] = np.array([heart_rate])
            else:
                features['heart_rate'] = np.array([60])  # Default to 60 bpm
            
            # HRV time domain
            features['hrv_sdnn'] = np.std(rr_intervals)
            features['hrv_rmssd'] = np.sqrt(np.mean(np.square(np.diff(rr_intervals))))
            
            # HRV frequency domain
            if len(rr_intervals) > 3:
                freqs, psd = signal.welch(rr_intervals, fs=1.0/np.mean(rr_intervals))
                lf_mask = (freqs >= 0.04) & (freqs <= 0.15)
                hf_mask = (freqs >= 0.15) & (freqs <= 0.4)
                features['hrv_lf'] = np.sum(psd[lf_mask])
                features['hrv_hf'] = np.sum(psd[hf_mask])
                if features['hrv_hf'] > 0:
                    features['hrv_lf_hf_ratio'] = features['hrv_lf'] / features['hrv_hf']
                else:
                    features['hrv_lf_hf_ratio'] = np.array([1.0])
        
        return features

    def _extract_eeg_features(self, window: np.ndarray) -> Dict[str, np.ndarray]:
        """Extract EEG-specific features."""
        features = {}
        
        # Compute power spectrum
        freqs, psd = signal.welch(window, fs=self.sampling_rate, axis=0)
        
        # Band powers
        for band_name, (low_freq, high_freq) in self.freq_bands.items():
            mask = (freqs >= low_freq) & (freqs <= high_freq)
            features[f'{band_name}_power'] = np.sum(psd[mask], axis=0)
            
        # Hjorth parameters
        diff1 = np.diff(window, axis=0)
        diff2 = np.diff(diff1, axis=0)
        
        activity = np.var(window, axis=0)
        mobility = np.sqrt(np.var(diff1, axis=0) / activity)
        complexity = np.sqrt(np.var(diff2, axis=0) / np.var(diff1, axis=0)) / mobility
        
        features['hjorth_activity'] = activity
        features['hjorth_mobility'] = mobility
        features['hjorth_complexity'] = complexity
        
        return features

    def _extract_emg_features(self, window: np.ndarray) -> Dict[str, np.ndarray]:
        """Extract EMG-specific features."""
        features = {}
        
        # Time domain features
        features['rms'] = np.sqrt(np.mean(np.square(window), axis=0))
        features['mav'] = np.mean(np.abs(window), axis=0)
        features['wl'] = np.sum(np.abs(np.diff(window, axis=0)), axis=0)
        
        # Zero crossings
        zero_crossings = np.diff((window > self.zc_threshold).astype(int), axis=0)
        features['zc'] = np.sum(np.abs(zero_crossings), axis=0)
        
        # Slope sign changes
        diff = np.diff(window, axis=0)
        slope_changes = np.diff(diff > 0, axis=0)
        features['ssc'] = np.sum(slope_changes, axis=0)
        
        # Frequency domain
        freqs, psd = signal.welch(window, fs=self.sampling_rate, axis=0)
        features['frequency_median'] = np.median(freqs)
        
        return features

    def _extract_ppg_features(self, window: np.ndarray) -> Dict[str, np.ndarray]:
        """Extract PPG-specific features."""
        features = {}
        
        # Find peaks
        peaks, _ = signal.find_peaks(window[:, 0], height=self.peak_threshold)
        
        if len(peaks) > 1:
            # Pulse rate
            peak_intervals = np.diff(peaks) / self.sampling_rate
            # Convert to beats per minute
            pulse_rate = 60 / np.mean(peak_intervals) if np.mean(peak_intervals) > 0 else 0
            
            # Ensure pulse rate is within physiological limits (40-200 bpm)
            if 40 <= pulse_rate <= 200:
                features['pulse_rate'] = np.array([pulse_rate])
            else:
                features['pulse_rate'] = np.array([72])  # Default to 72 bpm
            
            # Peak characteristics
            peak_amplitudes = window[peaks, 0]
            features['peak_amplitude_mean'] = np.mean(peak_amplitudes)
            features['peak_amplitude_std'] = np.std(peak_amplitudes)
            
            # Reflection index
            troughs = signal.find_peaks(-window[:, 0])[0]
            if len(troughs) > 0:
                features['reflection_index'] = np.mean(peak_amplitudes) / np.abs(np.mean(window[troughs, 0]))
        
        return features

    def _extract_gsr_features(self, window: np.ndarray) -> Dict[str, np.ndarray]:
        """Extract GSR-specific features."""
        features = {}
        
        # Skin conductance level
        features['scl_mean'] = np.mean(window, axis=0)
        features['scl_std'] = np.std(window, axis=0)
        
        # Skin conductance response
        peaks, _ = signal.find_peaks(window[:, 0], height=self.scr_threshold)
        if len(peaks) > 0:
            features['scr_rate'] = len(peaks) / (self.window_size / self.sampling_rate)
            features['scr_amplitude_mean'] = np.mean(window[peaks, 0])
            
            # Response characteristics
            rise_times = []
            recovery_times = []
            for peak_idx in peaks:
                # Find onset (previous trough)
                onset = peak_idx - 1
                while onset > 0 and window[onset, 0] > window[onset-1, 0]:
                    onset -= 1
                rise_times.append((peak_idx - onset) / self.sampling_rate)
                
                # Find recovery (return to baseline)
                recovery = peak_idx + 1
                while recovery < len(window) and window[recovery, 0] > window[onset, 0]:
                    recovery += 1
                if recovery < len(window):
                    recovery_times.append((recovery - peak_idx) / self.sampling_rate)
            
            if rise_times:
                features['scr_rise_time'] = np.mean(rise_times)
            if recovery_times:
                features['scr_recovery_time'] = np.mean(recovery_times)
        
        return features

    def _extract_resp_features(self, window: np.ndarray) -> Dict[str, np.ndarray]:
        """Extract respiratory-specific features."""
        features = {}
        
        # Find peaks (inspiration) and troughs (expiration)
        peaks, _ = signal.find_peaks(window[:, 0], height=self.resp_threshold)
        troughs, _ = signal.find_peaks(-window[:, 0], height=self.resp_threshold)
        
        if len(peaks) > 1:
            # Respiratory rate
            breath_intervals = np.diff(peaks) / self.sampling_rate
            respiratory_rate = 60 / np.mean(breath_intervals) if np.mean(breath_intervals) > 0 else 0
            
            # Ensure respiratory rate is within physiological limits (5-60 bpm)
            if 5 <= respiratory_rate <= 60:
                features['respiratory_rate'] = np.array([respiratory_rate])
            else:
                features['respiratory_rate'] = np.array([15])  # Default to 15 bpm
            
            # Timing
            if len(troughs) > 0:
                # Match each peak with following trough for inspiration time
                inspiration_times = []
                expiration_times = []
                
                for i, peak in enumerate(peaks[:-1]):
                    # Find next trough
                    next_trough_idx = np.searchsorted(troughs, peak)
                    if next_trough_idx < len(troughs):
                        next_trough = troughs[next_trough_idx]
                        inspiration_times.append((next_trough - peak) / self.sampling_rate)
                        
                        # Find next peak for expiration time
                        if i + 1 < len(peaks):
                            next_peak = peaks[i + 1]
                            expiration_times.append((next_peak - next_trough) / self.sampling_rate)
                
                if inspiration_times:
                    features['inspiration_time'] = np.mean(inspiration_times)
                if expiration_times:
                    features['expiration_time'] = np.mean(expiration_times)
                
                # Volume estimation (if calibrated signal)
                # Make sure we have matching peaks and troughs
                if len(peaks) > 0 and len(troughs) > 0:
                    # Calculate tidal volumes from matched peaks and troughs
                    tidal_volumes = []
                    for i in range(min(len(peaks), len(troughs))):
                        tidal_volumes.append(abs(window[peaks[i], 0] - window[troughs[i], 0]))
                    
                    if tidal_volumes:
                        features['tidal_volume'] = np.mean(tidal_volumes)
                        if 'respiratory_rate' in features:
                            features['minute_ventilation'] = features['tidal_volume'] * features['respiratory_rate']
        
        return features

    def get_feature_names(self) -> List[str]:
        """Get list of available feature names.
        
        Returns:
            List of feature names that can be extracted
        """
        return self.features.copy()

    def validate_signal(self, signal: np.ndarray) -> bool:
        """Validate input signal format and properties.
        
        Args:
            signal: Input signal to validate
            
        Returns:
            True if signal is valid, False otherwise
        """
        if not isinstance(signal, np.ndarray):
            return False
        
        if signal.ndim not in [1, 2]:
            return False
            
        if len(signal) < self.window_size:
            return False
            
        return True 