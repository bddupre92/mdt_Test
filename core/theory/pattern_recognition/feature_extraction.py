"""Feature Extraction Framework

This module implements various feature extraction methods for physiological signal analysis.
It provides specialized extractors for different types of features:
- Time-domain features
- Frequency-domain features
- Statistical features
- Physiological features

Each feature extractor implements the FeatureExtractor interface defined in __init__.py.
"""

import numpy as np
from scipy import stats, signal as scipy_signal
from typing import Dict, List, Optional, Union, Tuple
from . import FeatureExtractor

class TimeDomainFeatures(FeatureExtractor):
    """Time-domain feature extraction for physiological signals."""
    
    def __init__(self, window_size: int = 256, overlap: float = 0.5):
        """Initialize time domain feature extractor.
        
        Args:
            window_size: Size of the sliding window for feature extraction
            overlap: Overlap between consecutive windows (0 to 1)
        """
        self.window_size = window_size
        self.overlap = overlap
        
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
        if signal.ndim != 1:
            raise ValueError("Signal must be 1-dimensional")
        if len(signal) < self.window_size:
            raise ValueError("Signal length must be >= window_size")
        return True
        
    def extract(self, signal: np.ndarray, **kwargs) -> Dict[str, np.ndarray]:
        """Extract time-domain features from the signal.
        
        Features include:
        - Peak-to-peak amplitude
        - Zero crossings
        - Mean absolute value
        - Waveform length
        - Slope sign changes
        
        Args:
            signal: Input physiological signal
            **kwargs: Additional parameters
            
        Returns:
            Dictionary of extracted features
        """
        self.validate_input(signal)
        
        # Calculate step size for overlapping windows
        step = int(self.window_size * (1 - self.overlap))
        n_windows = (len(signal) - self.window_size) // step + 1
        
        features = {
            'peak_to_peak': np.zeros(n_windows),
            'zero_crossings': np.zeros(n_windows),
            'mean_abs_value': np.zeros(n_windows),
            'waveform_length': np.zeros(n_windows),
            'slope_changes': np.zeros(n_windows)
        }
        
        for i in range(n_windows):
            start = i * step
            end = start + self.window_size
            window = signal[start:end]
            
            # Extract features
            features['peak_to_peak'][i] = np.ptp(window)
            features['zero_crossings'][i] = np.sum(np.diff(np.signbit(window)))
            features['mean_abs_value'][i] = np.mean(np.abs(window))
            features['waveform_length'][i] = np.sum(np.abs(np.diff(window)))
            features['slope_changes'][i] = np.sum(np.diff(np.sign(np.diff(window))) != 0)
            
        return features

class FrequencyDomainFeatures(FeatureExtractor):
    """Frequency-domain feature extraction for physiological signals."""
    
    def __init__(self, sampling_rate: float = 100.0, window_size: int = 256,
                 overlap: float = 0.5, nperseg: Optional[int] = None):
        """Initialize frequency domain feature extractor.
        
        Args:
            sampling_rate: Signal sampling rate in Hz
            window_size: Size of the sliding window
            overlap: Overlap between consecutive windows (0 to 1)
            nperseg: Length of each segment for spectral estimation
        """
        self.fs = sampling_rate
        self.window_size = window_size
        self.overlap = overlap
        self.nperseg = nperseg or window_size
        
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
        if signal.ndim != 1:
            raise ValueError("Signal must be 1-dimensional")
        if len(signal) < self.window_size:
            raise ValueError("Signal length must be >= window_size")
        return True
        
    def extract(self, signal: np.ndarray, **kwargs) -> Dict[str, np.ndarray]:
        """Extract frequency-domain features from the signal.
        
        Features include:
        - Power spectral density
        - Spectral centroid
        - Spectral bandwidth
        - Dominant frequency
        - Band powers (delta, theta, alpha, beta, gamma)
        
        Args:
            signal: Input physiological signal
            **kwargs: Additional parameters
            
        Returns:
            Dictionary of extracted features
        """
        self.validate_input(signal)
        
        # Calculate step size for overlapping windows
        step = int(self.window_size * (1 - self.overlap))
        n_windows = (len(signal) - self.window_size) // step + 1
        
        features = {
            'spectral_centroid': np.zeros(n_windows),
            'spectral_bandwidth': np.zeros(n_windows),
            'dominant_frequency': np.zeros(n_windows),
            'delta_power': np.zeros(n_windows),  # 0.5-4 Hz
            'theta_power': np.zeros(n_windows),  # 4-8 Hz
            'alpha_power': np.zeros(n_windows),  # 8-13 Hz
            'beta_power': np.zeros(n_windows),   # 13-30 Hz
            'gamma_power': np.zeros(n_windows)   # >30 Hz
        }
        
        for i in range(n_windows):
            start = i * step
            end = start + self.window_size
            window = signal[start:end]
            
            # Compute power spectral density
            frequencies, psd = scipy_signal.welch(window, fs=self.fs, nperseg=self.nperseg)
            
            # Extract features
            features['spectral_centroid'][i] = np.sum(frequencies * psd) / np.sum(psd)
            features['spectral_bandwidth'][i] = np.sqrt(np.sum(((frequencies - features['spectral_centroid'][i])**2) * psd) / np.sum(psd))
            features['dominant_frequency'][i] = frequencies[np.argmax(psd)]
            
            # Calculate band powers
            features['delta_power'][i] = np.sum(psd[(frequencies >= 0.5) & (frequencies < 4)])
            features['theta_power'][i] = np.sum(psd[(frequencies >= 4) & (frequencies < 8)])
            features['alpha_power'][i] = np.sum(psd[(frequencies >= 8) & (frequencies < 13)])
            features['beta_power'][i] = np.sum(psd[(frequencies >= 13) & (frequencies < 30)])
            features['gamma_power'][i] = np.sum(psd[frequencies >= 30])
            
        return features

class StatisticalFeatures(FeatureExtractor):
    """Statistical feature extraction for physiological signals."""
    
    def __init__(self, window_size: int = 256, overlap: float = 0.5):
        """Initialize statistical feature extractor.
        
        Args:
            window_size: Size of the sliding window
            overlap: Overlap between consecutive windows (0 to 1)
        """
        self.window_size = window_size
        self.overlap = overlap
        
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
        if signal.ndim != 1:
            raise ValueError("Signal must be 1-dimensional")
        if len(signal) < self.window_size:
            raise ValueError("Signal length must be >= window_size")
        return True
        
    def extract(self, signal: np.ndarray, **kwargs) -> Dict[str, np.ndarray]:
        """Extract statistical features from the signal.
        
        Features include:
        - Mean
        - Standard deviation
        - Skewness
        - Kurtosis
        - Median
        - IQR
        - Entropy
        
        Args:
            signal: Input physiological signal
            **kwargs: Additional parameters
            
        Returns:
            Dictionary of extracted features
        """
        self.validate_input(signal)
        
        # Calculate step size for overlapping windows
        step = int(self.window_size * (1 - self.overlap))
        n_windows = (len(signal) - self.window_size) // step + 1
        
        features = {
            'mean': np.zeros(n_windows),
            'std': np.zeros(n_windows),
            'skewness': np.zeros(n_windows),
            'kurtosis': np.zeros(n_windows),
            'median': np.zeros(n_windows),
            'iqr': np.zeros(n_windows),
            'entropy': np.zeros(n_windows)
        }
        
        for i in range(n_windows):
            start = i * step
            end = start + self.window_size
            window = signal[start:end]
            
            # Center the window before extracting features
            window = window - np.mean(window)
            
            # Extract features
            features['mean'][i] = np.mean(window)
            features['std'][i] = np.std(window)
            features['skewness'][i] = stats.skew(window)
            features['kurtosis'][i] = stats.kurtosis(window)
            features['median'][i] = np.median(window)
            features['iqr'][i] = np.percentile(window, 75) - np.percentile(window, 25)
            
            # Calculate entropy
            hist, _ = np.histogram(window, bins='auto', density=True)
            hist = hist[hist > 0]  # Remove zero probabilities
            features['entropy'][i] = -np.sum(hist * np.log2(hist))
            
        return features

class PhysiologicalFeatures(FeatureExtractor):
    """Physiological-specific feature extraction for biosignals."""
    
    def __init__(self, signal_type: str, sampling_rate: float = 100.0,
                 window_size: int = 256, overlap: float = 0.5):
        """Initialize physiological feature extractor.
        
        Args:
            signal_type: Type of physiological signal ('ecg', 'eeg', 'emg', etc.)
            sampling_rate: Signal sampling rate in Hz
            window_size: Size of the sliding window
            overlap: Overlap between consecutive windows (0 to 1)
        """
        self.signal_type = signal_type.lower()
        self.fs = sampling_rate
        self.window_size = window_size
        self.overlap = overlap
        
        # Validate signal type
        valid_types = {'ecg', 'eeg', 'emg', 'ppg', 'gsr', 'resp'}
        if self.signal_type not in valid_types:
            raise ValueError(f"Signal type must be one of: {valid_types}")
        
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
        if signal.ndim != 1:
            raise ValueError("Signal must be 1-dimensional")
        if len(signal) < self.window_size:
            raise ValueError("Signal length must be >= window_size")
        return True
        
    def extract(self, signal: np.ndarray, **kwargs) -> Dict[str, np.ndarray]:
        """Extract physiological-specific features from the signal.
        
        Features depend on signal type and include:
        - ECG: Heart rate variability features
        - EEG: Brain wave band powers
        - EMG: Muscle activation features
        - PPG: Blood volume pulse features
        - GSR: Skin conductance features
        - RESP: Respiratory features
        
        Args:
            signal: Input physiological signal
            **kwargs: Additional parameters
            
        Returns:
            Dictionary of extracted features
        """
        self.validate_input(signal)
        
        # Calculate step size for overlapping windows
        step = int(self.window_size * (1 - self.overlap))
        n_windows = (len(signal) - self.window_size) // step + 1
        
        # Initialize features based on signal type
        if self.signal_type == 'ecg':
            return self._extract_ecg_features(signal, n_windows, step)
        elif self.signal_type == 'eeg':
            return self._extract_eeg_features(signal, n_windows, step)
        elif self.signal_type == 'emg':
            return self._extract_emg_features(signal, n_windows, step)
        elif self.signal_type == 'ppg':
            return self._extract_ppg_features(signal, n_windows, step)
        elif self.signal_type == 'gsr':
            return self._extract_gsr_features(signal, n_windows, step)
        else:  # resp
            return self._extract_resp_features(signal, n_windows, step)
    
    def _extract_ecg_features(self, signal: np.ndarray, n_windows: int, step: int) -> Dict[str, np.ndarray]:
        """Extract ECG-specific features."""
        features = {
            'heart_rate': np.zeros(n_windows),
            'rr_intervals': np.zeros(n_windows),
            'hrv_sdnn': np.zeros(n_windows),
            'hrv_rmssd': np.zeros(n_windows)
        }
        
        for i in range(n_windows):
            start = i * step
            end = start + self.window_size
            window = signal[start:end]
            
            # Find R-peaks with more lenient parameters
            peaks, _ = scipy_signal.find_peaks(window, height=0.1, distance=int(0.2 * self.fs))
            
            if len(peaks) > 1:
                # Calculate features
                rr_intervals = np.diff(peaks) / self.fs
                features['heart_rate'][i] = 60 / np.mean(rr_intervals)
                features['rr_intervals'][i] = np.mean(rr_intervals)
                features['hrv_sdnn'][i] = np.std(rr_intervals)
                features['hrv_rmssd'][i] = np.sqrt(np.mean(np.diff(rr_intervals) ** 2))
            else:
                # Set default values if no peaks found
                features['heart_rate'][i] = 60  # Default 60 BPM
                features['rr_intervals'][i] = 1.0  # Default 1 second interval
                features['hrv_sdnn'][i] = 0.0
                features['hrv_rmssd'][i] = 0.0
                
        return features
    
    def _extract_eeg_features(self, signal: np.ndarray, n_windows: int, step: int) -> Dict[str, np.ndarray]:
        """Extract EEG-specific features."""
        freq_extractor = FrequencyDomainFeatures(self.fs, self.window_size, self.overlap)
        return freq_extractor.extract(signal)
    
    def _extract_emg_features(self, signal: np.ndarray, n_windows: int, step: int) -> Dict[str, np.ndarray]:
        """Extract EMG-specific features."""
        features = {
            'rms': np.zeros(n_windows),
            'mav': np.zeros(n_windows),
            'zc_rate': np.zeros(n_windows),
            'ssc': np.zeros(n_windows)
        }
        
        for i in range(n_windows):
            start = i * step
            end = start + self.window_size
            window = signal[start:end]
            
            # Calculate features
            features['rms'][i] = np.sqrt(np.mean(window ** 2))
            features['mav'][i] = np.mean(np.abs(window))
            features['zc_rate'][i] = np.sum(np.diff(np.signbit(window))) / len(window)
            features['ssc'][i] = np.sum(np.diff(np.sign(np.diff(window))) != 0) / len(window)
            
        return features
    
    def _extract_ppg_features(self, signal: np.ndarray, n_windows: int, step: int) -> Dict[str, np.ndarray]:
        """Extract PPG-specific features."""
        features = {
            'pulse_rate': np.zeros(n_windows),
            'pulse_amplitude': np.zeros(n_windows),
            'augmentation_index': np.zeros(n_windows)
        }
        
        for i in range(n_windows):
            start = i * step
            end = start + self.window_size
            window = signal[start:end]
            
            # Find peaks
            peaks, _ = scipy_signal.find_peaks(window, distance=int(0.2 * self.fs))
            
            if len(peaks) > 1:
                # Calculate features
                features['pulse_rate'][i] = 60 * len(peaks) / (len(window) / self.fs)
                features['pulse_amplitude'][i] = np.mean(window[peaks])
                # Simplified augmentation index
                features['augmentation_index'][i] = np.std(window[peaks]) / np.mean(window[peaks])
            else:
                # Set default values if no peaks found
                features['pulse_rate'][i] = 60  # Default 60 BPM
                features['pulse_amplitude'][i] = np.mean(window)
                features['augmentation_index'][i] = 0.0
                
        return features
    
    def _extract_gsr_features(self, signal: np.ndarray, n_windows: int, step: int) -> Dict[str, np.ndarray]:
        """Extract GSR-specific features."""
        features = {
            'mean_level': np.zeros(n_windows),
            'response_rate': np.zeros(n_windows),
            'response_amplitude': np.zeros(n_windows)
        }
        
        for i in range(n_windows):
            start = i * step
            end = start + self.window_size
            window = signal[start:end]
            
            # Calculate features
            features['mean_level'][i] = np.mean(window)
            # Detect SCRs (Skin Conductance Responses)
            peaks, _ = scipy_signal.find_peaks(window, height=0.05, distance=int(1 * self.fs))
            features['response_rate'][i] = len(peaks) / (len(window) / self.fs)
            if len(peaks) > 0:
                features['response_amplitude'][i] = np.mean(window[peaks])
            else:
                features['response_amplitude'][i] = 0.0
                
        return features
    
    def _extract_resp_features(self, signal: np.ndarray, n_windows: int, step: int) -> Dict[str, np.ndarray]:
        """Extract respiratory-specific features."""
        features = {
            'breathing_rate': np.zeros(n_windows),
            'tidal_volume': np.zeros(n_windows),
            'inspiration_time': np.zeros(n_windows),
            'expiration_time': np.zeros(n_windows)
        }
        
        for i in range(n_windows):
            start = i * step
            end = start + self.window_size
            window = signal[start:end]
            
            # Find peaks (inspiration) and troughs (expiration)
            peaks, _ = scipy_signal.find_peaks(window, distance=int(1 * self.fs))
            troughs, _ = scipy_signal.find_peaks(-window, distance=int(1 * self.fs))
            
            if len(peaks) > 1 and len(troughs) > 0:
                # Calculate features
                features['breathing_rate'][i] = 60 * len(peaks) / (len(window) / self.fs)
                
                # Calculate tidal volume using min of peak and trough lengths
                min_len = min(len(peaks), len(troughs))
                features['tidal_volume'][i] = np.mean(window[peaks[:min_len]] - window[troughs[:min_len]])
                
                if len(peaks) > 1 and len(troughs) > 1:
                    # Calculate inspiration and expiration times
                    insp_times = np.diff(peaks) / self.fs
                    exp_times = np.diff(troughs) / self.fs
                    features['inspiration_time'][i] = np.mean(insp_times)
                    features['expiration_time'][i] = np.mean(exp_times)
                else:
                    features['inspiration_time'][i] = 2.0  # Default 2 seconds
                    features['expiration_time'][i] = 3.0  # Default 3 seconds
            else:
                # Set default values if no peaks found
                features['breathing_rate'][i] = 12  # Default 12 breaths per minute
                features['tidal_volume'][i] = 0.0
                features['inspiration_time'][i] = 2.0  # Default 2 seconds
                features['expiration_time'][i] = 3.0  # Default 3 seconds
                
        return features 