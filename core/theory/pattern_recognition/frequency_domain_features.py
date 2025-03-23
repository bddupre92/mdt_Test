"""Frequency-domain feature extraction for physiological signals.

This module implements various frequency-domain features for physiological signal analysis,
focusing on spectral characteristics and frequency-based properties of the signals.
"""

import numpy as np
from typing import Dict, List, Optional, Union, Tuple
from scipy import signal
from .feature_extraction import FeatureExtractor

class FrequencyDomainFeatures(FeatureExtractor):
    """Extracts frequency-domain features from physiological signals."""
    
    def __init__(self, sampling_rate: float, window_size: int = 256,
                 overlap: float = 0.5, features: Optional[List[str]] = None,
                 freq_bands: Optional[Dict[str, Tuple[float, float]]] = None) -> None:
        """Initialize frequency domain feature extractor.
        
        Args:
            sampling_rate: Sampling rate of the signal in Hz
            window_size: Number of samples in each analysis window
            overlap: Fraction of overlap between consecutive windows (0 to 1)
            features: List of features to extract. If None, extracts all features.
            freq_bands: Dictionary of frequency bands to analyze {name: (low_freq, high_freq)}
        """
        super().__init__()
        self.sampling_rate = sampling_rate
        self.window_size = window_size
        self.overlap = overlap
        
        # Default frequency bands (can be overridden)
        self.freq_bands = freq_bands or {
            'delta': (0.5, 4),
            'theta': (4, 8),
            'alpha': (8, 13),
            'beta': (13, 30),
            'gamma': (30, 100)
        }
        
        self.available_features = [
            'spectral_centroid', 'spectral_bandwidth',
            'spectral_rolloff', 'spectral_flatness',
            'band_powers', 'dominant_frequency',
            'median_frequency', 'mean_frequency',
            'power_ratio', 'spectral_edge'
        ]
        self.features = features or self.available_features
        
        # Prepare window function
        self.window = signal.windows.hann(window_size)

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
        """Extract frequency-domain features from the signal.
        
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
        """Extract features from a single window of data.
        
        Args:
            window: Signal window of shape (window_size, n_channels)
            
        Returns:
            Dictionary of computed features for the window
        """
        features = {}
        
        # Compute FFT for the window
        freqs = np.fft.rfftfreq(self.window_size, 1/self.sampling_rate)
        fft = np.fft.rfft(window * self.window[:, np.newaxis], axis=0)
        power = np.abs(fft) ** 2
        
        if 'spectral_centroid' in self.features:
            features['spectral_centroid'] = self._compute_spectral_centroid(freqs, power)
            
        if 'spectral_bandwidth' in self.features:
            features['spectral_bandwidth'] = self._compute_spectral_bandwidth(freqs, power)
            
        if 'spectral_rolloff' in self.features:
            features['spectral_rolloff'] = self._compute_spectral_rolloff(freqs, power)
            
        if 'spectral_flatness' in self.features:
            features['spectral_flatness'] = self._compute_spectral_flatness(power)
            
        if 'band_powers' in self.features:
            features.update(self._compute_band_powers(freqs, power))
            
        if 'dominant_frequency' in self.features:
            features['dominant_frequency'] = freqs[np.argmax(power, axis=0)]
            
        if 'median_frequency' in self.features:
            features['median_frequency'] = self._compute_median_frequency(freqs, power)
            
        if 'mean_frequency' in self.features:
            # Calculate mean frequency for each channel separately
            mean_freqs = []
            for i in range(power.shape[1]):
                mean_freqs.append(np.average(freqs, weights=power[:, i]))
            features['mean_frequency'] = np.array(mean_freqs)
            
        if 'power_ratio' in self.features:
            features['power_ratio'] = self._compute_power_ratio(freqs, power)
            
        if 'spectral_edge' in self.features:
            features['spectral_edge'] = self._compute_spectral_edge(freqs, power)
            
        return features

    def _compute_spectral_centroid(self, freqs: np.ndarray, power: np.ndarray) -> np.ndarray:
        """Compute spectral centroid."""
        return np.sum(freqs[:, np.newaxis] * power, axis=0) / np.sum(power, axis=0)

    def _compute_spectral_bandwidth(self, freqs: np.ndarray, power: np.ndarray) -> np.ndarray:
        """Compute spectral bandwidth."""
        centroid = self._compute_spectral_centroid(freqs, power)
        return np.sqrt(np.sum(((freqs[:, np.newaxis] - centroid) ** 2) * power, axis=0) / np.sum(power, axis=0))

    def _compute_spectral_rolloff(self, freqs: np.ndarray, power: np.ndarray,
                                percentile: float = 0.85) -> np.ndarray:
        """Compute frequency below which percentile% of the power is contained."""
        cumsum = np.cumsum(power, axis=0)
        threshold = percentile * cumsum[-1]
        indices = np.argmax(cumsum >= threshold, axis=0)
        return freqs[indices]

    def _compute_spectral_flatness(self, power: np.ndarray) -> np.ndarray:
        """Compute spectral flatness (ratio of geometric mean to arithmetic mean)."""
        return np.exp(np.mean(np.log(power + 1e-10), axis=0)) / np.mean(power, axis=0)

    def _compute_band_powers(self, freqs: np.ndarray, power: np.ndarray) -> Dict[str, np.ndarray]:
        """Compute power in different frequency bands."""
        band_powers = {}
        for band_name, (low_freq, high_freq) in self.freq_bands.items():
            mask = (freqs >= low_freq) & (freqs <= high_freq)
            band_power = np.sum(power[mask], axis=0)
            band_powers[f'{band_name}_power'] = band_power
        return band_powers

    def _compute_median_frequency(self, freqs: np.ndarray, power: np.ndarray) -> np.ndarray:
        """Compute median frequency."""
        cumsum = np.cumsum(power, axis=0)
        total_power = cumsum[-1]
        indices = np.argmax(cumsum >= total_power/2, axis=0)
        return freqs[indices]

    def _compute_power_ratio(self, freqs: np.ndarray, power: np.ndarray,
                           cutoff: float = 10.0) -> np.ndarray:
        """Compute ratio of high-frequency to low-frequency power."""
        mask_low = freqs <= cutoff
        mask_high = freqs > cutoff
        power_low = np.sum(power[mask_low], axis=0)
        power_high = np.sum(power[mask_high], axis=0)
        return power_high / (power_low + 1e-10)

    def _compute_spectral_edge(self, freqs: np.ndarray, power: np.ndarray,
                             percentile: float = 0.95) -> np.ndarray:
        """Compute spectral edge frequency."""
        cumsum = np.cumsum(power, axis=0)
        threshold = percentile * cumsum[-1]
        indices = np.argmax(cumsum >= threshold, axis=0)
        return freqs[indices]

    def get_feature_names(self) -> List[str]:
        """Get list of available feature names.
        
        Returns:
            List of feature names that can be extracted
        """
        feature_names = self.features.copy()
        if 'band_powers' in feature_names:
            feature_names.remove('band_powers')
            feature_names.extend([f'{band}_power' for band in self.freq_bands.keys()])
        return feature_names

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