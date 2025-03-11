"""Time-domain feature extraction for physiological signals.

This module implements various time-domain features for physiological signal analysis,
focusing on temporal characteristics and morphological properties of the signals.
"""

import numpy as np
from typing import Dict, List, Optional, Union
from .feature_extraction import FeatureExtractor

class TimeDomainFeatures(FeatureExtractor):
    """Extracts time-domain features from physiological signals."""
    
    def __init__(self, window_size: int = 256, overlap: float = 0.5,
                 features: Optional[List[str]] = None) -> None:
        """Initialize time domain feature extractor.
        
        Args:
            window_size: Number of samples in each analysis window
            overlap: Fraction of overlap between consecutive windows (0 to 1)
            features: List of features to extract. If None, extracts all features.
        """
        super().__init__()
        self.window_size = window_size
        self.overlap = overlap
        self.available_features = [
            'mean_amplitude', 'peak_to_peak', 'zero_crossings',
            'slope_changes', 'rms', 'waveform_length',
            'integrated_emg', 'variance', 'max_amplitude',
            'min_amplitude', 'mean_absolute_value'
        ]
        self.features = features or self.available_features

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
        """Extract time-domain features from the signal.
        
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
        
        # Convert lists to numpy arrays
        return {k: np.array(v) for k, v in features.items()}

    def _extract_window_features(self, window: np.ndarray) -> Dict[str, float]:
        """Extract features from a single window of data.
        
        Args:
            window: Signal window of shape (window_size, n_channels)
            
        Returns:
            Dictionary of computed features for the window
        """
        features = {}
        
        if 'mean_amplitude' in self.features:
            features['mean_amplitude'] = np.mean(window, axis=0)
            
        if 'peak_to_peak' in self.features:
            features['peak_to_peak'] = np.ptp(window, axis=0)
            
        if 'zero_crossings' in self.features:
            features['zero_crossings'] = self._count_zero_crossings(window)
            
        if 'slope_changes' in self.features:
            features['slope_changes'] = self._count_slope_changes(window)
            
        if 'rms' in self.features:
            features['rms'] = np.sqrt(np.mean(np.square(window), axis=0))
            
        if 'waveform_length' in self.features:
            features['waveform_length'] = np.sum(np.abs(np.diff(window, axis=0)), axis=0)
            
        if 'integrated_emg' in self.features:
            features['integrated_emg'] = np.sum(np.abs(window), axis=0)
            
        if 'variance' in self.features:
            features['variance'] = np.var(window, axis=0)
            
        if 'max_amplitude' in self.features:
            features['max_amplitude'] = np.max(window, axis=0)
            
        if 'min_amplitude' in self.features:
            features['min_amplitude'] = np.min(window, axis=0)
            
        if 'mean_absolute_value' in self.features:
            features['mean_absolute_value'] = np.mean(np.abs(window), axis=0)
            
        return features

    def _count_zero_crossings(self, window: np.ndarray) -> np.ndarray:
        """Count number of zero crossings in the signal window."""
        return np.sum(np.diff(window > 0, axis=0).astype(bool), axis=0)

    def _count_slope_changes(self, window: np.ndarray) -> np.ndarray:
        """Count number of slope sign changes in the signal window."""
        diff = np.diff(window, axis=0)
        return np.sum(np.diff(diff > 0, axis=0).astype(bool), axis=0)

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