"""Statistical feature extraction for physiological signals.

This module implements various statistical features for physiological signal analysis,
focusing on distribution properties and statistical measures of the signals.
"""

import numpy as np
from scipy import stats
from typing import Dict, List, Optional, Union
from .feature_extraction import FeatureExtractor

class StatisticalFeatures(FeatureExtractor):
    """Extracts statistical features from physiological signals."""
    
    def __init__(self, window_size: int = 256, overlap: float = 0.5,
                 features: Optional[List[str]] = None) -> None:
        """Initialize statistical feature extractor.
        
        Args:
            window_size: Number of samples in each analysis window
            overlap: Fraction of overlap between consecutive windows (0 to 1)
            features: List of features to extract. If None, extracts all features.
        """
        super().__init__()
        self.window_size = window_size
        self.overlap = overlap
        self.available_features = [
            'mean', 'std', 'var', 'skewness', 'kurtosis',
            'median', 'iqr', 'range', 'entropy',
            'percentiles', 'mode', 'coefficient_variation',
            'rms', 'energy', 'max_min_ratio',
            'mean_abs_deviation', 'median_abs_deviation'
        ]
        self.features = features or self.available_features
        self.percentile_values = [25, 50, 75, 90, 95]

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
        """Extract statistical features from the signal.
        
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
        
        if 'mean' in self.features:
            features['mean'] = np.mean(window, axis=0)
            
        if 'std' in self.features:
            features['std'] = np.std(window, axis=0)
            
        if 'var' in self.features:
            features['var'] = np.var(window, axis=0)
            
        if 'skewness' in self.features:
            features['skewness'] = stats.skew(window, axis=0)
            
        if 'kurtosis' in self.features:
            features['kurtosis'] = stats.kurtosis(window, axis=0)
            
        if 'median' in self.features:
            features['median'] = np.median(window, axis=0)
            
        if 'iqr' in self.features:
            q75, q25 = np.percentile(window, [75, 25], axis=0)
            features['iqr'] = q75 - q25
            
        if 'range' in self.features:
            features['range'] = np.ptp(window, axis=0)
            
        if 'entropy' in self.features:
            features['entropy'] = self._compute_entropy(window)
            
        if 'percentiles' in self.features:
            for p in self.percentile_values:
                features[f'percentile_{p}'] = np.percentile(window, p, axis=0)
                
        if 'mode' in self.features:
            features['mode'] = stats.mode(window, axis=0)[0][0]
            
        if 'coefficient_variation' in self.features:
            features['coefficient_variation'] = np.std(window, axis=0) / (np.mean(window, axis=0) + 1e-10)
            
        if 'rms' in self.features:
            features['rms'] = np.sqrt(np.mean(np.square(window), axis=0))
            
        if 'energy' in self.features:
            features['energy'] = np.sum(np.square(window), axis=0)
            
        if 'max_min_ratio' in self.features:
            features['max_min_ratio'] = (np.max(window, axis=0) + 1e-10) / (np.min(window, axis=0) + 1e-10)
            
        if 'mean_abs_deviation' in self.features:
            features['mean_abs_deviation'] = np.mean(np.abs(window - np.mean(window, axis=0)), axis=0)
            
        if 'median_abs_deviation' in self.features:
            features['median_abs_deviation'] = np.median(np.abs(window - np.median(window, axis=0)), axis=0)
            
        return features

    def _compute_entropy(self, window: np.ndarray, bins: int = 50) -> np.ndarray:
        """Compute Shannon entropy of the signal window.
        
        Args:
            window: Signal window
            bins: Number of bins for histogram
            
        Returns:
            Entropy value for each channel
        """
        entropies = []
        for channel in range(window.shape[1]):
            hist, _ = np.histogram(window[:, channel], bins=bins)
            prob = hist / np.sum(hist)
            prob = prob[prob > 0]  # Remove zero probabilities
            entropy = -np.sum(prob * np.log2(prob))
            entropies.append(entropy)
        return np.array(entropies)

    def get_feature_names(self) -> List[str]:
        """Get list of available feature names.
        
        Returns:
            List of feature names that can be extracted
        """
        feature_names = []
        for feature in self.features:
            if feature == 'percentiles':
                feature_names.extend([f'percentile_{p}' for p in self.percentile_values])
            else:
                feature_names.append(feature)
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