"""
Drift Detection Module

This module provides algorithms for detecting drift in data streams and optimization processes.
"""

import numpy as np
from typing import Tuple, List, Union, Optional
from scipy import stats
import logging
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor

class DriftDetector:
    """
    Class for detecting concept drift in data streams and optimization processes.
    
    Methods include:
    - Distribution-based drift detection (KS-test, AD-test)
    - Error rate monitoring
    - Window-based comparison
    - Adaptive windowing
    """
    
    def __init__(
        self,
        window_size: int = 50,
        drift_threshold: float = 0.05,
        significance_level: float = 0.05,
        method: str = 'ks'
    ):
        """
        Initialize drift detector.
        
        Args:
            window_size: Size of window for drift detection
            drift_threshold: Threshold for drift detection
            significance_level: Significance level for statistical tests
            method: Detection method (ks, ad, error_rate, window_comparison)
        """
        self.window_size = window_size
        self.drift_threshold = drift_threshold
        self.significance_level = significance_level
        self.method = method
        self.reference_window = None
        self.drift_score_history = []
    
    def detect_drift(
        self, 
        current_window_X: np.ndarray, 
        current_window_y: Optional[np.ndarray] = None
    ) -> Tuple[bool, float, float]:
        """
        Detect drift based on the specified method.
        
        Args:
            current_window_X: Current window of data features
            current_window_y: Current window of target values (optional)
            
        Returns:
            Tuple of (is_drift, drift_score, p_value)
        """
        # Set reference window if not set
        if self.reference_window is None:
            self.reference_window = current_window_X
            return False, 0.0, 1.0
            
        # Calculate drift based on method
        if self.method == 'ks':
            return self._detect_drift_ks(current_window_X)
        elif self.method == 'ad':
            return self._detect_drift_ad(current_window_X)
        elif self.method == 'error_rate':
            if current_window_y is None:
                raise ValueError("Target values required for error_rate method")
            return self._detect_drift_error_rate(current_window_X, current_window_y)
        elif self.method == 'window_comparison':
            return self._detect_drift_window_comparison(current_window_X)
        else:
            raise ValueError(f"Unknown drift detection method: {self.method}")
    
    def _detect_drift_ks(self, current_window: np.ndarray) -> Tuple[bool, float, float]:
        """
        Detect drift using the Kolmogorov-Smirnov test.
        
        Args:
            current_window: Current window of data
            
        Returns:
            Tuple of (is_drift, drift_score, p_value)
        """
        # Get the average p-value across all features
        p_values = []
        
        for feature_idx in range(current_window.shape[1]):
            ref_feature = self.reference_window[:, feature_idx]
            curr_feature = current_window[:, feature_idx]
            
            # Perform KS test
            try:
                _, p_value = stats.ks_2samp(ref_feature, curr_feature)
                p_values.append(p_value)
            except Exception as e:
                logging.warning(f"Error in KS test for feature {feature_idx}: {e}")
                p_values.append(1.0)
        
        avg_p_value = np.mean(p_values)
        drift_score = 1.0 - avg_p_value
        
        self.drift_score_history.append(drift_score)
        
        # Update reference window
        if len(self.drift_score_history) > 10:
            self.reference_window = current_window
        
        is_drift = avg_p_value < self.significance_level
        
        return is_drift, drift_score, avg_p_value
    
    def _detect_drift_ad(self, current_window: np.ndarray) -> Tuple[bool, float, float]:
        """
        Detect drift using the Anderson-Darling test.
        
        Args:
            current_window: Current window of data
            
        Returns:
            Tuple of (is_drift, drift_score, p_value)
        """
        # Get the average drift score across all features
        drift_scores = []
        
        for feature_idx in range(current_window.shape[1]):
            ref_feature = self.reference_window[:, feature_idx]
            curr_feature = current_window[:, feature_idx]
            
            # Perform AD test
            try:
                result = stats.anderson_ksamp([ref_feature, curr_feature])
                p_value = result.significance_level
                drift_scores.append(1.0 - p_value)
            except Exception as e:
                logging.warning(f"Error in AD test for feature {feature_idx}: {e}")
                drift_scores.append(0.0)
        
        avg_drift_score = np.mean(drift_scores)
        
        self.drift_score_history.append(avg_drift_score)
        
        # Update reference window
        if len(self.drift_score_history) > 10:
            self.reference_window = current_window
        
        is_drift = avg_drift_score > self.drift_threshold
        p_value = 1.0 - avg_drift_score
        
        return is_drift, avg_drift_score, p_value
    
    def _detect_drift_error_rate(
        self, 
        current_window_X: np.ndarray, 
        current_window_y: np.ndarray
    ) -> Tuple[bool, float, float]:
        """
        Detect drift using error rate monitoring.
        
        Args:
            current_window_X: Current window of data features
            current_window_y: Current window of target values
            
        Returns:
            Tuple of (is_drift, drift_score, p_value)
        """
        # Train a simple model on reference window
        from sklearn.ensemble import RandomForestRegressor
        
        model = RandomForestRegressor(n_estimators=10)
        model.fit(self.reference_window, current_window_y[:len(self.reference_window)])
        
        # Predict on current window
        y_pred = model.predict(current_window_X)
        error = mean_squared_error(current_window_y, y_pred)
        
        # Compare with threshold
        p_value = np.exp(-error)  # Convert error to a p-value like score
        drift_score = error
        
        self.drift_score_history.append(drift_score)
        
        # Update reference window
        if len(self.drift_score_history) > 10:
            self.reference_window = current_window_X
        
        is_drift = drift_score > self.drift_threshold
        
        return is_drift, drift_score, p_value
    
    def _detect_drift_window_comparison(self, current_window: np.ndarray) -> Tuple[bool, float, float]:
        """
        Detect drift by direct window comparison.
        
        Args:
            current_window: Current window of data
            
        Returns:
            Tuple of (is_drift, drift_score, p_value)
        """
        # Compute distance between windows
        ref_mean = np.mean(self.reference_window, axis=0)
        curr_mean = np.mean(current_window, axis=0)
        
        distance = np.linalg.norm(ref_mean - curr_mean)
        max_distance = np.linalg.norm(np.max(self.reference_window, axis=0) - np.min(self.reference_window, axis=0))
        
        normalized_distance = distance / max_distance if max_distance > 0 else distance
        
        p_value = np.exp(-normalized_distance)  # Convert distance to a p-value like score
        drift_score = normalized_distance
        
        self.drift_score_history.append(drift_score)
        
        # Update reference window
        if len(self.drift_score_history) > 10:
            self.reference_window = current_window
        
        is_drift = drift_score > self.drift_threshold
        
        return is_drift, drift_score, p_value
    
    def reset(self):
        """Reset the detector state."""
        self.reference_window = None
        self.drift_score_history = []
        
    def get_drift_score_history(self) -> List[float]:
        """
        Get the history of drift scores.
        
        Returns:
            List of drift scores over time
        """
        return self.drift_score_history
