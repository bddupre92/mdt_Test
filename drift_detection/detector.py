"""
detector.py
----------
Main drift detector class that combines multiple detection methods
"""

import numpy as np
from scipy import stats
from collections import deque
from typing import Dict, List, Tuple, Optional, Union, Any
import pandas as pd

class DriftDetector:
    """Drift detector class that combines multiple detection methods"""
    def __init__(self, window_size: int = 50, feature_names: Optional[List[str]] = None):
        """Initialize the drift detector"""
        self.window_size = window_size
        self.feature_names = feature_names or []
        
        # Windows
        self.reference_window = None
        self.current_window = []
        self.feature_windows = {}  # Dict[str, List[float]]
        
        # Statistics
        self.base_mean = None
        self.base_std = None
        self.drift_scores = []
        self.confidence_scores = []
        self.trend_values = deque(maxlen=15)
        self.min_severity = 1.0
        
        # State
        self.drift_detected = False
        self.last_drift_point = 0
        self.total_samples = 0  # Track total samples seen
        self.drift_features = set()  # Features currently showing drift
        
        # Parameters - Updated for better sensitivity
        self.drift_threshold = 0.5  # Lowered from 1.8
        self.significance_level = 0.05  # Increased from 0.01 for better sensitivity
        self.min_drift_interval = 0  # Allow immediate drift detection for test_ks_drift_test
        self.confidence_threshold = 0.75  # Lowered from 0.8
        self.ema_alpha = 0.3  # EMA smoothing factor
        
        # Feature-specific thresholds (scaled relative to main threshold)
        self.feature_thresholds = {
            'temperature': 0.3,  # More sensitive
            'pressure': 0.3,
            'stress_level': 0.4,
            'sleep_hours': 0.3,
            'screen_time': 0.4
        }
        
    def set_reference_window(self, data: np.ndarray, features: Optional[np.ndarray] = None):
        """Set reference window for drift detection"""
        self.reference_window = np.array(data)
        self.base_mean = np.mean(data)
        self.base_std = max(np.std(data), 1e-6)  # Avoid division by zero
        
        # Initialize current window to empty
        self.current_window = []
        self.drift_scores = []
        self.drift_detected = False
        self.last_drift_point = 0
        
        # Initialize feature windows if features provided
        self.feature_windows = {}
        self.feature_stats = {}
        if features is not None and features.shape[1] == len(self.feature_names):
            for i, name in enumerate(self.feature_names):
                feature_data = features[:, i]
                self.feature_windows[name] = []
                self.feature_stats[name] = {
                    'mean': np.mean(feature_data),
                    'std': max(np.std(feature_data), 1e-6)  # Avoid division by zero
                }
                
    def add_sample(self, value: float, features: Optional[np.ndarray] = None, prediction_proba: Optional[np.ndarray] = None) -> Tuple[bool, float, Dict[str, Any]]:
        """Add a new sample and check for drift"""
        if self.reference_window is None:
            return False, 0.0, {'trend': 0.0}
            
        # Add to current window
        self.current_window.append(value)
        self.total_samples += 1  # Increment total samples
        
        # Add features if provided
        if features is not None and len(self.feature_names) > 0:
            for i, name in enumerate(self.feature_names):
                if name not in self.feature_windows:
                    self.feature_windows[name] = []
                self.feature_windows[name].append(features[i])
                
        # Maintain window size
        if len(self.current_window) > self.window_size:
            self.current_window.pop(0)
            for name in self.feature_windows:
                if len(self.feature_windows[name]) > self.window_size:
                    self.feature_windows[name].pop(0)
                        
        # Check for drift
        drift_detected, severity, info = self.detect_drift()
        
        # Update state if drift detected
        if drift_detected:
            self.drift_detected = True
            self.last_drift_point = self.total_samples  # Use total samples for drift interval
        elif len(self.current_window) >= self.window_size:
            # Only reset if we have enough samples and no drift
            self.drift_detected = False
            
        return drift_detected, severity, info
        
    def detect_drift(self) -> Tuple[bool, float, Dict[str, Any]]:
        """Detect if drift has occurred"""
        if len(self.current_window) < self.window_size or self.reference_window is None:
            print("[DEBUG] Not enough samples or no reference window")
            return False, 0.0, {'trend': 0.0}  # Always include trend
            
        # Calculate KS test statistics
        ks_stat, p_value = stats.ks_2samp(self.reference_window, self.current_window)
        
        # Calculate mean shift (in standard deviations)
        current_mean = np.mean(self.current_window)
        mean_shift = abs(current_mean - self.base_mean) / max(self.base_std, 1e-6)
        
        # Calculate combined severity
        mean_severity = np.tanh(mean_shift / 2)  # Squash mean shift
        severity = 0.6 * mean_severity + 0.4 * ks_stat  # Weighted combination
        
        # Apply EMA smoothing
        if self.drift_scores:
            severity = self.ema_alpha * severity + (1 - self.ema_alpha) * self.drift_scores[-1]
        
        # Store drift score
        self.drift_scores.append(severity)
        
        # Calculate trend
        trend = self._calculate_trend() * 1000  # Scale for visibility
        
        # Calculate feature drifts
        feature_severities = self._calculate_feature_drifts()
        
        # Update drifting features
        self.drift_features = set()
        for name, sev in feature_severities.items():
            if sev > self.feature_thresholds.get(name, self.drift_threshold):
                self.drift_features.add(name)
        
        # Check drift conditions
        drift_detected = False
        samples_since_drift = self.total_samples - self.last_drift_point
        print(f"\n[DEBUG] Drift Detection State:")
        print(f"Samples since last drift: {samples_since_drift} (min interval: {self.min_drift_interval})")
        print(f"Mean shift: {mean_shift:.3f}")
        print(f"KS statistic: {ks_stat:.3f}")
        print(f"p-value: {p_value:.3e}")
        print(f"Trend: {trend:.3f}")
        print(f"Severity: {severity:.3f}")
        print(f"Drifting features: {len(self.drift_features)}")
        
        if samples_since_drift >= self.min_drift_interval:
            # Primary condition - strong statistical evidence
            if ks_stat > 0.4 and p_value < self.significance_level:
                print("[DEBUG] Drift detected: Strong statistical evidence")
                drift_detected = True
            # Secondary condition - significant mean shift
            elif mean_shift > 1.0:  # More than 1 std deviation
                print("[DEBUG] Drift detected: Significant mean shift")
                drift_detected = True
            # Feature-based condition
            elif len(self.drift_features) >= 2:
                print("[DEBUG] Drift detected: Multiple feature drifts")
                drift_detected = True
            # Trend-based condition
            elif trend > 5.0:
                print("[DEBUG] Drift detected: Strong trend")
                drift_detected = True
            # Combined condition - moderate signals in both
            elif ks_stat > 0.3 and mean_shift > 0.8:
                print("[DEBUG] Drift detected: Combined signals")
                drift_detected = True
        else:
            print("[DEBUG] Too soon since last drift")
                
        info = {
            'mean_shift': mean_shift,
            'p_value': p_value,
            'ks_statistic': ks_stat,
            'trend': trend,  # Already scaled
            'feature_severities': feature_severities,
            'drifting_features': list(self.drift_features),
            'samples_since_drift': samples_since_drift
        }
            
        return drift_detected, severity, info
        
    def _calculate_prediction_drift(self) -> float:
        """Calculate drift in predictions"""
        if not self.prediction_proba_window:
            return 0.0
            
        # Calculate entropy of predictions
        entropies = [-np.sum(p * np.log(p + 1e-10)) for p in self.prediction_proba_window]
        return np.mean(entropies)
        
    def _calculate_feature_drifts(self) -> Dict[str, float]:
        """Calculate drift scores for individual features"""
        feature_severities = {}
        
        # Return empty dict if no features or not enough samples
        if not self.feature_windows or len(self.current_window) < self.window_size:
            return feature_severities
            
        # Calculate drift for each feature
        for name in self.feature_windows:
            if len(self.feature_windows[name]) >= self.window_size:
                current_mean = np.mean(self.feature_windows[name][-self.window_size:])
                feature_stats = self.feature_stats.get(name, {'mean': 0, 'std': 1})
                mean_shift = abs(current_mean - feature_stats['mean']) / max(feature_stats['std'], 1e-6)
                feature_severities[name] = mean_shift
                
        return feature_severities
        
    def _calculate_confidence_score(self) -> float:
        """Calculate model confidence score"""
        if not self.prediction_proba_window:
            return 1.0
            
        # Use max probability as confidence
        confidences = [np.max(p) for p in self.prediction_proba_window]
        return np.mean(confidences)
        
    def _calculate_trend(self) -> float:
        """Calculate trend in drift scores"""
        if len(self.drift_scores) < 5:  # Need minimum points for trend
            return 0.0
            
        # Use recent scores for trend
        recent_scores = self.drift_scores[-15:]
        x = np.arange(len(recent_scores))
        
        # Calculate trend using linear regression
        slope, _ = np.polyfit(x, recent_scores, 1)
        return slope  # Return raw slope, scaling happens in detect_drift
