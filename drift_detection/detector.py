"""
detector.py
----------
Main drift detector class that combines multiple detection methods
"""

import numpy as np
from scipy import stats
from collections import deque

class DriftDetector:
    """Drift detector class that combines multiple detection methods"""
    def __init__(self, window_size=50):
        """Initialize drift detector
        Args:
            window_size (int): Size of the sliding window
        """
        self.window_size = window_size
        self.current_window = []
        self.reference_window = None
        self.base_distribution = None
        self.base_mean = None
        self.base_std = None
        self.drift_scores = []
        self.trend_values = deque(maxlen=15)
        self.min_severity = 1.0  # Start at 1.0 to avoid division issues
        self.drift_detected = False
        self.last_drift_point = 0
        
        # Fixed parameters
        self.drift_threshold = 1.8  # Lowered threshold for mean shift
        self.significance_level = 0.01  # Keep stringent p-value
        self.min_drift_interval = 40  # Slightly reduced interval
        
    def set_reference_window(self, data):
        """Set reference window for drift detection
        Args:
            data (array-like): Data to use as reference
        """
        self.reference_window = np.array(data)
        if self.base_distribution is None:
            self.base_distribution = np.array(data)
            self.base_mean = np.mean(data)
            self.base_std = np.std(data)
            
        # Reset state
        self.drift_detected = False
        self.last_drift_point = 0
        self.min_severity = 1.0  # Reset to 1.0
        self.drift_scores = []
        self.trend_values.clear()
            
    def add_sample(self, value):
        """Add new sample to current window
        Args:
            value (float): New sample value
        """
        self.current_window.append(value)
        if len(self.current_window) > self.window_size:
            self.current_window.pop(0)
            
    def detect_drift(self):
        """
        Detect if drift has occurred using multiple methods
        Returns:
            bool: True if drift detected
            float: Drift severity score
        """
        if len(self.current_window) < self.window_size or self.reference_window is None:
            return False, 0.0
            
        # Calculate means and standard deviations
        ref_mean = np.mean(self.reference_window)
        cur_mean = np.mean(self.current_window)
        ref_std = np.std(self.reference_window)
        cur_std = np.std(self.current_window)
        
        # Calculate KS test
        ks_stat, p_value = stats.ks_2samp(self.reference_window, self.current_window)
        
        # Calculate mean shift relative to pooled standard deviation
        pooled_std = np.sqrt((ref_std**2 + cur_std**2) / 2)
        mean_shift = abs(cur_mean - ref_mean) / (pooled_std + 1e-10)
        
        # Calculate severity based on statistical measures
        severity = (
            0.6 * float(np.tanh(mean_shift / 2)) +  # Squash mean shift with tanh
            0.4 * float(ks_stat)                    # KS statistic is already [0,1]
        )
        
        # Update minimum severity
        self.min_severity = float(min(self.min_severity, severity))
            
        # Normalize severity relative to minimum
        normalized_severity = float(severity / (self.min_severity + 1e-10))
        normalized_severity = float(np.clip(normalized_severity, 0.0, 1.0))
        
        # Update drift scores with exponential moving average
        alpha = 0.3
        if self.drift_scores:
            last_score = self.drift_scores[-1]
            score = alpha * normalized_severity + (1 - alpha) * last_score
        else:
            score = normalized_severity
            
        self.drift_scores.append(float(score))
        if len(self.drift_scores) > self.window_size:
            self.drift_scores.pop(0)
            
        # Calculate trend using linear regression on recent scores
        if len(self.drift_scores) >= 5:  # Need more points for stable trend
            x = np.arange(len(self.drift_scores))
            y = np.array(self.drift_scores)
            slope, _, _, _, _ = stats.linregress(x, y)
            self.trend_values.append(float(slope))
        else:
            self.trend_values.append(0.0)
            
        # Detect drift based on severity and minimum interval
        samples_since_drift = len(self.current_window) - self.last_drift_point
        if samples_since_drift >= self.min_drift_interval:
            # Primary criterion: significant mean shift
            if mean_shift > self.drift_threshold and p_value < self.significance_level:
                self.drift_detected = True
                self.last_drift_point = len(self.current_window)
                return True, float(normalized_severity)
                
            # Secondary criterion: very strong statistical evidence
            if p_value < 1e-10 and ks_stat > 0.5:
                self.drift_detected = True
                self.last_drift_point = len(self.current_window)
                return True, float(normalized_severity)
                
        return False, float(severity)  # Return raw severity for better interpretability
        
    def get_trend(self):
        """Get current trend in drift scores"""
        if len(self.trend_values) < 5:  # Need more points for stable trend
            return 0.0
        values = np.array(list(self.trend_values))
        # Use exponential moving average for trend
        alpha = 0.3
        trend = values[0]
        for v in values[1:]:
            trend = alpha * v + (1 - alpha) * trend
        return float(trend * 1000.0)  # Scale for better visibility
