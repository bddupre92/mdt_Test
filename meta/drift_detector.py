"""
Drift detection module for meta-learning optimization.
"""
from typing import Dict, List, Optional, Tuple, Union, Any
import numpy as np
from scipy import stats
import logging
from datetime import datetime
from visualization.drift_analysis import DriftAnalyzer

logger = logging.getLogger(__name__)

class DriftDetector:
    def __init__(self, 
                 window_size: int = 50, 
                 drift_threshold: float = 1.8, 
                 significance_level: float = 0.01,
                 min_drift_interval: int = 40,
                 ema_alpha: float = 0.3):
        """Initialize drift detector with parameter validation
        
        Args:
            window_size: Size of the sliding window for drift detection
            drift_threshold: Threshold for mean shift to detect drift
            significance_level: P-value threshold for statistical significance
            min_drift_interval: Minimum number of samples between drift detections
            ema_alpha: Alpha parameter for exponential moving average
        """
        # Validate parameters
        if not isinstance(window_size, int) or window_size < 10:
            raise ValueError("window_size must be an integer >= 10")
            
        if not isinstance(drift_threshold, (int, float)) or drift_threshold <= 0:
            raise ValueError("drift_threshold must be a positive number")
            
        if not 0 < significance_level < 1:
            raise ValueError("significance_level must be between 0 and 1")
            
        if not isinstance(min_drift_interval, int) or min_drift_interval < 1:
            raise ValueError("min_drift_interval must be a positive integer")
            
        if not 0 < ema_alpha < 1:
            raise ValueError("ema_alpha must be between 0 and 1")
            
        # Store parameters
        self.window_size = window_size
        self.drift_threshold = drift_threshold
        self.significance_level = significance_level
        self.min_drift_interval = min_drift_interval
        self.ema_alpha = ema_alpha
        
        # Initialize tracking
        self.reference_window = None
        self.current_window = []
        self.samples_since_drift = 0
        self.drift_scores = []
        self.mean_shifts = []
        self.ks_stats = []
        self.p_values = []
        self.last_drift_detected = False
        
        # Initialize analyzer
        self.analyzer = DriftAnalyzer()
        self.analyzer.drift_threshold = drift_threshold
        
    def detect_drift(self, y_true: np.ndarray, y_pred: np.ndarray) -> bool:
        """Detect drift between true values and predictions
        
        Args:
            y_true: True target values
            y_pred: Predicted target values
            
        Returns:
            True if drift is detected, False otherwise
        """
        # Convert inputs to numpy arrays if needed
        if isinstance(y_true, list):
            y_true = np.array(y_true)
        if isinstance(y_pred, list):
            y_pred = np.array(y_pred)
            
        # Calculate prediction errors
        errors = y_true - y_pred
        
        # Update current window
        self.current_window.extend(errors.flatten())
        
        # Ensure current window doesn't grow too large
        if len(self.current_window) > 2 * self.window_size:
            self.current_window = self.current_window[-2 * self.window_size:]
            
        # Initialize reference window if needed
        if self.reference_window is None:
            if len(self.current_window) >= self.window_size:
                self.reference_window = np.array(self.current_window[:self.window_size])
                self.current_window = self.current_window[self.window_size:]
            else:
                # Not enough data yet
                self.samples_since_drift += len(errors)
                return False
                
        # Check if we have enough data for drift detection
        if len(self.current_window) < self.window_size:
            self.samples_since_drift += len(errors)
            return False
            
        # Get current window for comparison
        curr_window = np.array(self.current_window[-self.window_size:])
        
        # Calculate statistics
        mean_shift = abs(np.mean(curr_window) - np.mean(self.reference_window))
        mean_shift_normalized = mean_shift / (np.std(self.reference_window) if np.std(self.reference_window) > 0 else 1.0)
        
        # Perform KS test
        try:
            ks_stat, p_value = stats.ks_2samp(curr_window, self.reference_window)
        except ValueError:
            # Handle potential errors in KS test
            ks_stat = 0.0
            p_value = 1.0
            
        # Calculate severity score
        # Use tanh to squash mean shift and prevent it from dominating
        mean_shift_component = np.tanh(mean_shift_normalized / 2)
        # Combine with KS statistic using weights
        severity = 0.6 * mean_shift_component + 0.4 * ks_stat
        
        # Update history
        self.drift_scores.append(severity)
        self.mean_shifts.append(mean_shift_normalized)
        self.ks_stats.append(ks_stat)
        self.p_values.append(p_value)
        
        # Limit history size
        max_history = 100
        if len(self.drift_scores) > max_history:
            self.drift_scores = self.drift_scores[-max_history:]
            self.mean_shifts = self.mean_shifts[-max_history:]
            self.ks_stats = self.ks_stats[-max_history:]
            self.p_values = self.p_values[-max_history:]
            
        # Calculate trend if we have enough data
        trend = 0.0
        if len(self.drift_scores) >= 5:
            # Use linear regression to calculate trend
            x = np.arange(len(self.drift_scores[-5:]))
            y = np.array(self.drift_scores[-5:])
            trend = np.polyfit(x, y, 1)[0] * 1000  # Scale for better visibility
            
        # Check drift conditions
        drift_detected = False
        
        # Increment samples since last drift
        self.samples_since_drift += len(errors)
        
        # Primary condition: significant mean shift AND low p-value
        primary_condition = (mean_shift_normalized > self.drift_threshold and 
                            p_value < self.significance_level)
                            
        # Secondary condition: very strong statistical evidence
        secondary_condition = (p_value < 1e-10 and ks_stat > 0.5)
        
        # Check if minimum interval has passed
        interval_ok = self.samples_since_drift >= self.min_drift_interval
        
        # Detect drift if conditions are met
        if (primary_condition or secondary_condition) and interval_ok:
            drift_detected = True
            self.samples_since_drift = 0
            
            # Update reference window
            self.reference_window = curr_window
            
            # Track drift event in analyzer
            self.analyzer.add_drift_event(
                timestamp=datetime.now(),
                severity=severity,
                error=p_value,
                predictions=y_pred,
                true_values=y_true
            )
            
            logger.info(f"Drift detected - Mean shift: {mean_shift_normalized:.3f}, "
                       f"KS stat: {ks_stat:.3f}, p-value: {p_value:.3e}")
                       
        self.last_drift_detected = drift_detected
        return drift_detected
        
    def get_statistics(self) -> Dict[str, Any]:
        """Get current drift statistics
        
        Returns:
            Dictionary with current drift statistics
        """
        if not self.drift_scores:
            return {
                'mean_shift': 0.0,
                'ks_statistic': 0.0,
                'p_value': 1.0,
                'severity': 0.0,
                'trend': 0.0
            }
            
        return {
            'mean_shift': self.mean_shifts[-1] if self.mean_shifts else 0.0,
            'ks_statistic': self.ks_stats[-1] if self.ks_stats else 0.0,
            'p_value': self.p_values[-1] if self.p_values else 1.0,
            'severity': self.drift_scores[-1] if self.drift_scores else 0.0,
            'trend': self._calculate_trend()
        }
        
    def _calculate_trend(self) -> float:
        """Calculate trend in drift scores
        
        Returns:
            Trend value
        """
        if len(self.drift_scores) < 5:
            return 0.0
            
        x = np.arange(len(self.drift_scores[-5:]))
        y = np.array(self.drift_scores[-5:])
        return np.polyfit(x, y, 1)[0] * 1000
        
    def plot_detection_results(self, save_path: Optional[str] = None) -> None:
        """Plot drift detection results."""
        self.analyzer.plot_drift_severity(save_path)
    
    def plot_drift_analysis(self, save_path: Optional[str] = None) -> None:
        """Plot comprehensive drift analysis."""
        self.analyzer.plot_drift_analysis(save_path)
        
    def save_analysis(self, save_dir: str) -> None:
        """Save all drift analysis plots."""
        self.analyzer.save_analysis(save_dir)
