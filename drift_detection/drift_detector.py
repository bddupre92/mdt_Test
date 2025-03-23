"""
detector.py
----------
Drift detection implementation
"""

from typing import Tuple, Dict, Any, Optional, List, Union
import numpy as np
from scipy import stats
import logging
import time

logger = logging.getLogger(__name__)

class DriftDetector:
    def __init__(self, 
                 window_size: int = 50, 
                 drift_threshold: float = 1.8, 
                 significance_level: float = 0.01,
                 min_drift_interval: int = 40,
                 ema_alpha: float = 0.3,
                 confidence_threshold: float = 0.8,
                 feature_names: Optional[List[str]] = None,
                 feature_thresholds: Optional[Dict[str, float]] = None,
                 feature_significance: Optional[Dict[str, float]] = None,
                 logger: Optional[logging.Logger] = None,
                 max_history_size: int = 20):
        """Initialize drift detector with parameter validation"""
        # Validate window size
        if not isinstance(window_size, int) or window_size < 10:
            raise ValueError("window_size must be an integer >= 10")
            
        # Validate thresholds and levels
        if not isinstance(drift_threshold, (int, float)) or drift_threshold <= 0:
            raise ValueError("drift_threshold must be a positive number")
        if not 0 < significance_level < 1:
            raise ValueError("significance_level must be between 0 and 1")
        if not isinstance(min_drift_interval, int) or min_drift_interval < 1:
            raise ValueError("min_drift_interval must be a positive integer")
            
        # Validate smoothing parameters
        if not 0 < ema_alpha < 1:
            raise ValueError("ema_alpha must be between 0 and 1")
        if not 0 < confidence_threshold < 1:
            raise ValueError("confidence_threshold must be between 0 and 1")
            
        # Validate feature configurations
        if feature_names is not None and not isinstance(feature_names, list):
            raise ValueError("feature_names must be a list or None")
        if feature_thresholds is not None:
            if not isinstance(feature_thresholds, dict):
                raise ValueError("feature_thresholds must be a dictionary or None")
            if not all(isinstance(v, (int, float)) and v > 0 for v in feature_thresholds.values()):
                raise ValueError("feature_threshold values must be positive numbers")
        if feature_significance is not None:
            if not isinstance(feature_significance, dict):
                raise ValueError("feature_significance must be a dictionary or None")
            if not all(0 < v < 1 for v in feature_significance.values()):
                raise ValueError("feature_significance values must be between 0 and 1")
                
        # Store parameters
        self.window_size = window_size
        self.drift_threshold = drift_threshold
        self.significance_level = significance_level
        self.min_drift_interval = min_drift_interval
        self.ema_alpha = ema_alpha
        self.confidence_threshold = confidence_threshold
        self.feature_names = feature_names if feature_names is not None else []
        self.feature_thresholds = feature_thresholds if feature_thresholds is not None else {}
        self.feature_significance = feature_significance if feature_significance is not None else {}
        self.max_history_size = max_history_size
        
        # Initialize tracking
        self.drift_detected = False
        self.reference_window = None
        self.current_window = []
        self.original_reference = None
        self.last_severity = 0.0
        self.samples_since_drift = 0
        self.last_drift = 0
        self.last_reference_update = 0
        self.last_drift_detected = False
        self.last_info = {'mean_shift': 0.0, 'ks_statistic': 0.0, 'p_value': 1.0, 'trend': 0.0}
        
        # Statistics tracking with memory limitation
        self.drift_scores = []
        self.mean_shifts = []
        self.ks_stats = []
        self.p_values = []
        self.severity_history = []
        self.trend = 0.0
        
        # Additional history tracking attributes
        self.mean_shift_history = []
        self.ks_stat_history = []
        self.p_value_history = []
        self.drifting_features = []
        self.drifting_features_history = []
        
        # For backward compatibility
        self.scores = []
        
        # Set up logging
        if logger is None:
            self.logger = logging.getLogger(__name__)
        else:
            self.logger = logger
            
        self.logger.info(
            f"Initializing DriftDetector - Window: {window_size}, "
            f"Threshold: {drift_threshold:.3f}, "
            f"Alpha: {ema_alpha:.3f}, "
            f"Min Interval: {min_drift_interval}"
        )
    
    def set_reference_window(self, data: np.ndarray, original: Optional[np.ndarray] = None):
        """Set reference window for drift detection.
        
        Args:
            data: Reference data window
            original: Original reference data for comparison
        """
        # Convert to numpy array if needed
        if isinstance(data, list):
            data = np.array(data)
            
        # Store reference window
        self.reference_window = data
        
        # Store original reference if provided
        if original is not None:
            if isinstance(original, list):
                original = np.array(original)
            self.original_reference = original
            
            # Extract feature names if not already set
            if hasattr(self, 'feature_names') and not self.feature_names and original.ndim > 1:
                self.feature_names = [f"feature_{i}" for i in range(original.shape[1])]
                self.logger.debug(f"Auto-generated feature names: {self.feature_names}")
        else:
            self.original_reference = data.copy()
            
        self.logger.info(f"Reference window set with {len(data)} samples")
    
    def add_sample(self, point: float, features: Optional[np.ndarray] = None, 
                  prediction_proba: Optional[np.ndarray] = None) -> Tuple[bool, float, Dict[str, float]]:
        """Add a sample and check for drift.
        
        Args:
            point: New data point (typically prediction probability)
            features: Optional feature vector for feature-level drift detection
            prediction_proba: Optional prediction probability vector
            
        Returns:
            Tuple of (drift_detected, severity, info_dict)
        """
        # Add point to current window
        self.current_window.append(point)
        
        # Check if we have enough data
        if len(self.current_window) < self.window_size:
            return False, 0.0, {'mean_shift': 0.0, 'ks_statistic': 0.0, 'p_value': 1.0}
        
        # Ensure current window doesn't exceed window size
        if len(self.current_window) > self.window_size:
            self.current_window = self.current_window[-self.window_size:]
        
        # Initialize reference window if needed
        if self.reference_window is None:
            self.reference_window = self.current_window.copy()
            self.original_reference = self.current_window.copy()
            return False, 0.0, {'mean_shift': 0.0, 'ks_statistic': 0.0, 'p_value': 1.0}
        
        # Check for drift using detection logic
        drift_detected, severity, info = self.detect_drift(
            curr_data=self.current_window[-self.window_size:],
            ref_data=self.reference_window,
            features=features,
            prediction_proba=prediction_proba
        )
            
        # Apply EMA smoothing to severity
        if not hasattr(self, 'ema_score'):
            self.ema_score = severity
        else:
            self.ema_score = (1 - self.ema_alpha) * self.ema_score + self.ema_alpha * severity
            
        # Track drift scores
        if not hasattr(self, 'drift_scores'):
            self.drift_scores = []
        self.drift_scores.append(severity)
        if len(self.drift_scores) > self.max_history_size:
            self.drift_scores.pop(0)
        
        # Update last results
        self.last_severity = severity
        self.last_drift_detected = drift_detected
        self.last_info = info
        
        # Update reference window if drift is detected
        if drift_detected:
            self.logger.info(f"Drift detected - Updating reference window")
            self.reference_window = self.current_window.copy()
            self.samples_since_drift = 0
        else:
            self.samples_since_drift += 1
            
        # Return detection result
        return drift_detected, severity, info
    
    def calculate_severity(self, mean_shift: float, ks_stat: float):
        """Calculate drift severity using weighted combination of mean shift and KS statistic.
        
        Args:
            mean_shift: Absolute mean shift between windows
            ks_stat: KS statistic value
            
        Returns:
            Combined severity score
        """
        # Apply tanh to squash mean shift
        squashed_mean = np.tanh(mean_shift/2)
        
        # Calculate weighted severity (0.6 for mean shift, 0.4 for KS)
        severity = 0.6 * squashed_mean + 0.4 * ks_stat
        
        # Log detailed calculation
        self.logger.debug(
            f"Severity calculation:\n"
            f"  Raw mean shift: {mean_shift:.4f}\n"
            f"  Squashed mean (tanh): {squashed_mean:.4f}\n"
            f"  KS statistic: {ks_stat:.4f}\n"
            f"  Final severity (0.6*mean + 0.4*ks): {severity:.4f}"
        )
        
        return severity

    def detect_drift(self, curr_data: Union[List[float], np.ndarray], 
                    ref_data: Union[List[float], np.ndarray],
                    features: Optional[np.ndarray] = None,
                    prediction_proba: Optional[np.ndarray] = None) -> Tuple[bool, float, Dict[str, Any]]:
        """Detect drift between current and reference data.
        
        Args:
            curr_data: Current data window
            ref_data: Reference data window
            features: Optional feature vector for feature-level drift detection
            prediction_proba: Optional prediction probability vector
            
        Returns:
            Tuple of (drift_detected, severity, info_dict)
        """
        # Convert to numpy arrays if needed
        if isinstance(curr_data, list):
            curr_data = np.array(curr_data)
        if isinstance(ref_data, list):
            ref_data = np.array(ref_data)
            
        # Calculate mean shift
        mean_shift = np.abs(np.mean(curr_data) - np.mean(ref_data))
        
        # Normalize mean shift by standard deviation of reference window
        ref_std = np.std(ref_data)
        if ref_std > 0:
            mean_shift_normalized = mean_shift / ref_std
        else:
            mean_shift_normalized = mean_shift
            
        # Perform KS test
        try:
            ks_stat, p_value = stats.ks_2samp(curr_data, ref_data)
        except ValueError:
            self.logger.warning("KS test failed, using default values")
            ks_stat, p_value = 0.0, 1.0
            
        # Calculate severity score
        # Use tanh to squash mean shift and prevent it from dominating
        mean_shift_component = np.tanh(mean_shift_normalized / 2)
        # Combine with KS statistic using weights
        severity = 0.6 * mean_shift_component + 0.4 * ks_stat
        
        # Apply EMA smoothing to severity
        if not hasattr(self, 'ema_severity'):
            self.ema_severity = severity
        else:
            self.ema_severity = (1 - self.ema_alpha) * self.ema_severity + self.ema_alpha * severity
            
        # Update statistics history
        self.mean_shift_history.append(mean_shift_normalized)
        self.ks_stat_history.append(ks_stat)
        self.p_value_history.append(p_value)
        
        # Check if we have enough data to update our tracking history 
        if np.size(severity) > 0:
            # Ensure we're storing a scalar value
            if isinstance(severity, np.ndarray):
                if severity.size == 1:
                    severity_scalar = float(severity.item())
                else:
                    severity_scalar = float(severity[0]) if severity.size > 0 else 0.0
            else:
                severity_scalar = float(severity)
                
            # Limit history size
            if len(self.severity_history) >= self.max_history_size:
                self.severity_history.pop(0)
            self.severity_history.append(severity_scalar)
            
        # Calculate trend if we have enough data
        trend = 0.0
        if len(self.severity_history) >= 5:
            # Use linear regression to calculate trend
            x = np.arange(len(self.severity_history[-5:]))
            y = np.array(self.severity_history[-5:])
            try:
                trend_coeffs = np.polyfit(x, y, 1)
                trend = trend_coeffs[0] * 1000.0
                
                # Ensure trend is a scalar
                if isinstance(trend, np.ndarray):
                    if trend.size == 1:
                        trend = float(trend.item())
                    else:
                        trend = float(trend[0]) if trend.size > 0 else 0.0
                
                # Weight trend by r-squared to account for fit quality
                r_squared = trend_coeffs[1] ** 2
                if isinstance(r_squared, np.ndarray):
                    r_squared = float(r_squared.item()) if r_squared.size == 1 else 0.0
                weighted_trend = trend * r_squared
                
                # Apply EMA smoothing if we have history
                if hasattr(self, 'trend_history') and len(self.trend_history) > 0:
                    self.trend = self.ema_alpha * weighted_trend + (1 - self.ema_alpha) * self.trend_history[-1]
                else:
                    self.trend = weighted_trend
                
                # Store trend history
                if not hasattr(self, 'trend_history'):
                    self.trend_history = []
                if len(self.trend_history) > self.max_history_size:
                    self.trend_history.pop(0)
                self.trend_history.append(self.trend)
                
                smoothed_trend = self.trend
                
                # Log trend calculation
                self.logger.debug(
                    f"Trend calculation - Points: {len(self.severity_history[-5:])}, "
                    f"Raw slope: {trend:.5f}, "
                    f"Scaled: {smoothed_trend:.3f}, "
                    f"R-squared: {r_squared:.3f}, "
                    f"Weighted: {weighted_trend:.3f}, "
                    f"Smoothed: {smoothed_trend:.3f}, "
                    f"Recent scores: {[f'{s:.3f}' for s in self.severity_history[-5:]]}"
                )
                
                # Store the smoothed trend instead of returning it
                smoothed_trend = self.trend
                
            except Exception as e:
                self.logger.warning(f"Error calculating trend: {e}")
                trend = 0.0
        
        # Check drift conditions
        drift_detected = False
        
        # Increment samples since drift
        self.samples_since_drift += 1
        
        # Check if minimum interval has passed
        interval_ok = self.samples_since_drift >= self.min_drift_interval
        
        # Handle case when p_value is an array
        if isinstance(p_value, np.ndarray):
            if p_value.size > 0:
                p_value = float(p_value[0])  # Take the first value
            else:
                p_value = 1.0  # Default to no statistical significance
        
        # Handle case when ks_stat is an array
        if isinstance(ks_stat, np.ndarray):
            if ks_stat.size > 0:
                ks_stat = float(ks_stat[0])  # Take the first value
            else:
                ks_stat = 0.0  # Default to no statistical significance
        
        # Handle case when severity is an array
        if isinstance(severity, np.ndarray):
            if severity.size > 0:
                severity = float(severity[0])  # Take the first value
            else:
                severity = 0.0  # Default to no severity
        
        # Handle case when mean_shift_normalized is an array
        if isinstance(mean_shift_normalized, np.ndarray):
            if mean_shift_normalized.size > 0:
                mean_shift_normalized = float(mean_shift_normalized[0])  # Take the first value
            else:
                mean_shift_normalized = 0.0  # Default to no mean shift
        
        # Handle case when trend is an array
        if isinstance(trend, np.ndarray):
            if trend.size > 0:
                trend = float(trend[0])  # Take the first value
            else:
                trend = 0.0
        
        # Multiple drift detection conditions for robustness
        # Condition 1: Significant mean shift AND low p-value
        condition1 = (mean_shift_normalized > self.drift_threshold and 
                     p_value < self.significance_level)
                     
        # Condition 2: Very strong statistical evidence
        condition2 = (p_value < 1e-5 and ks_stat > 0.3)
        
        # Condition 3: Persistent high severity
        condition3 = False
        if len(self.severity_history) >= 3:
            # Make sure all elements in severity_history are scalar values
            recent_severities = []
            for sev in self.severity_history[-3:]:
                if isinstance(sev, (np.ndarray, list)):
                    if len(sev) > 0:
                        recent_severities.append(float(sev[0]))  # Take the first element
                    else:
                        recent_severities.append(0.0)  # Default value if empty
                else:
                    recent_severities.append(float(sev))  # Convert to float
            
            avg_severity = sum(recent_severities) / len(recent_severities)
            condition3 = (severity > self.drift_threshold * 0.8 and 
                         avg_severity > self.drift_threshold * 0.7)
        
        # Condition 4: Strong trend upward in severity
        condition4 = (trend > 10.0 and severity > self.drift_threshold * 0.5)
        
        # Condition 5: Extremely high KS statistic
        condition5 = (ks_stat > 0.5 and p_value < 0.1)
        
        # Detect drift if any condition is met and minimum interval has passed
        if interval_ok and (condition1 or condition2 or condition3 or condition4 or condition5):
            drift_detected = True
            self.samples_since_drift = 0
            self.drift_detected = True
            self.last_drift = time.time()
            
            # Log detailed drift information
            self.logger.info(
                f"Drift detected - Mean shift: {mean_shift_normalized:.3f}, "
                f"KS stat: {ks_stat:.3f}, p-value: {p_value:.3e}, "
                f"Severity: {severity:.3f}, Trend: {trend:.3f}"
            )
            
            # Log which condition triggered the drift
            conditions = []
            if condition1: conditions.append("Mean shift + p-value")
            if condition2: conditions.append("Strong statistical evidence")
            if condition3: conditions.append("Persistent high severity")
            if condition4: conditions.append("Strong upward trend")
            if condition5: conditions.append("High KS statistic")
            self.logger.debug(f"Drift triggered by conditions: {', '.join(conditions)}")
            
        # Return results
        info = {
            'mean_shift': mean_shift_normalized,
            'ks_statistic': ks_stat,
            'p_value': p_value,
            'trend': trend,
            'severity': severity
        }
        
        return drift_detected, severity, info

    def get_state(self) -> Dict[str, Any]:
        """Get current state of drift detection"""
        return {
            'config': {
                'window_size': self.window_size,
                'drift_threshold': self.drift_threshold,
                'significance_level': self.significance_level,
                'min_drift_interval': self.min_drift_interval,
                'ema_alpha': self.ema_alpha,
                'confidence_threshold': self.confidence_threshold,
                'feature_names': self.feature_names,
                'feature_thresholds': self.feature_thresholds,
                'feature_significance': self.feature_significance
            },
            'state': {
                'samples_since_drift': self.samples_since_drift,
                'total_samples': len(self.current_window),
                'last_reference_update': self.last_reference_update,
                'last_drift_detected': self.last_drift_detected,
                'trend': self.trend
            },
            'windows': {
                'reference_shape': self.reference_window.shape if self.reference_window is not None else None,
                'original_reference_shape': self.original_reference.shape if self.original_reference is not None else None,
                'current_window_size': len(self.current_window)
            },
            'history': {
                'drift_scores': self.drift_scores[-100:],  # Keep last 100 points
                'mean_shifts': self.mean_shifts[-100:],
                'ks_stats': self.ks_stats[-100:],
                'p_values': self.p_values[-100:]
            },
            'statistics': {
                'drift_score_mean': np.mean(self.drift_scores) if self.drift_scores else 0.0,
                'drift_score_std': np.std(self.drift_scores) if self.drift_scores else 0.0,
                'mean_shift_mean': np.mean(self.mean_shifts) if self.mean_shifts else 0.0,
                'mean_shift_std': np.std(self.mean_shifts) if self.mean_shifts else 0.0,
                'ks_stat_mean': np.mean(self.ks_stats) if self.ks_stats else 0.0,
                'ks_stat_std': np.std(self.ks_stats) if self.ks_stats else 0.0
            }
        }
        
    def set_state(self, state: Dict[str, Any]) -> None:
        """Set detector state from dictionary"""
        try:
            # Validate state dictionary structure
            required_keys = ['config', 'state', 'windows', 'history', 'statistics']
            if not all(key in state for key in required_keys):
                raise ValueError(f"State dictionary missing required keys: {required_keys}")
            
            # Update configuration if provided
            config = state.get('config', {})
            for key, value in config.items():
                if hasattr(self, key):
                    setattr(self, key, value)
            
            # Update state variables
            state_vars = state.get('state', {})
            for key, value in state_vars.items():
                if hasattr(self, key):
                    setattr(self, key, value)
            
            # Update history (with bounds checking)
            history = state.get('history', {})
            max_history = 1000  # Prevent memory issues
            for key, value in history.items():
                if hasattr(self, key):
                    if isinstance(value, list) and len(value) > max_history:
                        value = value[-max_history:]  # Keep most recent
                    setattr(self, key, value)
            
            self.logger.info("Detector state successfully restored")
            
        except Exception as e:
            self.logger.error(f"Error restoring detector state: {str(e)}")
            raise ValueError(f"Failed to restore detector state: {str(e)}")
    
    def _empty_info(self) -> Dict[str, Any]:
        return {
            'mean_shift': 0.0,
            'ks_statistic': 0.0,
            'p_value': 1.0,
            'severity': 0.0,
            'trend': 0.0,
            'drifting_features': [],
            'confidence': 0.0,
            'confidence_warning': False
        }
    
    def process_point(self, point: float, feature_idx: Optional[int] = None) -> Tuple[bool, float]:
        """Process a new data point for drift detection.
        
        Args:
            point: New data point
            feature_idx: Optional feature index for multivariate data
            
        Returns:
            Tuple of (drift_detected, severity)
        """
        # Initialize tracking if needed
        if not hasattr(self, 'window'):
            self.window = []
            
        # Initialize all history arrays if not present
        for attr in ['drift_scores', 'scores', 'mean_shifts', 'ks_stats', 'p_values',
                     'mean_shift_history', 'ks_stat_history', 'p_value_history']:
            if not hasattr(self, attr):
                setattr(self, attr, [])
            
        self.last_drift = 0 if not hasattr(self, 'last_drift') else self.last_drift
            
        # Convert to float if needed
        point = float(point)
        
        # Add point to window
        self.window.append(point)
        
        # Ensure window doesn't exceed window_size + a small buffer
        max_window = self.window_size + 10
        if len(self.window) > max_window:
            self.window = self.window[-max_window:]
        
        # Check if we have enough data
        if len(self.window) < self.window_size:
            return False, 0.0
        
        # Split window for comparison
        split_idx = max(0, len(self.window) - self.window_size)
        ref_window = self.window[:split_idx]
        cur_window = self.window[split_idx:]
        
        # Calculate statistics
        if len(ref_window) >= self.window_size:
            ref_mean = float(np.mean(ref_window))
            ref_std = float(np.std(ref_window))
            cur_mean = float(np.mean(cur_window))
            cur_std = float(np.std(cur_window))
            
            # Calculate normalized mean shift
            mean_shift = abs(cur_mean - ref_mean)
            normalized_shift = mean_shift / (ref_std + cur_std + 1e-6)
            
            # Run KS test
            ks_stat, p_value = stats.ks_2samp(ref_window, cur_window)
            
            # Store statistics
            self.mean_shifts.append(normalized_shift)
            self.ks_stats.append(ks_stat)
            self.p_values.append(p_value)
            self.mean_shift_history.append(normalized_shift)
            self.ks_stat_history.append(ks_stat)
            self.p_value_history.append(p_value)
        else:
            # Not enough reference data yet
            normalized_shift = 0.0
            ks_stat = 0.0
            p_value = 1.0
        
        # Calculate severity
        severity = self.calculate_severity(normalized_shift, ks_stat)
        
        # Apply EMA smoothing for stability
        if len(self.scores) > 0:
            alpha = self.ema_alpha
            ema_score = alpha * severity + (1 - alpha) * self.scores[-1]
            self.ema_score = ema_score
        else:
            self.ema_score = severity
            
        # Store severity score
        self.scores.append(self.ema_score)
        
        # Check interval since last drift
        interval_ok = len(self.scores) - self.last_drift >= self.min_drift_interval
        
        # Get appropriate thresholds for detection
        drift_threshold = self.drift_threshold
        significance_level = self.significance_level
        
        # Primary drift condition - check both mean shift and significance
        # Note: We compare raw mean_shift with threshold since threshold is designed for raw values
        drift_by_mean = normalized_shift > drift_threshold
        significant = p_value < significance_level
        
        # Check for gradual drift (trend-based detection)
        if hasattr(self, 'drift_scores') and len(self.drift_scores) >= 5:
            score_trend = self.calculate_trend()
            trend_threshold = drift_threshold / 2
            drift_by_trend = (score_trend > 5.0 and normalized_shift > trend_threshold)
            self.logger.debug(f"Trend detection - Score trend: {score_trend:.3f}, Threshold: {trend_threshold:.3f}")
        else:
            score_trend = 0.0
            drift_by_trend = False
        
        # Secondary condition - extreme statistical significance
        extreme_significance = p_value < 1e-10 and ks_stat > 0.5
        
        self.logger.debug(
            f"Drift analysis - Shift: {normalized_shift:.3f}, "
            f"KS: {ks_stat:.3f}, P-value: {p_value:.3e}, "
            f"Threshold: {drift_threshold:.3f}, "
            f"Detected: {drift_by_mean or drift_by_trend}"
        )
        
        # Detect drift
        drift_detected = ((drift_by_mean or drift_by_trend) and significant) or extreme_significance
        
        # Update last drift if detected
        if drift_detected:
            self.last_drift = len(self.scores) - 1
            self.logger.info(
                f"Drift detected - Shift: {normalized_shift:.3f}, "
                f"KS: {ks_stat:.3f}, P-value: {p_value:.3e}"
            )
            
        # Limit history size
        if len(self.scores) > self.max_history_size:
            if self.scores:
                self.scores.pop(0)
            if self.mean_shifts:
                self.mean_shifts.pop(0)
            if self.ks_stats:
                self.ks_stats.pop(0)
            if self.p_values:
                self.p_values.pop(0)
            if self.mean_shift_history:
                self.mean_shift_history.pop(0)
            if self.ks_stat_history:
                self.ks_stat_history.pop(0)
            if self.p_value_history:
                self.p_value_history.pop(0)
        
        self.logger.debug(
            f"Severity scores - Raw: {severity:.3f}, EMA: {self.ema_score:.3f}, "
            f"Total scores: {len(self.scores)}"
        )
        
        return drift_detected, float(self.ema_score)
    
    def _calculate_severity(self, mean_shift: float, ks_stat: float, trend_change: float = 0.0) -> float:
        """Calculate drift severity score.
        
        Args:
            mean_shift: Normalized mean shift between windows
            ks_stat: KS statistic value
            trend_change: Change in trend between windows
            
        Returns:
            Combined severity score
        """
        # Squash mean shift and trend to prevent domination
        mean_component = np.tanh(mean_shift / 2)
        trend_component = np.tanh(trend_change)
        
        # Combine mean shift, trend, and KS statistic
        severity = 0.4 * mean_component + 0.2 * trend_component + 0.4 * ks_stat
        
        return float(severity)
        
    def reset(self) -> None:
        """Reset detector state."""
        self.logger.info("Resetting detector state")
        if hasattr(self, 'window'):
            self.window = []
        if hasattr(self, 'scores'):
            self.scores = []
        self.last_drift = 0
        self.ema_score = None

    def _process_feature(self, 
                         curr_feature: np.ndarray, 
                         ref_feature: np.ndarray, 
                         orig_feature: np.ndarray, 
                         feature_idx: int = 0) -> Tuple[bool, float, Dict[str, Any]]:
        """Process a single feature for drift detection.
        
        Args:
            curr_feature: Current window data for this feature
            ref_feature: Reference window data for this feature
            orig_feature: Original reference window data for this feature
            feature_idx: Index of the feature
            
        Returns:
            Tuple of (drift_detected, severity, info_dict)
        """
        # Get feature name if available
        feature_name = self.feature_names[feature_idx] if self.feature_names and feature_idx < len(self.feature_names) else f"feature_{feature_idx}"
        
        # Get feature-specific threshold and significance level
        drift_threshold = self.feature_thresholds.get(feature_name, self.drift_threshold)
        significance_level = self.feature_significance.get(feature_name, self.significance_level)
        
        self.logger.debug(f"Processing feature {feature_name} with threshold={drift_threshold:.3f}, significance={significance_level:.3e}")
        
        # Calculate mean shifts
        curr_mean = float(np.mean(curr_feature))
        ref_mean = float(np.mean(ref_feature))
        orig_mean = float(np.mean(orig_feature))
        mean_shift = float(abs(curr_mean - ref_mean))
        orig_shift = float(abs(curr_mean - orig_mean))
        
        # Use maximum mean shift
        feature_shift = max(mean_shift, orig_shift)
        
        # Calculate KS statistics for both windows
        ks_stat, p_value = stats.ks_2samp(curr_feature, ref_feature)
        orig_ks, orig_p = stats.ks_2samp(curr_feature, orig_feature)
        
        # Ensure values are scalar
        try:
            if hasattr(ks_stat, '__len__') and len(ks_stat) > 0:
                ks_stat = float(ks_stat[0])
            else:
                ks_stat = float(ks_stat)
                
            if hasattr(p_value, '__len__') and len(p_value) > 0:
                p_value = float(p_value[0])
            else:
                p_value = float(p_value)
                
            if hasattr(orig_ks, '__len__') and len(orig_ks) > 0:
                orig_ks = float(orig_ks[0])
            else:
                orig_ks = float(orig_ks)
                
            if hasattr(orig_p, '__len__') and len(orig_p) > 0:
                orig_p = float(orig_p[0])
            else:
                orig_p = float(orig_p)
        except (TypeError, ValueError, IndexError):
            self.logger.warning(f"Error converting KS statistics to float for feature {feature_name}")
            ks_stat = 0.0
            p_value = 1.0
            orig_ks = 0.0
            orig_p = 1.0
        
        # Use more significant result
        if orig_p < p_value:
            ks_stat = float(orig_ks)
            p_value = float(orig_p)
        else:
            ks_stat = float(ks_stat)
            p_value = float(p_value)
        
        # Calculate feature severity
        feature_severity = self.calculate_severity(feature_shift, ks_stat)
        
        # Check for drift
        drift_by_mean = feature_shift > drift_threshold
        significant = p_value < significance_level
        
        # Secondary condition - extreme statistical significance
        extreme_significance = p_value < 1e-10 and ks_stat > 0.5
        
        self.logger.debug(
            f"Feature {feature_name} stats - Mean shift: {feature_shift:.3f}, "
            f"KS stat: {ks_stat:.3f}, p-value: {p_value:.3e}, "
            f"Drift by mean: {drift_by_mean}, Significant: {significant}"
        )
        
        # Store feature info
        feature_info = {
            'mean_shift': feature_shift,
            'ks_statistic': ks_stat,
            'p_value': p_value,
            'threshold': drift_threshold,
            'significance': significance_level
        }
        
        # Check if this feature is drifting
        drift_detected = (drift_by_mean and significant) or extreme_significance
        
        # Prepare info dictionary
        info = {
            'drifting_features': [feature_name] if drift_detected else [],
            'feature_info': {feature_name: feature_info},
            'mean_shift': feature_shift,
            'ks_statistic': ks_stat,
            'p_value': p_value,
            'trend': self.trend,
            'samples_since_drift': self.samples_since_drift,
            'confidence': 1.0 - p_value if p_value < 0.5 else 0.5,  # Convert p-value to confidence
            'confidence_warning': p_value > 0.1  # Warn if p-value is high
        }
        
        return drift_detected, feature_severity, info
