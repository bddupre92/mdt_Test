"""
detector.py
----------
Drift detection implementation
"""

from typing import Tuple, Dict, Any, Optional, List, Union
import numpy as np
from scipy import stats
import logging

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
    
    def calculate_severity(self, mean_shift: float, ks_stat: float, sensitivity: float = 1.0) -> float:
        """Calculate drift severity score.
        
        Args:
            mean_shift: Mean shift (normalized)
            ks_stat: KS statistic
            sensitivity: Sensitivity multiplier
            
        Returns:
            Severity score (0.0-1.0)
        """
        # Squash mean shift using tanh to prevent domination
        squashed_mean_shift = np.tanh(mean_shift / 2)
        
        # Combine mean shift and KS statistic
        mean_shift_weight = 0.6
        ks_weight = 0.4
        raw_severity = (mean_shift_weight * squashed_mean_shift + ks_weight * ks_stat) * sensitivity
        
        # Apply EMA smoothing if we have history
        if hasattr(self, 'severity_history') and len(self.severity_history) > 0:
            alpha = getattr(self, 'ema_alpha', 0.3)
            smoothed_severity = alpha * raw_severity + (1 - alpha) * self.severity_history[-1]
        else:
            smoothed_severity = raw_severity
            
        # Log calculation details
        self.logger.debug(
            f"Severity calculation - Mean shift: {mean_shift:.3f}, Squashed: {squashed_mean_shift:.3f}, "
            f"KS stat: {ks_stat:.3f}, Raw: {raw_severity:.3f}, Smoothed: {smoothed_severity:.3f}"
        )
            
        return float(smoothed_severity)
    
    def calculate_trend(self) -> float:
        """Calculate trend in recent drift scores.
        
        Returns:
            Trend value (positive indicates increasing drift)
        """
        # Initialize drift_scores if not present
        if not hasattr(self, 'drift_scores'):
            self.drift_scores = []
            
        # Check if we have enough scores for trend calculation
        if len(self.drift_scores) < 5:
            self.logger.debug("Not enough drift scores for trend calculation (need at least 5)")
            return 0.0
            
        # Use last 10 points or fewer if not available
        n_points = min(10, len(self.drift_scores))
        recent_scores = self.drift_scores[-n_points:]
        
        # Create x indices (0, 1, 2, ...)
        x = np.arange(len(recent_scores))
        
        # Calculate linear regression slope (1000x scale for better visibility)
        if len(recent_scores) >= 2:
            try:
                slope, _, _, _, _ = stats.linregress(x, recent_scores)
                scaled_trend = slope * 1000.0
            except Exception as e:
                self.logger.warning(f"Error calculating trend: {e}")
                return 0.0
                
            # Apply EMA smoothing if we have history
            if hasattr(self, 'trend_history') and len(self.trend_history) > 0:
                alpha = getattr(self, 'ema_alpha', 0.3)
                smoothed_trend = alpha * scaled_trend + (1 - alpha) * self.trend_history[-1]
                
                # Log trend calculation
                self.logger.debug(
                    f"Trend calculation - Points: {n_points}, "
                    f"Raw slope: {slope:.5f}, "
                    f"Scaled: {scaled_trend:.3f}, "
                    f"Smoothed: {smoothed_trend:.3f}, "
                    f"Recent scores: {[f'{s:.3f}' for s in recent_scores]}"
                )
                
                # Store trend history
                if not hasattr(self, 'trend_history'):
                    self.trend_history = []
                self.trend_history.append(smoothed_trend)
                
                # Limit history size
                if len(self.trend_history) > self.max_history_size:
                    self.trend_history.pop(0)
                    
                return float(smoothed_trend)
            else:
                # Log trend calculation without smoothing
                self.logger.debug(
                    f"Trend calculation (no smoothing) - Points: {n_points}, "
                    f"Raw slope: {slope:.5f}, "
                    f"Scaled: {scaled_trend:.3f}, "
                    f"Recent scores: {[f'{s:.3f}' for s in recent_scores]}"
                )
                
                # Initialize trend history
                self.trend_history = [scaled_trend]
                return float(scaled_trend)
        else:
            self.logger.debug("Not enough points for regression (need at least 2)")
            return 0.0
        
    def detect_drift(self, curr_data=None, ref_data=None, features=None, prediction_proba=None, orig_data=None) -> Tuple[bool, float, Dict[str, Any]]:
        """Detect drift between current and reference windows.
        
        This method supports both old and new parameter naming for compatibility.
        
        Args:
            curr_data: Current window data (window_size samples) or None
            ref_data: Reference window data or None
            features: Optional feature vector for feature-level drift detection
            prediction_proba: Optional prediction probability vector
            orig_data: Original reference window data or None
            
        Returns:
            Tuple of (drift_detected, severity, info_dict)
        """
        # Convert current_window to numpy if needed
        if curr_data is None and hasattr(self, 'current_window') and len(self.current_window) >= self.window_size:
            curr_data = np.array(self.current_window[-self.window_size:]).reshape(-1, 1)
            
        # Use reference_window if ref_data not provided
        if ref_data is None and hasattr(self, 'reference_window'):
            ref_data = self.reference_window
            
        # Use original_reference if orig_data not provided
        if orig_data is None and hasattr(self, 'original_reference'):
            orig_data = self.original_reference
            
        # Check for required data
        if curr_data is None or ref_data is None:
            self.logger.warning("Missing data for drift detection")
            return False, 0.0, {'mean_shift': 0.0, 'ks_statistic': 0.0, 'p_value': 1.0, 'trend': 0.0, 'drifting_features': [], 'confidence': 0.0, 'confidence_warning': False}
            
        # Convert to numpy arrays if needed
        if isinstance(curr_data, list):
            curr_data = np.array(curr_data)
        if isinstance(ref_data, list):
            ref_data = np.array(ref_data)
        if features is not None and isinstance(features, list):
            features = np.array(features)
        if prediction_proba is not None and isinstance(prediction_proba, list):
            prediction_proba = np.array(prediction_proba)
            
        # Flatten arrays if needed
        if hasattr(curr_data, 'ndim') and curr_data.ndim > 1 and curr_data.shape[1] == 1:
            curr_data = curr_data.flatten()
        if hasattr(ref_data, 'ndim') and ref_data.ndim > 1 and ref_data.shape[1] == 1:
            ref_data = ref_data.flatten()
        if features is not None and hasattr(features, 'ndim') and features.ndim > 1 and features.shape[1] == 1:
            features = features.flatten()
        if prediction_proba is not None and hasattr(prediction_proba, 'ndim') and prediction_proba.ndim > 1 and prediction_proba.shape[1] == 1:
            prediction_proba = prediction_proba.flatten()
        
        # Initialize statistics arrays
        mean_shifts = []
        ks_stats = []
        p_values = []
        drifting_features = []
        severities = []
        
        # Process feature-level drift if features are provided
        if features is not None and hasattr(self, 'feature_names') and len(self.feature_names) > 0:
            feature_drifts = []
            
            for i, feature_name in enumerate(self.feature_names):
                if i < len(features):
                    # Get feature-specific threshold if available
                    feature_threshold = self.feature_thresholds.get(feature_name, self.drift_threshold)
                    feature_significance = self.feature_significance.get(feature_name, self.significance_level)
                    
                    # Check for drift in this feature
                    feature_value = features[i]
                    if isinstance(feature_value, (int, float)) and not (isinstance(feature_value, str) or np.isnan(feature_value)):
                        # Simple threshold check for numeric features
                        feature_drift = abs(feature_value) > feature_threshold
                        if feature_drift:
                            drifting_features.append(feature_name)
                            feature_drifts.append((feature_name, feature_value, feature_threshold))
            
            if feature_drifts:
                self.logger.debug(f"Feature-level drifts detected: {feature_drifts}")
        
        # Calculate statistics for each feature
        n_features = curr_data.shape[1] if len(curr_data.shape) > 1 else 1
        
        for i in range(n_features):
            # Get feature data
            if n_features == 1:
                curr_feature = curr_data.flatten()
                ref_feature = ref_data.flatten()
                orig_feature = orig_data.flatten()
            else:
                curr_feature = curr_data[:, i]
                ref_feature = ref_data[:, i]
                orig_feature = orig_data[:, i]
            
            # Calculate mean shifts
            curr_mean = float(np.mean(curr_feature))
            ref_mean = float(np.mean(ref_feature))
            orig_mean = float(np.mean(orig_feature))
            mean_shift = float(abs(curr_mean - ref_mean))
            orig_shift = float(abs(curr_mean - orig_mean))
            
            # Use maximum mean shift
            feature_shift = max(mean_shift, orig_shift)
            mean_shifts.append(feature_shift)
            
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
                self.logger.warning(f"Error converting KS statistics to float for feature {i}")
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
            
            ks_stats.append(ks_stat)
            p_values.append(p_value)
            
            # Calculate feature severity
            feature_severity = self.calculate_severity(feature_shift, ks_stat)
            severities.append(feature_severity)
            
            feature_name = self.feature_names[i] if i < len(self.feature_names) else f"feature_{i}"
            self.logger.debug(f"Feature {feature_name}:")
            self.logger.debug(f"  Current: mean={curr_mean:.3f}, std={float(np.std(curr_feature)):.3f}")
            self.logger.debug(f"  Reference: mean={ref_mean:.3f}, std={float(np.std(ref_feature)):.3f}")
            self.logger.debug(f"  Original: mean={orig_mean:.3f}, std={float(np.std(orig_feature)):.3f}")
            self.logger.debug(f"  Mean shift: current={mean_shift:.3f}, original={orig_shift:.3f}, max={feature_shift:.3f}")
            self.logger.debug(f"  KS stat: current={ks_stat:.3f}, original={orig_ks:.3f}")
            self.logger.debug(f"  p-value: current={p_value:.3e}, original={orig_p:.3e}")
            self.logger.debug(f"  Severity: {feature_severity:.3f}")
            
            # Get feature-specific threshold
            drift_threshold = self.feature_thresholds.get(feature_name, self.drift_threshold)
            feature_significance = self.feature_significance.get(feature_name, self.significance_level)
            
            # Adjust thresholds based on feature variability and history
            feature_std = float(np.std(curr_feature))
            base_threshold = drift_threshold * (1.0 + feature_std)
            base_significance = feature_significance * (1.0 + feature_std)
            
            # Make thresholds more sensitive if we haven't detected drift in a while
            sensitivity_factor = min(2.0, 1.0 + self.samples_since_drift / (2 * self.min_drift_interval))
            adjusted_threshold = base_threshold / sensitivity_factor
            adjusted_significance = base_significance * sensitivity_factor
            
            # Further adjust thresholds based on window size
            window_factor = min(1.0, len(self.current_window) / self.window_size)
            adjusted_threshold *= window_factor
            adjusted_significance /= window_factor
            
            self.logger.debug(f"  Thresholds:")
            self.logger.debug(f"    Base threshold: {drift_threshold:.3f}")
            self.logger.debug(f"    Variability adjusted: {base_threshold:.3f}")
            self.logger.debug(f"    Final threshold: {adjusted_threshold:.3f}")
            self.logger.debug(f"    Base significance: {feature_significance:.3e}")
            self.logger.debug(f"    Final significance: {adjusted_significance:.3e}")
            self.logger.debug(f"    Sensitivity factor: {sensitivity_factor:.3f}")
            self.logger.debug(f"    Window factor: {window_factor:.3f}")
            
            # Check if feature is drifting using multiple criteria
            drift_conditions = []
            
            # Primary condition: significant mean shift AND low p-value
            if feature_shift > adjusted_threshold and p_value < adjusted_significance:
                drift_conditions.append("primary")
            
            # Secondary condition: very strong statistical evidence
            if p_value < 1e-5 and ks_stat > 0.3:
                drift_conditions.append("statistical")
            
            # Tertiary condition: extreme mean shift
            if feature_shift > 1.5 * adjusted_threshold:
                drift_conditions.append("extreme")
            
            # Additional condition: consistent drift pattern
            if len(self.mean_shifts) >= 3:
                recent_shifts = self.mean_shifts[-3:]
                if all(shift > adjusted_threshold * 0.8 for shift in recent_shifts):
                    drift_conditions.append("consistent")
            
            # Additional condition: increasing trend
            if len(self.mean_shifts) >= 5:
                recent_shifts = self.mean_shifts[-5:]
                if all(recent_shifts[i] < recent_shifts[i+1] for i in range(len(recent_shifts)-1)):
                    drift_conditions.append("trend")
            
            if drift_conditions:
                drifting_features.append(feature_name)
                self.logger.debug(f"  {feature_name} is drifting ({', '.join(drift_conditions)})")
        
        # Compute statistics
        # Ensure arrays have compatible shapes for KS test
        if hasattr(curr_data, 'shape') and hasattr(ref_data, 'shape'):
            if len(curr_data.shape) != len(ref_data.shape):
                # Make sure both arrays have the same number of dimensions
                if len(curr_data.shape) > len(ref_data.shape):
                    curr_data = curr_data.flatten()
                else:
                    ref_data = ref_data.flatten()
                    
            # Ensure both arrays are 1D for KS test
            if len(curr_data.shape) > 1 or len(ref_data.shape) > 1:
                curr_data = curr_data.flatten()
                ref_data = ref_data.flatten()
                
            self.logger.debug(f"Array shapes for KS test: curr_data={curr_data.shape}, ref_data={ref_data.shape}")
            
        try:
            ks_stat, p_value = stats.ks_2samp(curr_data, ref_data)
        except ValueError as e:
            self.logger.warning(f"Error in KS test: {e}. Using default values.")
            ks_stat = 0.0
            p_value = 1.0
        
        mean_shift = abs(np.mean(curr_data) - np.mean(ref_data)) / (np.std(ref_data) if np.std(ref_data) > 0 else 1.0)
        
        # Log detailed statistics for debugging
        self.logger.debug(
            f"Drift statistics - Mean shift: {mean_shift:.3f}, "
            f"KS statistic: {ks_stat:.3f}, p-value: {p_value:.3e}, "
            f"Current mean: {np.mean(curr_data):.3f}, "
            f"Reference mean: {np.mean(ref_data):.3f}, "
            f"Current std: {np.std(curr_data):.3f}, "
            f"Reference std: {np.std(ref_data):.3f}"
        )
        
        # Calculate overall statistics using max mean shift and min p-value
        mean_shift = float(np.max(mean_shifts))
        ks_stat = float(np.max(ks_stats))
        p_value = float(np.min(p_values))
        severity = float(np.max(severities))
        
        self.logger.debug(f"Overall statistics:")
        self.logger.debug(f"  Max mean shift: {mean_shift:.3f}")
        self.logger.debug(f"  Max KS stat: {ks_stat:.3f}")
        self.logger.debug(f"  Min p-value: {p_value:.3e}")
        self.logger.debug(f"  Max severity: {severity:.3f}")
        
        # Update history
        if hasattr(self, 'drift_scores'):
            self.drift_scores.append(float(severity))
        if hasattr(self, 'mean_shifts'):
            self.mean_shifts.append(mean_shift)
        if hasattr(self, 'ks_stats'):
            self.ks_stats.append(ks_stat)
        if hasattr(self, 'p_values'):
            self.p_values.append(p_value)
        if hasattr(self, 'severity_history'):
            self.severity_history.append(severity)
        if hasattr(self, 'drifting_features_history'):
            self.drifting_features_history.append(drifting_features)
        
        # Limit history size
        if len(self.drift_scores) > self.max_history_size:
            self.drift_scores.pop(0)
            self.mean_shifts.pop(0)
            self.ks_stats.pop(0)
            self.p_values.pop(0)
            self.severity_history.pop(0)
            self.drifting_features_history.pop(0)
        
        # Calculate trend
        self.trend = self.calculate_trend()
        
        # Initialize detection result
        drift_detected = False
        
        # Get feature-specific threshold and significance level if applicable
        drift_threshold = self.drift_threshold
        significance_level = self.significance_level
        
        if drifting_features and self.feature_thresholds:
            for feature_name in drifting_features:
                if feature_name in self.feature_thresholds:
                    drift_threshold = self.feature_thresholds[feature_name]
                    self.logger.debug(f"Using feature-specific threshold for {feature_name}: {drift_threshold:.3f}")
                    
            for feature_name in drifting_features:
                if feature_name in self.feature_significance:
                    significance_level = self.feature_significance[feature_name]
                    self.logger.debug(f"Using feature-specific significance for {feature_name}: {significance_level:.3e}")
        
        # Primary drift condition - check both mean shift and significance
        drift_by_mean = mean_shift > drift_threshold
        significant = p_value < significance_level
        interval_ok = self.samples_since_drift >= self.min_drift_interval
        
        # Check for gradual drift (trend-based detection)
        if hasattr(self, 'drift_scores') and len(self.drift_scores) >= 5:
            score_trend = self.calculate_trend()
            trend_threshold = drift_threshold / 2
            drift_by_trend = (score_trend > 5.0 and mean_shift > trend_threshold)
            self.logger.debug(f"Trend detection - Score trend: {score_trend:.3f}, Threshold: {trend_threshold:.3f}")
        else:
            score_trend = 0.0
            drift_by_trend = False
        
        # Secondary condition - extreme statistical significance
        extreme_significance = p_value < 1e-10 and ks_stat > 0.5
        
        # Special case for seasonal patterns - detect based on statistical significance alone
        # if the detector is configured with higher thresholds (like the 'robust' detector)
        seasonal_drift = False
        if drift_threshold > 1.5 and significance_level < 0.01:
            seasonal_drift = p_value < 0.05 and ks_stat > 0.2 and interval_ok
            self.logger.debug(f"Seasonal drift check: {seasonal_drift}")
            
        # Special case for KS test - detect based on high KS statistic and low p-value
        ks_drift = p_value < 1e-5 and ks_stat > 0.3 and interval_ok
        self.logger.debug(f"KS drift check: {ks_drift}, KS stat: {ks_stat:.3f}, p-value: {p_value:.3e}")
        
        self.logger.debug(
            f"Drift conditions - Mean: {drift_by_mean}, "
            f"Trend: {drift_by_trend}, "
            f"Significant: {significant}, Interval OK: {interval_ok}, "
            f"Extreme significance: {extreme_significance}, "
            f"Seasonal: {seasonal_drift}, "
            f"KS drift: {ks_drift}"
        )
        
        # Check combined drift conditions
        if (((drift_by_mean or drift_by_trend) and significant) or 
            extreme_significance or seasonal_drift or ks_drift) and interval_ok:
            drift_detected = True
            self.last_reference_update = len(self.drift_scores) - 1 if hasattr(self, 'drift_scores') else 0
            
            if drifting_features:
                self.logger.info(
                    f"Drift detected in features {', '.join(drifting_features)} - "
                    f"Mean shift: {mean_shift:.3f}, "
                    f"KS stat: {ks_stat:.3f}, "
                    f"p-value: {p_value:.3e}"
                )
            elif drift_by_mean:
                self.logger.info(
                    f"Mean shift drift - Shift: {mean_shift:.3f}, "
                    f"p-value: {p_value:.3e}"
                )
            elif drift_by_trend:
                self.logger.info(
                    f"Trend drift - Trend: {score_trend:.3f}, "
                    f"Mean shift: {mean_shift:.3f}, "
                    f"p-value: {p_value:.3e}"
                )
            elif seasonal_drift:
                self.logger.info(
                    f"Seasonal drift - KS stat: {ks_stat:.3f}, "
                    f"p-value: {p_value:.3e}"
                )
            elif ks_drift:
                self.logger.info(
                    f"KS drift - KS stat: {ks_stat:.3f}, "
                    f"p-value: {p_value:.3e}"
                )
            else:
                self.logger.info(
                    f"Statistical drift - KS stat: {ks_stat:.3f}, "
                    f"p-value: {p_value:.3e}"
                )
        
        # Create info dictionary
        info = {
            'mean_shift': mean_shift,
            'ks_statistic': ks_stat,
            'p_value': p_value,
            'trend': self.trend,
            'drifting_features': drifting_features,
            'confidence': 1.0 - p_value if p_value < 0.5 else 0.5,  # Convert p-value to confidence
            'confidence_warning': p_value > 0.1  # Warn if p-value is high
        }
        
        # Store last info
        self.last_info = info
        self.last_severity = severity
        self.last_drift_detected = drift_detected
        
        # Return detection result
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
