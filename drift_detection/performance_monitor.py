"""
performance_monitor.py
----------------------
Implements DDM/EDDM-like drift detection based on model error rate.
Reference: 'Learning in Non-Stationary Environments: DDM and EDDM'
"""

import numpy as np
from scipy import stats

class DDM:
    """Drift Detection Method"""
    def __init__(self, min_samples=30, warning_threshold=2.0, drift_threshold=3.0):
        self.min_samples = min_samples
        self.warning_threshold = warning_threshold
        self.drift_threshold = drift_threshold
        self.reset()
        
    def reset(self):
        """Reset the detector"""
        self.n_samples = 0
        self.sum_error = 0
        self.sum_error_squared = 0
        self.error_rate = float('inf')
        self.std_dev = float('inf')
        self.min_error = float('inf')
        self.min_std = float('inf')
        self.min_error_count = 0
        self.warning_zone = False
        self.drift_detected = False
        
    def update(self, correct):
        """
        Update the detector with a new sample
        Args:
            correct: bool, True if prediction was correct
        Returns:
            bool: True if drift detected
        """
        self.n_samples += 1
        error = 0 if correct else 1
        
        # Update statistics
        self.sum_error += error
        self.sum_error_squared += error * error
        
        if self.n_samples < self.min_samples:
            return False
            
        # Calculate mean and standard deviation
        self.error_rate = self.sum_error / self.n_samples
        variance = (self.sum_error_squared / self.n_samples) - (self.error_rate * self.error_rate)
        self.std_dev = np.sqrt(variance) if variance > 0 else 0
        
        # Update minimum statistics
        if self.error_rate + self.std_dev < self.min_error + self.min_std:
            self.min_error = self.error_rate
            self.min_std = self.std_dev
            self.min_error_count = self.n_samples
            self.warning_zone = False
            self.drift_detected = False
            
        # Check for warning zone
        warning_level = self.min_error + self.warning_threshold * self.min_std
        if self.error_rate + self.std_dev > warning_level:
            self.warning_zone = True
            
        # Check for drift
        drift_level = self.min_error + self.drift_threshold * self.min_std
        if self.error_rate + self.std_dev > drift_level:
            self.drift_detected = True
            return True
            
        return False
        
    def detected_warning_zone(self):
        """Check if in warning zone"""
        return self.warning_zone
        
    def detected_drift(self):
        """Check if drift detected"""
        return self.drift_detected

class EDDM:
    """Early Drift Detection Method"""
    def __init__(self, min_samples=30, warning_threshold=0.95, drift_threshold=0.90):
        self.min_samples = min_samples
        self.warning_threshold = warning_threshold
        self.drift_threshold = drift_threshold
        self.reset()
        
    def reset(self):
        """Reset the detector"""
        self.n_samples = 0
        self.n_errors = 0
        self.distances = []
        self.last_error_index = None
        self.max_distance_avg = 0
        self.max_distance_std = 0
        self.warning_zone = False
        self.drift_detected = False
        self.min_distance = float('inf')
        self.consecutive_errors = 0
        
    def update(self, correct):
        """
        Update the detector with a new sample
        Args:
            correct: bool, True if prediction was correct
        Returns:
            bool: True if drift detected
        """
        self.n_samples += 1
        
        if not correct:
            self.n_errors += 1
            self.consecutive_errors += 1
            
            if self.last_error_index is not None:
                distance = self.n_samples - self.last_error_index
                self.distances.append(distance)
                if distance < self.min_distance:
                    self.min_distance = distance
            self.last_error_index = self.n_samples
        else:
            self.consecutive_errors = 0
            
        if len(self.distances) < self.min_samples:
            return False
            
        # Calculate statistics over recent errors
        window = self.distances[-self.min_samples:]
        distance_avg = np.mean(window)
        distance_std = np.std(window)
        
        # Update maximum values if performance improves
        if distance_avg + 2 * distance_std > self.max_distance_avg + 2 * self.max_distance_std:
            self.max_distance_avg = distance_avg
            self.max_distance_std = distance_std
            self.warning_zone = False
            self.drift_detected = False
            
        # Calculate ratio to maximum
        if self.max_distance_avg == 0:
            ratio = 1.0
        else:
            ratio = (distance_avg + 2 * distance_std) / (self.max_distance_avg + 2 * self.max_distance_std)
            
        # Adjust thresholds based on error patterns
        warning_threshold = self.warning_threshold
        drift_threshold = self.drift_threshold
        
        if self.consecutive_errors > 3:
            # More sensitive thresholds if errors are consecutive
            warning_threshold *= 1.05
            drift_threshold *= 1.05
            
        if distance_avg < self.min_distance * 2:
            # More sensitive if errors are very close
            warning_threshold *= 1.1
            drift_threshold *= 1.1
            
        # Check for warning zone
        if ratio < warning_threshold:
            self.warning_zone = True
            
        # Check for drift
        drift_detected = (
            ratio < drift_threshold and
            (self.consecutive_errors > 2 or  # Multiple consecutive errors
             len(self.distances) >= self.min_samples * 2)  # Enough samples
        )
        
        if drift_detected:
            self.drift_detected = True
            return True
            
        return False
        
    def detected_warning_zone(self):
        """Check if in warning zone"""
        return self.warning_zone
        
    def detected_drift(self):
        """Check if drift detected"""
        return self.drift_detected
