"""
Drift detection implementation for migraine prediction.
"""
from typing import Dict, List, Optional, Tuple
import numpy as np
from scipy import stats
from dataclasses import dataclass
import pandas as pd

@dataclass
class DriftResult:
    detected: bool
    p_value: float
    feature: Optional[str] = None
    drift_type: Optional[str] = None
    severity: Optional[float] = None

class DriftDetector:
    def __init__(self, 
                window_size: int = 30,
                significance_level: float = 0.05):
        """
        Initialize drift detector.
        
        Args:
            window_size: Size of the sliding window for drift detection
            significance_level: P-value threshold for drift detection
        """
        self.window_size = window_size
        self.significance_level = significance_level
        self.reference_data: Optional[pd.DataFrame] = None
        
    def initialize_reference(self, data: pd.DataFrame):
        """Initialize reference window with initial data."""
        self.reference_data = data.copy()
        
    def detect_drift(self, new_data: pd.DataFrame) -> List[DriftResult]:
        """
        Detect drift in new data compared to reference window.
        
        Args:
            new_data: New data to check for drift
            
        Returns:
            List of DriftResult objects for each feature
        """
        if self.reference_data is None:
            self.initialize_reference(new_data)
            return []
            
        results = []
        
        # Check each feature for drift
        for column in new_data.columns:
            if column in ['patient_id', 'date', 'migraine_occurred']:
                continue
                
            # Get clean data
            ref_values = self.reference_data[column].dropna()
            new_values = new_data[column].dropna()
            
            if len(ref_values) < 2 or len(new_values) < 2:
                continue
            
            # Perform statistical tests
            ks_stat, p_value = stats.ks_2samp(ref_values, new_values)
            
            # Check for distribution drift
            if p_value < self.significance_level:
                severity = self._calculate_drift_severity(
                    ref_values, new_values
                )
                
                results.append(DriftResult(
                    detected=True,
                    p_value=p_value,
                    feature=column,
                    drift_type='distribution',
                    severity=severity
                ))
                
            # Check for trend drift
            trend_detected, trend_p_value = self._detect_trend_drift(
                new_values
            )
            
            if trend_detected:
                results.append(DriftResult(
                    detected=True,
                    p_value=trend_p_value,
                    feature=column,
                    drift_type='trend',
                    severity=None
                ))
                
        return results
    
    def update_reference(self, new_data: pd.DataFrame):
        """Update reference window with new data."""
        if self.reference_data is None:
            self.reference_data = new_data.copy()
        else:
            # Concatenate and keep most recent window_size samples
            self.reference_data = pd.concat(
                [self.reference_data, new_data]
            ).tail(self.window_size)
    
    def _calculate_drift_severity(self,
                                ref_values: np.ndarray,
                                new_values: np.ndarray) -> float:
        """
        Calculate severity of distribution drift.
        
        Returns:
            Severity score between 0 and 1
        """
        # Calculate normalized difference in mean and variance
        mean_diff = abs(np.mean(ref_values) - np.mean(new_values))
        std_diff = abs(np.std(ref_values) - np.std(new_values))
        
        # Normalize by reference statistics
        mean_severity = mean_diff / (abs(np.mean(ref_values)) + 1e-10)
        std_severity = std_diff / (np.std(ref_values) + 1e-10)
        
        return float(np.clip((mean_severity + std_severity) / 2, 0, 1))
    
    def _detect_trend_drift(self,
                          values: np.ndarray) -> Tuple[bool, float]:
        """
        Detect if there's a significant trend in the data.
        
        Returns:
            Tuple of (trend_detected, p_value)
        """
        # Use Mann-Kendall test for trend detection
        trend, p_value = stats.kendalltau(
            np.arange(len(values)), values
        )
        
        return bool(p_value < self.significance_level), float(p_value)