"""
test_drift_detection.py
----------------------
Tests for drift detection mechanisms
"""

import numpy as np
import pandas as pd
from scipy import stats
import pytest

from drift_detection.detector import DriftDetector
from drift_detection.performance_monitor import DDM, EDDM
from drift_detection.statistical import chi2_drift_test, ks_drift_test

class TestDriftDetection:
    @classmethod
    def setup_class(cls):
        """Setup test data with different drift patterns"""
        np.random.seed(42)
        cls.n_samples = 1000
        cls.drift_point = 500
        
        # Generate datasets
        cls.datasets = {
            'sudden': cls._generate_sudden_drift(),
            'gradual': cls._generate_gradual_drift(),
            'seasonal': cls._generate_seasonal_drift()
        }
        
    @classmethod
    def _generate_sudden_drift(cls):
        """Generate data with sudden drift"""
        pre_drift = np.random.normal(0, 1, cls.drift_point)
        post_drift = np.random.normal(2, 1, cls.n_samples - cls.drift_point)
        labels = np.zeros(cls.n_samples)
        return np.concatenate([pre_drift, post_drift]), labels
        
    @classmethod
    def _generate_gradual_drift(cls):
        """Generate data with gradual drift"""
        x = np.linspace(0, 10, cls.n_samples)
        noise = np.random.normal(0, 0.5, cls.n_samples)
        signal = np.where(x < 5, np.sin(x), np.sin(x) + x-5)
        labels = np.zeros(cls.n_samples)
        return signal + noise, labels
        
    @classmethod
    def _generate_seasonal_drift(cls):
        """Generate data with seasonal drift"""
        x = np.linspace(0, 4*np.pi, cls.n_samples)
        amplitude = np.where(x < 2*np.pi, 1, 2)
        signal = amplitude * np.sin(x)
        noise = np.random.normal(0, 0.3, cls.n_samples)
        labels = np.zeros(cls.n_samples)
        return signal + noise, labels
        
    def test_ks_drift_test(self):
        """Test KS drift test"""
        # Generate data from different distributions
        n_points = 100
        data1 = np.random.normal(0, 1, n_points)
        data2 = np.random.normal(2, 1, n_points)  # Different mean
        
        detector = DriftDetector(window_size=50)
        detector.set_reference_window(data1)
        
        for value in data2:
            detector.add_sample(value)
            
        drift_detected, severity = detector.detect_drift()
        
        # Print debug info
        print(f"\nKS Test Debug Info:")
        print(f"Reference Mean: {np.mean(data1):.3f}, Std: {np.std(data1):.3f}")
        print(f"Current Mean: {np.mean(data2):.3f}, Std: {np.std(data2):.3f}")
        print(f"Severity: {severity:.3f}")
        print(f"Drift Threshold: {detector.drift_threshold:.3f}")
        
        ks_stat, p_value = stats.ks_2samp(data1, data2)
        print(f"KS Statistic: {ks_stat:.3f}, p-value: {p_value:.3e}")
        
        assert drift_detected, "Failed to detect drift with KS test"
        assert severity > 0.0, "Expected non-zero severity"
        
    def test_chi2_drift_test(self):
        """Test chi-square drift test"""
        # Generate categorical data
        n_points = 100
        categories = ['A', 'B', 'C']
        data1 = np.random.choice(categories, n_points, p=[0.6, 0.3, 0.1])
        data2 = np.random.choice(categories, n_points, p=[0.2, 0.3, 0.5])  # Different distribution
        
        # Convert to pandas Series
        df1 = pd.Series(data1)
        df2 = pd.Series(data2)
        
        # Test chi-square drift test
        drift_detected = chi2_drift_test(df1, df2, 0)
        assert drift_detected, "Failed to detect drift with chi-square test"
        
    def test_ddm_different_drifts(self):
        """Test DDM with different types of drifts"""
        ddm = DDM()
        
        # Generate data with sudden drift
        n_points = 200
        errors = np.zeros(n_points)
        errors[100:] = 1  # Sudden drift at point 100
        
        drift_points = []
        for i, error in enumerate(errors):
            ddm.update(error == 0)  # Convert to correct/incorrect
            if ddm.detected_warning_zone() or ddm.detected_drift():
                drift_points.append(i)
                
        assert len(drift_points) > 0, "Failed to detect sudden drift"
        
    def test_eddm_concept_evolution(self):
        """Test EDDM with concept evolution"""
        eddm = EDDM()
        
        # Generate data with gradual drift
        n_points = 300
        errors = np.zeros(n_points)
        for i in range(100, 200):  # Gradual drift from 100 to 200
            errors[i] = i / 200.0
        errors[200:] = 1  # Complete drift after 200
        
        drift_points = []
        for i, error in enumerate(errors):
            eddm.update(error == 0)
            if eddm.detected_warning_zone() or eddm.detected_drift():
                drift_points.append(i)
                
        assert len(drift_points) > 0, "Failed to detect gradual drift"
        
    def test_drift_detection_ensemble(self):
        """Test ensemble of drift detectors"""
        detector = DriftDetector(window_size=50)
        ddm = DDM()
        eddm = EDDM()
        
        # Generate data with multiple types of drift
        n_points = 200
        data = []
        errors = np.zeros(n_points)
        
        # Add sudden drift
        data.extend(np.random.normal(0, 1, 100))
        data.extend(np.random.normal(3, 1, 100))
        errors[100:] = 1
        
        # Initialize detector
        detector.set_reference_window(data[:50])
        
        drift_points = []
        for i in range(50, n_points):
            # Update all detectors
            detector.add_sample(data[i])
            ddm.update(errors[i] == 0)
            eddm.update(errors[i] == 0)
            
            # Check for drift
            drift1, _ = detector.detect_drift()
            drift2 = ddm.detected_drift()
            drift3 = eddm.detected_drift()
            
            if drift1 or drift2 or drift3:
                drift_points.append(i)
                
        assert len(drift_points) > 0, "Failed to detect drift with ensemble"
        
    def test_drift_recovery(self):
        """Test detection of multiple concept drifts"""
        detector = DriftDetector(window_size=50)

        # Generate data with multiple drifts
        n_points = 200  # Reduced points per concept
        data = []
        concepts = [
            (0, 1),   # Initial distribution
            (3, 1),   # First drift - increased mean difference
            (0, 1),   # Return to initial
            (4, 1)    # Second drift - further increased mean
        ]

        for mean, std in concepts:
            data.extend(np.random.normal(mean, std, n_points))

        # Initialize reference window
        detector.set_reference_window(data[:50])

        drift_points = []
        for i in range(50, len(data)):
            detector.add_sample(data[i])
            drift_detected, severity = detector.detect_drift()

            # Debug info every 50 points
            if i % 50 == 0:
                print(f"\nPoint {i}:")
                print(f"Current Mean: {np.mean(data[i-50:i]):.3f}")
                print(f"Reference Mean: {np.mean(detector.reference_window):.3f}")
                print(f"Severity: {severity:.3f}")
                print(f"Drift Threshold: {detector.drift_threshold:.3f}")
                ks_stat, p_value = stats.ks_2samp(detector.reference_window, data[i-50:i])
                print(f"KS Statistic: {ks_stat:.3f}, p-value: {p_value:.3e}")

            if drift_detected:
                print(f"\nDrift detected at point {i}")
                print(f"Current Mean: {np.mean(data[i-50:i]):.3f}")
                print(f"Reference Mean: {np.mean(detector.reference_window):.3f}")
                drift_points.append(i)
                detector.set_reference_window(data[i-50:i])

        assert len(drift_points) == 3, "Failed to detect multiple concept drifts"
        
def test_trend_drift_detection():
    """Test trend detection in drift patterns"""
    detector = DriftDetector(window_size=50)
    
    # Generate trending data
    x = np.linspace(0, 10, 200)
    trend = 0.5 * x
    noise = np.random.normal(0, 0.5, 200)  # Reduced noise
    data = trend + noise
    
    # Initialize reference
    detector.set_reference_window(data[:50])
    
    # Track drift over time
    trends = []
    for i in range(50, len(data)):
        detector.add_sample(data[i])
        _, severity = detector.detect_drift()
        trend = detector.get_trend()
        trends.append(trend)
        
    # Should detect positive trend
    assert np.mean(trends) > 0, "Failed to detect positive trend"
    
def test_drift_severity_calculation():
    """Test drift severity calculation"""
    detector = DriftDetector(window_size=50)
    
    # Generate data with known drift
    ref_data = np.random.normal(0, 1, 50)
    test_data = np.random.normal(0.2, 1, 50)  # Very mild drift
    
    detector.set_reference_window(ref_data)
    for value in test_data:
        detector.add_sample(value)
        
    _, severity = detector.detect_drift()
    assert severity < 0.5, "Expected minimal drift"
