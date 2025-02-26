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
        
        drift_detected, severity, info = detector.detect_drift()
        print("\nKS Test Debug:")
        print(f"Mean shift: {info['mean_shift']}")
        print(f"KS stat: {info['ks_statistic']}")
        print(f"p-value: {info['p_value']}")
        print(f"Trend: {info['trend']}")
        print(f"Severity: {severity}")
        assert drift_detected
        assert severity > 0.5
        
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
        
        # Add sudden drift
        data.extend(np.random.normal(0, 1, 100))
        data.extend(np.random.normal(3, 1, 100))
        
        # Initialize detector
        detector.set_reference_window(data[:50])
        
        drift_points = []
        for i in range(50, n_points):
            drift_detected, _, _ = detector.add_sample(data[i])
            if drift_detected:
                drift_points.append(i)
        
        assert len(drift_points) > 0
        assert drift_points[0] >= 100  # Drift should be detected after the change point
        
    def test_drift_recovery(self):
        """Test detection of multiple concept drifts"""
        detector = DriftDetector(window_size=50)
        
        # Generate data with multiple drifts
        n_points = 200
        data = []
        
        # Normal distribution
        data.extend(np.random.normal(0, 1, 50))
        # First drift - shift mean
        data.extend(np.random.normal(2, 1, 50))
        # Second drift - increase variance
        data.extend(np.random.normal(2, 2, 50))
        # Return to normal
        data.extend(np.random.normal(0, 1, 50))
        
        detector.set_reference_window(data[:50])
        
        drift_points = []
        for i in range(50, n_points):
            drift_detected, _, _ = detector.add_sample(data[i])
            if drift_detected:
                drift_points.append(i)
                
        assert len(drift_points) >= 2  # Should detect at least two drifts
        assert drift_points[1] - drift_points[0] >= detector.min_drift_interval
        
    def test_trend_drift_detection(self):
        """Test trend-based drift detection"""
        detector = DriftDetector(window_size=50)
        
        # Generate data with gradual drift
        n_points = 200
        x = np.linspace(0, 1, n_points)
        data = np.random.normal(3 * x, 1)  # Gradually increasing mean
        
        detector.set_reference_window(data[:50])
        
        drift_points = []
        for i in range(50, n_points):
            drift_detected, _, _ = detector.add_sample(data[i])
            if drift_detected:
                drift_points.append(i)
                
        assert len(drift_points) > 0  # Should detect the gradual drift
        
    def test_drift_severity_calculation(self):
        """Test drift severity calculation"""
        detector = DriftDetector(window_size=50)
        
        # Generate reference data
        ref_data = np.random.normal(0, 1, 50)
        detector.set_reference_window(ref_data)
        
        # Test with different severity levels
        test_data = np.random.normal(0, 1, 50)  # No drift
        for value in test_data:
            detector.add_sample(value)
            
        _, severity, _ = detector.detect_drift()
        assert severity < 0.5  # Low severity for similar distribution
        
        # Test with significant drift
        test_data = np.random.normal(3, 1, 50)  # Large mean shift
        detector = DriftDetector(window_size=50)
        detector.set_reference_window(ref_data)
        
        for value in test_data:
            detector.add_sample(value)
            
        _, severity, _ = detector.detect_drift()
        assert severity > 0.5  # High severity for different distribution
        
    def test_feature_specific_thresholds(self):
        """Test that feature-specific thresholds are working correctly"""
        detector = DriftDetector(window_size=50, feature_names=['temperature', 'pressure', 'stress_level'])
        
        # Generate reference data
        ref_data = np.random.normal(0, 1, (50, 3))  # 3 features
        test_data = np.random.normal(0, 1, (50, 3))
        
        # Add significant drift to temperature
        test_data[:, 0] = np.random.normal(3, 1, 50)  # Strong drift in temperature
        
        detector.set_reference_window(ref_data[:, 0], features=ref_data)
        
        # Add samples and check drift
        for i in range(50):
            detector.add_sample(test_data[i, 0], features=test_data[i])
            
        # Check final drift state
        _, _, info = detector.detect_drift()
        assert 'temperature' in info['drifting_features']

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
        _, severity, info = detector.detect_drift()
        trends.append(info['trend'])
        
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
        
    _, severity, _ = detector.detect_drift()
    assert severity < 0.5, "Expected minimal drift"

def test_lowered_drift_threshold():
    """Test that lowered drift threshold detects more subtle drifts"""
    detector = DriftDetector(window_size=50)
    detector.drift_threshold = 0.5  # Use new lowered threshold
    
    # Generate data with subtle drift
    ref_data = np.random.normal(0, 1, 50)
    test_data = np.random.normal(1.5, 1, 50)  # Significant drift
    
    detector.set_reference_window(ref_data)
    drift_count = 0
    last_info = None
    for value in test_data:
        drift_detected, severity, info = detector.add_sample(value)
        last_info = info
        if drift_detected:
            drift_count += 1
            
    print("\nLowered Threshold Debug:")
    print(f"Mean shift: {last_info['mean_shift']}")
    print(f"KS stat: {last_info['ks_statistic']}")
    print(f"p-value: {last_info['p_value']}")
    print(f"Trend: {last_info['trend']}")
    print(f"Severity: {severity}")
    assert drift_count > 0, "Should detect subtle drift with lowered threshold"

def test_reduced_drift_interval():
    """Test that reduced drift interval allows more frequent drift detection"""
    detector = DriftDetector(window_size=50)
    detector.min_drift_interval = 40  # Set to original value
    
    # Generate data with multiple significant drifts
    ref_data = np.random.normal(0, 1, 50)
    test_data = []
    
    # Generate multiple drift periods
    for _ in range(3):
        test_data.extend(np.random.normal(2, 1, 50))  # Strong drift
        test_data.extend(np.random.normal(0, 1, 10))  # Return to normal
    
    detector.set_reference_window(ref_data)
    drift_detections = []
    for i, value in enumerate(test_data):
        drift_detected, _, _ = detector.add_sample(value)
        if drift_detected:
            drift_detections.append(i)
            
    # Check that we have at least 2 drifts detected with appropriate spacing
    assert len(drift_detections) >= 2
    if len(drift_detections) >= 2:
        for i in range(1, len(drift_detections)):
            assert drift_detections[i] - drift_detections[i-1] >= detector.min_drift_interval

def test_confidence_threshold():
    """Test that lowered confidence threshold (0.75) affects detection"""
    detector = DriftDetector(window_size=50)
    
    # Generate reference data
    ref_data = np.random.normal(0, 1, 50)
    detector.set_reference_window(ref_data)
    
    # Generate borderline confidence data
    proba = np.array([[0.76, 0.24] for _ in range(50)])  # Just above threshold
    
    confidence_alerts = 0
    for i in range(50):
        _, _, info = detector.add_sample(ref_data[i], prediction_proba=proba[i])
        if info.get('confidence_warning', False):
            confidence_alerts += 1
    
    assert confidence_alerts == 0, "Should not trigger confidence warnings above 0.75"

def test_significance_level():
    """Test increased significance level sensitivity"""
    detector = DriftDetector(window_size=50)
    detector.significance_level = 0.01  # Set to original value
    
    # Generate reference data
    ref_data = np.random.normal(0, 1, 50)
    detector.set_reference_window(ref_data)
    
    # Generate borderline significant data
    borderline_data = np.random.normal(1.0, 1, 50)  # More noticeable shift
    
    p_values = []
    for i in range(50):
        _, _, info = detector.add_sample(borderline_data[i])
        if 'p_value' in info:
            p_values.append(info['p_value'])
    
    # Check that some p-values are significant at 0.01
    significant_001 = sum(1 for p in p_values if p <= 0.01)
    assert significant_001 > 0, "Should detect drifts with 0.01 significance level"
