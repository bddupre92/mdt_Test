"""
test_drift_detection.py
-----------------------
Tests various drift detection mechanisms and their ability to handle different types of drift.
"""

import unittest
import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier

from drift_detection.statistical import ks_drift_test, chi2_drift_test
from drift_detection.performance_monitor import DDM, EDDM
from drift_detection.adwin import ADWIN

def generate_drift_data(n_samples=1000, drift_point=500, drift_type='sudden'):
    """Generate data with different types of drift"""
    if drift_type == 'sudden':
        pre_drift = norm.rvs(loc=0, scale=1, size=drift_point)
        post_drift = norm.rvs(loc=2, scale=1, size=n_samples-drift_point)
    elif drift_type == 'gradual':
        pre_drift = norm.rvs(loc=0, scale=1, size=drift_point)
        transition = np.linspace(0, 2, n_samples-drift_point)
        post_drift = norm.rvs(loc=transition, scale=1, size=n_samples-drift_point)
    elif drift_type == 'incremental':
        x = np.linspace(0, 1, n_samples)
        data = norm.rvs(loc=2*x, scale=1, size=n_samples)
        return data, x > 0.5
    
    return np.concatenate([pre_drift, post_drift]), np.arange(n_samples) >= drift_point

class TestDriftDetection(unittest.TestCase):
    def setUp(self):
        """Initialize test data"""
        self.n_samples = 1000
        self.drift_point = 500
        self.drift_types = ['sudden', 'gradual', 'incremental']
        
        # Generate different drift scenarios
        self.datasets = {}
        for drift_type in self.drift_types:
            data, drift_points = generate_drift_data(
                n_samples=self.n_samples,
                drift_point=self.drift_point,
                drift_type=drift_type
            )
            self.datasets[drift_type] = (data, drift_points)
    
    def test_ks_drift_test(self):
        """Test Kolmogorov-Smirnov drift detection"""
        for drift_type, (data, _) in self.datasets.items():
            # Split data into pre and post drift
            pre_drift = pd.DataFrame({'feat': data[:self.drift_point]})
            post_drift = pd.DataFrame({'feat': data[self.drift_point:]})
            
            p_value = ks_drift_test(pre_drift, post_drift, 'feat')
            self.assertLess(p_value, 0.05, 
                           f"KS test failed to detect {drift_type} drift")
    
    def test_chi2_drift_test(self):
        """Test Chi-square drift detection"""
        # Generate categorical data with drift
        X_pre, y_pre = make_classification(n_samples=500, n_features=5, n_classes=2)
        X_post, y_post = make_classification(n_samples=500, n_features=5, n_classes=2,
                                           weights=[0.8, 0.2])  # Class imbalance drift
        
        pre_df = pd.DataFrame(X_pre, columns=[f'f{i}' for i in range(5)])
        post_df = pd.DataFrame(X_post, columns=[f'f{i}' for i in range(5)])
        
        # Discretize features
        for df in [pre_df, post_df]:
            for col in df.columns:
                df[col] = pd.qcut(df[col], q=5, labels=False)
        
        p_value = chi2_drift_test(pre_df, post_df, 'f0')
        self.assertLess(p_value, 0.05, "Chi2 test failed to detect categorical drift")
    
    def test_ddm_different_drifts(self):
        """Test DDM on different types of drift"""
        for drift_type, (data, drift_points) in self.datasets.items():
            ddm = DDM()
            drift_detected = False
            detection_point = None
            
            for i, value in enumerate(data):
                correct = value < 0.5  # Simulate classification
                if ddm.update(correct):
                    drift_detected = True
                    detection_point = i
                    break
            
            self.assertTrue(drift_detected,
                          f"DDM failed to detect {drift_type} drift")
            if drift_type == 'sudden':
                self.assertGreater(detection_point, self.drift_point - 100,
                                 "DDM detected drift too early")
    
    def test_eddm_concept_evolution(self):
        """Test EDDM's ability to detect gradual concept evolution"""
        _, (data, _) = self.datasets['gradual']
        
        eddm = EDDM()
        drift_detected = False
        last_warning = None
        
        for i, value in enumerate(data):
            correct = value < 0.5
            if eddm.update(correct):
                if eddm.warning_zone:
                    last_warning = i
                else:
                    drift_detected = True
                    break
        
        self.assertTrue(drift_detected, "EDDM failed to detect gradual drift")
        self.assertIsNotNone(last_warning, "EDDM never entered warning zone")
    
    def test_adwin_memory_usage(self):
        """Test ADWIN's memory usage with stream processing"""
        import psutil
        process = psutil.Process()
        initial_memory = process.memory_info().rss
        
        # Process a long stream
        adwin = ADWIN(delta=0.002)
        stream_length = 10000
        
        for i in range(stream_length):
            if i < stream_length//2:
                value = np.random.normal(0, 1)
            else:
                value = np.random.normal(1, 1)
            adwin.update(value)
        
        final_memory = process.memory_info().rss
        memory_increase = (final_memory - initial_memory) / 1024 / 1024  # MB
        
        self.assertLess(memory_increase, 50, 
                       "ADWIN memory usage too high")
    
    def test_drift_detection_ensemble(self):
        """Test ensemble of drift detectors"""
        # Combine multiple detectors
        detectors = {
            'ddm': DDM(),
            'eddm': EDDM(),
            'adwin': ADWIN()
        }
        
        # Test on sudden drift data
        data, _ = self.datasets['sudden']
        detection_points = {}
        
        for name, detector in detectors.items():
            for i, value in enumerate(data):
                correct = value < 0.5
                if name == 'adwin':
                    if detector.update(value):
                        detection_points[name] = i
                        break
                else:
                    if detector.update(correct):
                        detection_points[name] = i
                        break
        
        # Check if detectors agree within a reasonable range
        detection_times = list(detection_points.values())
        max_diff = max(detection_times) - min(detection_times)
        self.assertLess(max_diff, 200, 
                       "Drift detectors disagree significantly")
    
    def test_drift_recovery(self):
        """Test drift detection reset and recovery"""
        ddm = DDM()
        
        # Generate data with multiple drift points
        n_samples = 2000
        data = np.concatenate([
            norm.rvs(loc=0, scale=1, size=500),   # Initial concept
            norm.rvs(loc=2, scale=1, size=500),   # First drift
            norm.rvs(loc=0, scale=1, size=500),   # Return to initial concept
            norm.rvs(loc=3, scale=1, size=500)    # Second drift
        ])
        
        drift_points = []
        for i, value in enumerate(data):
            correct = value < 0.5
            if ddm.update(correct):
                drift_points.append(i)
                ddm.reset()  # Reset after drift
        
        self.assertEqual(len(drift_points), 3,
                        "Failed to detect multiple concept drifts")
