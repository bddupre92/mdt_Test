"""
Unit tests for physiological signal adapters.

Tests the functionality of various physiological signal adapters including:
- ECG/HRV processing
- EEG signal processing
- Skin conductance processing
- Respiratory signal processing
- Temperature signal processing
"""

import unittest
import numpy as np
from scipy import signal
import sys
import os

# Add the project root to the path so we can import the modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from core.theory.migraine_adaptation.physiological_adapters import (
    ECGAdapter,
    EEGAdapter,
    SkinConductanceAdapter,
    RespiratoryAdapter,
    TemperatureAdapter
)

# Test data generation functions
def generate_ecg_data(sampling_rate: float = 250.0, duration: float = 10.0) -> np.ndarray:
    """Generate synthetic ECG-like signal."""
    t = np.arange(0, duration, 1/sampling_rate)
    # Create synthetic QRS complexes
    signal = np.zeros_like(t)
    for i in range(int(duration)):
        peak_loc = i * sampling_rate + sampling_rate/2
        signal += 1.0 * np.exp(-(t*sampling_rate - peak_loc)**2 / 100)
    return signal

def generate_eeg_data(sampling_rate: float = 250.0, duration: float = 10.0) -> np.ndarray:
    """Generate synthetic EEG-like signal."""
    t = np.arange(0, duration, 1/sampling_rate)
    # Combine different frequency components
    alpha = np.sin(2 * np.pi * 10 * t)  # 10 Hz alpha
    beta = 0.5 * np.sin(2 * np.pi * 20 * t)  # 20 Hz beta
    theta = 0.3 * np.sin(2 * np.pi * 6 * t)  # 6 Hz theta
    return alpha + beta + theta + 0.1 * np.random.randn(len(t))

def generate_skin_conductance_data(sampling_rate: float = 100.0, duration: float = 10.0) -> np.ndarray:
    """Generate synthetic skin conductance signal."""
    t = np.arange(0, duration, 1/sampling_rate)
    # Base level + responses
    base = 5 + 0.1 * np.random.randn(len(t))
    responses = np.zeros_like(t)
    for i in range(3):
        peak_loc = (i + 1) * sampling_rate * 2
        responses += 2.0 * np.exp(-(t*sampling_rate - peak_loc)**2 / 10000)
    return base + responses

def generate_respiratory_data(sampling_rate: float = 100.0, duration: float = 10.0) -> np.ndarray:
    """Generate synthetic respiratory signal."""
    t = np.arange(0, duration, 1/sampling_rate)
    # Simulate breathing at 0.2 Hz (12 breaths per minute)
    return np.sin(2 * np.pi * 0.2 * t) + 0.1 * np.random.randn(len(t))

def generate_temperature_data(sampling_rate: float = 1.0, duration: float = 100.0) -> np.ndarray:
    """Generate synthetic temperature signal."""
    t = np.arange(0, duration, 1/sampling_rate)
    # Base temperature with slow drift and noise
    return 37.0 + 0.2 * np.sin(2 * np.pi * 0.01 * t) + 0.05 * np.random.randn(len(t))

class TestECGAdapter(unittest.TestCase):
    """Test suite for ECG signal adapter."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.adapter = ECGAdapter()
        self.test_data = generate_ecg_data()
        self.sampling_rate = 250.0
    
    def test_preprocess(self):
        """Test ECG preprocessing."""
        processed = self.adapter.preprocess(self.test_data, self.sampling_rate)
        self.assertIsInstance(processed, np.ndarray)
        self.assertEqual(len(processed), len(self.test_data))
        self.assertFalse(np.allclose(processed, self.test_data))  # Should be different after filtering
    
    def test_extract_features(self):
        """Test ECG feature extraction."""
        processed = self.adapter.preprocess(self.test_data, self.sampling_rate)
        features = self.adapter.extract_features(processed, self.sampling_rate)
        
        self.assertIsInstance(features, dict)
        self.assertIn('rr_intervals', features)
        self.assertIn('sdnn', features)
        self.assertIn('rmssd', features)
        self.assertIn('pnn50', features)
        
        self.assertIsInstance(features['sdnn'], (float, np.floating))
        self.assertGreaterEqual(features['pnn50'], 0)
        self.assertLessEqual(features['pnn50'], 1)
    
    def test_assess_quality(self):
        """Test ECG quality assessment."""
        quality = self.adapter.assess_quality(self.test_data, self.sampling_rate)
        self.assertIsInstance(quality, float)
        self.assertGreaterEqual(quality, 0)
        self.assertLessEqual(quality, 1)

class TestEEGAdapter(unittest.TestCase):
    """Test suite for EEG signal adapter."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.adapter = EEGAdapter()
        self.test_data = generate_eeg_data()
        self.sampling_rate = 250.0
    
    def test_preprocess(self):
        """Test EEG preprocessing."""
        processed = self.adapter.preprocess(self.test_data, self.sampling_rate)
        self.assertIsInstance(processed, np.ndarray)
        self.assertEqual(len(processed), len(self.test_data))
    
    def test_extract_features(self):
        """Test EEG feature extraction."""
        processed = self.adapter.preprocess(self.test_data, self.sampling_rate)
        features = self.adapter.extract_features(processed, self.sampling_rate)
        
        expected_bands = ['delta_power', 'theta_power', 'alpha_power', 'beta_power', 'gamma_power']
        for band in expected_bands:
            self.assertIn(band, features)
            self.assertGreaterEqual(features[band], 0)
        
        self.assertIn('theta_beta_ratio', features)
        self.assertIn('alpha_beta_ratio', features)
    
    def test_assess_quality(self):
        """Test EEG quality assessment."""
        quality = self.adapter.assess_quality(self.test_data, self.sampling_rate)
        self.assertIsInstance(quality, float)
        self.assertGreaterEqual(quality, 0)
        self.assertLessEqual(quality, 1)

class TestSkinConductanceAdapter(unittest.TestCase):
    """Test suite for skin conductance signal adapter."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.adapter = SkinConductanceAdapter()
        self.test_data = generate_skin_conductance_data()
        self.sampling_rate = 100.0
    
    def test_preprocess(self):
        """Test skin conductance preprocessing."""
        processed = self.adapter.preprocess(self.test_data, self.sampling_rate)
        self.assertIsInstance(processed, np.ndarray)
        self.assertEqual(len(processed), len(self.test_data))
    
    def test_extract_features(self):
        """Test skin conductance feature extraction."""
        processed = self.adapter.preprocess(self.test_data, self.sampling_rate)
        features = self.adapter.extract_features(processed, self.sampling_rate)
        
        expected_features = ['scr_rate', 'mean_amplitude', 'max_amplitude', 'tonic_level', 'standard_deviation']
        for feature in expected_features:
            self.assertIn(feature, features)
            self.assertIsInstance(features[feature], (float, np.floating))
    
    def test_assess_quality(self):
        """Test skin conductance quality assessment."""
        quality = self.adapter.assess_quality(self.test_data, self.sampling_rate)
        self.assertIsInstance(quality, float)
        self.assertGreaterEqual(quality, 0)
        self.assertLessEqual(quality, 1)

class TestRespiratoryAdapter(unittest.TestCase):
    """Test suite for respiratory signal adapter."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.adapter = RespiratoryAdapter()
        self.test_data = generate_respiratory_data()
        self.sampling_rate = 100.0
    
    def test_preprocess(self):
        """Test respiratory preprocessing."""
        processed = self.adapter.preprocess(self.test_data, self.sampling_rate)
        self.assertIsInstance(processed, np.ndarray)
        self.assertEqual(len(processed), len(self.test_data))
    
    def test_extract_features(self):
        """Test respiratory feature extraction."""
        processed = self.adapter.preprocess(self.test_data, self.sampling_rate)
        features = self.adapter.extract_features(processed, self.sampling_rate)
        
        expected_features = ['breathing_rate', 'breath_interval_std', 'depth_variation', 'irregularity', 'amplitude']
        for feature in expected_features:
            self.assertIn(feature, features)
            self.assertIsInstance(features[feature], (float, np.floating))
        
        # Check breathing rate is physiologically plausible
        self.assertGreaterEqual(features['breathing_rate'], 0)
        self.assertLessEqual(features['breathing_rate'], 60)
    
    def test_assess_quality(self):
        """Test respiratory quality assessment."""
        quality = self.adapter.assess_quality(self.test_data, self.sampling_rate)
        self.assertIsInstance(quality, float)
        self.assertGreaterEqual(quality, 0)
        self.assertLessEqual(quality, 1)

class TestTemperatureAdapter(unittest.TestCase):
    """Test suite for temperature signal adapter."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.adapter = TemperatureAdapter()
        self.test_data = generate_temperature_data()
        self.sampling_rate = 1.0
    
    def test_preprocess(self):
        """Test temperature preprocessing."""
        processed = self.adapter.preprocess(self.test_data, self.sampling_rate)
        self.assertIsInstance(processed, np.ndarray)
        self.assertEqual(len(processed), len(self.test_data))
    
    def test_extract_features(self):
        """Test temperature feature extraction."""
        processed = self.adapter.preprocess(self.test_data, self.sampling_rate)
        features = self.adapter.extract_features(processed, self.sampling_rate)
        
        expected_features = ['mean_temp', 'temp_std', 'temp_range', 'temp_gradient', 'temp_variability']
        for feature in expected_features:
            self.assertIn(feature, features)
            self.assertIsInstance(features[feature], (float, np.floating))
        
        # Check temperature range is physiologically plausible
        self.assertGreaterEqual(features['mean_temp'], 25)
        self.assertLessEqual(features['mean_temp'], 42)
    
    def test_assess_quality(self):
        """Test temperature quality assessment."""
        quality = self.adapter.assess_quality(self.test_data, self.sampling_rate)
        self.assertIsInstance(quality, float)
        self.assertGreaterEqual(quality, 0)
        self.assertLessEqual(quality, 1)

class TestErrorHandling(unittest.TestCase):
    """Test error handling across all adapters."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.adapters = [
            ECGAdapter(),
            EEGAdapter(),
            SkinConductanceAdapter(),
            RespiratoryAdapter(),
            TemperatureAdapter()
        ]
    
    def test_empty_signal(self):
        """Test handling of empty signals."""
        empty_signal = np.array([])
        for adapter in self.adapters:
            with self.assertRaises(ValueError):
                adapter.preprocess(empty_signal, 100.0)
    
    def test_invalid_sampling_rate(self):
        """Test handling of invalid sampling rates."""
        signal = np.random.randn(100)
        for adapter in self.adapters:
            with self.assertRaises(ValueError):
                adapter.preprocess(signal, -1.0)
            with self.assertRaises(ValueError):
                adapter.preprocess(signal, 0.0)
    
    def test_nan_values(self):
        """Test handling of NaN values in signals."""
        signal = np.random.randn(100)
        signal[50] = np.nan
        for adapter in self.adapters:
            with self.assertRaises(ValueError):
                adapter.preprocess(signal, 100.0)

if __name__ == "__main__":
    # Create a test suite
    suite = unittest.TestLoader().loadTestsFromModule(sys.modules[__name__])
    
    # Run the tests
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    
    # Exit with appropriate code
    sys.exit(not result.wasSuccessful()) 