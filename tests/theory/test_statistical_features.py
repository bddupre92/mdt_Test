"""Unit tests for statistical feature extraction.

This module tests the StatisticalFeatures class to ensure correct
functionality for extracting statistical features from signals.
"""

import unittest
import numpy as np
from scipy import stats
from core.theory.pattern_recognition.statistical_features import StatisticalFeatures

class TestStatisticalFeatures(unittest.TestCase):
    """Test cases for StatisticalFeatures class."""

    def setUp(self):
        """Set up test fixtures."""
        # Create sample signals for testing
        self.duration = 5  # seconds
        self.sample_rate = 100  # Hz
        t = np.linspace(0, self.duration, int(self.duration * self.sample_rate))
        
        # Normal distribution
        np.random.seed(42)  # For reproducibility
        self.normal_signal = np.random.normal(0, 1, len(t))
        
        # Skewed distribution
        self.skewed_signal = np.random.exponential(1, len(t))
        
        # Signal with known statistics
        self.known_stats_signal = np.linspace(-5, 5, 500)
        
        # Multi-channel signal
        self.multi_channel = np.column_stack([
            self.normal_signal,
            self.skewed_signal,
            np.sin(2 * np.pi * 2 * t)  # Sine wave
        ])

    def test_initialization(self):
        """Test initialization with various parameters."""
        # Default initialization
        extractor = StatisticalFeatures()
        self.assertEqual(extractor.window_size, 256)
        self.assertEqual(extractor.overlap, 0.5)
        self.assertListEqual(extractor.features, extractor.available_features)
        
        # Custom parameters
        extractor = StatisticalFeatures(window_size=512, overlap=0.75, 
                                       features=['mean', 'std', 'skewness'])
        self.assertEqual(extractor.window_size, 512)
        self.assertEqual(extractor.overlap, 0.75)
        self.assertListEqual(extractor.features, ['mean', 'std', 'skewness'])

    def test_validate_input(self):
        """Test signal validation."""
        extractor = StatisticalFeatures(window_size=100)
        
        # Valid signals
        self.assertTrue(extractor.validate_input(self.normal_signal))
        self.assertTrue(extractor.validate_input(self.skewed_signal))
        self.assertTrue(extractor.validate_input(self.multi_channel))
        
        # Test with invalid inputs
        with self.assertRaises(ValueError):
            extractor.validate_input("not_an_array")
        
        with self.assertRaises(ValueError):
            extractor.validate_input(np.array([1, 2, 3]))  # Too short
            
        with self.assertRaises(ValueError):
            extractor.validate_input(np.ones((10, 10, 10)))  # Wrong dimensions

    def test_extract_normal_signal(self):
        """Test feature extraction with normal distribution signal."""
        extractor = StatisticalFeatures(window_size=128, overlap=0)
        features = extractor.extract(self.normal_signal)
        
        # Check if all expected features are extracted
        for feature_name in extractor.features:
            if feature_name == 'percentiles':
                for p in extractor.percentile_values:
                    self.assertIn(f'percentile_{p}', features)
            else:
                self.assertIn(feature_name, features)
        
        # Basic checks on values
        # Normal distribution should have near-zero skewness and kurtosis close to 3
        mean_skewness = np.mean(features['skewness'])
        mean_kurtosis = np.mean(features['kurtosis'])
        
        self.assertAlmostEqual(mean_skewness, 0, delta=0.3)
        # For kurtosis, scipy uses Fisher's definition which is normalized to 0 for normal distribution
        self.assertAlmostEqual(mean_kurtosis, 0, delta=0.5)
        
    def test_extract_skewed_signal(self):
        """Test feature extraction with skewed distribution signal."""
        extractor = StatisticalFeatures(window_size=128, overlap=0)
        features = extractor.extract(self.skewed_signal)
        
        # Exponential distribution should have positive skewness
        mean_skewness = np.mean(features['skewness'])
        self.assertGreater(mean_skewness, 0.5)
        
    def test_extract_known_stats(self):
        """Test extraction with a signal that has known statistics."""
        # Linear range from -5 to 5 has:
        # - mean = 0
        # - median = 0
        # - std = ~2.89 (for continuous uniform distribution)
        extractor = StatisticalFeatures(
            window_size=len(self.known_stats_signal), 
            overlap=0,
            features=['mean', 'median', 'std']
        )
        features = extractor.extract(self.known_stats_signal)
        
        self.assertAlmostEqual(features['mean'][0], 0, delta=0.1)
        self.assertAlmostEqual(features['median'][0], 0, delta=0.1)
        # The standard deviation of a uniform distribution from -5 to 5 is (b-a)/sqrt(12) ≈ 2.89
        self.assertAlmostEqual(features['std'][0], 2.89, delta=0.1)
        
    def test_extract_multi_channel(self):
        """Test feature extraction with multi-channel signal."""
        extractor = StatisticalFeatures(window_size=128, overlap=0)
        features = extractor.extract(self.multi_channel)
        
        # Check dimensions - should have features for each channel
        self.assertEqual(features['mean'].shape, (3, 3))  # 3 windows, 3 channels
        
        # First channel is normal distribution, should have near-zero skewness
        self.assertAlmostEqual(np.mean(features['skewness'][:, 0]), 0, delta=0.5)
        
        # Second channel is exponential, should have positive skewness
        self.assertGreater(np.mean(features['skewness'][:, 1]), 0.5)
        
    def test_specific_features(self):
        """Test extraction of specific features."""
        extractor = StatisticalFeatures(
            window_size=128,
            overlap=0, 
            features=['mean', 'std', 'entropy']
        )
        features = extractor.extract(self.normal_signal)
        
        # Only requested features should be present
        self.assertIn('mean', features)
        self.assertIn('std', features)
        self.assertIn('entropy', features)
        self.assertNotIn('skewness', features)
        self.assertNotIn('kurtosis', features)
        
    def test_entropy(self):
        """Test entropy calculation."""
        # Constant signal has entropy = 0
        constant_signal = np.ones(500)
        
        # Random uniform signal has high entropy
        np.random.seed(42)
        uniform_signal = np.random.uniform(0, 1, 500)
        
        extractor = StatisticalFeatures(
            window_size=500,
            overlap=0, 
            features=['entropy']
        )
        
        # Extract features
        const_features = extractor.extract(constant_signal)
        uniform_features = extractor.extract(uniform_signal)
        
        # Constant signal should have minimal entropy
        self.assertLess(const_features['entropy'][0], 0.1)
        
        # Uniform signal should have higher entropy
        self.assertGreater(uniform_features['entropy'][0], const_features['entropy'][0])

if __name__ == '__main__':
    unittest.main() 