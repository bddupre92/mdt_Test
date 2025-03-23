"""Unit tests for time domain feature extraction.

This module tests the TimeDomainFeatures class to ensure correct
functionality for extracting time-domain features from signals.
"""

import unittest
import numpy as np
from core.theory.pattern_recognition.time_domain_features import TimeDomainFeatures

class TestTimeDomainFeatures(unittest.TestCase):
    """Test cases for TimeDomainFeatures class."""

    def setUp(self):
        """Set up test fixtures."""
        # Create sample signals for testing
        self.sample_rate = 100  # Hz
        self.duration = 5  # seconds
        t = np.linspace(0, self.duration, int(self.duration * self.sample_rate))
        
        # Simple sine wave
        self.sine_wave = np.sin(2 * np.pi * 2 * t)  # 2 Hz sine wave
        
        # Compound signal (sine + noise)
        self.compound_signal = np.sin(2 * np.pi * 2 * t) + 0.5 * np.random.randn(len(t))
        
        # Multi-channel signal
        self.multi_channel = np.column_stack([
            np.sin(2 * np.pi * 2 * t),  # 2 Hz sine
            np.sin(2 * np.pi * 5 * t),  # 5 Hz sine
            np.random.randn(len(t))     # Noise
        ])

    def test_initialization(self):
        """Test initialization with various parameters."""
        # Default initialization
        extractor = TimeDomainFeatures()
        self.assertEqual(extractor.window_size, 256)
        self.assertEqual(extractor.overlap, 0.5)
        self.assertListEqual(extractor.features, extractor.available_features)
        
        # Custom parameters
        extractor = TimeDomainFeatures(window_size=512, overlap=0.75, 
                                      features=['mean_amplitude', 'peak_to_peak'])
        self.assertEqual(extractor.window_size, 512)
        self.assertEqual(extractor.overlap, 0.75)
        self.assertListEqual(extractor.features, ['mean_amplitude', 'peak_to_peak'])

    def test_validate_signal(self):
        """Test signal validation."""
        extractor = TimeDomainFeatures(window_size=100)
        
        # Valid signals
        self.assertTrue(extractor.validate_signal(self.sine_wave))
        self.assertTrue(extractor.validate_signal(self.compound_signal))
        self.assertTrue(extractor.validate_signal(self.multi_channel))
        
        # Invalid signals
        self.assertFalse(extractor.validate_signal("not_an_array"))
        self.assertFalse(extractor.validate_signal(np.array([1, 2, 3])))  # Too short
        self.assertFalse(extractor.validate_signal(np.ones((10, 10, 10))))  # Wrong dimensions

    def test_extract_single_channel(self):
        """Test feature extraction with single-channel signal."""
        extractor = TimeDomainFeatures(window_size=128, overlap=0.5)
        features = extractor.extract(self.sine_wave)
        
        # Check if all expected features are extracted
        for feature_name in extractor.features:
            self.assertIn(feature_name, features)
            
        # Check dimensions
        n_windows = len(self.sine_wave) // 64 - 1  # Considering overlap
        self.assertEqual(features['mean_amplitude'].shape[0], n_windows)
        
        # Basic checks on values
        self.assertAlmostEqual(np.mean(features['peak_to_peak']), 2.0, delta=0.2)
        
    def test_extract_multi_channel(self):
        """Test feature extraction with multi-channel signal."""
        extractor = TimeDomainFeatures(window_size=128, overlap=0.5)
        features = extractor.extract(self.multi_channel)
        
        # Check dimensions - should have features for each channel
        n_windows = len(self.multi_channel) // 64 - 1  # Considering overlap
        self.assertEqual(features['mean_amplitude'].shape, (n_windows, 3))
        
    def test_specific_features(self):
        """Test extraction of specific features."""
        extractor = TimeDomainFeatures(window_size=128, overlap=0.5, 
                                     features=['rms', 'zero_crossings'])
        features = extractor.extract(self.sine_wave)
        
        # Only requested features should be present
        self.assertIn('rms', features)
        self.assertIn('zero_crossings', features)
        self.assertNotIn('mean_amplitude', features)
        self.assertNotIn('peak_to_peak', features)
        
    def test_zero_crossings(self):
        """Test zero crossings calculation."""
        # Create a simple sine wave with known zero crossings
        t = np.linspace(0, 1, 100)
        sine = np.sin(2 * np.pi * 5 * t)  # 5 Hz sine = 10 zero crossings
        
        extractor = TimeDomainFeatures(window_size=100, overlap=0)
        features = extractor.extract(sine)
        
        # For a clean sine wave, we expect around 10 zero crossings
        self.assertTrue(8 <= features['zero_crossings'][0] <= 12)
        
    def test_slope_changes(self):
        """Test slope sign changes calculation."""
        # Create a simple sine wave with known slope changes
        t = np.linspace(0, 1, 100)
        sine = np.sin(2 * np.pi * 5 * t)  # 5 Hz sine = 10 slope changes
        
        extractor = TimeDomainFeatures(window_size=100, overlap=0)
        features = extractor.extract(sine)
        
        # For a clean sine wave, we expect around 10 slope changes
        self.assertTrue(8 <= features['slope_changes'][0] <= 12)

if __name__ == '__main__':
    unittest.main() 