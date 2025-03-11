"""Unit tests for frequency domain feature extraction.

This module tests the FrequencyDomainFeatures class to ensure correct
functionality for extracting frequency-domain features from signals.
"""

import unittest
import numpy as np
from core.theory.pattern_recognition.frequency_domain_features import FrequencyDomainFeatures

class TestFrequencyDomainFeatures(unittest.TestCase):
    """Test cases for FrequencyDomainFeatures class."""

    def setUp(self):
        """Set up test fixtures."""
        # Create sample signals for testing
        self.sample_rate = 100  # Hz
        self.duration = 5  # seconds
        t = np.linspace(0, self.duration, int(self.duration * self.sample_rate))
        
        # Simple sine wave (5 Hz)
        self.sine_wave = np.sin(2 * np.pi * 5 * t)
        
        # Two-component signal (5Hz + 15Hz)
        self.two_component = np.sin(2 * np.pi * 5 * t) + 0.5 * np.sin(2 * np.pi * 15 * t)
        
        # Multi-channel signal
        self.multi_channel = np.column_stack([
            np.sin(2 * np.pi * 2 * t),   # 2 Hz sine
            np.sin(2 * np.pi * 10 * t),  # 10 Hz sine
            np.sin(2 * np.pi * 25 * t)   # 25 Hz sine
        ])

    def test_initialization(self):
        """Test initialization with various parameters."""
        # Default initialization
        extractor = FrequencyDomainFeatures(sampling_rate=100)
        self.assertEqual(extractor.sampling_rate, 100)
        self.assertEqual(extractor.window_size, 256)
        self.assertEqual(extractor.overlap, 0.5)
        self.assertListEqual(extractor.features, extractor.available_features)
        
        # Custom parameters
        extractor = FrequencyDomainFeatures(
            sampling_rate=200, 
            window_size=512, 
            overlap=0.75, 
            features=['spectral_centroid', 'spectral_bandwidth']
        )
        self.assertEqual(extractor.sampling_rate, 200)
        self.assertEqual(extractor.window_size, 512)
        self.assertEqual(extractor.overlap, 0.75)
        self.assertListEqual(extractor.features, ['spectral_centroid', 'spectral_bandwidth'])
        
        # Custom frequency bands
        custom_bands = {'low': (1, 10), 'high': (10, 40)}
        extractor = FrequencyDomainFeatures(
            sampling_rate=100,
            freq_bands=custom_bands
        )
        self.assertEqual(extractor.freq_bands, custom_bands)

    def test_validate_input(self):
        """Test signal validation."""
        extractor = FrequencyDomainFeatures(sampling_rate=100, window_size=100)
        
        # Valid signals
        self.assertTrue(extractor.validate_input(self.sine_wave))
        self.assertTrue(extractor.validate_input(self.two_component))
        self.assertTrue(extractor.validate_input(self.multi_channel))
        
        # Test with invalid inputs
        with self.assertRaises(ValueError):
            extractor.validate_input("not_an_array")
        
        with self.assertRaises(ValueError):
            extractor.validate_input(np.array([1, 2, 3]))  # Too short
            
        with self.assertRaises(ValueError):
            extractor.validate_input(np.ones((10, 10, 10)))  # Wrong dimensions

    def test_extract_simple_signal(self):
        """Test feature extraction with simple signal (sine wave)."""
        extractor = FrequencyDomainFeatures(sampling_rate=self.sample_rate, window_size=128)
        features = extractor.extract(self.sine_wave)
        
        # Check if all expected features are extracted
        for feature_name in extractor.features:
            if feature_name == 'band_powers':
                for band in extractor.freq_bands:
                    self.assertIn(f'{band}_power', features)
            else:
                self.assertIn(feature_name, features)
        
        # Check dimensions
        n_windows = len(self.sine_wave) // 64 - 1  # Considering overlap
        self.assertEqual(features['spectral_centroid'].shape[0], n_windows)
        
        # Check specific values for sine wave
        # Dominant frequency should be close to 5 Hz
        self.assertAlmostEqual(np.mean(features['dominant_frequency']), 5.0, delta=1.0)
        
    def test_extract_complex_signal(self):
        """Test feature extraction with complex signal (two components)."""
        extractor = FrequencyDomainFeatures(sampling_rate=self.sample_rate, window_size=128)
        features = extractor.extract(self.two_component)
        
        # For a two-component signal, dominant frequency can be either component
        # depending on the window, but the spectral centroid should be between them
        mean_centroid = np.mean(features['spectral_centroid'])
        self.assertTrue(5 < mean_centroid < 15)
        
    def test_extract_multi_channel(self):
        """Test feature extraction with multi-channel signal."""
        extractor = FrequencyDomainFeatures(sampling_rate=self.sample_rate, window_size=128)
        features = extractor.extract(self.multi_channel)
        
        # Check dimensions - should have features for each channel
        n_windows = len(self.multi_channel) // 64 - 1  # Considering overlap
        self.assertEqual(features['spectral_centroid'].shape, (n_windows, 3))
        
        # Check that dominant frequencies match the input frequencies
        # Allow some leakage/imprecision due to windowing
        expected_freqs = np.array([2, 10, 25])
        # We'll check the mean across windows
        mean_dom_freqs = np.mean(features['dominant_frequency'], axis=0)
        
        # Test with tolerance
        for i, expected in enumerate(expected_freqs):
            self.assertAlmostEqual(mean_dom_freqs[i], expected, delta=1.5)
            
    def test_specific_features(self):
        """Test extraction of specific features."""
        extractor = FrequencyDomainFeatures(
            sampling_rate=self.sample_rate,
            window_size=128,
            features=['spectral_centroid', 'dominant_frequency']
        )
        features = extractor.extract(self.sine_wave)
        
        # Only requested features should be present
        self.assertIn('spectral_centroid', features)
        self.assertIn('dominant_frequency', features)
        self.assertNotIn('spectral_bandwidth', features)
        self.assertNotIn('spectral_rolloff', features)
        
    def test_band_powers(self):
        """Test frequency band power calculation."""
        # Create a signal with energy in specific bands
        t = np.linspace(0, 10, 1000)
        delta_wave = 0.5 * np.sin(2 * np.pi * 2 * t)   # 2 Hz (delta: 0.5-4 Hz)
        alpha_wave = 1.0 * np.sin(2 * np.pi * 10 * t)  # 10 Hz (alpha: 8-13 Hz)
        beta_wave = 0.3 * np.sin(2 * np.pi * 20 * t)   # 20 Hz (beta: 13-30 Hz)
        
        # Combined signal with most energy in alpha band
        signal = delta_wave + alpha_wave + beta_wave
        
        extractor = FrequencyDomainFeatures(
            sampling_rate=100,
            window_size=100,
            overlap=0,
            features=['band_powers']
        )
        features = extractor.extract(signal)
        
        # Alpha power should be highest
        alpha_power = np.mean(features['alpha_power'])
        delta_power = np.mean(features['delta_power'])
        beta_power = np.mean(features['beta_power'])
        
        self.assertTrue(alpha_power > delta_power)
        self.assertTrue(alpha_power > beta_power)

if __name__ == '__main__':
    unittest.main() 