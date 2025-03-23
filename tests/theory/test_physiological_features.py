"""Unit tests for physiological feature extraction.

This module tests the PhysiologicalFeatures class to ensure correct
functionality for extracting features from various physiological signals.
"""

import unittest
import numpy as np
from scipy import signal
from core.theory.pattern_recognition.physiological_features import PhysiologicalFeatures

class TestPhysiologicalFeatures(unittest.TestCase):
    """Test cases for PhysiologicalFeatures class."""

    def setUp(self):
        """Set up test fixtures."""
        # Common parameters
        self.duration = 10  # seconds
        self.sample_rate = 100  # Hz
        self.n_samples = int(self.duration * self.sample_rate)
        t = np.linspace(0, self.duration, self.n_samples)
        
        # ECG-like signal (QRS complexes)
        # Create a periodic signal with sharp peaks
        self.ecg_signal = self.generate_ecg_like_signal(t)
        
        # EEG-like signal (frequency mixtures)
        self.eeg_signal = (
            0.5 * np.sin(2 * np.pi * 3 * t) +    # delta wave (3 Hz)
            0.3 * np.sin(2 * np.pi * 6 * t) +    # theta wave (6 Hz)
            0.8 * np.sin(2 * np.pi * 10 * t) +   # alpha wave (10 Hz)
            0.4 * np.sin(2 * np.pi * 20 * t) +   # beta wave (20 Hz)
            0.1 * np.sin(2 * np.pi * 45 * t)     # gamma wave (45 Hz)
        )
        
        # EMG-like signal (bursts of activity)
        self.emg_signal = self.generate_emg_like_signal(t)
        
        # PPG-like signal (smooth periodic)
        self.ppg_signal = 2 + np.sin(2 * np.pi * 1.2 * t) + 0.2 * np.random.randn(len(t))
        
        # GSR-like signal (slow changes with spikes)
        base_gsr = 5 + 0.05 * np.cumsum(np.random.randn(len(t)))
        spikes = np.zeros_like(t)
        for i in range(5):  # Add 5 random SCRs
            spike_loc = np.random.randint(100, self.n_samples - 100)
            spike_width = np.random.randint(50, 100)
            spikes[spike_loc:spike_loc+spike_width] = np.exp(-np.linspace(0, 5, spike_width))
        self.gsr_signal = base_gsr + spikes
        
        # Respiratory signal (slow sine with variations)
        self.resp_signal = np.sin(2 * np.pi * 0.25 * t) + 0.1 * np.sin(2 * np.pi * 0.05 * t) + 0.05 * np.random.randn(len(t))

    def generate_ecg_like_signal(self, t):
        """Generate a simplified ECG-like signal with R peaks."""
        # Heart rate: about 60 bpm (1 Hz)
        hr = 1.0  
        ecg = np.zeros_like(t)
        
        # Create QRS complexes at regular intervals
        for i in range(int(self.duration * hr)):
            center = i / hr
            # R peak
            mask = (t >= center - 0.05) & (t <= center + 0.05)
            ecg[mask] = 3.0 * np.exp(-100 * (t[mask] - center)**2)
            
            # Q and S waves
            q_mask = (t >= center - 0.1) & (t < center - 0.05)
            s_mask = (t > center + 0.05) & (t <= center + 0.1)
            ecg[q_mask] = -0.5 * np.exp(-100 * (t[q_mask] - (center - 0.1))**2)
            ecg[s_mask] = -0.5 * np.exp(-100 * (t[s_mask] - (center + 0.1))**2)
            
            # T wave
            t_mask = (t > center + 0.1) & (t <= center + 0.4)
            ecg[t_mask] = 1.0 * np.exp(-20 * (t[t_mask] - (center + 0.25))**2)
        
        return ecg + 0.1 * np.random.randn(len(t))

    def generate_emg_like_signal(self, t):
        """Generate an EMG-like signal with bursts of activity."""
        emg = np.random.randn(len(t)) * 0.1  # Background noise
        
        # Add activity bursts
        for i in range(3):
            start = np.random.randint(100, self.n_samples - 500)
            end = start + np.random.randint(200, 400)
            emg[start:end] = np.random.randn(end - start) * 1.0
            
        return emg

    def test_initialization(self):
        """Test initialization with various signal types."""
        # Test all supported signal types
        for signal_type in PhysiologicalFeatures.SUPPORTED_SIGNALS:
            extractor = PhysiologicalFeatures(signal_type=signal_type, sampling_rate=100)
            self.assertEqual(extractor.signal_type, signal_type)
            self.assertEqual(extractor.sampling_rate, 100)
            self.assertEqual(extractor.window_size, 256)
            self.assertEqual(extractor.overlap, 0.5)
            
        # Test with unsupported signal type
        with self.assertRaises(ValueError):
            PhysiologicalFeatures(signal_type='unsupported', sampling_rate=100)
            
        # Test with custom parameters
        extractor = PhysiologicalFeatures(
            signal_type='ecg',
            sampling_rate=250,
            window_size=512,
            overlap=0.75
        )
        self.assertEqual(extractor.signal_type, 'ecg')
        self.assertEqual(extractor.sampling_rate, 250)
        self.assertEqual(extractor.window_size, 512)
        self.assertEqual(extractor.overlap, 0.75)

    def test_validate_input(self):
        """Test signal validation."""
        extractor = PhysiologicalFeatures(signal_type='ecg', sampling_rate=100, window_size=100)
        
        # Valid signals
        self.assertTrue(extractor.validate_input(self.ecg_signal))
        self.assertTrue(extractor.validate_input(self.eeg_signal))
        
        # Test with invalid inputs
        with self.assertRaises(ValueError):
            extractor.validate_input("not_an_array")
        
        with self.assertRaises(ValueError):
            extractor.validate_input(np.array([1, 2, 3]))  # Too short
            
        with self.assertRaises(ValueError):
            extractor.validate_input(np.ones((10, 10, 10)))  # Wrong dimensions

    def test_ecg_features(self):
        """Test ECG feature extraction."""
        extractor = PhysiologicalFeatures(
            signal_type='ecg',
            sampling_rate=self.sample_rate,
            window_size=256,
            overlap=0.5
        )
        
        features = extractor.extract(self.ecg_signal)
        
        # Check if features were extracted
        self.assertGreater(len(features), 0)
        
        # Heart rate should be approximately 60 bpm
        if 'heart_rate' in features:
            mean_hr = np.mean(features['heart_rate'])
            self.assertAlmostEqual(mean_hr, 60, delta=15)  # Allow some margin due to peak detection

    def test_eeg_features(self):
        """Test EEG feature extraction."""
        extractor = PhysiologicalFeatures(
            signal_type='eeg',
            sampling_rate=self.sample_rate,
            window_size=512,  # Larger window for better frequency resolution
            overlap=0.5
        )
        
        features = extractor.extract(self.eeg_signal)
        
        # Check if features were extracted
        self.assertGreater(len(features), 0)
        
        # Alpha power should be higher (we made alpha waves the strongest)
        if 'alpha_power' in features and 'beta_power' in features:
            self.assertGreater(np.mean(features['alpha_power']), np.mean(features['beta_power']))

    def test_emg_features(self):
        """Test EMG feature extraction."""
        extractor = PhysiologicalFeatures(
            signal_type='emg',
            sampling_rate=self.sample_rate,
            window_size=256,
            overlap=0.5
        )
        
        features = extractor.extract(self.emg_signal)
        
        # Check if features were extracted
        self.assertGreater(len(features), 0)

    def test_ppg_features(self):
        """Test PPG feature extraction."""
        extractor = PhysiologicalFeatures(
            signal_type='ppg',
            sampling_rate=self.sample_rate,
            window_size=256,
            overlap=0.5
        )
        
        features = extractor.extract(self.ppg_signal)
        
        # Check if features were extracted
        self.assertGreater(len(features), 0)
        
        # Pulse rate should be close to the signal frequency (1.2 Hz * 60 = 72 bpm)
        if 'pulse_rate' in features:
            mean_pr = np.mean(features['pulse_rate'])
            self.assertAlmostEqual(mean_pr, 72, delta=15)

    def test_gsr_features(self):
        """Test GSR feature extraction."""
        extractor = PhysiologicalFeatures(
            signal_type='gsr',
            sampling_rate=self.sample_rate,
            window_size=256,
            overlap=0.5
        )
        
        features = extractor.extract(self.gsr_signal)
        
        # Check if features were extracted
        self.assertGreater(len(features), 0)

    def test_resp_features(self):
        """Test respiratory feature extraction."""
        extractor = PhysiologicalFeatures(
            signal_type='resp',
            sampling_rate=self.sample_rate,
            window_size=512,  # Larger window for better frequency resolution
            overlap=0.5
        )
        
        features = extractor.extract(self.resp_signal)
        
        # Check if features were extracted
        self.assertGreater(len(features), 0)
        
        # Respiratory rate should be close to signal frequency (0.25 Hz * 60 = 15 bpm)
        if 'respiratory_rate' in features:
            mean_rr = np.mean(features['respiratory_rate'])
            self.assertAlmostEqual(mean_rr, 15, delta=5)

    def test_multi_channel(self):
        """Test feature extraction with multi-channel signals."""
        # Create a multi-channel EEG (3 channels)
        np.random.seed(42)
        t = np.linspace(0, 5, 500)
        multi_eeg = np.column_stack([
            np.sin(2 * np.pi * 10 * t) + 0.1 * np.random.randn(len(t)),  # Alpha (10 Hz)
            np.sin(2 * np.pi * 20 * t) + 0.1 * np.random.randn(len(t)),  # Beta (20 Hz)
            np.sin(2 * np.pi * 5 * t) + 0.1 * np.random.randn(len(t))    # Theta (5 Hz)
        ])
        
        extractor = PhysiologicalFeatures(
            signal_type='eeg',
            sampling_rate=100,
            window_size=256,
            overlap=0.5
        )
        
        features = extractor.extract(multi_eeg)
        
        # Check dimensions - should have features for each channel
        for feature_name, feature_array in features.items():
            if feature_array.ndim > 1:  # Skip scalar features
                self.assertEqual(feature_array.shape[1], 3)  # 3 channels

if __name__ == '__main__':
    unittest.main() 