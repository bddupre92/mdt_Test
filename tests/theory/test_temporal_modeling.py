"""
Tests for the Temporal Modeling components.

This module contains tests for the theoretical components related to
temporal modeling, including spectral analysis, state space models, 
and causal inference mechanisms for physiological time series.
"""

import unittest
import numpy as np
from typing import Dict, List, Any

from core.theory.temporal_modeling import SpectralAnalyzer


class TestSpectralAnalyzer(unittest.TestCase):
    """Tests for the SpectralAnalyzer class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.eeg_analyzer = SpectralAnalyzer("eeg", "Test EEG analyzer")
        self.hrv_analyzer = SpectralAnalyzer("hrv", "Test HRV analyzer")
        self.general_analyzer = SpectralAnalyzer("general", "Test general analyzer")
        
        # Generate synthetic time series for testing
        t = np.linspace(0, 10, 1000)
        
        # EEG-like signal with mixed frequency components
        alpha_component = 10 * np.sin(2 * np.pi * 10 * t)  # Alpha ~10 Hz
        beta_component = 5 * np.sin(2 * np.pi * 20 * t)    # Beta ~20 Hz
        noise = 2 * np.random.randn(len(t))
        self.eeg_signal = alpha_component + beta_component + noise
        
        # HRV-like signal with slower components
        vlf_component = 3 * np.sin(2 * np.pi * 0.01 * t)   # Very low freq
        lf_component = 2 * np.sin(2 * np.pi * 0.1 * t)     # Low freq
        hf_component = 1 * np.sin(2 * np.pi * 0.25 * t)    # High freq
        noise = 0.5 * np.random.randn(len(t))
        self.hrv_signal = vlf_component + lf_component + hf_component + noise
        
        # Set sampling rates
        self.eeg_sampling_rate = 100  # Hz
        self.hrv_sampling_rate = 4     # Hz
    
    def test_initialization(self):
        """Test initialization with different data types."""
        self.assertEqual(self.eeg_analyzer.data_type, "eeg")
        self.assertEqual(self.hrv_analyzer.data_type, "hrv")
        
        # Check if relevant bands are set correctly
        self.assertIn("alpha", self.eeg_analyzer.relevant_bands)
        self.assertIn("beta", self.eeg_analyzer.relevant_bands)
        self.assertIn("delta", self.eeg_analyzer.relevant_bands)
        self.assertIn("theta", self.eeg_analyzer.relevant_bands)
        self.assertIn("gamma", self.eeg_analyzer.relevant_bands)
        
        self.assertIn("ultra_low", self.hrv_analyzer.relevant_bands)
        self.assertIn("very_low", self.hrv_analyzer.relevant_bands)
        self.assertIn("low", self.hrv_analyzer.relevant_bands)
        self.assertIn("high", self.hrv_analyzer.relevant_bands)
        
        # General analyzer should have all bands
        self.assertEqual(len(self.general_analyzer.relevant_bands), 
                         len(self.general_analyzer.SPECTRAL_METRICS["bands"]))
    
    def test_analyze_fft(self):
        """Test FFT analysis method."""
        # Analyze EEG signal with FFT
        analysis = self.eeg_analyzer.analyze(self.eeg_signal, 
                                            self.eeg_sampling_rate, 
                                            method="fft")
        
        # Check if the analysis contains expected keys
        self.assertIn("frequencies", analysis)
        self.assertIn("power_spectrum", analysis)
        self.assertIn("band_powers", analysis)
        self.assertIn("entropy", analysis)
        self.assertIn("theoretical_insights", analysis)
        
        # Check if the frequencies array has correct length and range
        freqs = analysis["frequencies"]
        self.assertEqual(len(freqs), len(self.eeg_signal) // 2)
        self.assertLessEqual(freqs[-1], self.eeg_sampling_rate / 2)  # Nyquist frequency
        
        # Check if band powers were calculated
        self.assertIn("absolute", analysis["band_powers"])
        self.assertIn("relative", analysis["band_powers"])
        
        # Alpha power should be significant due to our synthetic signal
        self.assertIn("alpha", analysis["band_powers"]["relative"])
        
        # Entropy should be a valid value between 0 and 1
        self.assertGreaterEqual(analysis["entropy"]["spectral_entropy"], 0)
        self.assertLessEqual(analysis["entropy"]["spectral_entropy"], 1)
        
        # Should have at least one theoretical insight
        self.assertGreater(len(analysis["theoretical_insights"]), 0)
    
    def test_analyze_welch(self):
        """Test Welch's method for spectral analysis."""
        # Analyze HRV signal with Welch's method
        analysis = self.hrv_analyzer.analyze(self.hrv_signal, 
                                           self.hrv_sampling_rate, 
                                           method="welch",
                                           window_size=128)
        
        # Check if the analysis contains expected keys
        self.assertIn("frequencies", analysis)
        self.assertIn("power_spectrum", analysis)
        self.assertIn("method", analysis)
        self.assertEqual(analysis["method"], "welch")
        
        # Check if frequency resolution makes sense
        self.assertAlmostEqual(analysis["freq_resolution"], 
                              self.hrv_sampling_rate / 128, delta=0.01)
        
        # VLF and LF bands should be significant in our synthetic HRV signal
        self.assertIn("very_low", analysis["band_powers"]["relative"])
        self.assertIn("low", analysis["band_powers"]["relative"])
    
    def test_analyze_wavelet(self):
        """Test wavelet analysis method."""
        analysis = self.eeg_analyzer.analyze(self.eeg_signal, 
                                           self.eeg_sampling_rate, 
                                           method="wavelet")
        
        # Check if the analysis contains expected keys
        self.assertIn("frequencies", analysis)
        self.assertIn("power_spectrum", analysis)
        self.assertIn("wavelet_coefficients", analysis)
        self.assertIn("wavelet_scales", analysis)
        self.assertEqual(analysis["method"], "wavelet")
        
        # Frequencies should be in ascending order
        self.assertTrue(np.all(np.diff(analysis["frequencies"]) > 0))
        
        # Check wavelet coefficients dimensions
        self.assertEqual(analysis["wavelet_coefficients"].shape[1], len(self.eeg_signal))
    
    def test_compare_signals(self):
        """Test signal comparison functionality."""
        # Compare EEG and HRV signals
        signals = [self.eeg_signal, self.hrv_signal]
        labels = ["EEG", "HRV"]
        
        # Common sampling rate for comparison
        comparison_fs = 100
        eeg_signal = self.eeg_signal
        
        # Resample HRV to match EEG sampling rate
        t_orig = np.linspace(0, len(self.hrv_signal)/self.hrv_sampling_rate, len(self.hrv_signal))
        t_new = np.linspace(0, len(self.hrv_signal)/self.hrv_sampling_rate, len(self.hrv_signal)*comparison_fs//self.hrv_sampling_rate)
        hrv_signal = np.interp(t_new, t_orig, self.hrv_signal)
        
        comparison = self.general_analyzer.compare_signals(
            [eeg_signal, hrv_signal], 
            labels, 
            comparison_fs,
            method="welch"
        )
        
        # Check if comparison contains expected keys
        self.assertIn("individual_analyses", comparison)
        self.assertIn("band_power_comparison", comparison)
        self.assertIn("entropy_comparison", comparison)
        self.assertIn("coherence", comparison)
        self.assertIn("theoretical_implications", comparison)
        
        # Should have two individual analyses
        self.assertEqual(len(comparison["individual_analyses"]), 2)
        
        # Check if labels are correct
        self.assertEqual(comparison["individual_analyses"][0]["label"], "EEG")
        self.assertEqual(comparison["individual_analyses"][1]["label"], "HRV")
        
        # Coherence should be calculated
        coherence_key = "EEG_HRV"
        self.assertIn(coherence_key, comparison["coherence"])
        
        # Should have some theoretical implications
        self.assertGreater(len(comparison["theoretical_implications"]), 0)
    
    def test_entropy_calculation(self):
        """Test entropy calculation methods."""
        # Create signals with different complexity
        t = np.linspace(0, 10, 1000)
        
        # Simple signal (sine wave)
        simple_signal = 10 * np.sin(2 * np.pi * 1 * t)
        
        # Complex signal (multiple components + noise)
        complex_signal = 3 * np.sin(2 * np.pi * 1 * t) + \
                         2 * np.sin(2 * np.pi * 3.5 * t) + \
                         5 * np.sin(2 * np.pi * 7 * t) + \
                         2 * np.random.randn(len(t))
        
        # Analyze both signals
        simple_analysis = self.general_analyzer.analyze(simple_signal, 100, "fft")
        complex_analysis = self.general_analyzer.analyze(complex_signal, 100, "fft")
        
        # Complex signal should have higher spectral entropy
        simple_entropy = simple_analysis["entropy"]["spectral_entropy"]
        complex_entropy = complex_analysis["entropy"]["spectral_entropy"]
        
        self.assertGreater(complex_entropy, simple_entropy)
    
    def test_formal_definition(self):
        """Test retrieval of formal definition."""
        definition = self.eeg_analyzer.get_formal_definition()
        
        # Should be a substantial string
        self.assertIsInstance(definition, str)
        self.assertGreater(len(definition), 200)
        
        # Should contain key terms
        self.assertIn("fourier", definition.lower())
        self.assertIn("power spectral density", definition.lower())
        self.assertIn("wavelet", definition.lower())
        self.assertIn("entropy", definition.lower())


if __name__ == '__main__':
    unittest.main() 