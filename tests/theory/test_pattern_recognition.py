"""Tests for the pattern recognition framework.

This module contains tests for both feature extraction and pattern classification components.
"""

import unittest
import numpy as np
from scipy import signal as scipy_signal
from sklearn.datasets import make_classification
from core.theory.pattern_recognition import (
    TimeDomainFeatures,
    FrequencyDomainFeatures,
    StatisticalFeatures,
    PhysiologicalFeatures,
    BinaryClassifier,
    EnsembleClassifier,
    ProbabilisticClassifier
)

def create_sample_signal():
    """Generate a sample signal for testing."""
    t = np.linspace(0, 10, 1000)
    # Create a complex signal with multiple frequency components
    signal = (
        np.sin(2 * np.pi * 1 * t) +  # 1 Hz component
        0.5 * np.sin(2 * np.pi * 10 * t) +  # 10 Hz component
        0.25 * np.sin(2 * np.pi * 20 * t)  # 20 Hz component
    )
    # Ensure the signal is centered around zero
    return signal - np.mean(signal)

def create_ecg_signal():
    """Generate a synthetic ECG-like signal."""
    t = np.linspace(0, 10, 1000)
    # Create synthetic R-peaks with higher amplitude
    peaks = 2 * scipy_signal.gausspulse(t - 5, fc=2)
    # Add some baseline variation
    baseline = 0.2 * np.sin(2 * np.pi * 0.1 * t)
    return peaks + baseline

class TestTimeDomainFeatures(unittest.TestCase):
    """Test suite for TimeDomainFeatures class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.sample_signal = create_sample_signal()
    
    def test_initialization(self):
        """Test initialization with default and custom parameters."""
        extractor = TimeDomainFeatures()
        self.assertEqual(extractor.window_size, 256)
        self.assertEqual(extractor.overlap, 0.5)
        
        extractor = TimeDomainFeatures(window_size=512, overlap=0.75)
        self.assertEqual(extractor.window_size, 512)
        self.assertEqual(extractor.overlap, 0.75)
    
    def test_input_validation(self):
        """Test input validation."""
        extractor = TimeDomainFeatures()
        
        # Valid input
        self.assertTrue(extractor.validate_input(self.sample_signal))
        
        # Invalid inputs
        with self.assertRaises(ValueError):
            extractor.validate_input(self.sample_signal.reshape(-1, 1))  # 2D array
        with self.assertRaises(ValueError):
            extractor.validate_input(self.sample_signal[:100])  # Too short
    
    def test_feature_extraction(self):
        """Test feature extraction."""
        extractor = TimeDomainFeatures(window_size=256, overlap=0.5)
        features = extractor.extract(self.sample_signal)
        
        # Check feature names
        expected_features = {'peak_to_peak', 'zero_crossings', 'mean_abs_value',
                           'waveform_length', 'slope_changes'}
        self.assertEqual(set(features.keys()), expected_features)
        
        # Check feature dimensions
        n_windows = (len(self.sample_signal) - 256) // 128 + 1  # Based on window_size and overlap
        for feature in features.values():
            self.assertEqual(len(feature), n_windows)

class TestFrequencyDomainFeatures(unittest.TestCase):
    """Test suite for FrequencyDomainFeatures class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.sample_signal = create_sample_signal()
    
    def test_initialization(self):
        """Test initialization with default and custom parameters."""
        extractor = FrequencyDomainFeatures()
        self.assertEqual(extractor.fs, 100.0)
        self.assertEqual(extractor.window_size, 256)
        self.assertEqual(extractor.overlap, 0.5)
        
        extractor = FrequencyDomainFeatures(sampling_rate=200.0, window_size=512)
        self.assertEqual(extractor.fs, 200.0)
        self.assertEqual(extractor.window_size, 512)
    
    def test_feature_extraction(self):
        """Test feature extraction."""
        extractor = FrequencyDomainFeatures(sampling_rate=100.0)
        features = extractor.extract(self.sample_signal)
        
        # Check feature names
        expected_features = {'spectral_centroid', 'spectral_bandwidth',
                           'dominant_frequency', 'delta_power', 'theta_power',
                           'alpha_power', 'beta_power', 'gamma_power'}
        self.assertEqual(set(features.keys()), expected_features)
        
        # Check if dominant frequencies are detected
        # Our sample signal has components at 1, 10, and 20 Hz
        dominant_freqs = features['dominant_frequency']
        self.assertTrue(np.any(np.abs(dominant_freqs - 1.0) < 1.0))  # Allow 1 Hz tolerance

class TestStatisticalFeatures(unittest.TestCase):
    """Test suite for StatisticalFeatures class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.sample_signal = create_sample_signal()
    
    def test_initialization(self):
        """Test initialization."""
        extractor = StatisticalFeatures()
        self.assertEqual(extractor.window_size, 256)
        self.assertEqual(extractor.overlap, 0.5)
    
    def test_feature_extraction(self):
        """Test feature extraction."""
        extractor = StatisticalFeatures()
        features = extractor.extract(self.sample_signal)
        
        # Check feature names
        expected_features = {'mean', 'std', 'skewness', 'kurtosis',
                           'median', 'iqr', 'entropy'}
        self.assertEqual(set(features.keys()), expected_features)
        
        # Basic statistical checks
        self.assertTrue(np.allclose(features['mean'], 0.0, atol=0.1))  # Signal should be roughly centered
        self.assertTrue(np.all(features['std'] > 0))  # Standard deviation should be positive

class TestPhysiologicalFeatures(unittest.TestCase):
    """Test suite for PhysiologicalFeatures class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.sample_signal = create_sample_signal()
        self.ecg_signal = create_ecg_signal()
    
    def test_initialization(self):
        """Test initialization."""
        extractor = PhysiologicalFeatures('ecg')
        self.assertEqual(extractor.signal_type, 'ecg')
        self.assertEqual(extractor.fs, 100.0)
        
        # Test invalid signal type
        with self.assertRaises(ValueError):
            PhysiologicalFeatures('invalid_type')
    
    def test_ecg_features(self):
        """Test ECG feature extraction."""
        extractor = PhysiologicalFeatures('ecg')
        features = extractor.extract(self.ecg_signal)
        
        # Check feature names
        expected_features = {'heart_rate', 'rr_intervals', 'hrv_sdnn', 'hrv_rmssd'}
        self.assertEqual(set(features.keys()), expected_features)
        
        # Basic physiological checks
        self.assertTrue(np.all(features['heart_rate'] > 0))  # Heart rate should be positive
        self.assertTrue(np.all(features['hrv_sdnn'] >= 0))  # HRV measures should be non-negative
    
    def test_eeg_features(self):
        """Test EEG feature extraction."""
        extractor = PhysiologicalFeatures('eeg')
        features = extractor.extract(self.sample_signal)
        
        # EEG features should include frequency bands
        expected_features = {'delta_power', 'theta_power', 'alpha_power',
                           'beta_power', 'gamma_power'}
        self.assertTrue(all(name in features for name in expected_features))
    
    def test_feature_extraction_all_types(self):
        """Test feature extraction for all signal types."""
        signal_types = ['ecg', 'eeg', 'emg', 'ppg', 'gsr', 'resp']
        
        for sig_type in signal_types:
            extractor = PhysiologicalFeatures(sig_type)
            features = extractor.extract(self.sample_signal)
            
            # All feature extractors should return a dictionary
            self.assertIsInstance(features, dict)
            # All features should be numpy arrays
            self.assertTrue(all(isinstance(v, np.ndarray) for v in features.values()))
            # All features should have the same length
            lengths = [len(v) for v in features.values()]
            self.assertEqual(len(set(lengths)), 1)  # All lengths should be equal

class TestBinaryClassifier(unittest.TestCase):
    """Test suite for BinaryClassifier class."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Generate synthetic binary classification data
        X, y = make_classification(
            n_samples=100,
            n_features=20,
            n_informative=10,
            n_redundant=5,
            random_state=42
        )
        self.X = X
        self.y = y
    
    def test_initialization(self):
        """Test initialization with different classifier types."""
        # Test random forest classifier
        clf = BinaryClassifier('rf')
        self.assertEqual(clf.classifier_type, 'rf')
        
        # Test SVM classifier
        clf = BinaryClassifier('svm', kernel='linear')
        self.assertEqual(clf.classifier_type, 'svm')
        
        # Test invalid classifier type
        with self.assertRaises(ValueError):
            BinaryClassifier('invalid')
    
    def test_training_and_prediction(self):
        """Test model training and prediction."""
        clf = BinaryClassifier('rf')
        
        # Train the model
        clf.fit(self.X, self.y)
        
        # Make predictions
        y_pred = clf.predict(self.X)
        y_proba = clf.predict_proba(self.X)
        
        # Check predictions
        self.assertEqual(y_pred.shape, (len(self.X),))
        self.assertEqual(y_proba.shape, (len(self.X), 2))
        self.assertTrue(np.all((y_proba >= 0) & (y_proba <= 1)))
    
    def test_evaluation(self):
        """Test model evaluation."""
        clf = BinaryClassifier('rf')
        clf.fit(self.X, self.y)
        
        # Evaluate the model
        metrics = clf.evaluate(self.X, self.y, cv=3)
        
        # Check metrics
        self.assertIn('accuracy_mean', metrics)
        self.assertIn('accuracy_std', metrics)
        self.assertTrue(0 <= metrics['accuracy_mean'] <= 1)

class TestEnsembleClassifier(unittest.TestCase):
    """Test suite for EnsembleClassifier class."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Generate synthetic binary classification data
        X, y = make_classification(
            n_samples=100,
            n_features=20,
            n_informative=10,
            n_redundant=5,
            random_state=42
        )
        self.X = X
        self.y = y
    
    def test_initialization(self):
        """Test initialization with different configurations."""
        # Test default initialization
        clf = EnsembleClassifier()
        self.assertEqual(len(clf.base_classifiers), 3)
        self.assertTrue(np.allclose(clf.weights, [1/3, 1/3, 1/3]))
        
        # Test custom weights
        weights = [0.5, 0.3, 0.2]
        clf = EnsembleClassifier(weights=weights)
        self.assertTrue(np.allclose(clf.weights, weights))
        
        # Test invalid weights
        with self.assertRaises(ValueError):
            EnsembleClassifier(weights=[0.5, 0.5])  # Wrong length
        with self.assertRaises(ValueError):
            EnsembleClassifier(weights=[0.5, 0.5, 0.5])  # Don't sum to 1
    
    def test_training_and_prediction(self):
        """Test ensemble training and prediction."""
        clf = EnsembleClassifier()
        
        # Train the ensemble
        clf.fit(self.X, self.y)
        
        # Make predictions
        y_pred = clf.predict(self.X)
        y_proba = clf.predict_proba(self.X)
        
        # Check predictions
        self.assertEqual(y_pred.shape, (len(self.X),))
        self.assertEqual(y_proba.shape, (len(self.X), 2))
        self.assertTrue(np.all((y_proba >= 0) & (y_proba <= 1)))

class TestProbabilisticClassifier(unittest.TestCase):
    """Test suite for ProbabilisticClassifier class."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Generate synthetic binary classification data
        X, y = make_classification(
            n_samples=100,
            n_features=20,
            n_informative=10,
            n_redundant=5,
            random_state=42
        )
        self.X = X
        self.y = y
    
    def test_initialization(self):
        """Test initialization."""
        # Test default initialization
        clf = ProbabilisticClassifier()
        self.assertEqual(clf.n_bootstrap, 100)
        self.assertIsInstance(clf.base_classifier, BinaryClassifier)
        
        # Test custom initialization
        base_clf = BinaryClassifier('svm')
        clf = ProbabilisticClassifier(base_classifier=base_clf, n_bootstrap=50)
        self.assertEqual(clf.n_bootstrap, 50)
        self.assertEqual(clf.base_classifier, base_clf)
    
    def test_training_and_prediction(self):
        """Test probabilistic training and prediction."""
        clf = ProbabilisticClassifier(n_bootstrap=10)  # Use fewer bootstraps for testing
        
        # Train the model
        clf.fit(self.X, self.y)
        
        # Make predictions
        y_pred = clf.predict(self.X)
        y_proba = clf.predict_proba(self.X)
        
        # Check predictions
        self.assertEqual(y_pred.shape, (len(self.X),))
        self.assertEqual(y_proba.shape, (len(self.X), 2))
        self.assertTrue(np.all((y_proba >= 0) & (y_proba <= 1)))
        
        # Check uncertainty estimates
        uncertainty = clf.get_uncertainty()
        self.assertEqual(uncertainty.shape, (len(self.X), 2))
        self.assertTrue(np.all(uncertainty >= 0))
    
    def test_uncertainty_access(self):
        """Test uncertainty access without prediction."""
        clf = ProbabilisticClassifier()
        
        # Accessing uncertainty before prediction should raise error
        with self.assertRaises(RuntimeError):
            clf.get_uncertainty() 