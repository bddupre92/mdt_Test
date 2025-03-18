"""
Test Cases for Pattern Recognition Components.

This module provides test cases for validating the theoretical correctness
of pattern recognition components using synthetic data generators.
"""

import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Any

from tests.theory.validation.test_harness import TestCase, CoreLayerHarness
from tests.theory.validation.synthetic_generators.patient_generators import (
    PatientGenerator, LongitudinalDataGenerator
)
from tests.theory.validation.synthetic_generators.signal_generators import (
    ECGGenerator, EEGGenerator, SkinConductanceGenerator
)
from core.theory.pattern_recognition.feature_extraction import FeatureExtractor
from core.theory.temporal_modeling.spectral_analysis import SpectralAnalyzer
from core.theory.pattern_recognition.pattern_classifier import PatternClassifier

class TestFeatureExtraction:
    """Test cases for feature extraction theoretical components."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.extractor = FeatureExtractor()
        self.spectral = SpectralAnalyzer()
        
        # Create synthetic data generators
        self.ecg_gen = ECGGenerator(sampling_rate=250.0)
        self.eeg_gen = EEGGenerator(sampling_rate=250.0)
        self.sc_gen = SkinConductanceGenerator(sampling_rate=128.0)
    
    def test_time_domain_features(self):
        """Test extraction of time-domain features from physiological signals."""
        # Generate synthetic ECG with known properties
        duration = 300  # 5 minutes
        stress_events = [
            (60, 0.3),   # Mild stress at 1min
            (180, 0.8)   # Strong stress at 3min
        ]
        
        time, ecg_data = self.ecg_gen.generate(duration, stress_level=0.0)
        
        # Create test case
        test_case = TestCase(
            name="ecg_time_features",
            inputs={
                "signal": ecg_data['ecg'],
                "sampling_rate": 250.0,
                "signal_type": "ecg"
            },
            expected_outputs={
                "mean_hr": 70.0,  # Baseline HR
                "hrv_features": {
                    "sdnn": (0.03, 0.15),    # Normal HRV range
                    "rmssd": (0.02, 0.10),   # Normal RMSSD range
                    "pnn50": (0.1, 0.5)      # Normal pNN50 range
                },
                "qrs_features": {
                    "amplitude": (0.8, 1.5),  # Normal QRS amplitude
                    "duration": (0.06, 0.12)  # Normal QRS duration in seconds
                }
            },
            tolerance={
                "mean_hr": 5.0,      # ±5 BPM
                "hrv_sdnn": 0.02,    # ±0.02 tolerance
                "hrv_rmssd": 0.02,   # ±0.02 tolerance
                "qrs_amp": 0.2,      # ±0.2 mV
                "qrs_dur": 0.02      # ±20ms
            },
            metadata={
                "description": "Validate ECG time-domain feature extraction",
                "stress_events": stress_events,
                "true_hr": ecg_data['hr']
            }
        )
        
        # Create validation function
        def validate_time_features(
            signal: np.ndarray,
            sampling_rate: float,
            signal_type: str
        ) -> Dict[str, Any]:
            # Extract time domain features
            features = self.extractor.extract_time_features(
                signal,
                sampling_rate=sampling_rate,
                signal_type=signal_type
            )
            
            # Validate feature ranges
            hrv_features = features['hrv']
            qrs_features = features['qrs']
            
            return {
                "mean_hr": features['mean_hr'],
                "hrv_features": {
                    "sdnn": hrv_features['sdnn'],
                    "rmssd": hrv_features['rmssd'],
                    "pnn50": hrv_features['pnn50']
                },
                "qrs_features": {
                    "amplitude": qrs_features['amplitude'],
                    "duration": qrs_features['duration']
                }
            }
        
        # Create harness and validate
        harness = CoreLayerHarness("time_domain_features", validate_time_features)
        harness.add_test_case(test_case)
        results = harness.run_all()
        
        return results
    
    def test_frequency_domain_features(self):
        """Test extraction of frequency-domain features from physiological signals."""
        # Generate synthetic EEG with known frequency components
        duration = 60  # seconds
        band_weights = {
            'delta': 0.8,  # Strong delta (sleep)
            'theta': 0.5,  # Moderate theta
            'alpha': 1.0,  # Strong alpha (relaxation)
            'beta': 0.3,   # Weak beta
            'gamma': 0.1   # Very weak gamma
        }
        
        time, eeg_data = self.eeg_gen.generate(duration, band_weights)
        
        # Create test case
        test_case = TestCase(
            name="eeg_frequency_features",
            inputs={
                "signal": eeg_data['eeg'],
                "sampling_rate": 250.0,
                "signal_type": "eeg"
            },
            expected_outputs={
                "band_powers": {
                    "delta": (0.7, 0.9),
                    "theta": (0.4, 0.6),
                    "alpha": (0.9, 1.1),
                    "beta": (0.2, 0.4),
                    "gamma": (0.05, 0.15)
                },
                "peak_frequencies": {
                    "alpha": (9.5, 10.5),  # Expected ~10 Hz
                    "beta": (19.5, 20.5)   # Expected ~20 Hz
                },
                "spectral_edge": (30.0, 35.0)  # 95% of power below this frequency
            },
            tolerance={
                "band_powers": 0.1,      # ±10% power
                "peak_freq": 1.0,        # ±1 Hz
                "spectral_edge": 5.0     # ±5 Hz
            },
            metadata={
                "description": "Validate EEG frequency-domain feature extraction",
                "true_weights": band_weights
            }
        )
        
        # Create validation function
        def validate_frequency_features(
            signal: np.ndarray,
            sampling_rate: float,
            signal_type: str
        ) -> Dict[str, Any]:
            # Extract frequency domain features
            features = self.extractor.extract_frequency_features(
                signal,
                sampling_rate=sampling_rate,
                signal_type=signal_type
            )
            
            # Get spectral properties
            spectral = self.spectral.analyze(
                signal,
                sampling_rate=sampling_rate,
                method="welch"
            )
            
            return {
                "band_powers": features['band_powers'],
                "peak_frequencies": features['peak_frequencies'],
                "spectral_edge": spectral['spectral_edge_95']
            }
        
        # Create harness and validate
        harness = CoreLayerHarness("frequency_domain_features", validate_frequency_features)
        harness.add_test_case(test_case)
        results = harness.run_all()
        
        return results
    
    def test_statistical_features(self):
        """Test extraction of statistical features from physiological signals."""
        # Generate synthetic skin conductance data with stress responses
        duration = 300  # 5 minutes
        stress_events = [
            (60, 0.5),   # Moderate stress at 1min
            (180, 0.9),  # High stress at 3min
            (240, 0.3)   # Mild stress at 4min
        ]
        
        time, sc_data = self.sc_gen.generate(duration, stress_events)
        
        # Create test case
        test_case = TestCase(
            name="sc_statistical_features",
            inputs={
                "signal": sc_data['sc'],
                "sampling_rate": 128.0,
                "signal_type": "gsr"
            },
            expected_outputs={
                "basic_stats": {
                    "mean": (1.5, 3.0),      # Expected SC range
                    "std": (0.2, 0.8),       # Expected variability
                    "skewness": (0.5, 2.0),  # Right-skewed due to stress responses
                    "kurtosis": (2.0, 5.0)   # Peaked due to stress events
                },
                "complexity": {
                    "sample_entropy": (0.5, 1.5),
                    "approximate_entropy": (0.6, 1.6)
                },
                "stationarity": {
                    "adf_statistic": (-4.0, -2.0),  # Non-stationary expected
                    "is_stationary": False
                }
            },
            tolerance={
                "mean": 0.5,        # ±0.5 µS
                "std": 0.2,         # ±0.2 µS
                "entropy": 0.3,     # ±0.3 entropy units
                "adf": 1.0         # ±1.0 statistic units
            },
            metadata={
                "description": "Validate statistical feature extraction",
                "stress_events": stress_events,
                "baseline": sc_data['tonic']
            }
        )
        
        # Create validation function
        def validate_statistical_features(
            signal: np.ndarray,
            sampling_rate: float,
            signal_type: str
        ) -> Dict[str, Any]:
            # Extract statistical features
            features = self.extractor.extract_statistical_features(
                signal,
                sampling_rate=sampling_rate,
                signal_type=signal_type
            )
            
            return {
                "basic_stats": {
                    "mean": features['mean'],
                    "std": features['std'],
                    "skewness": features['skewness'],
                    "kurtosis": features['kurtosis']
                },
                "complexity": {
                    "sample_entropy": features['sample_entropy'],
                    "approximate_entropy": features['approximate_entropy']
                },
                "stationarity": {
                    "adf_statistic": features['adf_statistic'],
                    "is_stationary": features['is_stationary']
                }
            }
        
        # Create harness and validate
        harness = CoreLayerHarness("statistical_features", validate_statistical_features)
        harness.add_test_case(test_case)
        results = harness.run_all()
        
        return results 

class TestPatternClassification:
    """Test cases for pattern classification theoretical components."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.classifier = PatternClassifier()
        self.patient_gen = PatientGenerator(num_patients=100)  # Large enough for training
        self.feature_extractor = FeatureExtractor()
    
    def test_binary_classification(self):
        """Test binary classification for migraine prediction."""
        # Generate synthetic patient data with known migraine patterns
        duration_days = 30
        data_gen = LongitudinalDataGenerator(self.patient_gen.generate_profile())
        
        # Generate training data with balanced classes
        train_data = data_gen.generate(
            duration_days=duration_days,
            include_physiological=True,
            include_environmental=True
        )
        
        # Extract features and prepare labels
        X_train = self.feature_extractor.extract_all_features(train_data)
        y_train = train_data['labels']['migraine_onset']  # Binary labels
        
        # Create test case
        test_case = TestCase(
            name="binary_migraine_prediction",
            inputs={
                "features": X_train,
                "labels": y_train,
                "validation_split": 0.2,
                "random_seed": 42
            },
            expected_outputs={
                "accuracy": 0.75,        # Minimum expected accuracy
                "precision": 0.70,       # Minimum expected precision
                "recall": 0.70,          # Minimum expected recall
                "f1_score": 0.70,        # Minimum expected F1
                "roc_auc": 0.80         # Minimum expected AUC-ROC
            },
            tolerance={
                "metrics": 0.05          # ±5% tolerance for all metrics
            },
            metadata={
                "description": "Validate binary classification performance",
                "feature_names": list(X_train.columns),
                "class_distribution": np.bincount(y_train)
            }
        )
        
        # Create validation function
        def validate_binary_classification(
            features: np.ndarray,
            labels: np.ndarray,
            validation_split: float,
            random_seed: int
        ) -> Dict[str, float]:
            # Train and evaluate classifier
            results = self.classifier.train_evaluate(
                features,
                labels,
                validation_split=validation_split,
                random_seed=random_seed
            )
            
            return {
                "accuracy": results['accuracy'],
                "precision": results['precision'],
                "recall": results['recall'],
                "f1_score": results['f1_score'],
                "roc_auc": results['roc_auc']
            }
        
        # Create harness and validate
        harness = CoreLayerHarness("binary_classification", validate_binary_classification)
        harness.add_test_case(test_case)
        results = harness.run_all()
        
        return results
    
    def test_ensemble_methods(self):
        """Test ensemble classification methods for robust prediction."""
        # Generate synthetic data with complex patterns
        duration_days = 60  # Longer duration for more complex patterns
        profiles = [
            self.patient_gen.generate_profile(complexity='low'),
            self.patient_gen.generate_profile(complexity='medium'),
            self.patient_gen.generate_profile(complexity='high')
        ]
        
        # Generate data for different patient profiles
        all_data = []
        for profile in profiles:
            data_gen = LongitudinalDataGenerator(profile)
            patient_data = data_gen.generate(
                duration_days=duration_days,
                include_physiological=True,
                include_environmental=True
            )
            all_data.append(patient_data)
        
        # Combine and prepare data
        X_combined = np.vstack([
            self.feature_extractor.extract_all_features(data)
            for data in all_data
        ])
        y_combined = np.concatenate([
            data['labels']['migraine_onset']
            for data in all_data
        ])
        
        # Create test case
        test_case = TestCase(
            name="ensemble_classification",
            inputs={
                "features": X_combined,
                "labels": y_combined,
                "n_estimators": 10,
                "validation_split": 0.2
            },
            expected_outputs={
                "ensemble_accuracy": 0.80,     # Expected ensemble accuracy
                "diversity_score": 0.3,        # Expected prediction diversity
                "reliability_score": 0.75,     # Expected reliability
                "individual_performances": {    # Expected base classifier performance
                    "mean": 0.75,
                    "std": 0.05
                }
            },
            tolerance={
                "accuracy": 0.05,      # ±5% tolerance
                "diversity": 0.1,      # ±0.1 tolerance
                "reliability": 0.05    # ±5% tolerance
            },
            metadata={
                "description": "Validate ensemble classification methods",
                "profile_complexities": ["low", "medium", "high"],
                "ensemble_type": "heterogeneous"
            }
        )
        
        # Create validation function
        def validate_ensemble(
            features: np.ndarray,
            labels: np.ndarray,
            n_estimators: int,
            validation_split: float
        ) -> Dict[str, Any]:
            # Train and evaluate ensemble
            results = self.classifier.train_evaluate_ensemble(
                features,
                labels,
                n_estimators=n_estimators,
                validation_split=validation_split
            )
            
            return {
                "ensemble_accuracy": results['ensemble_accuracy'],
                "diversity_score": results['diversity_score'],
                "reliability_score": results['reliability_score'],
                "individual_performances": {
                    "mean": np.mean(results['individual_accuracies']),
                    "std": np.std(results['individual_accuracies'])
                }
            }
        
        # Create harness and validate
        harness = CoreLayerHarness("ensemble_classification", validate_ensemble)
        harness.add_test_case(test_case)
        results = harness.run_all()
        
        return results
    
    def test_probabilistic_classification(self):
        """Test probabilistic classification with uncertainty estimation."""
        # Generate synthetic data with known uncertainty regions
        duration_days = 30
        profile = self.patient_gen.generate_profile(
            uncertainty_regions=[
                (5, 7),    # Days 5-7: High uncertainty
                (15, 17),  # Days 15-17: High uncertainty
                (25, 27)   # Days 25-27: High uncertainty
            ]
        )
        
        data_gen = LongitudinalDataGenerator(profile)
        train_data = data_gen.generate(
            duration_days=duration_days,
            include_physiological=True,
            include_environmental=True
        )
        
        # Extract features and prepare data
        X_train = self.feature_extractor.extract_all_features(train_data)
        y_train = train_data['labels']['migraine_onset']
        
        # Create test case
        test_case = TestCase(
            name="probabilistic_classification",
            inputs={
                "features": X_train,
                "labels": y_train,
                "uncertainty_threshold": 0.2,  # Threshold for high uncertainty
                "calibration_split": 0.2
            },
            expected_outputs={
                "calibration_score": 0.85,    # Expected calibration
                "uncertainty_detection": {
                    "precision": 0.7,         # Uncertainty detection precision
                    "recall": 0.7            # Uncertainty detection recall
                },
                "confidence_metrics": {
                    "mean_confidence": (0.6, 0.9),  # Expected confidence range
                    "confidence_calibration": 0.8   # Expected calibration
                },
                "reliability_diagram": {
                    "max_deviation": 0.15     # Maximum calibration error
                }
            },
            tolerance={
                "calibration": 0.1,     # ±10% tolerance
                "uncertainty": 0.1,     # ±10% tolerance
                "confidence": 0.1      # ±10% tolerance
            },
            metadata={
                "description": "Validate probabilistic classification",
                "uncertainty_regions": profile.uncertainty_regions,
                "calibration_method": "isotonic"
            }
        )
        
        # Create validation function
        def validate_probabilistic(
            features: np.ndarray,
            labels: np.ndarray,
            uncertainty_threshold: float,
            calibration_split: float
        ) -> Dict[str, Any]:
            # Train and evaluate probabilistic classifier
            results = self.classifier.train_evaluate_probabilistic(
                features,
                labels,
                uncertainty_threshold=uncertainty_threshold,
                calibration_split=calibration_split
            )
            
            return {
                "calibration_score": results['calibration_score'],
                "uncertainty_detection": {
                    "precision": results['uncertainty_precision'],
                    "recall": results['uncertainty_recall']
                },
                "confidence_metrics": {
                    "mean_confidence": results['mean_confidence'],
                    "confidence_calibration": results['confidence_calibration']
                },
                "reliability_diagram": {
                    "max_deviation": results['max_calibration_error']
                }
            }
        
        # Create harness and validate
        harness = CoreLayerHarness("probabilistic_classification", validate_probabilistic)
        harness.add_test_case(test_case)
        results = harness.run_all()
        
        return results 