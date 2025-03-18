"""
Test Cases for Physiological Adapter Components.

This module provides test cases for validating the theoretical correctness
of physiological signal adapters using synthetic data generators.
"""

import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Any

from tests.theory.validation.test_harness import TestCase, CoreLayerHarness
from tests.theory.validation.synthetic_generators.signal_generators import (
    ECGGenerator, EEGGenerator, SkinConductanceGenerator
)
from core.theory.migraine_adaptation.physiological_adapters import (
    ECGAdapter,
    EEGAdapter,
    GSRAdapter,
    SignalQualityAnalyzer
)

class TestECGAdapter:
    """Test cases for ECG signal adaptation."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.adapter = ECGAdapter()
        self.generator = ECGGenerator(sampling_rate=250.0)
        self.quality_analyzer = SignalQualityAnalyzer()
    
    def test_hrv_feature_extraction(self):
        """Test extraction of HRV features from ECG signals."""
        # Generate synthetic ECG with known HRV characteristics
        duration = 300  # 5 minutes (standard for HRV)
        stress_levels = [
            (0, 60, 0.1),    # First minute: low stress
            (60, 180, 0.6),  # 2-3 minutes: moderate stress
            (180, 300, 0.2)  # 4-5 minutes: mild stress
        ]
        
        time, ecg_data = self.generator.generate(
            duration=duration,
            stress_profile=stress_levels,
            hrv_characteristics={
                "mean_hr": 75,
                "sdnn": 45,
                "rmssd": 35,
                "pnn50": 0.15,
                "lf_hf_ratio": 2.5
            }
        )
        
        # Create test case
        test_case = TestCase(
            name="hrv_extraction",
            inputs={
                "ecg_signal": ecg_data['ecg'],
                "sampling_rate": 250.0,
                "window_size": 300,  # 5-minute window
                "quality_threshold": 0.8
            },
            expected_outputs={
                "time_domain_features": {
                    "mean_hr": (70, 80),      # Expected range
                    "sdnn": (40, 50),
                    "rmssd": (30, 40),
                    "pnn50": (0.1, 0.2)
                },
                "frequency_domain_features": {
                    "lf_power": (0.4, 0.6),   # Normalized units
                    "hf_power": (0.2, 0.3),
                    "lf_hf_ratio": (2.0, 3.0)
                },
                "stress_indicators": {
                    "stress_index": (0.3, 0.5),
                    "parasympathetic_tone": (0.4, 0.6),
                    "sympathetic_tone": (0.5, 0.7)
                }
            },
            tolerance={
                "time_domain": 0.1,     # ±10% tolerance
                "frequency_domain": 0.1,
                "stress_metrics": 0.15   # ±15% tolerance for stress metrics
            },
            metadata={
                "description": "Validate HRV feature extraction",
                "true_stress_levels": stress_levels,
                "signal_quality": "high"
            }
        )
        
        # Create validation function
        def validate_hrv_features(
            ecg_signal: np.ndarray,
            sampling_rate: float,
            window_size: int,
            quality_threshold: float
        ) -> Dict[str, Any]:
            # Check signal quality
            quality_score = self.quality_analyzer.analyze_ecg_quality(
                signal=ecg_signal,
                sampling_rate=sampling_rate
            )
            
            if quality_score < quality_threshold:
                raise ValueError(f"ECG signal quality ({quality_score}) below threshold ({quality_threshold})")
            
            # Extract HRV features
            features = self.adapter.extract_hrv_features(
                signal=ecg_signal,
                sampling_rate=sampling_rate,
                window_size=window_size
            )
            
            # Calculate stress indicators
            stress_metrics = self.adapter.calculate_stress_metrics(
                hrv_features=features
            )
            
            return {
                "time_domain_features": {
                    "mean_hr": features['mean_hr'],
                    "sdnn": features['sdnn'],
                    "rmssd": features['rmssd'],
                    "pnn50": features['pnn50']
                },
                "frequency_domain_features": {
                    "lf_power": features['lf_power'],
                    "hf_power": features['hf_power'],
                    "lf_hf_ratio": features['lf_hf_ratio']
                },
                "stress_indicators": stress_metrics
            }
        
        # Create harness and validate
        harness = CoreLayerHarness("hrv_features", validate_hrv_features)
        harness.add_test_case(test_case)
        results = harness.run_all()
        
        return results
    
    def test_artifact_handling(self):
        """Test handling of common ECG artifacts."""
        # Generate synthetic ECG with various artifacts
        duration = 60  # 1 minute
        artifacts = [
            {
                "type": "baseline_wander",
                "start": 10,
                "duration": 5,
                "amplitude": 0.2
            },
            {
                "type": "muscle_noise",
                "start": 25,
                "duration": 3,
                "intensity": 0.3
            },
            {
                "type": "powerline_interference",
                "start": 40,
                "duration": 5,
                "frequency": 50,
                "amplitude": 0.1
            }
        ]
        
        time, ecg_data = self.generator.generate(
            duration=duration,
            artifacts=artifacts
        )
        
        # Create test case
        test_case = TestCase(
            name="artifact_handling",
            inputs={
                "ecg_signal": ecg_data['ecg'],
                "sampling_rate": 250.0,
                "artifact_types": ["baseline_wander", "muscle_noise", "powerline"],
                "correction_methods": {
                    "baseline_wander": "cubic_spline",
                    "muscle_noise": "wavelet_filter",
                    "powerline": "notch_filter"
                }
            },
            expected_outputs={
                "signal_improvement": {
                    "snr_improvement": 6.0,      # dB improvement
                    "baseline_stability": 0.9,    # Baseline stability score
                    "beat_detection_accuracy": 0.95  # QRS detection accuracy
                },
                "artifact_detection": {
                    "sensitivity": 0.9,
                    "specificity": 0.9,
                    "detection_latency": 0.2  # seconds
                },
                "correction_quality": {
                    "morphology_preservation": 0.9,
                    "feature_stability": 0.85,
                    "clinical_acceptability": 0.9
                }
            },
            tolerance={
                "improvement": 1.0,    # ±1 dB tolerance
                "detection": 0.1,      # ±10% tolerance
                "quality": 0.1         # ±10% tolerance
            },
            metadata={
                "description": "Validate ECG artifact handling",
                "true_artifacts": artifacts,
                "original_snr": ecg_data['snr']
            }
        )
        
        # Create validation function
        def validate_artifact_handling(
            ecg_signal: np.ndarray,
            sampling_rate: float,
            artifact_types: List[str],
            correction_methods: Dict[str, str]
        ) -> Dict[str, Any]:
            # Process signal and handle artifacts
            cleaned_signal = self.adapter.remove_artifacts(
                signal=ecg_signal,
                sampling_rate=sampling_rate,
                artifact_types=artifact_types,
                methods=correction_methods
            )
            
            # Calculate signal quality metrics
            quality_metrics = self.quality_analyzer.compute_quality_metrics(
                original=ecg_signal,
                cleaned=cleaned_signal,
                sampling_rate=sampling_rate
            )
            
            # Evaluate artifact detection
            detection_results = self.adapter.evaluate_artifact_detection(
                signal=ecg_signal,
                true_artifacts=test_case.metadata['true_artifacts']
            )
            
            # Assess correction quality
            correction_assessment = self.adapter.assess_correction_quality(
                original=ecg_signal,
                cleaned=cleaned_signal,
                sampling_rate=sampling_rate
            )
            
            return {
                "signal_improvement": {
                    "snr_improvement": quality_metrics['snr_improvement'],
                    "baseline_stability": quality_metrics['baseline_stability'],
                    "beat_detection_accuracy": quality_metrics['beat_detection_accuracy']
                },
                "artifact_detection": {
                    "sensitivity": detection_results['sensitivity'],
                    "specificity": detection_results['specificity'],
                    "detection_latency": detection_results['latency']
                },
                "correction_quality": correction_assessment
            }
        
        # Create harness and validate
        harness = CoreLayerHarness("artifact_handling", validate_artifact_handling)
        harness.add_test_case(test_case)
        results = harness.run_all()
        
        return results

class TestEEGAdapter:
    """Test cases for EEG signal adaptation."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.adapter = EEGAdapter()
        self.generator = EEGGenerator(sampling_rate=250.0)
        self.quality_analyzer = SignalQualityAnalyzer()
    
    def test_spectral_feature_extraction(self):
        """Test extraction of spectral features from EEG signals."""
        # Generate synthetic EEG with known spectral characteristics
        duration = 60  # 1 minute
        band_powers = {
            'delta': 0.3,  # 0.5-4 Hz
            'theta': 0.2,  # 4-8 Hz
            'alpha': 0.3,  # 8-13 Hz
            'beta': 0.15,  # 13-30 Hz
            'gamma': 0.05  # 30-100 Hz
        }
        
        time, eeg_data = self.generator.generate(
            duration=duration,
            band_powers=band_powers,
            cognitive_state="relaxed"
        )
        
        # Create test case
        test_case = TestCase(
            name="eeg_spectral_features",
            inputs={
                "eeg_signal": eeg_data['eeg'],
                "sampling_rate": 250.0,
                "window_size": 4,    # 4-second windows
                "overlap": 0.5,      # 50% overlap
                "method": "welch"
            },
            expected_outputs={
                "band_powers": {
                    "delta": (0.25, 0.35),
                    "theta": (0.15, 0.25),
                    "alpha": (0.25, 0.35),
                    "beta": (0.1, 0.2),
                    "gamma": (0.03, 0.07)
                },
                "spectral_features": {
                    "spectral_edge": (30, 40),  # Hz
                    "median_frequency": (8, 12),
                    "spectral_entropy": (0.7, 0.9)
                },
                "cognitive_indicators": {
                    "alertness": (0.3, 0.5),
                    "relaxation": (0.7, 0.9),
                    "concentration": (0.4, 0.6)
                }
            },
            tolerance={
                "powers": 0.05,      # ±5% tolerance for band powers
                "features": 0.1,     # ±10% tolerance for spectral features
                "indicators": 0.15   # ±15% tolerance for cognitive indicators
            },
            metadata={
                "description": "Validate EEG spectral feature extraction",
                "true_band_powers": band_powers,
                "cognitive_state": "relaxed"
            }
        )
        
        # Create validation function
        def validate_spectral_features(
            eeg_signal: np.ndarray,
            sampling_rate: float,
            window_size: int,
            overlap: float,
            method: str
        ) -> Dict[str, Any]:
            # Extract spectral features
            features = self.adapter.extract_spectral_features(
                signal=eeg_signal,
                sampling_rate=sampling_rate,
                window_size=window_size,
                overlap=overlap,
                method=method
            )
            
            # Calculate cognitive state indicators
            cognitive_state = self.adapter.assess_cognitive_state(
                spectral_features=features
            )
            
            return {
                "band_powers": features['band_powers'],
                "spectral_features": {
                    "spectral_edge": features['spectral_edge'],
                    "median_frequency": features['median_frequency'],
                    "spectral_entropy": features['spectral_entropy']
                },
                "cognitive_indicators": cognitive_state
            }
        
        # Create harness and validate
        harness = CoreLayerHarness("spectral_features", validate_spectral_features)
        harness.add_test_case(test_case)
        results = harness.run_all()
        
        return results

class TestGSRAdapter:
    """Test cases for Galvanic Skin Response (GSR) signal adaptation."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.adapter = GSRAdapter()
        self.generator = SkinConductanceGenerator(sampling_rate=128.0)
        self.quality_analyzer = SignalQualityAnalyzer()
    
    def test_stress_response_detection(self):
        """Test detection of stress responses in GSR signals."""
        # Generate synthetic GSR with known stress responses
        duration = 300  # 5 minutes
        stress_events = [
            (60, 0.5, 10),   # At 1min, moderate response, 10s duration
            (180, 0.8, 15),  # At 3min, strong response, 15s duration
            (240, 0.3, 8)    # At 4min, mild response, 8s duration
        ]
        
        time, gsr_data = self.generator.generate(
            duration=duration,
            stress_events=stress_events,
            noise_level=0.1
        )
        
        # Create test case
        test_case = TestCase(
            name="gsr_stress_detection",
            inputs={
                "gsr_signal": gsr_data['gsr'],
                "sampling_rate": 128.0,
                "detection_params": {
                    "min_amplitude": 0.2,
                    "min_duration": 5,
                    "max_rise_time": 4
                }
            },
            expected_outputs={
                "event_detection": {
                    "precision": 0.9,
                    "recall": 0.9,
                    "f1_score": 0.9
                },
                "response_characteristics": {
                    "amplitude_error": 0.1,    # Maximum relative error
                    "timing_error": 1.0,       # Maximum error in seconds
                    "duration_error": 1.0      # Maximum error in seconds
                },
                "stress_metrics": {
                    "mean_response": (0.4, 0.6),
                    "response_frequency": (0.5, 0.7),
                    "recovery_rate": (0.3, 0.5)
                }
            },
            tolerance={
                "detection": 0.1,    # ±10% tolerance
                "characteristics": 0.2,  # ±20% tolerance
                "metrics": 0.15      # ±15% tolerance
            },
            metadata={
                "description": "Validate GSR stress response detection",
                "true_events": stress_events,
                "baseline_scl": gsr_data['tonic']
            }
        )
        
        # Create validation function
        def validate_stress_detection(
            gsr_signal: np.ndarray,
            sampling_rate: float,
            detection_params: Dict[str, float]
        ) -> Dict[str, Any]:
            # Detect stress responses
            detected_responses = self.adapter.detect_stress_responses(
                signal=gsr_signal,
                sampling_rate=sampling_rate,
                **detection_params
            )
            
            # Evaluate detection accuracy
            detection_metrics = self.adapter.evaluate_detection(
                detected=detected_responses,
                true_events=test_case.metadata['true_events'],
                sampling_rate=sampling_rate
            )
            
            # Analyze response characteristics
            response_analysis = self.adapter.analyze_responses(
                signal=gsr_signal,
                responses=detected_responses,
                sampling_rate=sampling_rate
            )
            
            # Calculate stress metrics
            stress_metrics = self.adapter.calculate_stress_metrics(
                signal=gsr_signal,
                responses=detected_responses,
                baseline=test_case.metadata['baseline_scl']
            )
            
            return {
                "event_detection": {
                    "precision": detection_metrics['precision'],
                    "recall": detection_metrics['recall'],
                    "f1_score": detection_metrics['f1_score']
                },
                "response_characteristics": {
                    "amplitude_error": response_analysis['amplitude_error'],
                    "timing_error": response_analysis['timing_error'],
                    "duration_error": response_analysis['duration_error']
                },
                "stress_metrics": stress_metrics
            }
        
        # Create harness and validate
        harness = CoreLayerHarness("stress_detection", validate_stress_detection)
        harness.add_test_case(test_case)
        results = harness.run_all()
        
        return results 