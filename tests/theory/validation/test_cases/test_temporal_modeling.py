"""
Test Cases for Temporal Modeling Components.

This module provides test cases for validating the theoretical correctness
of temporal modeling components using synthetic data generators.
"""

import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple

from tests.theory.validation.test_harness import TestCase, CoreLayerHarness
from tests.theory.validation.synthetic_generators.patient_generators import (
    PatientGenerator, LongitudinalDataGenerator
)
from tests.theory.validation.synthetic_generators.signal_generators import (
    ECGGenerator, EEGGenerator
)
from tests.theory.validation.synthetic_generators.trigger_generators import (
    TriggerGenerator, SymptomGenerator, TriggerProfile
)
from core.theory.temporal_modeling.spectral_analysis import SpectralAnalyzer
from core.theory.temporal_modeling.state_space_models import StateSpaceModeler
from core.theory.temporal_modeling.causal_inference import CausalAnalyzer

class TestSpectralAnalysis:
    """Test cases for spectral analysis theoretical components."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.analyzer = SpectralAnalyzer(data_type="eeg")
        
        # Create synthetic data generators
        self.eeg_gen = EEGGenerator(sampling_rate=250.0)
        self.patient_gen = PatientGenerator(num_patients=1)
    
    def test_known_frequency_components(self):
        """Test detection of known frequency components in synthetic EEG."""
        # Generate synthetic EEG with known frequency components
        duration = 60  # seconds
        band_weights = {
            'alpha': 1.0,  # Strong alpha (8-13 Hz)
            'beta': 0.3,   # Moderate beta (13-30 Hz)
            'theta': 0.2,  # Weak theta (4-8 Hz)
            'delta': 0.1,  # Very weak delta (0.5-4 Hz)
            'gamma': 0.1   # Very weak gamma (30-100 Hz)
        }
        
        time, eeg_data = self.eeg_gen.generate(duration, band_weights)
        
        # Create test case
        test_case = TestCase(
            name="eeg_frequency_components",
            inputs={
                "time_series": eeg_data['eeg'],
                "sampling_rate": 250.0,
                "method": "welch"
            },
            expected_outputs={
                "dominant_band": "alpha",
                "band_power_order": ["alpha", "beta", "theta", "delta", "gamma"]
            },
            tolerance={
                "alpha_power": 0.1,  # 10% tolerance for power values
                "beta_power": 0.1
            },
            metadata={
                "description": "Validate detection of known frequency components",
                "data_type": "eeg",
                "duration": duration,
                "true_weights": band_weights
            }
        )
        
        # Create validation function
        def validate_spectral_components(
            time_series: np.ndarray,
            sampling_rate: float,
            method: str
        ) -> Dict:
            # Analyze signal
            results = self.analyzer.analyze(time_series, sampling_rate, method)
            
            # Get band powers
            band_powers = results['band_powers']
            
            # Determine dominant band
            dominant_band = max(band_powers.items(), key=lambda x: x[1]['absolute'])[0]
            
            # Sort bands by power
            sorted_bands = sorted(
                band_powers.items(),
                key=lambda x: x[1]['absolute'],
                reverse=True
            )
            band_power_order = [band[0] for band in sorted_bands]
            
            return {
                "dominant_band": dominant_band,
                "band_power_order": band_power_order,
                "alpha_power": band_powers['alpha']['relative'],
                "beta_power": band_powers['beta']['relative']
            }
        
        # Create harness and validate
        harness = CoreLayerHarness("spectral_analysis", validate_spectral_components)
        harness.add_test_case(test_case)
        results = harness.run_all()
        
        return results
    
    def test_circadian_rhythm_detection(self):
        """Test detection of circadian rhythms in longitudinal data."""
        # Generate patient with known circadian patterns
        profile = self.patient_gen.generate_profile()
        data_gen = LongitudinalDataGenerator(profile)
        
        # Generate 7 days of data with hourly samples
        patient_data = data_gen.generate(
            duration_days=7,
            include_physiological=True
        )
        
        # Extract symptom intensity (should have circadian pattern)
        symptoms = patient_data['symptoms']['symptoms']
        headache = symptoms['headache']
        timestamps = patient_data['symptoms']['timestamps']
        
        # Create test case
        test_case = TestCase(
            name="circadian_rhythm_detection",
            inputs={
                "time_series": headache,
                "sampling_rate": 1/3600,  # hourly samples
                "method": "fft"
            },
            expected_outputs={
                "has_circadian": True,
                "circadian_period": 24.0,  # hours
                "has_significant_ultra_low": True
            },
            tolerance={
                "circadian_period": 1.0  # ±1 hour tolerance
            },
            metadata={
                "description": "Validate circadian rhythm detection",
                "data_type": "symptoms",
                "duration_days": 7
            }
        )
        
        # Create validation function
        def validate_circadian(
            time_series: np.ndarray,
            sampling_rate: float,
            method: str
        ) -> Dict:
            # Analyze signal
            results = self.analyzer.analyze(time_series, sampling_rate, method)
            
            # Find peaks in power spectrum
            freqs = results['frequencies']
            power = results['power_spectrum']
            
            # Convert frequency to period (hours)
            periods = 1 / (freqs + 1e-10)  # Avoid division by zero
            
            # Find peaks in power spectrum
            from scipy.signal import find_peaks
            peaks, _ = find_peaks(power, height=np.max(power)*0.1)
            peak_periods = periods[peaks]
            
            # Check for circadian rhythm (24±2 hours)
            has_circadian = any(22 <= p <= 26 for p in peak_periods)
            circadian_period = next(
                (p for p in peak_periods if 22 <= p <= 26),
                0.0
            )
            
            # Check for ultra-low frequency components
            ultra_low_mask = freqs < 1/3600  # Less than once per hour
            has_significant_ultra_low = np.any(
                power[ultra_low_mask] > np.max(power) * 0.1
            )
            
            return {
                "has_circadian": has_circadian,
                "circadian_period": circadian_period,
                "has_significant_ultra_low": has_significant_ultra_low
            }
        
        # Create harness and validate
        harness = CoreLayerHarness("circadian_detection", validate_circadian)
        harness.add_test_case(test_case)
        results = harness.run_all()
        
        return results
    
    def test_spectral_entropy(self):
        """Test spectral entropy calculation for different signal types."""
        # Generate synthetic signals with known complexity
        duration = 60  # seconds
        
        # Regular signal (low entropy)
        t = np.linspace(0, duration, int(250 * duration))
        regular_signal = np.sin(2 * np.pi * 10 * t)  # 10 Hz sine wave
        
        # Complex signal (high entropy)
        complex_weights = {
            'delta': 0.2,
            'theta': 0.2,
            'alpha': 0.2,
            'beta': 0.2,
            'gamma': 0.2
        }
        _, complex_data = self.eeg_gen.generate(duration, complex_weights)
        complex_signal = complex_data['eeg']
        
        # Create test cases
        test_cases = [
            TestCase(
                name="regular_signal_entropy",
                inputs={
                    "time_series": regular_signal,
                    "sampling_rate": 250.0,
                    "method": "welch"
                },
                expected_outputs={
                    "spectral_entropy": 0.3,  # Low entropy expected
                    "entropy_range": (0.0, 1.0)
                },
                tolerance={
                    "spectral_entropy": 0.2
                },
                metadata={
                    "description": "Validate entropy for regular signal",
                    "signal_type": "synthetic_sine"
                }
            ),
            TestCase(
                name="complex_signal_entropy",
                inputs={
                    "time_series": complex_signal,
                    "sampling_rate": 250.0,
                    "method": "welch"
                },
                expected_outputs={
                    "spectral_entropy": 0.8,  # High entropy expected
                    "entropy_range": (0.0, 1.0)
                },
                tolerance={
                    "spectral_entropy": 0.2
                },
                metadata={
                    "description": "Validate entropy for complex signal",
                    "signal_type": "synthetic_eeg"
                }
            )
        ]
        
        # Create validation function
        def validate_entropy(
            time_series: np.ndarray,
            sampling_rate: float,
            method: str
        ) -> Dict:
            # Analyze signal
            results = self.analyzer.analyze(time_series, sampling_rate, method)
            
            return {
                "spectral_entropy": results['entropy']['spectral_entropy'],
                "entropy_range": (0.0, 1.0)
            }
        
        # Create harness and validate
        harness = CoreLayerHarness("spectral_entropy", validate_entropy)
        for test_case in test_cases:
            harness.add_test_case(test_case)
        results = harness.run_all()
        
        return results

class TestStateSpaceModels:
    """Test cases for state space modeling theoretical components."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.modeler = StateSpaceModeler()
        self.ecg_gen = ECGGenerator(sampling_rate=250.0)
        self.patient_gen = PatientGenerator(num_patients=1)
    
    def test_kalman_filter_tracking(self):
        """Test Kalman filter's ability to track known state evolution."""
        # Generate synthetic ECG with known heart rate changes
        duration = 60  # seconds
        stress_events = [
            (20, 0.5),  # Moderate stress at 20s
            (40, 0.8)   # High stress at 40s
        ]
        
        time, ecg_data = self.ecg_gen.generate(duration, stress_level=0.0)
        
        # Create test case
        test_case = TestCase(
            name="kalman_hr_tracking",
            inputs={
                "observations": ecg_data['ecg'],
                "sampling_rate": 250.0,
                "state_dim": 2,  # HR and HR variability
                "stress_events": stress_events
            },
            expected_outputs={
                "tracking_error": 0.1,  # 10% max error
                "response_delay": 1.0,  # 1 second max delay
                "state_bounds_valid": True
            },
            tolerance={
                "tracking_error": 0.05,
                "response_delay": 0.5
            },
            metadata={
                "description": "Validate Kalman filter state tracking",
                "signal_type": "ecg",
                "true_hr": ecg_data['hr']
            }
        )
        
        # Create validation function
        def validate_kalman(
            observations: np.ndarray,
            sampling_rate: float,
            state_dim: int,
            stress_events: List[Tuple[float, float]]
        ) -> Dict:
            # Initialize and run Kalman filter
            results = self.modeler.kalman_filter(
                observations,
                sampling_rate=sampling_rate,
                state_dimension=state_dim
            )
            
            # Calculate tracking error
            estimated_hr = results['states'][:, 0]  # First state dimension is HR
            true_hr = ecg_data['hr']
            tracking_error = np.mean(np.abs(estimated_hr - true_hr) / true_hr)
            
            # Calculate response delay to stress events
            delays = []
            for event_time, _ in stress_events:
                event_idx = int(event_time * sampling_rate)
                response_idx = event_idx + np.argmax(
                    estimated_hr[event_idx:event_idx+int(5*sampling_rate)]
                )
                delay = (response_idx - event_idx) / sampling_rate
                delays.append(delay)
            
            # Verify state bounds
            state_bounds_valid = np.all(estimated_hr > 0) and np.all(estimated_hr < 200)
            
            return {
                "tracking_error": tracking_error,
                "response_delay": np.mean(delays),
                "state_bounds_valid": state_bounds_valid
            }
        
        # Create harness and validate
        harness = CoreLayerHarness("kalman_filter", validate_kalman)
        harness.add_test_case(test_case)
        results = harness.run_all()
        
        return results
    
    def test_state_prediction(self):
        """Test state space model's prediction capabilities."""
        # Generate patient data with known patterns
        profile = self.patient_gen.generate_profile()
        data_gen = LongitudinalDataGenerator(profile)
        
        # Generate 5 days of data
        patient_data = data_gen.generate(
            duration_days=5,
            include_physiological=True
        )
        
        # Extract symptom intensity
        symptoms = patient_data['symptoms']['symptoms']
        headache = symptoms['headache']
        
        # Split into training and testing
        train_size = int(len(headache) * 0.8)
        train_data = headache[:train_size]
        test_data = headache[train_size:]
        
        # Create test case
        test_case = TestCase(
            name="state_prediction",
            inputs={
                "training_data": train_data,
                "prediction_horizon": len(test_data),
                "state_dim": 3  # Level, trend, seasonal
            },
            expected_outputs={
                "prediction_error": 0.2,  # 20% max error
                "trend_correlation": 0.7,  # Strong trend correlation
                "prediction_bounds_valid": True
            },
            tolerance={
                "prediction_error": 0.1,
                "trend_correlation": 0.1
            },
            metadata={
                "description": "Validate state space prediction",
                "data_type": "symptoms",
                "true_values": test_data
            }
        )
        
        # Create validation function
        def validate_prediction(
            training_data: np.ndarray,
            prediction_horizon: int,
            state_dim: int
        ) -> Dict:
            # Fit model and generate predictions
            results = self.modeler.fit_predict(
                training_data,
                horizon=prediction_horizon,
                state_dimension=state_dim
            )
            
            predictions = results['predictions']
            prediction_bounds = results['prediction_bounds']
            
            # Calculate prediction error
            true_values = test_data
            prediction_error = np.mean(
                np.abs(predictions - true_values) / (true_values + 1e-6)
            )
            
            # Calculate trend correlation
            from scipy.stats import pearsonr
            trend_correlation = pearsonr(
                np.diff(predictions),
                np.diff(true_values)
            )[0]
            
            # Verify prediction bounds
            bounds_valid = np.all(prediction_bounds['lower'] <= predictions) and \
                         np.all(predictions <= prediction_bounds['upper'])
            
            return {
                "prediction_error": prediction_error,
                "trend_correlation": trend_correlation,
                "prediction_bounds_valid": bounds_valid
            }
        
        # Create harness and validate
        harness = CoreLayerHarness("state_prediction", validate_prediction)
        harness.add_test_case(test_case)
        results = harness.run_all()
        
        return results
    
    def test_model_selection(self):
        """Test state space model selection capabilities."""
        # Generate synthetic data with known complexity
        duration = 60  # seconds
        
        # Simple linear trend with noise
        t = np.linspace(0, duration, int(250 * duration))
        simple_signal = 0.1 * t + np.random.normal(0, 0.1, len(t))
        
        # Complex nonlinear dynamics
        complex_signal = np.sin(0.1 * t) + \
                        0.5 * np.sin(0.05 * t) * t + \
                        np.random.normal(0, 0.2, len(t))
        
        # Create test cases
        test_cases = [
            TestCase(
                name="simple_model_selection",
                inputs={
                    "observations": simple_signal,
                    "max_state_dim": 5
                },
                expected_outputs={
                    "selected_dim": 2,  # Level and trend only
                    "model_complexity": "low",
                    "residuals_white": True
                },
                tolerance={
                    "selected_dim": 0  # Must be exact
                },
                metadata={
                    "description": "Validate model selection for simple dynamics",
                    "signal_type": "linear_trend"
                }
            ),
            TestCase(
                name="complex_model_selection",
                inputs={
                    "observations": complex_signal,
                    "max_state_dim": 5
                },
                expected_outputs={
                    "selected_dim": 4,  # More states needed
                    "model_complexity": "high",
                    "residuals_white": True
                },
                tolerance={
                    "selected_dim": 1  # Allow ±1 state
                },
                metadata={
                    "description": "Validate model selection for complex dynamics",
                    "signal_type": "nonlinear_dynamics"
                }
            )
        ]
        
        # Create validation function
        def validate_model_selection(
            observations: np.ndarray,
            max_state_dim: int
        ) -> Dict:
            # Perform model selection
            results = self.modeler.select_model(
                observations,
                max_dimension=max_state_dim
            )
            
            # Get selected model properties
            selected_dim = results['selected_dimension']
            complexity = results['complexity_metrics']
            
            # Test residual whiteness
            from scipy.stats import normaltest
            residuals = results['residuals']
            _, p_value = normaltest(residuals)
            residuals_white = p_value > 0.05
            
            return {
                "selected_dim": selected_dim,
                "model_complexity": "high" if complexity['aic'] > 100 else "low",
                "residuals_white": residuals_white
            }
        
        # Create harness and validate
        harness = CoreLayerHarness("model_selection", validate_model_selection)
        for test_case in test_cases:
            harness.add_test_case(test_case)
        results = harness.run_all()
        
        return results

class TestCausalInference:
    """Test cases for causal inference theoretical components."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.analyzer = CausalAnalyzer()
        self.trigger_gen = TriggerGenerator([
            'stress', 'bright_light', 'weather_change'
        ])
        self.patient_gen = PatientGenerator(num_patients=1)
    
    def test_granger_causality(self):
        """Test Granger causality detection with known causal relationships."""
        # Generate synthetic trigger-symptom data with known delays
        duration_days = 30
        
        # Create trigger profile with known sensitivities
        trigger_profile = TriggerProfile(
            trigger_types=['stress', 'bright_light', 'weather_change'],
            base_sensitivities={
                'stress': 0.8,        # Strong causal effect
                'bright_light': 0.5,  # Moderate causal effect
                'weather_change': 0.1  # Weak/no causal effect
            }
        )
        
        # Generate trigger data
        trigger_data = self.trigger_gen.generate(
            duration_days,
            hourly=True
        )
        
        # Generate symptoms with known delays
        symptom_gen = SymptomGenerator(
            trigger_profile,
            latency_range=(2, 6)  # 2-6 hour delay
        )
        symptom_data = symptom_gen.generate(
            trigger_data['intensities'],
            trigger_data['timestamps']
        )
        
        # Create test case
        test_case = TestCase(
            name="granger_causality",
            inputs={
                "triggers": trigger_data['intensities'],
                "symptoms": symptom_data['symptoms'],
                "sampling_rate": 1/3600,  # hourly data
                "max_lag": 12  # Test up to 12 hours lag
            },
            expected_outputs={
                "causal_triggers": ['stress', 'bright_light'],
                "non_causal_triggers": ['weather_change'],
                "lag_detection_error": 2.0  # Max 2 hour error in lag detection
            },
            tolerance={
                "lag_detection_error": 1.0
            },
            metadata={
                "description": "Validate Granger causality detection",
                "true_delays": {
                    'stress': 3,        # hours
                    'bright_light': 4,  # hours
                    'weather_change': 0  # no causal relationship
                }
            }
        )
        
        # Create validation function
        def validate_granger(
            triggers: Dict[str, np.ndarray],
            symptoms: Dict[str, np.ndarray],
            sampling_rate: float,
            max_lag: int
        ) -> Dict:
            # Test each trigger for causality
            results = {}
            detected_lags = {}
            
            headache = symptoms['headache']
            for trigger_name, trigger_series in triggers.items():
                # Perform Granger causality test
                causality_result = self.analyzer.granger_test(
                    cause_series=trigger_series,
                    effect_series=headache,
                    max_lag=max_lag,
                    sampling_rate=sampling_rate
                )
                
                results[trigger_name] = causality_result
                if causality_result['is_causal']:
                    detected_lags[trigger_name] = causality_result['optimal_lag']
            
            # Separate causal and non-causal triggers
            causal_triggers = [
                name for name, res in results.items() 
                if res['is_causal']
            ]
            non_causal_triggers = [
                name for name, res in results.items() 
                if not res['is_causal']
            ]
            
            # Calculate lag detection error
            true_delays = test_case.metadata['true_delays']
            lag_errors = []
            for trigger in causal_triggers:
                if true_delays[trigger] > 0:  # Only for actual causal relationships
                    error = abs(detected_lags[trigger] - true_delays[trigger])
                    lag_errors.append(error)
            
            return {
                "causal_triggers": causal_triggers,
                "non_causal_triggers": non_causal_triggers,
                "lag_detection_error": max(lag_errors) if lag_errors else 0.0
            }
        
        # Create harness and validate
        harness = CoreLayerHarness("granger_causality", validate_granger)
        harness.add_test_case(test_case)
        results = harness.run_all()
        
        return results
    
    def test_transfer_entropy(self):
        """Test transfer entropy calculation for information flow detection."""
        # Generate synthetic data with known information flow
        duration_days = 14
        
        # Generate patient data with known trigger-symptom relationships
        profile = self.patient_gen.generate_profile()
        data_gen = LongitudinalDataGenerator(profile)
        
        patient_data = data_gen.generate(
            duration_days=duration_days,
            include_physiological=True
        )
        
        # Extract relevant time series
        ecg = patient_data['physiological']['ecg']['hr']
        symptoms = patient_data['symptoms']['symptoms']['headache']
        stress = patient_data['triggers']['intensities']['stress']
        
        # Create test cases for different information flows
        test_cases = [
            TestCase(
                name="stress_to_ecg",
                inputs={
                    "source": stress,
                    "target": ecg,
                    "sampling_rate": 1/3600,
                    "embedding_dim": 3
                },
                expected_outputs={
                    "significant_flow": True,
                    "flow_direction": "forward",
                    "entropy_ratio": 0.6  # Strong forward flow
                },
                tolerance={
                    "entropy_ratio": 0.2
                },
                metadata={
                    "description": "Validate stress→ECG information flow",
                    "flow_type": "physiological_response"
                }
            ),
            TestCase(
                name="stress_symptoms",
                inputs={
                    "source": stress,
                    "target": symptoms,
                    "sampling_rate": 1/3600,
                    "embedding_dim": 3
                },
                expected_outputs={
                    "significant_flow": True,
                    "flow_direction": "forward",
                    "entropy_ratio": 0.7  # Very strong forward flow
                },
                tolerance={
                    "entropy_ratio": 0.2
                },
                metadata={
                    "description": "Validate stress→symptoms information flow",
                    "flow_type": "trigger_effect"
                }
            )
        ]
        
        # Create validation function
        def validate_transfer_entropy(
            source: np.ndarray,
            target: np.ndarray,
            sampling_rate: float,
            embedding_dim: int
        ) -> Dict:
            # Calculate transfer entropy in both directions
            forward_te = self.analyzer.transfer_entropy(
                source, target,
                embedding_dimension=embedding_dim
            )
            backward_te = self.analyzer.transfer_entropy(
                target, source,
                embedding_dimension=embedding_dim
            )
            
            # Calculate entropy ratio and determine flow direction
            entropy_ratio = forward_te / (forward_te + backward_te)
            
            # Test significance against surrogate data
            significance = self.analyzer.test_te_significance(
                source, target,
                n_surrogates=100
            )
            
            return {
                "significant_flow": significance['is_significant'],
                "flow_direction": "forward" if entropy_ratio > 0.5 else "backward",
                "entropy_ratio": entropy_ratio
            }
        
        # Create harness and validate
        harness = CoreLayerHarness("transfer_entropy", validate_transfer_entropy)
        for test_case in test_cases:
            harness.add_test_case(test_case)
        results = harness.run_all()
        
        return results
    
    def test_convergent_cross_mapping(self):
        """Test convergent cross-mapping for detecting nonlinear causality."""
        # Generate synthetic data with nonlinear coupling
        duration = 60 * 24  # 60 days
        dt = 3600  # 1 hour
        t = np.arange(0, duration * 3600, dt)
        
        # Generate coupled Lorenz system with known causality
        def lorenz_coupled(state, t, coupling=0.3):
            x, y, z, w = state
            
            # First system (x, y, z) affects second (w)
            dx = 10 * (y - x)
            dy = x * (28 - z) - y
            dz = x * y - 8/3 * z
            dw = -w + coupling * x  # w is driven by x
            
            return [dx, dy, dz, dw]
        
        # Simulate system
        from scipy.integrate import odeint
        initial_state = [1.0, 1.0, 1.0, 1.0]
        solution = odeint(lorenz_coupled, initial_state, t)
        
        # Extract time series (x drives w)
        x = solution[:, 0]  # Driver
        w = solution[:, 3]  # Target
        
        # Create test case
        test_case = TestCase(
            name="nonlinear_causality",
            inputs={
                "series1": x,
                "series2": w,
                "sampling_rate": 1/3600,
                "embedding_dim": 3
            },
            expected_outputs={
                "causality_detected": True,
                "direction": "series1_to_series2",
                "convergence_rate": 0.8  # Strong convergence expected
            },
            tolerance={
                "convergence_rate": 0.2
            },
            metadata={
                "description": "Validate nonlinear causality detection",
                "true_coupling": 0.3
            }
        )
        
        # Create validation function
        def validate_ccm(
            series1: np.ndarray,
            series2: np.ndarray,
            sampling_rate: float,
            embedding_dim: int
        ) -> Dict:
            # Perform CCM analysis
            results = self.analyzer.convergent_cross_mapping(
                series1, series2,
                embedding_dimension=embedding_dim,
                tau=int(1/sampling_rate)
            )
            
            # Calculate convergence rate
            library_sizes = results['library_sizes']
            correlations = results['correlations']
            
            # Fit exponential convergence curve
            from scipy.optimize import curve_fit
            def conv_curve(x, a, b):
                return a * (1 - np.exp(-b * x))
            
            popt, _ = curve_fit(
                conv_curve, library_sizes, 
                correlations['series1_to_series2'],
                p0=[1.0, 0.01]
            )
            convergence_rate = popt[1]
            
            # Determine causality direction
            forward_corr = np.mean(correlations['series1_to_series2'])
            backward_corr = np.mean(correlations['series2_to_series1'])
            
            return {
                "causality_detected": forward_corr > 0.5,
                "direction": "series1_to_series2" 
                           if forward_corr > backward_corr 
                           else "series2_to_series1",
                "convergence_rate": convergence_rate
            }
        
        # Create harness and validate
        harness = CoreLayerHarness("convergent_cross_mapping", validate_ccm)
        harness.add_test_case(test_case)
        results = harness.run_all()
        
        return results 