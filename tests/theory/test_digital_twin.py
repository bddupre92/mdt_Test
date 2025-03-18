"""
Unit tests for digital twin foundation.

Tests the functionality of the digital twin model including:
- Patient state modeling
- Model initialization and updating
- Intervention simulation
- Accuracy assessment
- State prediction and confidence estimation
"""

import unittest
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Any

from core.theory.migraine_adaptation.digital_twin import (
    DigitalTwinModel,
    PatientState
)

def generate_synthetic_patient_data(n_samples: int = 1000) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
    """Generate synthetic patient data for testing."""
    # Generate timestamps
    timestamps = np.array([
        datetime.now() + timedelta(hours=i)
        for i in range(n_samples)
    ])
    
    # Generate physiological signals
    t = np.linspace(0, n_samples/24, n_samples)  # Time in days
    
    # ECG signal with daily pattern
    ecg = 0.7 * np.sin(2 * np.pi * t * 24) + \
          0.3 * np.sin(2 * np.pi * t * 60) + \
          0.2 * np.random.randn(n_samples)
    
    # EEG signal with circadian rhythm
    eeg = np.zeros((n_samples, 4))  # 4 channels
    for i in range(4):
        eeg[:, i] = 0.8 * np.sin(2 * np.pi * t + i*np.pi/4) + 0.2 * np.random.randn(n_samples)
    
    # Skin conductance with stress responses
    sc = 5 + np.sin(2 * np.pi * t * 12) + 0.5 * np.random.randn(n_samples)
    
    # Generate trigger levels
    stress = 0.6 * np.sin(2 * np.pi * t) + 0.4 * np.random.randn(n_samples)
    sleep = 0.7 * np.sin(2 * np.pi * t/7 + np.pi/3) + 0.3 * np.random.randn(n_samples)
    
    # Generate symptom intensities
    pain = (
        0.4 * np.roll(stress, 12) +  # 12-hour lag for stress
        0.3 * np.roll(sleep, 24)     # 24-hour lag for sleep
    )
    nausea = (
        0.3 * np.roll(stress, 8) +   # 8-hour lag for stress
        0.2 * np.roll(sleep, 16)     # 16-hour lag for sleep
    )
    
    # Combine into patient history
    patient_history = {
        'timestamps': timestamps,
        'phys_ecg': ecg,
        'phys_eeg': eeg,
        'phys_sc': sc,
        'trigger_stress': stress,
        'trigger_sleep': sleep,
        'symptom_pain': pain,
        'symptom_nausea': nausea
    }
    
    # Create patient metadata
    patient_metadata = {
        'patient_id': 'TEST001',
        'age': 35,
        'sex': 'F',
        'migraine_history_years': 10,
        'typical_frequency': 'monthly',
        'known_triggers': ['stress', 'sleep_disruption'],
        'medications': ['sumatriptan']
    }
    
    return patient_history, patient_metadata

def generate_intervention(intensity: float = 1.0) -> Dict[str, Any]:
    """Generate test intervention data."""
    return {
        'type': 'medication',
        'name': 'sumatriptan',
        'intensity': intensity,
        'route': 'oral',
        'timing': 'acute',
        'metadata': {
            'dose': '50mg',
            'max_daily': 2
        }
    }

class TestDigitalTwin(unittest.TestCase):
    """Test suite for digital twin model."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.model = DigitalTwinModel()
        self.patient_history, self.patient_metadata = generate_synthetic_patient_data()
        self.initial_state = self.model.initialize_twin(
            self.patient_history,
            self.patient_metadata
        )
    
    def test_initialization(self):
        """Test model initialization."""
        # Check model parameters
        self.assertEqual(self.model.state_prediction_horizon, 24)
        self.assertEqual(self.model.update_window, 12)
        self.assertEqual(self.model.confidence_threshold, 0.8)
        
        # Check initial state
        self.assertIsNotNone(self.model.current_state)
        self.assertIsNotNone(self.model.state_transition_model)
        self.assertIsNotNone(self.model.intervention_response_model)
        self.assertIsNotNone(self.model.anomaly_detector)
        
        # Check feature mapping
        self.assertTrue(self.model.feature_map)
        self.assertIn('phys_ecg', self.model.feature_map)
        self.assertIn('trigger_stress', self.model.feature_map)
        self.assertIn('symptom_pain', self.model.feature_map)
        
        # Check metadata
        self.assertEqual(
            self.model.model_metadata['patient_id'],
            self.patient_metadata['patient_id']
        )
    
    def test_state_conversion(self):
        """Test patient state vector conversion."""
        # Get current state
        state = self.model.current_state
        
        # Convert to vector
        vector = state.to_vector()
        
        # Convert back to state
        reconstructed = PatientState.from_vector(
            vector,
            self.model.feature_map,
            state.timestamp,
            state.metadata
        )
        
        # Check reconstruction
        self.assertEqual(
            len(reconstructed.physiological_state),
            len(state.physiological_state)
        )
        self.assertEqual(
            len(reconstructed.trigger_state),
            len(state.trigger_state)
        )
        self.assertEqual(
            len(reconstructed.symptom_state),
            len(state.symptom_state)
        )
        
        # Check values (allowing for small numerical differences)
        for key, value in state.trigger_state.items():
            self.assertAlmostEqual(
                reconstructed.trigger_state[key],
                value,
                places=5
            )
    
    def test_model_update(self):
        """Test model updating with new observations."""
        # Create new observations
        n_new = 24  # 1 day of data
        new_observations = {
            key: value[-n_new:] if isinstance(value, np.ndarray) else value
            for key, value in self.patient_history.items()
        }
        
        # Update model
        updated_state = self.model.update_twin(
            self.initial_state,
            new_observations
        )
        
        # Check update results
        self.assertIsNotNone(updated_state)
        self.assertGreater(
            self.model.model_metadata['n_historical_states'],
            self.initial_state['model_metadata']['n_historical_states']
        )
        self.assertGreater(
            self.model.model_metadata['last_update_time'],
            self.initial_state['model_metadata']['last_update_time']
        )
    
    def test_intervention_simulation(self):
        """Test intervention simulation."""
        # Create test intervention
        intervention = generate_intervention()
        
        # Run simulation
        results = self.model.simulate_intervention(
            self.initial_state,
            intervention,
            simulation_duration=48.0  # 2 days
        )
        
        # Check simulation results
        self.assertIn('state_trajectories', results)
        self.assertIn('confidence_scores', results)
        self.assertIn('simulation_times', results)
        self.assertIn('intervention_effects', results)
        
        # Check trajectory shape
        n_steps = len(results['simulation_times'])
        self.assertEqual(
            results['state_trajectories'].shape[0],
            n_steps
        )
        
        # Check confidence scores
        self.assertTrue(all(0 <= score <= 1 for score in results['confidence_scores']))
        
        # Check intervention effects
        effects = results['intervention_effects']
        self.assertIn('immediate', effects)
        self.assertIn('sustained', effects)
        self.assertIn('peak', effects)
        self.assertIn('time_to_peak', effects)
        self.assertIn('feature_effects', effects)
    
    def test_accuracy_assessment(self):
        """Test model accuracy assessment."""
        # Create test data with some variation
        n_test = 24  # 1 day of data
        t = np.linspace(0, 1, n_test)
        
        test_data = {
            'timestamps': np.array([
                datetime.now() + timedelta(hours=i)
                for i in range(n_test)
            ]),
            'phys_ecg': np.sin(2 * np.pi * t) + 0.1 * np.random.randn(n_test),
            'phys_eeg': np.random.randn(n_test, 4),
            'phys_sc': np.cos(2 * np.pi * t) + 0.1 * np.random.randn(n_test),
            'trigger_stress': 0.5 + 0.1 * np.random.randn(n_test),
            'trigger_sleep': 0.7 + 0.1 * np.random.randn(n_test),
            'symptom_pain': 0.3 + 0.1 * np.random.randn(n_test),
            'symptom_nausea': 0.2 + 0.1 * np.random.randn(n_test)
        }
        
        # Test with normal data
        metrics = self.model.assess_twin_accuracy(
            self.initial_state,
            test_data
        )
        
        # Check metrics exist and have valid values
        self.assertIn('mse', metrics)
        self.assertIn('mae', metrics)
        self.assertIn('r2', metrics)
        self.assertIn('temporal_correlation', metrics)
        self.assertIn('feature_metrics', metrics)
        
        # Check metric ranges
        self.assertGreaterEqual(metrics['mse'], 0.0)
        self.assertGreaterEqual(metrics['mae'], 0.0)
        self.assertGreaterEqual(metrics['r2'], -1.0)
        self.assertLessEqual(metrics['r2'], 1.0)
        self.assertGreaterEqual(metrics['temporal_correlation'], -1.0)
        self.assertLessEqual(metrics['temporal_correlation'], 1.0)
        
        # Test with constant data
        constant_data = {
            'timestamps': test_data['timestamps'],
            'phys_ecg': np.ones(n_test),
            'phys_eeg': np.ones((n_test, 4)),
            'phys_sc': np.ones(n_test),
            'trigger_stress': np.ones(n_test),
            'trigger_sleep': np.ones(n_test),
            'symptom_pain': np.ones(n_test),
            'symptom_nausea': np.ones(n_test)
        }
        
        constant_metrics = self.model.assess_twin_accuracy(
            self.initial_state,
            constant_data
        )
        
        # Check handling of constant data
        self.assertGreaterEqual(constant_metrics['temporal_correlation'], -1.0)
        self.assertLessEqual(constant_metrics['temporal_correlation'], 1.0)
        
        # Test with empty data
        empty_metrics = self.model.assess_twin_accuracy(
            self.initial_state,
            {'timestamps': np.array([])}
        )
        
        # Check handling of empty data
        self.assertEqual(empty_metrics['mse'], 0.0)
        self.assertEqual(empty_metrics['mae'], 0.0)
        self.assertEqual(empty_metrics['r2'], 0.0)
        self.assertEqual(empty_metrics['temporal_correlation'], 0.0)
        self.assertEqual(empty_metrics['feature_metrics'], {})
    
    def test_error_handling(self):
        """Test error handling for edge cases."""
        # Test with empty data
        empty_results = self.model.update_twin(
            self.initial_state,
            {'timestamps': []}
        )
        self.assertEqual(
            empty_results['model_metadata']['n_historical_states'],
            self.initial_state['model_metadata']['n_historical_states']
        )
        
        # Test with invalid intervention
        invalid_intervention = {'type': 'unknown'}
        results = self.model.simulate_intervention(
            self.initial_state,
            invalid_intervention,
            simulation_duration=24.0
        )
        self.assertTrue(all(score > 0 for score in results['confidence_scores']))
        
        # Test with mismatched data
        mismatched_data = {
            'timestamps': self.patient_history['timestamps'][:10],
            'phys_ecg': self.patient_history['phys_ecg'][:20]  # Different length
        }
        updated_state = self.model.update_twin(
            self.initial_state,
            mismatched_data
        )
        self.assertEqual(
            updated_state['model_metadata']['n_historical_states'],
            self.initial_state['model_metadata']['n_historical_states']
        )
    
    def test_prediction_confidence(self):
        """Test prediction confidence calculation."""
        # Get current state vector
        state_vector = self.model.current_state.to_vector()
        scaled_state = self.model.feature_scaler.transform([state_vector])[0]
        
        # Get next state prediction
        next_state = self.model._predict_next_state(scaled_state)
        
        # Calculate confidence at different horizons
        confidences = []
        horizons = [0, 6, 12, 24, 48]  # hours
        
        for horizon in horizons:
            confidence = self.model._calculate_prediction_confidence(
                scaled_state,
                next_state,
                horizon
            )
            confidences.append(confidence)
            
            # Check confidence is valid
            self.assertGreaterEqual(confidence, 0.0)
            self.assertLessEqual(confidence, 1.0)
        
        # Check confidence decreases with horizon
        self.assertTrue(all(c1 >= c2 for c1, c2 in zip(confidences[:-1], confidences[1:])))
    
    def test_intervention_effects(self):
        """Test intervention effect analysis."""
        # Generate trajectories
        n_steps = 24
        base_trajectory = np.random.randn(n_steps, len(self.model.current_state.to_vector()))
        intervention_trajectory = base_trajectory + 0.5  # Simulated effect
        
        # Analyze effects
        effects = self.model._analyze_intervention_effects(
            intervention_trajectory,
            generate_intervention()
        )
        
        # Check effect metrics
        self.assertGreater(effects['immediate'], 0)
        self.assertGreater(effects['sustained'], 0)
        self.assertGreater(effects['peak'], 0)
        self.assertGreaterEqual(effects['time_to_peak'], 0)
        
        # Check feature-specific effects
        for feature, effect in effects['feature_effects'].items():
            self.assertGreaterEqual(effect, 0)

if __name__ == '__main__':
    unittest.main() 