"""
Tests for the State Space Models components.

This module contains tests for the theoretical components related to
state space modeling, including Kalman filters, hidden Markov models,
and particle filtering for physiological time series.
"""

import unittest
import numpy as np
from typing import Dict, List, Any

from core.theory.temporal_modeling import StateSpaceModeler


class TestStateSpaceModeler(unittest.TestCase):
    """Tests for the StateSpaceModeler class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.linear_modeler = StateSpaceModeler("eeg", "linear", "Test linear Kalman filter")
        self.hmm_modeler = StateSpaceModeler("hrv", "hmm", "Test HMM modeler")
        self.general_modeler = StateSpaceModeler("general", "linear", "Test general modeler")
        
        # Generate synthetic time series for testing
        t = np.linspace(0, 10, 100)
        
        # Linear dynamical system
        # Position and velocity states
        x0 = np.array([0, 1])  # Initial state (position, velocity)
        process_noise = 0.01
        measurement_noise = 0.1
        
        # F matrix (state transition) for constant velocity model
        F = np.array([[1, 0.1], [0, 1]])
        
        # Initialize states and measurements
        x = x0.copy()
        self.eeg_signal = np.zeros(len(t))
        states = np.zeros((len(t), 2))
        
        # Generate the signal
        for i in range(len(t)):
            # State update with process noise
            x = F @ x + np.random.normal(0, process_noise, 2)
            states[i] = x
            
            # Measurement with noise (just observe position)
            self.eeg_signal[i] = x[0] + np.random.normal(0, measurement_noise)
        
        # HRV-like signal with state transitions
        # Generate a 3-state HMM-like signal
        state = 0  # Start in state 0
        self.hrv_signal = np.zeros(len(t))
        
        # State means and variances
        state_means = [0.5, 1.2, 2.0]
        state_vars = [0.05, 0.1, 0.2]
        
        # Transition probabilities (stay in state with p=0.95, else transition)
        for i in range(len(t)):
            # Random state transition with low probability
            if np.random.rand() > 0.95:
                state = np.random.choice([0, 1, 2])
            
            # Generate observation from current state
            self.hrv_signal[i] = np.random.normal(state_means[state], np.sqrt(state_vars[state]))
        
        # Set sampling rates (needed for some methods)
        self.eeg_sampling_rate = 10  # Hz
        self.hrv_sampling_rate = 4   # Hz
    
    def test_initialization(self):
        """Test initialization with different data types and model types."""
        self.assertEqual(self.linear_modeler.data_type, "eeg")
        self.assertEqual(self.linear_modeler.model_type, "linear")
        self.assertEqual(self.hmm_modeler.data_type, "hrv")
        self.assertEqual(self.hmm_modeler.model_type, "hmm")
        
        # Check default parameters
        self.assertIn("state_dim", self.linear_modeler.model_parameters)
        self.assertIn("process_noise", self.linear_modeler.model_parameters)
        self.assertIn("measurement_noise", self.linear_modeler.model_parameters)
        
        self.assertIn("n_states", self.hmm_modeler.model_parameters)
        self.assertIn("covariance_type", self.hmm_modeler.model_parameters)
        
    def test_analyze_linear_kalman(self):
        """Test linear Kalman filter analysis."""
        # Analyze EEG signal with linear Kalman filter
        results = self.linear_modeler.analyze(self.eeg_signal)
        
        # Check structure of results
        self.assertIn("model_type", results)
        self.assertEqual(results["model_type"], "linear_kalman")
        
        self.assertIn("states", results)
        self.assertIn("observations", results)
        self.assertIn("parameters", results)
        self.assertIn("evaluation", results)
        self.assertIn("theoretical_insights", results)
        
        # Check shapes
        self.assertEqual(results["states"].shape[0], len(self.eeg_signal))
        self.assertEqual(results["states"].shape[1], 2)  # Default 2-state model
        
        # Check evaluation metrics
        self.assertIn("rmse", results["evaluation"])
        self.assertIn("mae", results["evaluation"])
        self.assertIn("aic", results["evaluation"])
        self.assertIn("bic", results["evaluation"])
        
        # Check theoretical insights
        self.assertIsInstance(results["theoretical_insights"], list)
        self.assertGreater(len(results["theoretical_insights"]), 0)
        
    def test_analyze_hmm(self):
        """Test HMM analysis."""
        # Analyze HRV signal with HMM
        results = self.hmm_modeler.analyze(self.hrv_signal)
        
        # Check structure of results
        self.assertIn("model_type", results)
        self.assertEqual(results["model_type"], "hmm")
        
        self.assertIn("states", results)
        self.assertIn("observations", results)
        self.assertIn("parameters", results)
        self.assertIn("note", results)  # Should include a note about implementation
        
        # Check shapes
        self.assertEqual(results["states"].shape[0], len(self.hrv_signal))
        
        # Check theoretical insights
        self.assertIn("theoretical_insights", results)
        self.assertIsInstance(results["theoretical_insights"], list)
        self.assertGreater(len(results["theoretical_insights"]), 0)
        
    def test_predict(self):
        """Test prediction functionality."""
        # Train model and make predictions
        horizon = 10
        predictions, confidence_intervals = self.linear_modeler.predict(
            self.eeg_signal, horizon, confidence_level=0.95)
        
        # Check shapes
        self.assertEqual(predictions.shape, (horizon,))
        self.assertEqual(confidence_intervals.shape, (horizon, 2))
        
        # Check confidence intervals (lower < upper)
        self.assertTrue(np.all(confidence_intervals[:, 0] <= confidence_intervals[:, 1]))
        
    def test_compare_models(self):
        """Test model comparison functionality."""
        # Compare models on EEG signal
        comparison = self.general_modeler.compare_models(self.eeg_signal, model_types=["linear", "hmm"])
        
        # Check structure of results
        self.assertIn("model_results", comparison)
        self.assertIn("best_model", comparison)
        self.assertIn("comparison_insights", comparison)
        
        # Check model results
        self.assertIn("linear", comparison["model_results"])
        self.assertIn("hmm", comparison["model_results"])
        
        # Check comparison insights
        self.assertIsInstance(comparison["comparison_insights"], list)
        self.assertGreater(len(comparison["comparison_insights"]), 0)
        
    def test_formal_definition(self):
        """Test formal definition generation."""
        # Get formal definition for linear model
        linear_definition = self.linear_modeler.get_formal_definition()
        self.assertIsInstance(linear_definition, str)
        self.assertIn("State equation", linear_definition)
        
        # Get formal definition for HMM model
        hmm_definition = self.hmm_modeler.get_formal_definition()
        self.assertIsInstance(hmm_definition, str)
        self.assertIn("Hidden Markov Model", hmm_definition)


if __name__ == "__main__":
    unittest.main() 