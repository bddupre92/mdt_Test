"""
Direct tests for MoE metrics calculator.

This module tests the MoEMetricsCalculator functionality directly
without relying on complex dependencies.
"""

import os
import sys
import unittest
import numpy as np
import pandas as pd
import tempfile
import shutil
import json
from pathlib import Path

# Add parent directory to path to ensure imports work
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the metrics calculator directly
from baseline_comparison.moe_metrics import MoEMetricsCalculator


class TestMoEMetricsCalculator(unittest.TestCase):
    """Test the MoEMetricsCalculator directly."""
    
    def setUp(self):
        """Set up the test environment."""
        # Create a temporary directory for test outputs
        self.test_dir = tempfile.mkdtemp()
        
        # Generate synthetic data for testing
        np.random.seed(42)
        self.n_samples = 100
        
        # True values and predictions
        self.y_true = np.random.normal(50, 10, size=self.n_samples)
        self.y_pred = self.y_true + np.random.normal(0, 5, size=self.n_samples)
        
        # Expert weights - three experts with weights that sum to 1 for each sample
        raw_weights = np.random.random((self.n_samples, 3))
        self.expert_weights = {}
        self.expert_weights['expert1'] = (raw_weights[:, 0] / raw_weights.sum(axis=1)).tolist()
        self.expert_weights['expert2'] = (raw_weights[:, 1] / raw_weights.sum(axis=1)).tolist()
        self.expert_weights['expert3'] = (raw_weights[:, 2] / raw_weights.sum(axis=1)).tolist()
        
        # Expert predictions - each expert has its own prediction for each sample
        self.expert_predictions = {}
        self.expert_predictions['expert1'] = self.y_true + np.random.normal(0, 3, size=self.n_samples)
        self.expert_predictions['expert2'] = self.y_true + np.random.normal(0, 7, size=self.n_samples)
        self.expert_predictions['expert3'] = self.y_true + np.random.normal(0, 10, size=self.n_samples)
        
        # Confidence scores
        self.confidence_scores = np.random.beta(5, 2, size=self.n_samples)
        
        # Create output directory for visualizations
        self.output_dir = os.path.join(self.test_dir, "metrics_output")
        os.makedirs(self.output_dir, exist_ok=True)
        
    def tearDown(self):
        """Clean up after tests."""
        # Remove temporary directory
        shutil.rmtree(self.test_dir)
    
    def test_metrics_calculation(self):
        """Test that metrics are calculated correctly."""
        # Create metrics calculator
        calculator = MoEMetricsCalculator(output_dir=self.output_dir)
        
        # Calculate expert errors
        expert_errors = {}
        for expert_name, expert_preds in self.expert_predictions.items():
            expert_errors[expert_name] = np.abs(self.y_true - expert_preds)
        
        # Calculate metrics
        metrics = calculator.compute_all_metrics(
            predictions=self.y_pred,
            actual_values=self.y_true,
            expert_contributions=self.expert_weights,
            confidence_scores=self.confidence_scores,
            expert_errors=expert_errors
        )
        
        # Verify metrics structure
        self.assertIsInstance(metrics, dict)
        
        # Check that standard metrics are included
        self.assertIn('standard', metrics)
        self.assertIn('rmse', metrics['standard'])
        self.assertIn('mae', metrics['standard'])
        self.assertIn('r2', metrics['standard'])
        
        # Check that expert contribution metrics are included
        self.assertIn('expert_contribution', metrics)
        self.assertIn('normalized_entropy', metrics['expert_contribution'])
        self.assertIn('expert_dominance_counts', metrics['expert_contribution'])
        
        # Check confidence metrics
        self.assertIn('confidence', metrics)
        self.assertIn('mean_confidence', metrics['confidence'])
        self.assertIn('confidence_error_correlation', metrics['confidence'])
        self.assertIn('expected_calibration_error', metrics['confidence'])
        self.assertIn('bin_mean_errors', metrics['confidence'])
        
        # Check gating network metrics
        self.assertIn('gating_network', metrics)
        self.assertIn('optimal_expert_selection_rate', metrics['gating_network'])
        self.assertIn('mean_regret', metrics['gating_network'])
        self.assertIn('mean_weight_error_correlation', metrics['gating_network'])
    
    def test_metrics_visualization(self):
        """Test that visualizations are generated correctly."""
        # Create metrics calculator
        calculator = MoEMetricsCalculator(output_dir=self.output_dir)
        
        # Calculate expert errors
        expert_errors = {}
        for expert_name, expert_preds in self.expert_predictions.items():
            expert_errors[expert_name] = np.abs(self.y_true - expert_preds)
        
        # First compute the metrics
        metrics = calculator.compute_all_metrics(
            predictions=self.y_pred,
            actual_values=self.y_true,
            expert_contributions=self.expert_weights,
            confidence_scores=self.confidence_scores,
            expert_errors=expert_errors
        )
        
        # Generate visualizations
        try:
            visualization_files = calculator.visualize_metrics(
                metrics=metrics,
                name="test_run"
            )
            
            # Check that visualization files were created
            for file_path in visualization_files:
                self.assertTrue(os.path.exists(file_path))
                
        except NotImplementedError:
            # If visualizations are not implemented yet, this test should be skipped
            self.skipTest("Visualization not implemented yet")
    
    def test_save_metrics_to_file(self):
        """Test that metrics can be saved to a file."""
        # Create metrics calculator
        calculator = MoEMetricsCalculator(output_dir=self.output_dir)
        
        # Calculate expert errors
        expert_errors = {}
        for expert_name, expert_preds in self.expert_predictions.items():
            expert_errors[expert_name] = np.abs(self.y_true - expert_preds)
        
        # Calculate metrics
        metrics = calculator.compute_all_metrics(
            predictions=self.y_pred,
            actual_values=self.y_true,
            expert_contributions=self.expert_weights,
            confidence_scores=self.confidence_scores,
            expert_errors=expert_errors
        )
        
        # Use the calculator's built-in save method
        metrics_file_path = calculator.save_metrics(metrics, name="test_metrics")
        
        # Verify file was created
        self.assertTrue(os.path.exists(metrics_file_path))
        
        # Load metrics from file and verify
        with open(metrics_file_path, 'r') as f:
            loaded_metrics = json.load(f)
        
        # Check that loaded metrics match original
        self.assertEqual(metrics['standard']['rmse'], loaded_metrics['standard']['rmse'])
        self.assertEqual(metrics['standard']['mae'], loaded_metrics['standard']['mae'])


if __name__ == '__main__':
    unittest.main()
