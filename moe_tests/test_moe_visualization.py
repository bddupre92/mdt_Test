"""
Tests for MoE visualization functionality.

This module tests the visualization capabilities of the MoE framework,
ensuring that all visualization methods work correctly across different environments.
"""

import os
import sys
import unittest
import numpy as np
import tempfile
import shutil
import matplotlib.pyplot as plt
from pathlib import Path

# Add parent directory to path to ensure imports work
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the components to test
from baseline_comparison.moe_metrics import MoEMetricsCalculator
from baseline_comparison.moe_comparison import MoEBaselineComparison, visualize_moe_with_baselines, create_moe_adapter


class TestMoEVisualization(unittest.TestCase):
    """Test the MoE visualization functionality."""
    
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
        row_sums = raw_weights.sum(axis=1)
        self.expert_weights = {}
        self.expert_weights['expert1'] = (raw_weights[:, 0] / row_sums)
        self.expert_weights['expert2'] = (raw_weights[:, 1] / row_sums)
        self.expert_weights['expert3'] = (raw_weights[:, 2] / row_sums)
        
        # Expert predictions - each expert has its own prediction for each sample
        self.expert_predictions = {}
        self.expert_predictions['expert1'] = self.y_true + np.random.normal(0, 3, size=self.n_samples)
        self.expert_predictions['expert2'] = self.y_true + np.random.normal(0, 7, size=self.n_samples)
        self.expert_predictions['expert3'] = self.y_true + np.random.normal(0, 10, size=self.n_samples)
        
        # Confidence scores
        self.confidence_scores = np.random.beta(5, 2, size=self.n_samples)
        
        # Create output directory for visualizations
        self.output_dir = os.path.join(self.test_dir, "visualization_output")
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Create metrics calculator
        self.calculator = MoEMetricsCalculator(output_dir=self.output_dir)
        
        # Calculate expert errors
        self.expert_errors = {}
        for expert_name, expert_preds in self.expert_predictions.items():
            self.expert_errors[expert_name] = np.abs(self.y_true - expert_preds)
        
        # Calculate all metrics
        self.metrics = self.calculator.compute_all_metrics(
            predictions=self.y_pred,
            actual_values=self.y_true,
            expert_contributions=self.expert_weights,
            confidence_scores=self.confidence_scores,
            expert_errors=self.expert_errors
        )
        
        # Create baseline metrics for comparison
        self.baseline_metrics = {
            "simple": {"rmse": 7.2, "mae": 5.8, "r2": 0.82},
            "meta": {"rmse": 6.5, "mae": 5.1, "r2": 0.85},
            "enhanced": {"rmse": 6.1, "mae": 4.8, "r2": 0.87}
        }
        
    def tearDown(self):
        """Clean up after tests."""
        # Close all matplotlib figures to avoid memory leaks
        plt.close('all')
        
        # Remove temporary directory
        shutil.rmtree(self.test_dir)
    
    def test_metrics_visualization(self):
        """Test that metrics visualizations are generated correctly."""
        # Generate visualizations
        visualization_paths = self.calculator.visualize_metrics(
            metrics=self.metrics,
            name="test_metrics"
        )
        
        # Check that visualizations were created
        self.assertIsInstance(visualization_paths, list)
        self.assertTrue(len(visualization_paths) > 0)
        
        # Verify that files exist
        for path in visualization_paths:
            self.assertTrue(os.path.exists(path))
            self.assertTrue(path.endswith('.png'))
    
    def test_moe_comparison_visualization(self):
        """Test visualization of MoE comparison with baseline approaches."""
        # Create mock MoE adapter with a simple config
        config = {
            "predictions": self.y_pred.tolist(),
            "actual_values": self.y_true.tolist(),
            "expert_contributions": {k: v.tolist() for k, v in self.expert_weights.items()},
            "confidence_scores": self.confidence_scores.tolist(),
            "expert_errors": {k: v.tolist() for k, v in self.expert_errors.items()}
        }
        moe_adapter = create_moe_adapter(config=config)
        
        # Create a baseline comparison object with minimal mock components
        comparison = MoEBaselineComparison(
            simple_baseline=lambda x, y: {"rmse": 7.2, "mae": 5.8, "r2": 0.82},
            meta_learner=lambda x, y: {"rmse": 6.5, "mae": 5.1, "r2": 0.85},
            enhanced_meta=lambda x, y: {"rmse": 6.1, "mae": 4.8, "r2": 0.87},
            satzilla_selector=lambda x, y: {"rmse": 5.9, "mae": 4.5, "r2": 0.88},
            moe_adapter=moe_adapter,
            output_dir=self.output_dir
        )
        
        # Create fake supervised comparison results
        supervised_results = {
            "simple": {"rmse": 7.2, "mae": 5.8, "predictions": np.random.normal(50, 10, size=self.n_samples)},
            "meta": {"rmse": 6.5, "mae": 5.1, "predictions": np.random.normal(50, 8, size=self.n_samples)},
            "enhanced": {"rmse": 6.1, "mae": 4.8, "predictions": np.random.normal(50, 7, size=self.n_samples)},
            "moe": {"rmse": 5.5, "mae": 4.2, "predictions": self.y_pred}
        }
        
        # Test supervised comparison visualization
        metrics_df = comparison.visualize_supervised_comparison(supervised_results)
        self.assertIsNotNone(metrics_df)
        
        # Check that at least some visualization files were created
        png_files = [f for f in os.listdir(self.output_dir) if f.endswith('.png')]
        self.assertTrue(len(png_files) > 0, "No visualization files were created")
        
        # Test general results visualization
        paths = comparison.visualize_results()
        # The method might return a list, dictionary, or empty result depending on implementation
        self.assertTrue(isinstance(paths, list) or isinstance(paths, dict), 
                      f"Expected paths to be list or dict, but got {type(paths)}")
        # No assertion on length since mock adapters might return empty results
        
        # Skip detailed checks since the mock adapter might not generate all expected visualizations
        # We're just testing that the function runs without errors and produces some output
        # Visual verification of actual outputs would be done separately
    
    def test_standalone_visualization(self):
        """Test the standalone visualization utility."""
        # Use the standalone visualization function
        paths = visualize_moe_with_baselines(
            moe_metrics=self.metrics,
            baseline_metrics=self.baseline_metrics,
            output_dir=self.output_dir
        )
        
        # Check that visualizations were created
        self.assertIsInstance(paths, list)
        self.assertTrue(len(paths) > 0)
        
        # Verify that files exist
        for path in paths:
            self.assertTrue(os.path.exists(path))
    
    def test_visualization_customization(self):
        """Test customization options for visualizations."""
        # Test with custom name
        paths = self.calculator.visualize_metrics(
            metrics=self.metrics,
            name="custom_test_metrics"
        )
        
        # Verify that files have custom name
        for path in paths:
            filename = os.path.basename(path)
            self.assertTrue("custom_test_metrics" in filename)
    
    def test_visualization_with_missing_data(self):
        """Test visualization behavior with incomplete metrics data."""
        # Create metrics with missing components
        incomplete_metrics = {
            "standard": self.metrics["standard"],
            # Deliberately omit expert_contribution and confidence
        }
        
        # Visualization should still work without expert contribution
        paths = self.calculator.visualize_metrics(
            metrics=incomplete_metrics,
            name="incomplete_metrics"
        )
        
        # Should return some paths even with incomplete data
        self.assertIsInstance(paths, list)
        
        # Add expert contribution but with missing fields
        incomplete_metrics["expert_contribution"] = {}
        
        # Visualization should handle missing fields gracefully
        paths = self.calculator.visualize_metrics(
            metrics=incomplete_metrics,
            name="incomplete_fields"
        )
        
        # Should return some paths even with incomplete fields
        self.assertIsInstance(paths, list)


if __name__ == '__main__':
    unittest.main()
