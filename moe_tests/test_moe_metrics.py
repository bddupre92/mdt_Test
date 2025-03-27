"""
Unit tests for MoE-specific evaluation metrics.

This module tests the functionality of the MoE metrics calculator, ensuring
all metrics are calculated correctly and visualizations are generated properly.
"""

import os
import numpy as np
import pandas as pd
import unittest
import json
import tempfile
import shutil
from pathlib import Path
from unittest.mock import patch, MagicMock

from baseline_comparison.moe_metrics import MoEMetricsCalculator


class TestMoEMetricsCalculator(unittest.TestCase):
    """Tests for the MoE metrics calculator."""

    def setUp(self):
        """Create temporary directory for test outputs."""
        self.temp_dir = tempfile.mkdtemp()
        self.metrics_calculator = MoEMetricsCalculator(output_dir=self.temp_dir)
        
        # Sample test data
        np.random.seed(42)
        self.predictions = np.random.normal(0, 1, 100)
        self.actual_values = self.predictions + np.random.normal(0, 0.5, 100)
        self.errors = self.actual_values - self.predictions
        
        # Expert contributions (3 experts)
        self.expert_names = ["expert1", "expert2", "expert3"]
        
        # Create mock expert contributions that sum to 1.0 for each prediction
        contributions = np.random.random((100, 3))
        contributions = contributions / np.sum(contributions, axis=1, keepdims=True)
        
        self.expert_contributions = {
            self.expert_names[i]: contributions[:, i].tolist()
            for i in range(3)
        }
        
        # Expert individual errors
        self.expert_errors = {
            "expert1": (self.errors * 1.2).tolist(),
            "expert2": (self.errors * 0.8).tolist(),
            "expert3": (self.errors * 1.5).tolist()
        }
        
        # Confidence scores (higher = more confident)
        self.confidence_scores = 1.0 - np.abs(self.errors) / (np.max(np.abs(self.errors)) + 1e-10)
        
        # Timestamps for temporal analysis
        self.timestamps = np.arange(100)
        
        # Patient IDs for personalization analysis
        self.patient_ids = np.repeat(np.arange(10), 10)  # 10 patients with 10 samples each

    def tearDown(self):
        """Remove temporary directory."""
        shutil.rmtree(self.temp_dir)

    def test_compute_expert_contribution_metrics(self):
        """Test computation of expert contribution metrics."""
        metrics = self.metrics_calculator.compute_expert_contribution_metrics(
            self.expert_contributions,
            self.errors
        )
        
        # Check basic structure and values
        self.assertEqual(metrics["total_predictions"], 100)
        self.assertIn("expert_mean_contributions", metrics)
        self.assertIn("expert_dominance_count", metrics)
        self.assertIn("expert_dominance_percentage", metrics)
        
        # Check that mean contributions sum approximately to 1.0
        total_mean = sum(metrics["expert_mean_contributions"].values())
        self.assertAlmostEqual(total_mean, 1.0, places=1)
        
        # Check that dominance counts sum to the total
        total_dominance = sum(metrics["expert_dominance_count"].values())
        self.assertEqual(total_dominance, 100)
        
        # Check that normalized entropy is calculated (between 0 and 1)
        self.assertGreaterEqual(metrics["normalized_entropy"], 0.0)
        self.assertLessEqual(metrics["normalized_entropy"], 1.0)

    def test_compute_confidence_metrics(self):
        """Test computation of confidence metrics."""
        metrics = self.metrics_calculator.compute_confidence_metrics(
            self.confidence_scores,
            self.errors
        )
        
        # Check basic structure and values
        self.assertIn("mean_confidence", metrics)
        self.assertIn("confidence_error_correlation", metrics)
        self.assertIn("bin_edges", metrics)
        self.assertIn("bin_mean_errors", metrics)
        self.assertIn("expected_calibration_error", metrics)
        
        # Check ranges
        self.assertGreaterEqual(metrics["mean_confidence"], 0.0)
        self.assertLessEqual(metrics["mean_confidence"], 1.0)
        
        # Check that ECE is between 0 and 1
        if not np.isnan(metrics["expected_calibration_error"]):
            self.assertGreaterEqual(metrics["expected_calibration_error"], 0.0)
            self.assertLessEqual(metrics["expected_calibration_error"], 1.0)
        
        # Check bin edges are properly distributed
        self.assertEqual(len(metrics["bin_edges"]), 11)  # 10 bins = 11 edges
        self.assertEqual(metrics["bin_edges"][0], 0.0)
        self.assertEqual(metrics["bin_edges"][-1], 1.0)

    def test_compute_gating_network_metrics(self):
        """Test computation of gating network metrics."""
        metrics = self.metrics_calculator.compute_gating_network_metrics(
            self.expert_contributions,
            self.expert_errors
        )
        
        # Check basic structure and values
        self.assertIn("optimal_expert_selection_rate", metrics)
        self.assertIn("mean_regret", metrics)
        self.assertIn("mean_top_weight_ratio", metrics)
        self.assertIn("mean_weight_error_correlation", metrics)
        
        # Check reasonable ranges
        self.assertGreaterEqual(metrics["optimal_expert_selection_rate"], 0.0)
        self.assertLessEqual(metrics["optimal_expert_selection_rate"], 1.0)
        
        # Check weight ratio is between 0 and 1
        self.assertGreaterEqual(metrics["mean_top_weight_ratio"], 1.0 / len(self.expert_names))
        self.assertLessEqual(metrics["mean_top_weight_ratio"], 1.0)
        
        # Correlation should be between -1 and 1
        self.assertGreaterEqual(metrics["mean_weight_error_correlation"], -1.0)
        self.assertLessEqual(metrics["mean_weight_error_correlation"], 1.0)

    def test_compute_temporal_metrics(self):
        """Test computation of temporal metrics."""
        metrics = self.metrics_calculator.compute_temporal_metrics(
            self.predictions,
            self.actual_values,
            self.timestamps,
            self.expert_contributions
        )
        
        # Check basic structure and values
        self.assertIn("rmse", metrics)
        self.assertIn("mae", metrics)
        self.assertIn("r2", metrics)
        
        # Check that temporal segments exist
        self.assertIn("temporal_segment_rmse", metrics)
        self.assertIn("temporal_segment_mae", metrics)
        
        # Check error autocorrelation exists and is between -1 and 1
        self.assertIn("error_autocorrelation", metrics)
        self.assertGreaterEqual(metrics["error_autocorrelation"], -1.0)
        self.assertLessEqual(metrics["error_autocorrelation"], 1.0)
        
        # Check expert temporal trends
        self.assertIn("expert_temporal_trends", metrics)
        for expert in self.expert_names:
            self.assertIn(expert, metrics["expert_temporal_trends"])
            self.assertIn("slope", metrics["expert_temporal_trends"][expert])
            self.assertIn("change_percentage", metrics["expert_temporal_trends"][expert])

    def test_compute_personalization_metrics(self):
        """Test computation of personalization metrics."""
        metrics = self.metrics_calculator.compute_personalization_metrics(
            self.predictions,
            self.actual_values,
            self.patient_ids,
            self.expert_contributions
        )
        
        # Check basic structure and values
        self.assertIn("num_patients", metrics)
        self.assertEqual(metrics["num_patients"], 10)  # should be 10 unique patients
        
        self.assertIn("per_patient", metrics)
        self.assertEqual(len(metrics["per_patient"]), 10)  # one entry per patient
        
        # Check mean metrics exist
        self.assertIn("mean_patient_rmse", metrics)
        self.assertIn("patient_rmse_std", metrics)
        
        # Check expert specialization
        self.assertIn("expert_patient_specialization", metrics)
        for expert in self.expert_names:
            self.assertIn(expert, metrics["expert_patient_specialization"])
            
        # Check dominant expert distribution
        self.assertIn("dominant_expert_distribution", metrics)
        total_percentage = sum(metrics["dominant_expert_distribution"].values())
        self.assertAlmostEqual(total_percentage, 100.0, places=1)

    def test_compute_all_metrics(self):
        """Test computing all metrics together."""
        all_metrics = self.metrics_calculator.compute_all_metrics(
            self.predictions,
            self.actual_values,
            self.expert_contributions,
            self.confidence_scores,
            self.expert_errors,
            self.timestamps,
            self.patient_ids
        )
        
        # Check that all metric groups exist
        self.assertIn("standard", all_metrics)
        self.assertIn("expert_contribution", all_metrics)
        self.assertIn("confidence", all_metrics)
        self.assertIn("gating_network", all_metrics)
        self.assertIn("temporal", all_metrics)
        self.assertIn("personalization", all_metrics)

    def test_save_metrics(self):
        """Test saving metrics to JSON file."""
        test_metrics = {
            "test_value": 1.0,
            "nested": {
                "value": 2.0
            },
            "numpy_array": np.array([1, 2, 3]),
            "numpy_float": np.float32(4.5)
        }
        
        output_path = self.metrics_calculator.save_metrics(test_metrics, "test_metrics")
        
        # Check file exists
        self.assertTrue(os.path.exists(output_path))
        
        # Check content is valid JSON
        with open(output_path, 'r') as f:
            loaded = json.load(f)
        
        # Check values were correctly serialized
        self.assertEqual(loaded["test_value"], 1.0)
        self.assertEqual(loaded["nested"]["value"], 2.0)
        self.assertEqual(loaded["numpy_array"], [1, 2, 3])
        self.assertEqual(loaded["numpy_float"], 4.5)

    @patch('matplotlib.pyplot.savefig')
    def test_visualize_metrics(self, mock_savefig):
        """Test visualization generation."""
        test_metrics = {
            "expert_contribution": {
                "expert_dominance_percentage": {
                    "expert1": 40,
                    "expert2": 35,
                    "expert3": 25
                }
            },
            "confidence": {
                "bin_edges": [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
                "bin_mean_errors": [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.05]
            },
            "temporal": {
                "temporal_segment_rmse": [0.9, 0.8, 0.7, 0.6, 0.5],
                "temporal_segment_mae": [0.7, 0.6, 0.5, 0.4, 0.3]
            },
            "personalization": {
                "per_patient": {
                    "p1": {"rmse": 0.5},
                    "p2": {"rmse": 0.6},
                    "p3": {"rmse": 0.4}
                },
                "expert_patient_specialization": {
                    "expert1": {"mean": 0.4, "std": 0.1},
                    "expert2": {"mean": 0.3, "std": 0.05},
                    "expert3": {"mean": 0.3, "std": 0.15}
                },
                "dominant_expert_distribution": {
                    "expert1": 40,
                    "expert2": 35,
                    "expert3": 25
                }
            }
        }
        
        # Test visualization generation
        paths = self.metrics_calculator.visualize_metrics(test_metrics, "test_viz")
        
        # Check that the expected number of visualizations were created
        self.assertEqual(len(paths), 4)
        
        # Check that savefig was called the expected number of times
        self.assertEqual(mock_savefig.call_count, 4)


if __name__ == '__main__':
    unittest.main()
