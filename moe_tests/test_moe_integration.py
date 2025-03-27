"""
Test the integration of MoE metrics with the baseline comparison framework.

This module tests the MoEBaselineAdapter and MoEBaselineComparison's ability
to generate and compare MoE-specific metrics.
"""

import os
import sys
import unittest
import numpy as np
import pandas as pd
import tempfile
import shutil
from unittest.mock import MagicMock, patch
from pathlib import Path

# Add parent directory to path to ensure imports work
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import from baseline comparison
from baseline_comparison.moe_adapter import MoEBaselineAdapter
from baseline_comparison.moe_comparison import MoEBaselineComparison, create_moe_adapter
from baseline_comparison.moe_metrics import MoEMetricsCalculator

# Import mock components for testing
from moe_tests.test_utils import generate_test_data, MockExpert, MockGatingNetwork
from moe_tests.test_mocks import MoEEventTypes, MockMoEPipeline, MockMoEMetricsCalculator


class TestMoEBaselineAdapter(unittest.TestCase):
    """Test the MoEBaselineAdapter with focus on metrics integration."""
    
    def setUp(self):
        """Set up the test environment."""
        # Create a temporary directory for test outputs
        self.test_dir = tempfile.mkdtemp()
        
        # Mock MoEPipeline
        self.mock_moe_pipeline = MagicMock()
        
        # Create test data
        self.X_train, self.y_train, self.X_test, self.y_test = generate_test_data(
            n_samples=100,
            n_features=5,
            n_clusters=3,
            random_state=42
        )
        
        # Add timestamp and patient_id columns
        self.X_train['timestamp'] = pd.date_range(start='2023-01-01', periods=100, freq='D')
        self.X_train['patient_id'] = np.random.randint(1, 11, size=100)
        self.X_test['timestamp'] = pd.date_range(start='2023-04-10', periods=100, freq='D')
        self.X_test['patient_id'] = np.random.randint(1, 11, size=100)
        
        # Initialize adapter with mocked pipeline
        self.adapter_config = {
            'time_column': 'timestamp',
            'patient_column': 'patient_id',
            'experts': {
                'expert1': {'model_type': 'linear'},
                'expert2': {'model_type': 'tree'},
                'expert3': {'model_type': 'svm'}
            }
        }
        
        # Set up the metrics output directory
        self.metrics_output_dir = os.path.join(self.test_dir, "metrics")
        os.makedirs(self.metrics_output_dir, exist_ok=True)
        
    def tearDown(self):
        """Clean up after tests."""
        # Remove temporary directory
        shutil.rmtree(self.test_dir)
    
    @patch('baseline_comparison.moe_metrics.MoEMetricsCalculator')
    @patch('baseline_comparison.moe_adapter.MoEEventTypes', MoEEventTypes)
    @patch('baseline_comparison.moe_adapter.MoEPipeline')
    def test_adapter_metrics_calculation(self, mock_pipeline_class, mock_event_types, mock_metrics_calculator):
        """Test that the adapter correctly calculates metrics."""
        # Set up mock pipeline behavior
        # Configure the mock pipeline
        mock_pipeline_instance = MockMoEPipeline(config=self.adapter_config)
        mock_pipeline_class.return_value = mock_pipeline_instance
        
        # Configure the mock metrics calculator
        mock_calculator_instance = MockMoEMetricsCalculator()
        mock_metrics_calculator.return_value = mock_calculator_instance
        
        # Create adapter with mock pipeline
        adapter = MoEBaselineAdapter(
            config=self.adapter_config, 
            verbose=False,
            metrics_output_dir=self.metrics_output_dir
        )
        adapter.moe_pipeline = mock_pipeline
        
        # Train the adapter
        adapter.train(self.X_train, self.y_train)
        
        # Make predictions with ground truth
        predictions = adapter.predict(self.X_test, y=self.y_test)
        
        # Compute metrics
        metrics = adapter.compute_metrics("test_run")
        
        # Verify metrics were computed
        self.assertIsInstance(metrics, dict)
        self.assertIn('standard', metrics)
        self.assertIn('expert_contribution', metrics)
        self.assertIn('confidence', metrics)
        
        # Check if files were created
        self.assertTrue(os.path.exists(os.path.join(self.metrics_output_dir, "test_run_metrics.json")))
        
        # Get performance summary
        summary = adapter.get_performance_summary()
        self.assertIsInstance(summary, dict)
        
    @patch('baseline_comparison.moe_metrics.MoEMetricsCalculator', MockMoEMetricsCalculator)
    @patch('baseline_comparison.moe_adapter.MoEEventTypes', MoEEventTypes)
    def test_end_to_end_metrics_integration(self, mock_event_types, mock_metrics_class):
        """Test end-to-end integration with metrics calculation."""
        # Create a simplified adapter with mocked components
        with patch('baseline_comparison.moe_adapter.MoEPipeline', MockMoEPipeline):
            adapter = MoEBaselineAdapter(
                config=self.adapter_config, 
                verbose=False,
                metrics_output_dir=self.metrics_output_dir
            )
            
            # Train the adapter (this will use our mocked pipeline)
            adapter.train(self.X_train, self.y_train)
        
        # Make predictions
        predictions = adapter.predict(self.X_test, y=self.y_test)
        
        # Compute metrics
        metrics = adapter.compute_metrics("end_to_end_test")
        
        # Check metrics structure
        self.assertIn('expert_contribution', metrics)
        self.assertIn('normalized_entropy', metrics['expert_contribution'])
        self.assertIn('expert_dominance_counts', metrics['expert_contribution'])
        
        # Check confidence metrics
        self.assertIn('confidence', metrics)
        self.assertIn('mean_confidence', metrics['confidence'])
        
        # Check if visualization files were created
        self.assertTrue(os.path.exists(os.path.join(self.metrics_output_dir, "end_to_end_test_expert_contributions.png")))
        
        # Get performance summary
        summary = adapter.get_performance_summary()
        self.assertIn('expert_distribution', summary)
        self.assertIn('expert_diversity', summary)
        self.assertIn('mean_confidence', summary)


class TestMoEBaselineComparison(unittest.TestCase):
    """Test the MoEBaselineComparison with focus on metrics comparison."""
    
    def setUp(self):
        """Set up the test environment."""
        # Create a temporary directory for test outputs
        self.test_dir = tempfile.mkdtemp()
        
        # Create test data
        self.X_train, self.y_train, self.X_test, self.y_test = generate_test_data(
            n_samples=100,
            n_features=5,
            n_clusters=3,
            random_state=42
        )
        
        # Add timestamp and patient_id columns
        self.X_train['timestamp'] = pd.date_range(start='2023-01-01', periods=100, freq='D')
        self.X_train['patient_id'] = np.random.randint(1, 11, size=100)
        self.X_test['timestamp'] = pd.date_range(start='2023-04-10', periods=100, freq='D')
        self.X_test['patient_id'] = np.random.randint(1, 11, size=100)
        
        # Create mock components for comparison
        self.mock_simple_baseline = MagicMock()
        self.mock_meta_learner = MagicMock()
        self.mock_enhanced_meta = MagicMock()
        self.mock_satzilla = MagicMock()
        
        # Set up the metrics output directory
        self.metrics_output_dir = os.path.join(self.test_dir, "comparison_metrics")
        os.makedirs(self.metrics_output_dir, exist_ok=True)
        
        # Configure mock components to return predictable results
        self.mock_simple_baseline.predict.return_value = self.y_test.values + np.random.normal(0, 0.5, size=len(self.y_test))
        self.mock_meta_learner.predict.return_value = self.y_test.values + np.random.normal(0, 0.4, size=len(self.y_test))
        self.mock_enhanced_meta.predict.return_value = self.y_test.values + np.random.normal(0, 0.3, size=len(self.y_test))
        self.mock_satzilla.predict.return_value = self.y_test.values + np.random.normal(0, 0.35, size=len(self.y_test))
        
    def tearDown(self):
        """Clean up after tests."""
        # Remove temporary directory
        shutil.rmtree(self.test_dir)
    
    @patch('baseline_comparison.moe_metrics.MoEMetricsCalculator', MockMoEMetricsCalculator)
    @patch('baseline_comparison.moe_adapter.MoEEventTypes', MoEEventTypes)
    def test_supervised_comparison(self, mock_event_types, mock_metrics_class):
        """Test the supervised comparison with MoE metrics."""
        # Create MoE adapter with mocked pipeline
        adapter_config = {
            'time_column': 'timestamp',
            'patient_column': 'patient_id',
            'experts': {
                'expert1': {'model_type': 'linear'},
                'expert2': {'model_type': 'tree'},
                'expert3': {'model_type': 'svm'}
            }
        }
        
        # Use our mock pipeline instead of the real one
        with patch('baseline_comparison.moe_adapter.MoEPipeline', MockMoEPipeline):
            adapter = MoEBaselineAdapter(
                config=adapter_config, 
                verbose=False,
                metrics_output_dir=self.metrics_output_dir
            )
        
        # Create comparison framework with mocked baselines and real MoE adapter
        comparison = MoEBaselineComparison(
            simple_baseline=self.mock_simple_baseline,
            meta_learner=self.mock_meta_learner,
            enhanced_meta=self.mock_enhanced_meta,
            satzilla_selector=self.mock_satzilla,
            moe_adapter=adapter,
            output_dir=self.metrics_output_dir
        )
        
        # Run supervised comparison
        results = comparison.run_supervised_comparison(
            X_train=self.X_train,
            y_train=self.y_train,
            X_test=self.X_test,
            y_test=self.y_test
        )
        
        # Check results structure
        self.assertIn('simple_baseline', results)
        self.assertIn('meta_learner', results)
        self.assertIn('enhanced_meta', results)
        self.assertIn('satzilla_selector', results)
        self.assertIn('moe', results)
        
        # Check MoE metrics structure
        self.assertIn('standard', results['moe'])
        self.assertIn('moe_specific', results['moe'])
        
        # Visualize the results
        metrics_df = comparison.visualize_supervised_comparison(
            results=results,
            output_dir=self.metrics_output_dir,
            prefix="test_supervised"
        )
        
        # Check if visualization files were created
        self.assertTrue(os.path.exists(os.path.join(self.metrics_output_dir, "test_supervised_rmse_comparison.png")))
        self.assertTrue(os.path.exists(os.path.join(self.metrics_output_dir, "test_supervised_mae_comparison.png")))
        
        # Check if MoE-specific visualizations were created
        self.assertTrue(os.path.exists(os.path.join(self.metrics_output_dir, "test_supervised_full_results.json")))
        

if __name__ == '__main__':
    unittest.main()
