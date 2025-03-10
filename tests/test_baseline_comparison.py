"""
Unit tests for baseline comparison framework.
"""

import unittest
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import shutil
from unittest.mock import MagicMock, patch

from baseline_comparison import BaselineComparison, ComparisonVisualizer
from baseline_comparison.baseline_algorithms import SATzillaInspiredSelector


class TestProblem:
    """Mock problem class for testing."""
    
    def __init__(self, dimension=2, function_type='sphere'):
        self.dimension = dimension
        self.function_type = function_type
        self.lower_bounds = np.array([-5.0] * dimension)
        self.upper_bounds = np.array([5.0] * dimension)
        self.evaluations = 0
    
    def evaluate(self, x):
        """Simple test function evaluation."""
        self.evaluations += 1
        if self.function_type == 'sphere':
            return np.sum(x**2)
        elif self.function_type == 'rastrigin':
            return 10 * self.dimension + np.sum(x**2 - 10 * np.cos(2 * np.pi * x))
        else:
            return np.sum(x**2)  # Default to sphere
    
    def __str__(self):
        return f"{self.function_type.capitalize()} function ({self.dimension}D)"


class TestSATzillaInspiredSelector(unittest.TestCase):
    """Tests for the SATzilla-inspired algorithm selector."""
    
    def setUp(self):
        self.algorithms = ['DE', 'ES', 'ACO', 'GWO']
        self.selector = SATzillaInspiredSelector(self.algorithms)
        self.test_problem = TestProblem(dimension=2)
    
    def test_initialization(self):
        """Test proper initialization of the selector."""
        self.assertEqual(self.selector.algorithms, self.algorithms)
        self.assertFalse(self.selector.is_trained)
        self.assertIsNotNone(self.selector.feature_scaler)
        self.assertEqual(len(self.selector.performance_models), len(self.algorithms))
    
    def test_feature_extraction(self):
        """Test feature extraction from a problem."""
        features = self.selector.extract_features(self.test_problem)
        
        # Check that we get a dictionary with expected features
        self.assertIsInstance(features, dict)
        self.assertIn('dimension', features)
        self.assertIn('bound_range', features)
        self.assertIn('mean_value', features)
        self.assertIn('std_value', features)
        self.assertIn('modality_estimate', features)
        self.assertIn('hillclimb_progress', features)
        
        # Check specific values
        self.assertEqual(features['dimension'], 2)
        self.assertEqual(features['bound_range'], 10.0)  # upper_bound - lower_bound = 5 - (-5) = 10
    
    def test_train_and_predict(self):
        """Test training and performance prediction."""
        # Create mock training data
        problem_features = [
            {'dimension': 2, 'mean_value': 1.0, 'std_value': 0.5, 'modality_estimate': 1},
            {'dimension': 5, 'mean_value': 2.0, 'std_value': 1.0, 'modality_estimate': 2},
            {'dimension': 10, 'mean_value': 3.0, 'std_value': 1.5, 'modality_estimate': 3}
        ]
        
        algorithm_performances = {
            'DE': [0.1, 0.2, 0.3],
            'ES': [0.2, 0.1, 0.4],
            'ACO': [0.3, 0.3, 0.2],
            'GWO': [0.4, 0.4, 0.1]
        }
        
        # Train the selector
        self.selector.train(problem_features, algorithm_performances)
        
        # Check that it's marked as trained
        self.assertTrue(self.selector.is_trained)
        
        # Test prediction and selection
        test_features = {'dimension': 2, 'mean_value': 1.0, 'std_value': 0.5, 'modality_estimate': 1}
        predictions = self.selector.predict_performance(test_features)
        
        # Check that we get predictions for all algorithms
        self.assertEqual(set(predictions.keys()), set(self.algorithms))
        
        # Test algorithm selection
        selected_algorithm = self.selector.select_algorithm(test_features)
        self.assertIn(selected_algorithm, self.algorithms)
        
        # Test selection confidence
        confidence = self.selector.get_selection_confidence(test_features)
        self.assertEqual(set(confidence.keys()), set(self.algorithms))
        self.assertAlmostEqual(sum(confidence.values()), 1.0, places=5)


class TestBaselineComparison(unittest.TestCase):
    """Tests for the baseline comparison framework."""
    
    def setUp(self):
        self.algorithms = ['DE', 'ES', 'ACO', 'GWO']
        
        # Mock Meta Optimizer
        self.meta_optimizer = MagicMock()
        self.meta_optimizer.select_algorithm = MagicMock(return_value='DE')
        
        # Mock problem generator
        def problem_generator():
            return TestProblem(dimension=2)
        
        self.problem_generator = problem_generator
        
        # Create baseline comparison
        self.comparison = BaselineComparison(
            self.meta_optimizer, 
            self.problem_generator, 
            self.algorithms
        )
        
        # Mock the _run_algorithm method to avoid actual optimization
        self.comparison._run_algorithm = MagicMock(return_value={
            'algorithm': 'DE',
            'best_value': 0.1,
            'evaluations': 100,
            'runtime': 0.5,
            'convergence': [1.0, 0.5, 0.2, 0.1]
        })
        
        # Patch the actual optimization function
        self.patcher = patch(
            'baseline_comparison.comparison_runner.BaselineComparison._get_meta_optimizer_selection',
            return_value='DE'
        )
        self.mock_get_selection = self.patcher.start()
    
    def tearDown(self):
        self.patcher.stop()
    
    def test_initialization(self):
        """Test proper initialization of the comparison framework."""
        self.assertEqual(self.comparison.algorithms, self.algorithms)
        self.assertEqual(self.comparison.meta_optimizer, self.meta_optimizer)
        self.assertIsNotNone(self.comparison.satzilla_selector)
        self.assertIn('meta_optimizer', self.comparison.results)
        self.assertIn('satzilla', self.comparison.results)
        for algo in self.algorithms:
            self.assertIn(algo, self.comparison.results)
    
    def test_run_training_phase(self):
        """Test training phase with a small number of problems."""
        # Run training with just 3 problems
        self.comparison.run_training_phase(n_problems=3)
        
        # Check that features and performances were collected
        self.assertEqual(len(self.comparison.features_db), 3)
        for algo in self.algorithms:
            self.assertEqual(len(self.comparison.performance_db[algo]), 3)
    
    def test_run_comparison(self):
        """Test comparison phase with a small number of problems."""
        # First run training
        self.comparison.run_training_phase(n_problems=3)
        
        # Then run comparison
        results = self.comparison.run_comparison(n_problems=2, verbose=False)
        
        # Check that we have results for all methods
        self.assertIn('meta_optimizer', results)
        self.assertIn('satzilla', results)
        for algo in self.algorithms:
            self.assertIn(algo, results)
        
        # Check result lengths
        self.assertEqual(len(results['meta_optimizer']), 2)
        self.assertEqual(len(results['satzilla']), 2)
        for algo in self.algorithms:
            self.assertEqual(len(results[algo]), 2)
    
    def test_get_summary_dataframe(self):
        """Test getting a summary DataFrame."""
        # Populate results with dummy data
        for method in self.comparison.results:
            self.comparison.results[method] = [0.1, 0.2, 0.3]
        
        # Get summary DataFrame
        summary_df = self.comparison.get_summary_dataframe()
        
        # Check DataFrame structure
        self.assertIsInstance(summary_df, pd.DataFrame)
        self.assertIn('Method', summary_df.columns)
        self.assertIn('Mean', summary_df.columns)
        self.assertIn('Std', summary_df.columns)
        
        # Check methods are included
        methods = set(summary_df['Method'].values)
        self.assertIn('meta_optimizer', methods)
        self.assertIn('satzilla', methods)
        for algo in self.algorithms:
            self.assertIn(algo, methods)


class TestComparisonVisualizer(unittest.TestCase):
    """Tests for the comparison visualizer."""
    
    def setUp(self):
        # Create a temporary directory for visualizations
        self.test_dir = 'tests/tmp_visualization'
        os.makedirs(self.test_dir, exist_ok=True)
        
        # Create dummy results
        self.results = {
            'meta_optimizer': [0.1, 0.2, 0.15, 0.12],
            'satzilla': [0.2, 0.3, 0.25, 0.22],
            'DE': [0.3, 0.4, 0.35, 0.32],
            'ES': [0.4, 0.5, 0.45, 0.42]
        }
        
        # Create visualizer
        self.visualizer = ComparisonVisualizer(self.results, export_dir=self.test_dir)
    
    def tearDown(self):
        # Clean up temporary directory
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)
    
    def test_initialization(self):
        """Test proper initialization of the visualizer."""
        self.assertEqual(self.visualizer.results, self.results)
        self.assertEqual(self.visualizer.export_dir, self.test_dir)
        self.assertIsInstance(self.visualizer.df, pd.DataFrame)
        
        # Check all methods are included
        self.assertEqual(set(self.visualizer.methods), set(self.results.keys()))
    
    def test_head_to_head_comparison(self):
        """Test creation of head-to-head comparison plot."""
        # Generate plot
        fig = self.visualizer.head_to_head_comparison(save=True)
        
        # Check that the figure was created
        self.assertIsInstance(fig, plt.Figure)
        
        # Check that the file was saved
        self.assertTrue(os.path.exists(f"{self.test_dir}/head_to_head_comparison.png"))
    
    def test_performance_profile(self):
        """Test creation of performance profile plot."""
        # Generate plot
        fig = self.visualizer.performance_profile(save=True)
        
        # Check that the figure was created
        self.assertIsInstance(fig, plt.Figure)
        
        # Check that the file was saved
        self.assertTrue(os.path.exists(f"{self.test_dir}/performance_profile.png"))
    
    def test_rank_table(self):
        """Test creation of rank table."""
        # Generate table
        rank_df = self.visualizer.rank_table(save=True)
        
        # Check that the DataFrame was created
        self.assertIsInstance(rank_df, pd.DataFrame)
        
        # Check that the files were saved
        self.assertTrue(os.path.exists(f"{self.test_dir}/rank_table.csv"))
        self.assertTrue(os.path.exists(f"{self.test_dir}/rank_table.png"))
    
    def test_critical_difference_diagram(self):
        """Test creation of critical difference diagram."""
        # Generate plot
        fig = self.visualizer.critical_difference_diagram(save=True)
        
        # Check that the figure was created
        self.assertIsInstance(fig, plt.Figure)
        
        # Check that the file was saved
        self.assertTrue(os.path.exists(f"{self.test_dir}/critical_difference.png"))
    
    def test_improvement_heatmap(self):
        """Test creation of improvement heatmap."""
        # Generate plot
        fig = self.visualizer.improvement_heatmap(save=True)
        
        # Check that the figure was created
        self.assertIsInstance(fig, plt.Figure)
        
        # Check that the file was saved
        self.assertTrue(os.path.exists(f"{self.test_dir}/improvement_heatmap.png"))
    
    def test_create_all_visualizations(self):
        """Test creation of all visualizations at once."""
        # Generate all visualizations
        self.visualizer.create_all_visualizations()
        
        # Check that all files were saved
        self.assertTrue(os.path.exists(f"{self.test_dir}/head_to_head_comparison.png"))
        self.assertTrue(os.path.exists(f"{self.test_dir}/performance_profile.png"))
        self.assertTrue(os.path.exists(f"{self.test_dir}/rank_table.csv"))
        self.assertTrue(os.path.exists(f"{self.test_dir}/critical_difference.png"))
        self.assertTrue(os.path.exists(f"{self.test_dir}/improvement_heatmap.png"))


if __name__ == '__main__':
    unittest.main() 