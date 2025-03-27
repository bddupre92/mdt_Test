"""
Tests for the MoE Baseline Comparison framework.

This module contains tests for the MoEBaselineComparison class, which extends
the baseline comparison framework to include MoE as a selection approach.
"""

import os
import pytest
import numpy as np
import pandas as pd
from unittest.mock import MagicMock, patch

from baseline_comparison.moe_comparison import MoEBaselineComparison, create_moe_adapter
from baseline_comparison.moe_adapter import MoEBaselineAdapter


class MockSelector:
    """Mock selector for testing."""
    
    def __init__(self, algorithms=None):
        self.algorithms = algorithms or {}
        self.selected = []
    
    def select_algorithm(self, problem_instance):
        algorithm = list(self.algorithms.keys())[0] if self.algorithms else "default"
        self.selected.append(algorithm)
        return algorithm
    
    def run_optimizer(self, problem_instance, max_evaluations=1000):
        return {
            "best_fitness": 0.5,
            "evaluations": 100,
            "convergence_data": [0.9, 0.7, 0.5],
            "selected_algorithm": self.select_algorithm(problem_instance)
        }
    
    def cross_validate(self, X, y, n_splits=5):
        return {
            "mean_scores": {"rmse": 0.5, "mae": 0.4},
            "fold_scores": {"rmse": [0.5] * n_splits, "mae": [0.4] * n_splits}
        }


@pytest.fixture
def sample_data():
    """Create sample data for testing."""
    # Create a simple dataset with timestamped data for two patients
    np.random.seed(42)
    
    data = {
        'timestamp': pd.date_range(start='2025-01-01', periods=100, freq='H'),
        'patient_id': [1] * 50 + [2] * 50,
        'feature1': np.random.randn(100),
        'feature2': np.random.randn(100),
        'feature3': np.random.randn(100),
        'target': np.random.randn(100)
    }
    
    return pd.DataFrame(data)


@pytest.fixture
def mock_selectors():
    """Create mock selectors for testing."""
    return {
        "simple": MockSelector({"algorithm1": {}, "algorithm2": {}}),
        "meta": MockSelector({"algorithm1": {}, "algorithm2": {}}),
        "enhanced": MockSelector({"algorithm1": {}, "algorithm2": {}}),
        "satzilla": MockSelector({"algorithm1": {}, "algorithm2": {}})
    }


@pytest.fixture
def mock_moe_adapter():
    """Create a mock MoE adapter for testing."""
    adapter = MagicMock(spec=MoEBaselineAdapter)
    adapter.get_available_algorithms.return_value = ["expert1", "expert2"]
    adapter.run_optimizer.return_value = {
        "best_fitness": 0.4,
        "evaluations": 120,
        "convergence_data": [0.8, 0.6, 0.4],
        "selected_algorithm": "expert1"
    }
    adapter.cross_validate.return_value = {
        "mean_scores": {"rmse": 0.4, "mae": 0.3},
        "fold_scores": {"rmse": [0.4, 0.4, 0.4], "mae": [0.3, 0.3, 0.3]}
    }
    
    return adapter


def test_initialization(mock_selectors, mock_moe_adapter):
    """Test that the MoEBaselineComparison initializes correctly."""
    # Create comparison framework
    comparison = MoEBaselineComparison(
        simple_baseline=mock_selectors["simple"],
        meta_learner=mock_selectors["meta"],
        enhanced_meta=mock_selectors["enhanced"],
        satzilla_selector=mock_selectors["satzilla"],
        moe_adapter=mock_moe_adapter,
        verbose=False
    )
    
    # Check that MoE was added to optimizers
    assert "moe" in comparison.optimizers
    assert comparison.optimizers["moe"] == mock_moe_adapter
    
    # Check that MoE was added to results
    assert "moe" in comparison.results
    assert "best_fitness" in comparison.results["moe"]
    
    # Check that set_available_algorithms was called
    mock_moe_adapter.set_available_algorithms.assert_called_once()


def test_initialization_with_config():
    """Test initialization with a configuration instead of an adapter."""
    with patch('baseline_comparison.moe_comparison.MoEBaselineAdapter') as MockAdapter:
        # Setup mock adapter instance
        mock_instance = MagicMock()
        MockAdapter.return_value = mock_instance
        
        # Create mock selectors
        mock_selectors = {
            "simple": MockSelector(),
            "meta": MockSelector(),
            "enhanced": MockSelector(),
            "satzilla": MockSelector()
        }
        
        # Test config
        test_config = {"target_column": "target"}
        
        # Create comparison framework with config
        comparison = MoEBaselineComparison(
            simple_baseline=mock_selectors["simple"],
            meta_learner=mock_selectors["meta"],
            enhanced_meta=mock_selectors["enhanced"],
            satzilla_selector=mock_selectors["satzilla"],
            moe_config=test_config,
            verbose=False
        )
        
        # Check that MoEBaselineAdapter was created with the config
        MockAdapter.assert_called_once()
        assert MockAdapter.call_args[1]["config"] == test_config


def test_run_comparison(mock_selectors, mock_moe_adapter, sample_data):
    """Test running a comparison with MoE included."""
    # Create comparison framework
    comparison = MoEBaselineComparison(
        simple_baseline=mock_selectors["simple"],
        meta_learner=mock_selectors["meta"],
        enhanced_meta=mock_selectors["enhanced"],
        satzilla_selector=mock_selectors["satzilla"],
        moe_adapter=mock_moe_adapter,
        verbose=False
    )
    
    # Define a simple problem function
    def problem_func(dimensions=None):
        return sample_data
    
    # Patch the super().run_comparison method to avoid calling it
    with patch('baseline_comparison.comparison_runner.BaselineComparison.run_comparison') as mock_super_run:
        mock_super_run.return_value = {"simple": [], "meta": [], "enhanced": [], "satzilla": []}
        
        # Run comparison
        results = comparison.run_comparison(
            problem_name="test_problem",
            problem_func=problem_func,
            dimensions=sample_data.shape[1],
            max_evaluations=100,
            num_trials=1
        )
        
        # Check that super().run_comparison was called
        mock_super_run.assert_called_once()
        

def test_cross_validate_all(mock_selectors, mock_moe_adapter, sample_data):
    """Test cross-validation for all selectors including MoE."""
    # Create comparison framework
    comparison = MoEBaselineComparison(
        simple_baseline=mock_selectors["simple"],
        meta_learner=mock_selectors["meta"],
        enhanced_meta=mock_selectors["enhanced"],
        satzilla_selector=mock_selectors["satzilla"],
        moe_adapter=mock_moe_adapter,
        verbose=False
    )
    
    # Prepare data
    X = sample_data.drop(columns=['target'])
    y = sample_data['target']
    
    # Run cross-validation
    results = comparison.cross_validate_all(X, y, n_splits=3, method='patient_aware')
    
    # Check that MoE cross-validation was called
    mock_moe_adapter.cross_validate.assert_called_once()
    assert mock_moe_adapter.cross_validate.call_args[0][0].equals(X)
    assert mock_moe_adapter.cross_validate.call_args[0][1].equals(y)
    assert mock_moe_adapter.cross_validate.call_args[1]['n_splits'] == 3
    assert mock_moe_adapter.cross_validate.call_args[1]['method'] == 'patient_aware'
    
    # Check that results include MoE
    assert 'moe' in results


def test_prepare_dataset_for_moe(mock_selectors, mock_moe_adapter, sample_data):
    """Test the prepare_dataset_for_moe method."""
    # Create comparison framework
    comparison = MoEBaselineComparison(
        simple_baseline=mock_selectors["simple"],
        meta_learner=mock_selectors["meta"],
        enhanced_meta=mock_selectors["enhanced"],
        satzilla_selector=mock_selectors["satzilla"],
        moe_adapter=mock_moe_adapter,
        verbose=False
    )
    
    # Test with DataFrame
    df_result = comparison.prepare_dataset_for_moe(sample_data)
    assert df_result.equals(sample_data)
    
    # Test with object that has get_features method
    mock_problem = MagicMock()
    feature_dict = {'feature1': [1, 2], 'feature2': [3, 4]}
    target_vals = [10, 20]
    mock_problem.get_features.return_value = feature_dict
    mock_problem.get_target.return_value = target_vals
    
    obj_result = comparison.prepare_dataset_for_moe(mock_problem)
    assert 'feature1' in obj_result.columns
    assert 'feature2' in obj_result.columns
    assert 'target' in obj_result.columns


def test_get_summary_with_moe(mock_selectors, mock_moe_adapter):
    """Test getting a summary with MoE results included."""
    # Create comparison framework
    comparison = MoEBaselineComparison(
        simple_baseline=mock_selectors["simple"],
        meta_learner=mock_selectors["meta"],
        enhanced_meta=mock_selectors["enhanced"],
        satzilla_selector=mock_selectors["satzilla"],
        moe_adapter=mock_moe_adapter,
        verbose=False
    )
    
    # Add some results for MoE
    comparison.results["moe"]["best_fitness"] = [0.3, 0.4, 0.5]
    comparison.results["moe"]["evaluations"] = [100, 120, 140]
    comparison.results["moe"]["time"] = [1.0, 1.2, 1.4]
    
    # Add results for other selectors
    for selector in ["simple", "meta", "enhanced", "satzilla"]:
        comparison.results[selector]["best_fitness"] = [0.6, 0.7, 0.8]
        comparison.results[selector]["evaluations"] = [200, 220, 240]
        comparison.results[selector]["time"] = [2.0, 2.2, 2.4]
    
    # Mock the parent get_summary_dataframe method
    with patch('baseline_comparison.comparison_runner.BaselineComparison.get_summary_dataframe') as mock_get_summary:
        # Create a mock DataFrame for the parent method to return
        parent_summary = pd.DataFrame({
            'Selector': ['Simple', 'Meta', 'Enhanced', 'SATzilla'],
            'Best Fitness (mean)': [0.7, 0.7, 0.7, 0.7],
            'Best Fitness (std)': [0.1, 0.1, 0.1, 0.1],
            'Evaluations (mean)': [220, 220, 220, 220],
            'Time (mean)': [2.2, 2.2, 2.2, 2.2],
            'Success Rate': [0.67, 0.67, 0.67, 0.67],
            'Overall Rank': [2, 3, 4, 5]
        })
        mock_get_summary.return_value = parent_summary
        
        # Get summary
        summary = comparison.get_summary_with_moe()
        
        # Check that MoE was added
        assert 'MoE' in summary['Selector'].values
        
        # Check that MoE has better fitness (lower is better)
        moe_row = summary[summary['Selector'] == 'MoE']
        assert moe_row['Best Fitness (mean)'].values[0] == 0.4
        
        # Check that the overall rank was recalculated
        assert len(summary['Overall Rank'].unique()) == 5  # 5 different ranks


def test_create_moe_adapter():
    """Test the create_moe_adapter helper function."""
    with patch('baseline_comparison.moe_comparison.MoEBaselineAdapter') as MockAdapter:
        # Setup mock adapter instance
        mock_instance = MagicMock()
        MockAdapter.return_value = mock_instance
        
        # Test configuration
        test_config = {"target_column": "target"}
        
        # Create adapter with configuration
        adapter = create_moe_adapter(config=test_config, verbose=True)
        
        # Check that MoEBaselineAdapter was created with the config
        MockAdapter.assert_called_once()
        assert MockAdapter.call_args[1]["config"] == test_config
        assert MockAdapter.call_args[1]["verbose"] == True
        
        # Reset mock
        MockAdapter.reset_mock()
        
        # Test with config path (mocked file reading)
        with patch('os.path.exists', return_value=True):
            with patch('builtins.open', create=True) as mock_open:
                mock_open.return_value.__enter__.return_value.read.return_value = '{"key": "value"}'
                with patch('json.load', return_value={"key": "value"}):
                    adapter = create_moe_adapter(config_path="test_path.json")
                    
                    # Check that MoEBaselineAdapter was created with the loaded config
                    assert MockAdapter.call_args[1]["config"] == {"key": "value"}
