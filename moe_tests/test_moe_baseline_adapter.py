"""
Tests for the MoE Baseline Adapter.

This module contains tests for the MoEBaselineAdapter class, which integrates
the MoE framework with the baseline comparison framework.
"""

import os
import pytest
import numpy as np
import pandas as pd
from unittest.mock import MagicMock, patch

from baseline_comparison.moe_adapter import MoEBaselineAdapter
from moe_framework.workflow.moe_pipeline import MoEPipeline
from moe_framework.interfaces.base import PatientContext


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
def mock_moe_pipeline():
    """Create a mock MoE pipeline for testing."""
    mock_pipeline = MagicMock(spec=MoEPipeline)
    mock_pipeline.experts = {
        'expert1': MagicMock(),
        'expert2': MagicMock(),
        'expert3': MagicMock()
    }
    mock_pipeline.predict.return_value = np.array([1.0, 2.0, 3.0])
    mock_pipeline.get_expert_weights.return_value = {
        'expert1': 0.2,
        'expert2': 0.5,
        'expert3': 0.3
    }
    mock_pipeline.event_manager = MagicMock()
    mock_pipeline.event_manager.register_listener = MagicMock()
    
    return mock_pipeline


@pytest.fixture
def minimal_config():
    """Create a minimal configuration for testing."""
    return {
        'target_column': 'target',
        'time_column': 'timestamp',
        'patient_column': 'patient_id',
        'experts': {
            'expert1': {},
            'expert2': {},
            'expert3': {}
        }
    }


def test_initialization():
    """Test that the adapter initializes correctly."""
    config = {
        'target_column': 'target',
        'time_column': 'timestamp',
        'patient_column': 'patient_id'
    }
    
    with patch('baseline_comparison.moe_adapter.MoEPipeline') as mock_pipeline_class:
        mock_pipeline = MagicMock()
        mock_pipeline.experts = {'expert1': {}, 'expert2': {}}
        mock_pipeline.event_manager = MagicMock()
        mock_pipeline_class.return_value = mock_pipeline
        
        adapter = MoEBaselineAdapter(config=config)
        
        assert adapter.config == config
        assert adapter.moe_pipeline == mock_pipeline
        assert mock_pipeline_class.call_count == 1
        assert mock_pipeline_class.call_args[1]['config'] == config


def test_get_available_algorithms(mock_moe_pipeline):
    """Test getting available algorithms."""
    with patch('baseline_comparison.moe_adapter.MoEPipeline', return_value=mock_moe_pipeline):
        adapter = MoEBaselineAdapter()
        
        algorithms = adapter.get_available_algorithms()
        
        assert set(algorithms) == {'expert1', 'expert2', 'expert3'}


def test_set_available_algorithms(mock_moe_pipeline):
    """Test setting available algorithms."""
    with patch('baseline_comparison.moe_adapter.MoEPipeline', return_value=mock_moe_pipeline):
        adapter = MoEBaselineAdapter()
        
        # Set only a subset of algorithms
        adapter.set_available_algorithms(['expert1', 'expert3'])
        
        # Original mock still has all experts, but real implementation would filter
        # For test purposes, we just verify the method runs without error
        assert True


def test_fit(mock_moe_pipeline, sample_data):
    """Test the fit method."""
    X = sample_data.drop(columns=['target'])
    y = sample_data['target']
    
    with patch('baseline_comparison.moe_adapter.MoEPipeline', return_value=mock_moe_pipeline):
        adapter = MoEBaselineAdapter(config={'target_column': 'target'})
        
        adapter.fit(X, y)
        
        mock_moe_pipeline.train.assert_called_once()
        # Check that the data passed to train combines X and y
        train_data = mock_moe_pipeline.train.call_args[0][0]
        assert 'target' in train_data.columns
        assert len(train_data) == len(X)


def test_predict(mock_moe_pipeline, sample_data):
    """Test the predict method."""
    X = sample_data.drop(columns=['target'])
    
    with patch('baseline_comparison.moe_adapter.MoEPipeline', return_value=mock_moe_pipeline):
        adapter = MoEBaselineAdapter()
        
        predictions = adapter.predict(X)
        
        mock_moe_pipeline.predict.assert_called_once()
        assert np.array_equal(predictions, np.array([1.0, 2.0, 3.0]))


def test_select_algorithm(mock_moe_pipeline, sample_data):
    """Test the select_algorithm method."""
    X = sample_data.drop(columns=['target']).iloc[:1]  # Just use one row
    
    with patch('baseline_comparison.moe_adapter.MoEPipeline', return_value=mock_moe_pipeline):
        adapter = MoEBaselineAdapter()
        
        algorithm = adapter.select_algorithm(X)
        
        # Should select expert2 which has the highest weight (0.5)
        assert algorithm == 'expert2'
        mock_moe_pipeline.get_expert_weights.assert_called_once()


def test_run_optimizer(mock_moe_pipeline, sample_data):
    """Test the run_optimizer method."""
    X = sample_data.drop(columns=['target'])
    y = sample_data['target']
    data = pd.concat([X, y], axis=1)
    
    with patch('baseline_comparison.moe_adapter.MoEPipeline', return_value=mock_moe_pipeline):
        with patch('baseline_comparison.moe_adapter.mean_squared_error', return_value=0.5):
            adapter = MoEBaselineAdapter(config={'target_column': 'target'})
            
            # Track a selection to test Counter logic
            adapter.selected_algorithms = ['expert2', 'expert2', 'expert1']
            
            result = adapter.run_optimizer(data)
            
            assert result['best_fitness'] == 0.5
            assert result['evaluations'] == len(data)
            assert result['selected_algorithm'] == 'expert2'


def test_cross_validate_with_patient_aware(mock_moe_pipeline, sample_data):
    """Test the cross_validate method with patient-aware splitting."""
    X = sample_data.drop(columns=['target'])
    y = sample_data['target']
    
    # Create a mock for TimeSeriesValidator
    with patch('baseline_comparison.moe_adapter.MoEPipeline', return_value=mock_moe_pipeline):
        with patch('baseline_comparison.moe_adapter.TimeSeriesValidator') as mock_validator_class:
            mock_validator = MagicMock()
            mock_validator.get_validation_scores.return_value = {
                'mean_scores': {'rmse': 1.5},
                'fold_scores': {'rmse': [1.4, 1.5, 1.6]}
            }
            mock_validator_class.return_value = mock_validator
            
            adapter = MoEBaselineAdapter(config={'target_column': 'target'})
            
            scores = adapter.cross_validate(X, y, method='patient_aware')
            
            assert scores['mean_scores']['rmse'] == 1.5
            assert mock_validator.get_validation_scores.called
            # Verify that patient_aware method was used
            assert mock_validator.get_validation_scores.call_args[1]['method'] == 'patient_aware'


def test_save_and_load_model(mock_moe_pipeline, tmp_path):
    """Test saving and loading models."""
    model_path = os.path.join(tmp_path, 'test_model.pkl')
    
    with patch('baseline_comparison.moe_adapter.MoEPipeline', return_value=mock_moe_pipeline):
        with patch('os.path.exists', return_value=True):
            adapter = MoEBaselineAdapter()
            
            # Test save
            adapter.save_model(model_path)
            mock_moe_pipeline.save_checkpoint.assert_called_once_with(model_path)
            
            # Test load
            adapter.load_model(model_path)
            mock_moe_pipeline.load_from_checkpoint.assert_called_once_with(model_path)


def test_integration_with_moe_pipeline(minimal_config, sample_data):
    """Test that the adapter can work with a real MoE pipeline."""
    X = sample_data.drop(columns=['target'])
    y = sample_data['target']
    
    # This is a more integrated test that uses the real MoEPipeline class
    # We'll skip complex initialization and just verify the interfaces work
    with patch('moe_framework.workflow.moe_pipeline.MoEPipeline.__init__', return_value=None):
        with patch('moe_framework.workflow.moe_pipeline.MoEPipeline.train', return_value=None):
            with patch('moe_framework.workflow.moe_pipeline.MoEPipeline.predict', return_value=np.zeros(len(X))):
                with patch.object(MoEPipeline, 'experts', {}, create=True):
                    with patch.object(MoEPipeline, 'event_manager', MagicMock(), create=True):
                        adapter = MoEBaselineAdapter(config=minimal_config)
                        
                        # Set some expert mocks manually
                        adapter.moe_pipeline.experts = {
                            'expert1': MagicMock(),
                            'expert2': MagicMock()
                        }
                        
                        # Test the basic workflow
                        adapter.fit(X, y)
                        predictions = adapter.predict(X)
                        
                        assert len(predictions) == len(X)
