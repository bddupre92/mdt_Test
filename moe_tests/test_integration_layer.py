"""
Tests for the Integration Layer components.

This test suite covers:
1. WeightedAverageIntegration functionality
2. AdaptiveIntegration functionality
3. IntegrationConnector with mocked experts
4. Error handling scenarios
"""

import os
import pytest
import numpy as np
import pandas as pd
from unittest.mock import Mock, patch, MagicMock

from moe_framework.integration.integration_layer import (
    IntegrationLayer,
    WeightedAverageIntegration, 
    AdaptiveIntegration
)
from moe_framework.integration.pipeline_connector import IntegrationConnector
from moe_framework.interfaces.base import PatientContext


class TestWeightedAverageIntegration:
    """Tests for the WeightedAverageIntegration class."""
    
    def setup_method(self):
        """Set up test fixtures before each test method."""
        # Create a concrete implementation with the required abstract methods
        class TestWeightedIntegration(WeightedAverageIntegration):
            def get_config(self):
                return self.config
                
            def set_config(self, config):
                self.config = config
        
        self.integration = TestWeightedIntegration(config={'test_param': True})
        
    def test_integrate_scalar_predictions(self):
        """Test integration of scalar predictions."""
        # Setup
        expert_outputs = {
            'expert1': 5.0,
            'expert2': 7.0,
            'expert3': 3.0
        }
        
        weights = {
            'expert1': 0.5,
            'expert2': 0.3,
            'expert3': 0.2
        }
        
        # Expected: (5.0*0.5 + 7.0*0.3 + 3.0*0.2) = 5.2
        expected = 5.2
        
        # Execute
        result = self.integration.integrate(expert_outputs, weights)
        
        # Verify
        assert result == pytest.approx(expected)
    
    def test_integrate_array_predictions(self):
        """Test integration of array predictions."""
        # Setup
        expert_outputs = {
            'expert1': np.array([1.0, 2.0, 3.0]),
            'expert2': np.array([4.0, 5.0, 6.0]),
            'expert3': np.array([7.0, 8.0, 9.0])
        }
        
        weights = {
            'expert1': 0.5,
            'expert2': 0.3,
            'expert3': 0.2
        }
        
        # Expected calculations for each position
        # Position 0: 1.0*0.5 + 4.0*0.3 + 7.0*0.2 = 3.1
        # Position 1: 2.0*0.5 + 5.0*0.3 + 8.0*0.2 = 4.1
        # Position 2: 3.0*0.5 + 6.0*0.3 + 9.0*0.2 = 5.1
        expected = np.array([3.1, 4.1, 5.1])
        
        # Execute
        result = self.integration.integrate(expert_outputs, weights)
        
        # Verify
        np.testing.assert_array_almost_equal(result, expected)
    
    def test_missing_expert_prediction(self):
        """Test handling of missing expert predictions."""
        # Setup
        expert_outputs = {
            'expert1': 5.0,
            'expert2': 7.0
            # expert3 is missing
        }
        
        weights = {
            'expert1': 0.5,
            'expert2': 0.3,
            'expert3': 0.2  # Weight for missing expert
        }
        
        # Expected: Only consider available experts and renormalize weights
        # New weights: expert1 = 0.5/(0.5+0.3) = 0.625, expert2 = 0.3/(0.5+0.3) = 0.375
        # Result: 5.0*0.625 + 7.0*0.375 = 5.75
        expected = 5.75
        
        # Execute
        result = self.integration.integrate(expert_outputs, weights)
        
        # Verify - with scalar inputs, result will be a single-element array
        if isinstance(result, np.ndarray) and result.size == 1:
            result = float(result[0])  # Extract the scalar value from the array
        assert result == pytest.approx(expected)
    
    def test_empty_expert_outputs(self):
        """Test handling of empty expert outputs."""
        # Setup
        expert_outputs = {}
        weights = {'expert1': 0.5, 'expert2': 0.5}
        
        # Execute & Verify
        with pytest.raises(ValueError, match="No expert outputs provided"):
            self.integration.integrate(expert_outputs, weights)
    
    def test_mismatched_output_shapes(self):
        """Test handling of mismatched output shapes."""
        # Setup
        expert_outputs = {
            'expert1': np.array([1.0, 2.0]),
            'expert2': np.array([3.0, 4.0, 5.0])  # Different shape
        }
        
        weights = {
            'expert1': 0.5,
            'expert2': 0.5
        }
        
        # Execute & Verify
        with pytest.raises(ValueError, match="Expert outputs have inconsistent sample counts"):
            self.integration.integrate(expert_outputs, weights)


class TestAdaptiveIntegration:
    """Tests for the AdaptiveIntegration class."""
    
    def setup_method(self):
        """Set up test fixtures before each test method."""
        # Create a concrete implementation with the required abstract methods
        class TestAdaptiveImpl(AdaptiveIntegration):
            def get_config(self):
                return self.config
                
            def set_config(self, config):
                self.config = config
        
        self.integration = TestAdaptiveImpl(config={
            'confidence_threshold': 0.7,
            'quality_threshold': 0.5,
            'confidence_weight': 0.8
        })
        
    def test_integrate_with_quality_metrics(self):
        """Test integration with quality metrics."""
        # Setup
        expert_outputs = {
            'expert1': 5.0,
            'expert2': 7.0,
            'expert3': 3.0
        }
        
        weights = {
            'expert1': 0.4,
            'expert2': 0.35,
            'expert3': 0.25
        }
        
        # Mock context with quality metrics
        context = PatientContext()
        context.patient_id = "test_patient"
        context.quality_metrics = {
            'expert1': {'data_quality': 0.9},  # High quality
            'expert2': {'data_quality': 0.5},  # Low quality
            'expert3': {'data_quality': 0.8}   # Good quality
        }
        
        # Execute
        result = self.integration.integrate(expert_outputs, weights, context)
        
        # Verify: Should favor expert1 and expert3 due to higher quality
        if isinstance(result, np.ndarray) and result.size == 1:
            result = float(result[0])  # Extract the scalar value from the array
        assert result > 5.0  # Closer to expert1 and expert3 than to expert2
        
    def test_integrate_without_context(self):
        """Test integration without context (should fall back to weighted average)."""
        # Setup
        expert_outputs = {
            'expert1': 5.0,
            'expert2': 7.0
        }
        
        weights = {
            'expert1': 0.6,
            'expert2': 0.4
        }
        
        # Expected: Standard weighted average = 5.0*0.6 + 7.0*0.4 = 5.8
        expected = 5.8
        
        # Execute
        result = self.integration.integrate(expert_outputs, weights)
        
        # Verify
        if isinstance(result, np.ndarray) and result.size == 1:
            result = float(result[0])  # Extract the scalar value from the array
        assert result == pytest.approx(expected)
    
    def test_integrate_with_empty_quality_metrics(self):
        """Test integration with empty quality metrics."""
        # Setup
        expert_outputs = {
            'expert1': 5.0,
            'expert2': 7.0
        }
        
        weights = {
            'expert1': 0.6,
            'expert2': 0.4
        }
        
        # Mock context with empty quality metrics
        context = PatientContext()
        context.patient_id = "test_patient"
        context.quality_metrics = {}  # Empty
        
        # Expected: Standard weighted average = 5.0*0.6 + 7.0*0.4 = 5.8
        expected = 5.8
        
        # Execute
        result = self.integration.integrate(expert_outputs, weights, context)
        
        # Verify
        if isinstance(result, np.ndarray) and result.size == 1:
            result = float(result[0])  # Extract the scalar value from the array
        assert result == pytest.approx(expected)


class TestIntegrationConnector:
    """Tests for the IntegrationConnector class."""
    
    def setup_method(self):
        """Set up test fixtures before each test method."""
        # Create mocks
        self.mock_integration = Mock(spec=IntegrationLayer)
        self.mock_event_manager = Mock()
        self.mock_state_manager = Mock()
        
        # Create connector with mocks
        self.connector = IntegrationConnector(
            integration_layer=self.mock_integration,
            event_manager=self.mock_event_manager,
            state_manager=self.mock_state_manager
        )
    
    def test_integrate_predictions(self):
        """Test the integrate_predictions method."""
        # Setup
        expert_outputs = {'expert1': 5.0, 'expert2': 7.0}
        weights = {'expert1': 0.6, 'expert2': 0.4}
        # Create a PatientContext instance
        context = PatientContext()
        context.patient_id = "test_patient"
        
        # Configure mock to return expected value
        expected_result = 5.8
        self.mock_integration.integrate.return_value = expected_result
        
        # Execute
        result = self.connector.integrate_predictions(expert_outputs, weights, context)
        
        # Verify
        assert result == expected_result
        self.mock_integration.integrate.assert_called_once_with(
            expert_outputs=expert_outputs, weights=weights, context=context
        )
        self.mock_event_manager.emit_event.assert_called()  # Should emit an event
    
    def test_save_load_state(self):
        """Test the save and load state methods."""
        # Setup
        from moe_framework.interfaces.base import SystemState
        
        # Create a proper SystemState object
        state = SystemState()
        state.meta_learner_state = {"test_key": "test_value"}
        path = "/tmp/test_state"
        
        # Configure mocks
        self.mock_state_manager.save_state.return_value = path
        self.mock_state_manager.load_state.return_value = state
        
        # Execute save with the connector's save_pipeline_state method
        save_result = self.connector.save_pipeline_state(state, path)
        
        # Verify save
        assert save_result == path
        self.mock_state_manager.save_state.assert_called_once_with(state, path)
        self.mock_event_manager.emit_event.assert_called()
        
        # Reset event manager mock
        self.mock_event_manager.emit_event.reset_mock()
        
        # Execute load with the connector's load_pipeline_state method
        load_result = self.connector.load_pipeline_state(path)
        
        # Verify load
        assert load_result == state
        self.mock_state_manager.load_state.assert_called_once_with(path)
        self.mock_event_manager.emit_event.assert_called()
    
    def test_handle_integration_error(self):
        """Test error handling during integration."""
        # Setup
        expert_outputs = {'expert1': 5.0, 'expert2': 7.0}
        weights = {'expert1': 0.6, 'expert2': 0.4}
        
        # Configure mock to raise an exception
        self.mock_integration.integrate.side_effect = ValueError("Test error")
        
        # Execute & Verify
        with pytest.raises(ValueError, match="Test error"):
            self.connector.integrate_predictions(expert_outputs, weights)
        
        # Verify error event was emitted
        self.mock_event_manager.emit_event.assert_called()


if __name__ == "__main__":
    pytest.main(["-xvs", __file__])
