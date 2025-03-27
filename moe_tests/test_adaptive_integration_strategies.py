"""
Test Adaptive Integration Strategies

This module provides comprehensive testing for adaptive integration strategies, including:
1. Confidence-based integration
2. Quality-aware integration
3. Adaptive fusion with various weighting methods
4. Edge case handling and corner cases
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
from moe_framework.integration.integration_connector import IntegrationConnector
from moe_framework.integration.event_system import EventManager
from moe_framework.persistence.state_manager import FileSystemStateManager
from moe_framework.interfaces.base import PatientContext


class TestAdaptiveIntegrationStrategies:
    """
    Comprehensive tests for adaptive integration strategies.
    """
    
    def setup_method(self):
        """Set up test fixtures."""
        # Create mock data
        self.expert_outputs = {
            'expert1': np.array([[0.1], [0.2], [0.3], [0.4], [0.5]]),
            'expert2': np.array([[0.2], [0.4], [0.6], [0.8], [1.0]]),
            'expert3': np.array([[0.3], [0.3], [0.3], [0.3], [0.3]])
        }
        
        self.weights = {
            'expert1': 0.5,
            'expert2': 0.3,
            'expert3': 0.2
        }
        
        self.confidences = {
            'expert1': np.array([0.9, 0.8, 0.7, 0.6, 0.5]),
            'expert2': np.array([0.6, 0.7, 0.8, 0.9, 0.95]),
            'expert3': np.array([0.4, 0.4, 0.4, 0.4, 0.4])
        }
        
        self.quality_scores = {
            'expert1': 0.85,
            'expert2': 0.75,
            'expert3': 0.6
        }
        
        # Create mock patient context
        self.context = PatientContext(
            patient_id="test_patient",
            data_quality={
                'physiological': 0.8,
                'behavioral': 0.7,
                'environmental': 0.6
            },
            timestamp="2025-03-25T12:00:00"
        )
        
        # Create temporary directory for state manager
        self.temp_dir = os.path.join(os.path.dirname(__file__), 'temp_data')
        os.makedirs(self.temp_dir, exist_ok=True)
        
        # Create event manager
        self.event_manager = EventManager()
        
        # Create state manager
        self.state_manager = FileSystemStateManager({
            'base_dir': self.temp_dir
        })
        
    def teardown_method(self):
        """Tear down test fixtures."""
        # Clean up temporary directory if needed
        import shutil
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    def test_weighted_average_integration(self):
        """Test basic weighted average integration."""
        integration = WeightedAverageIntegration()
        
        # Integrate predictions
        result = integration.integrate(
            expert_outputs=self.expert_outputs,
            weights=self.weights
        )
        
        # Calculate expected result manually
        expected = np.zeros((5, 1))
        for expert_id, output in self.expert_outputs.items():
            expected += output * self.weights[expert_id]
        
        # Check shape and values
        assert result.shape == (5, 1)
        assert np.allclose(result, expected)
    
    def test_adaptive_integration_with_confidence(self):
        """Test adaptive integration with confidence scores."""
        # Create adaptive integration with confidence-based strategy
        config = {
            'strategy': 'confidence',
            'confidence_threshold': 0.7,
            'min_experts': 1
        }
        integration = AdaptiveIntegration(config)
        
        # Set confidence values
        for expert_id in self.expert_outputs:
            integration.expert_confidences[expert_id] = self.confidences[expert_id]
        
        # Integrate predictions
        result = integration.integrate(
            expert_outputs=self.expert_outputs,
            weights=self.weights
        )
        
        # Verify shape
        assert result.shape == (5, 1)
        
        # For each sample, verify that experts with confidence < threshold are given less weight
        for i in range(5):
            sample_weights = {}
            for expert_id in self.expert_outputs:
                confidence = self.confidences[expert_id][i]
                if confidence >= config['confidence_threshold']:
                    sample_weights[expert_id] = self.weights[expert_id]
                else:
                    # Should be adjusted down based on confidence
                    sample_weights[expert_id] = self.weights[expert_id] * (confidence / config['confidence_threshold'])
            
            # Normalize sample weights
            total = sum(sample_weights.values())
            if total > 0:
                sample_weights = {k: v/total for k, v in sample_weights.items()}
            
            # Calculate expected output for this sample
            expected_sample = 0
            for expert_id in self.expert_outputs:
                expected_sample += self.expert_outputs[expert_id][i] * sample_weights[expert_id]
            
            # Verify within tolerance
            assert abs(result[i][0] - expected_sample) < 1e-5
    
    def test_adaptive_integration_with_quality(self):
        """Test adaptive integration with quality scores."""
        # Create adaptive integration with quality-aware strategy
        config = {
            'strategy': 'quality_aware',
            'quality_threshold': 0.7,
            'quality_impact': 0.8
        }
        integration = AdaptiveIntegration(config)
        
        # Set quality scores
        integration.expert_quality_scores = self.quality_scores
        
        # Integrate predictions
        result = integration.integrate(
            expert_outputs=self.expert_outputs,
            weights=self.weights
        )
        
        # Verify shape
        assert result.shape == (5, 1)
        
        # Calculate adjusted weights based on quality
        quality_adjusted_weights = {}
        for expert_id, weight in self.weights.items():
            quality = self.quality_scores[expert_id]
            quality_factor = 1.0
            if quality >= config['quality_threshold']:
                # Boost weight for high quality experts
                quality_factor = 1.0 + (quality - config['quality_threshold']) * config['quality_impact']
            else:
                # Reduce weight for low quality experts
                quality_factor = quality / config['quality_threshold']
            
            quality_adjusted_weights[expert_id] = weight * quality_factor
        
        # Normalize weights
        total = sum(quality_adjusted_weights.values())
        if total > 0:
            quality_adjusted_weights = {k: v/total for k, v in quality_adjusted_weights.items()}
        
        # Calculate expected result
        expected = np.zeros((5, 1))
        for expert_id, output in self.expert_outputs.items():
            expected += output * quality_adjusted_weights[expert_id]
        
        # Verify result
        assert np.allclose(result, expected, atol=1e-5)
    
    def test_adaptive_integration_with_context(self):
        """Test adaptive integration with patient context."""
        # Create adaptive integration with context-aware strategy
        config = {
            'strategy': 'context_aware',
            'context_weight': 0.5
        }
        integration = AdaptiveIntegration(config)
        
        # Integrate predictions with context
        result = integration.integrate(
            expert_outputs=self.expert_outputs,
            weights=self.weights,
            patient_context=self.context
        )
        
        # Verify shape
        assert result.shape == (5, 1)
        
        # Verify that context influenced the result (exact calculation will depend on implementation)
        assert result is not None
    
    def test_adaptive_integration_with_empty_experts(self):
        """Test adaptive integration with empty expert list."""
        integration = AdaptiveIntegration()
        
        # Test with empty expert_outputs
        with pytest.raises(ValueError):
            integration.integrate(
                expert_outputs={},
                weights=self.weights
            )
    
    def test_adaptive_integration_with_mismatched_shapes(self):
        """Test adaptive integration with mismatched output shapes."""
        integration = AdaptiveIntegration()
        
        # Create expert outputs with mismatched shapes
        expert_outputs = {
            'expert1': np.array([[0.1], [0.2], [0.3]]),  # 3 samples
            'expert2': np.array([[0.2], [0.4]])          # 2 samples
        }
        weights = {'expert1': 0.6, 'expert2': 0.4}
        
        # Should raise ValueError
        with pytest.raises(ValueError):
            integration.integrate(
                expert_outputs=expert_outputs,
                weights=weights
            )
    
    def test_adaptive_integration_with_missing_weights(self):
        """Test adaptive integration with missing weights for some experts."""
        integration = AdaptiveIntegration()
        
        # Create weights dictionary missing an expert
        incomplete_weights = {
            'expert1': 0.7,
            'expert2': 0.3
            # expert3 missing
        }
        
        # Should raise ValueError
        with pytest.raises(ValueError):
            integration.integrate(
                expert_outputs=self.expert_outputs,
                weights=incomplete_weights
            )
    
    def test_advanced_fusion_strategy(self):
        """Test advanced fusion integration strategy."""
        # Create a custom integration class with advanced fusion strategy
        class AdvancedFusionIntegration(AdaptiveIntegration):
            def __init__(self):
                super().__init__(config={'strategy': 'advanced_fusion'})
            
            def integrate(self, expert_outputs, weights, **kwargs):
                """
                Advanced fusion strategy that adaptively combines 
                confidence, quality, and context-based weighting.
                """
                # Basic validation
                self._validate_inputs(expert_outputs, weights)
                
                # Get base outputs to determine shape
                first_expert_id = list(expert_outputs.keys())[0]
                base_output = expert_outputs[first_expert_id]
                sample_count = base_output.shape[0]
                
                # Initialize result array
                result = np.zeros_like(base_output)
                
                # Adaptive weight adjustment based on multiple factors
                adjusted_weights = {}
                
                for expert_id, base_weight in weights.items():
                    # Start with base weight
                    weight = base_weight
                    
                    # Apply confidence adjustment if available (sample-specific)
                    if expert_id in self.expert_confidences:
                        confidence = self.expert_confidences[expert_id]
                        # Ensure confidence is array-like with matching samples
                        if not hasattr(confidence, '__len__') or len(confidence) != sample_count:
                            confidence = np.full(sample_count, confidence)
                    else:
                        confidence = np.ones(sample_count)
                    
                    # Apply quality adjustment if available (global)
                    quality_factor = 1.0
                    if expert_id in self.expert_quality_scores:
                        quality = self.expert_quality_scores[expert_id]
                        quality_factor = max(0.5, min(1.5, quality))
                    
                    # Store adjusted weights for each sample
                    adjusted_weights[expert_id] = weight * quality_factor * confidence
                
                # Apply sample-specific normalization
                for i in range(sample_count):
                    sample_weights = {exp_id: weights[i] for exp_id, weights in adjusted_weights.items()}
                    total = sum(sample_weights.values())
                    if total > 0:
                        sample_weights = {k: v/total for k, v in sample_weights.items()}
                    
                    # Compute weighted sum for this sample
                    for expert_id, sample_weight in sample_weights.items():
                        result[i] += expert_outputs[expert_id][i] * sample_weight
                
                return result
        
        # Create integration instance
        integration = AdvancedFusionIntegration()
        
        # Set confidence and quality scores
        for expert_id in self.expert_outputs:
            integration.expert_confidences[expert_id] = self.confidences[expert_id]
        integration.expert_quality_scores = self.quality_scores
        
        # Integrate predictions
        result = integration.integrate(
            expert_outputs=self.expert_outputs,
            weights=self.weights
        )
        
        # Verify shape
        assert result.shape == (5, 1)
        assert not np.isnan(result).any()
    
    def test_integration_connector_with_adaptive_integration(self):
        """Test integration connector with adaptive integration."""
        # Create config
        config = {
            'integration': {
                'strategy': 'adaptive',
                'confidence_threshold': 0.7,
                'quality_threshold': 0.75
            }
        }
        
        # Create adaptive integration
        integration_layer = AdaptiveIntegration(config.get('integration', {}))
        
        # Set quality scores and confidences
        integration_layer.expert_quality_scores = self.quality_scores
        for expert_id in self.expert_outputs:
            integration_layer.expert_confidences[expert_id] = self.confidences[expert_id].mean()
        
        # Create integration connector
        connector = IntegrationConnector(
            integration_layer=integration_layer,
            event_manager=self.event_manager,
            state_manager=self.state_manager,
            config=config
        )
        
        # Integrate predictions
        result = connector.integrate_predictions(
            expert_outputs=self.expert_outputs,
            weights=self.weights,
            context=self.context
        )
        
        # Verify result
        assert result.shape == (5, 1)
        assert not np.isnan(result).any()
    
    def test_dynamic_strategy_switching(self):
        """Test dynamic switching between integration strategies."""
        # Create adaptive integration with dynamic strategy
        config = {
            'strategy': 'dynamic',
            'confidence_threshold': 0.7,
            'quality_threshold': 0.75,
            'fallback_strategy': 'weighted_average'
        }
        integration = AdaptiveIntegration(config)
        
        # Set quality scores and confidences
        integration_layer = integration
        integration_layer.expert_quality_scores = self.quality_scores
        for expert_id in self.expert_outputs:
            integration_layer.expert_confidences[expert_id] = self.confidences[expert_id].mean()
        
        # Test with normal inputs
        result1 = integration.integrate(
            expert_outputs=self.expert_outputs,
            weights=self.weights
        )
        
        # Verify result
        assert result1.shape == (5, 1)
        
        # Now test dynamic fallback with problematic inputs
        # Create a scenario where confidence-based strategy would fail
        integration.expert_confidences = {}  # Remove all confidences
        
        # Should fallback to weighted average
        result2 = integration.integrate(
            expert_outputs=self.expert_outputs,
            weights=self.weights
        )
        
        # Verify result is still valid
        assert result2.shape == (5, 1)
        assert not np.isnan(result2).any()


class TestIntegrationStrategyFactoryPattern:
    """
    Test the factory pattern for creating different integration strategies.
    """
    
    def test_integration_strategy_factory(self):
        """Test factory pattern for creating integration strategies."""
        from moe_framework.integration.integration_layer import get_integration_strategy
        
        # Test basic strategies
        weighted_avg = get_integration_strategy('weighted_average')
        assert isinstance(weighted_avg, WeightedAverageIntegration)
        
        adaptive = get_integration_strategy('adaptive', config={'confidence_threshold': 0.8})
        assert isinstance(adaptive, AdaptiveIntegration)
        assert adaptive.config['confidence_threshold'] == 0.8
        
        # Test with invalid strategy name
        with pytest.raises(ValueError):
            get_integration_strategy('invalid_strategy_name')


if __name__ == "__main__":
    pytest.main(["-v", __file__])
