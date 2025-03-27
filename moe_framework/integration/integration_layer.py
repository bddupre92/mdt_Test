"""
Integration layer implementations for the Mixture of Experts (MoE) framework.

This module provides concrete implementations of integration strategies
that combine predictions from multiple expert models based on weights
from a gating network.
"""

import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union

import numpy as np

from moe_framework.interfaces.expert import ExpertModel, ExpertRegistry
from moe_framework.interfaces.gating import GatingNetwork
from moe_framework.interfaces.base import Configurable, PatientContext

logger = logging.getLogger(__name__)


class IntegrationLayer(Configurable):
    """
    Abstract base class for integration layers in the MoE system.
    
    Integration layers are responsible for combining predictions from 
    multiple expert models based on weights from a gating network.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the integration layer.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        
    @abstractmethod
    def integrate(
        self,
        expert_outputs: Dict[str, np.ndarray],
        weights: Dict[str, float],
        context: Optional[PatientContext] = None
    ) -> np.ndarray:
        """
        Integrate outputs from multiple experts using provided weights.
        
        Args:
            expert_outputs: Dictionary mapping expert names to prediction outputs
            weights: Dictionary mapping expert names to weights
            context: Optional patient context for context-aware integration
            
        Returns:
            Integrated prediction
        """
        pass
    
    def validate_weights(self, weights: Dict[str, float]) -> Dict[str, float]:
        """
        Validate and normalize weights to ensure they sum to 1.0.
        
        Args:
            weights: Dictionary mapping expert names to weights
            
        Returns:
            Normalized weights dictionary
        """
        if not weights:
            logger.warning("Empty weights dictionary provided")
            return {}
            
        weight_sum = sum(weights.values())
        
        if weight_sum == 0:
            logger.warning("All weights are zero, using uniform weights")
            normalized = {name: 1.0 / len(weights) for name in weights}
        else:
            normalized = {name: w / weight_sum for name, w in weights.items()}
            
        return normalized


class WeightedAverageIntegration(IntegrationLayer):
    """
    Simple weighted average integration strategy.
    
    This integration layer combines expert outputs using a weighted average
    based on the weights provided by the gating network.
    """
    
    def get_config(self) -> Dict[str, Any]:
        """
        Get the current configuration of the integration layer.
        
        Returns:
            Dict containing the configuration parameters
        """
        return self.config.copy()
        
    def set_config(self, config: Dict[str, Any]) -> None:
        """
        Configure the integration layer with the provided parameters.
        
        Args:
            config: Dictionary containing configuration parameters
        """
        self.config = config or {}
    
    def integrate(
        self,
        expert_outputs: Dict[str, np.ndarray],
        weights: Dict[str, float],
        context: Optional[PatientContext] = None
    ) -> np.ndarray:
        """
        Integrate expert outputs using weighted average.
        
        Args:
            expert_outputs: Dictionary mapping expert names to prediction outputs
            weights: Dictionary mapping expert names to weights
            context: Optional patient context (not used in this implementation)
            
        Returns:
            Weighted average of expert outputs
        """
        # Validate inputs
        if not expert_outputs:
            raise ValueError("No expert outputs provided")
            
        # Filter to only include experts that have both outputs and weights
        valid_experts = set(expert_outputs.keys()).intersection(set(weights.keys()))
        
        if not valid_experts:
            raise ValueError("No valid experts found with both outputs and weights")
            
        # Extract weights for valid experts only and renormalize
        valid_weights = {name: weights[name] for name in valid_experts}
        normalized_weights = self.validate_weights(valid_weights)
            
        # Convert all values to numpy arrays if they are scalars
        expert_outputs = {name: np.array([output]) if np.isscalar(output) else output 
                        for name, output in expert_outputs.items() 
                        if name in valid_experts}
        
        # Extract shapes to verify compatibility
        output_shapes = {name: output.shape for name, output in expert_outputs.items()}
        
        if len(set(map(lambda x: x[0], output_shapes.values()))) > 1:
            # Different sample counts
            raise ValueError("Expert outputs have inconsistent sample counts")
            
        # Initialize weighted sum with zeros matching the shape of the first output
        first_expert = list(valid_experts)[0]
        result_shape = expert_outputs[first_expert].shape
        weighted_sum = np.zeros(result_shape)
        
        # Accumulate weighted outputs
        for expert_name in valid_experts:
            expert_weight = normalized_weights[expert_name]
            expert_output = expert_outputs[expert_name]
            weighted_sum += expert_output * expert_weight
            
        return weighted_sum


class AdaptiveIntegration(IntegrationLayer):
    """
    Advanced integration strategy with context-awareness.
    
    This integration layer adapts the combination strategy based on
    patient context and data quality information.
    """
    
    def get_config(self) -> Dict[str, Any]:
        """
        Get the current configuration of the integration layer.
        
        Returns:
            Dict containing the configuration parameters
        """
        return self.config.copy()
        
    def set_config(self, config: Dict[str, Any]) -> None:
        """
        Configure the integration layer with the provided parameters.
        
        Args:
            config: Dictionary containing configuration parameters
        """
        self.config = config or {}
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the adaptive integration layer.
        
        Args:
            config: Configuration dictionary with adaptation parameters
        """
        super().__init__(config)
        self.confidence_threshold = self.config.get('confidence_threshold', 0.7)
        self.quality_threshold = self.config.get('quality_threshold', 0.5)
        
    def integrate(
        self,
        expert_outputs: Dict[str, np.ndarray],
        weights: Dict[str, float],
        context: Optional[PatientContext] = None
    ) -> np.ndarray:
        """
        Integrate expert outputs with adaptive weighting.
        
        Args:
            expert_outputs: Dictionary mapping expert names to prediction outputs
            weights: Dictionary mapping expert names to weights
            context: Patient context with quality and confidence metadata
            
        Returns:
            Adaptively integrated prediction
        """
        # Filter to only include experts that have both outputs and weights
        valid_experts = set(expert_outputs.keys()).intersection(set(weights.keys()))
        
        if not valid_experts:
            raise ValueError("No valid experts found with both outputs and weights")
            
        # Extract weights for valid experts only and renormalize
        valid_weights = {name: weights[name] for name in valid_experts}
        normalized_weights = self.validate_weights(valid_weights)
        
        # Adapt weights based on context if available
        if context and hasattr(context, 'quality_metrics'):
            adjusted_weights = self._adjust_for_quality(
                normalized_weights, 
                context.quality_metrics
            )
        else:
            adjusted_weights = normalized_weights
            
        # Convert all values to numpy arrays if they are scalars
        expert_outputs = {name: np.array([output]) if np.isscalar(output) else output 
                        for name, output in expert_outputs.items() 
                        if name in valid_experts}
        
        first_expert = list(valid_experts)[0]
        result_shape = expert_outputs[first_expert].shape
        weighted_sum = np.zeros(result_shape)
        
        # Apply adjusted weights
        for expert_name in valid_experts:
            expert_weight = adjusted_weights[expert_name]
            expert_output = expert_outputs[expert_name]
            weighted_sum += expert_output * expert_weight
            
        return weighted_sum
    
    def _adjust_for_quality(
        self,
        weights: Dict[str, float],
        quality_metrics: Dict[str, float]
    ) -> Dict[str, float]:
        """
        Adjust weights based on data quality metrics.
        
        Args:
            weights: Original expert weights
            quality_metrics: Quality metrics for different data domains
            
        Returns:
            Adjusted weights accounting for data quality
        """
        adjusted = {}
        
        # Map expert names to their primary domains
        # This could be provided in configuration or inferred from expert name
        domain_map = self.config.get('expert_domain_map', {})
        
        for expert_name, weight in weights.items():
            # Determine the domain(s) this expert relies on
            domain = domain_map.get(expert_name)
            
            if domain and domain in quality_metrics:
                # Adjust weight based on quality
                quality = quality_metrics[domain]
                if quality < self.quality_threshold:
                    # Reduce weight for low quality data
                    adjustment_factor = max(0.1, quality / self.quality_threshold)
                    adjusted[expert_name] = weight * adjustment_factor
                else:
                    adjusted[expert_name] = weight
            else:
                # Keep original weight if no quality information
                adjusted[expert_name] = weight
                
        # Renormalize
        weight_sum = sum(adjusted.values())
        if weight_sum > 0:
            adjusted = {name: w / weight_sum for name, w in adjusted.items()}
            
        return adjusted
