"""
Optimizer interfaces for the Mixture of Experts (MoE) framework.

This module defines interfaces for optimizers that tune parameters
for expert models and other components of the MoE system.
"""

import logging
from abc import abstractmethod
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np

from .base import Configurable, Persistable

logger = logging.getLogger(__name__)


class Optimizer(Configurable, Persistable):
    """
    Base interface for optimizers in the MoE framework.
    
    Optimizers are responsible for tuning parameters of various
    components to maximize performance metrics.
    """
    
    @abstractmethod
    def optimize(
        self,
        objective_function: Callable[[Dict[str, Any]], float],
        parameter_space: Dict[str, Any],
        n_iterations: int = 50,
        **kwargs
    ) -> Tuple[Dict[str, Any], float]:
        """
        Optimize parameters to maximize objective function.
        
        Args:
            objective_function: Function that takes parameters and returns score
            parameter_space: Dictionary defining parameter search space
            n_iterations: Number of optimization iterations
            **kwargs: Additional optimizer-specific parameters
            
        Returns:
            Tuple of (best parameters, best score)
        """
        pass
    
    @abstractmethod
    def get_optimization_history(self) -> List[Tuple[Dict[str, Any], float]]:
        """
        Get history of parameters and scores from optimization process.
        
        Returns:
            List of (parameters, score) tuples from optimization history
        """
        pass
    
    @abstractmethod
    def update_bounds(self, parameter_bounds: Dict[str, Tuple[float, float]]) -> None:
        """
        Update bounds for continuous parameters.
        
        Args:
            parameter_bounds: Dictionary mapping parameter names to (min, max) tuples
        """
        pass


class ExpertSpecificOptimizer(Optimizer):
    """
    Extension of Optimizer for expert-specific optimization.
    
    This interface adds methods specific to optimizing expert models
    based on their specialty domains.
    """
    
    @abstractmethod
    def get_expert_domain(self) -> str:
        """
        Get the expert domain this optimizer is specialized for.
        
        Returns:
            String identifier for expert domain
        """
        pass
    
    @abstractmethod
    def is_compatible_with_expert(self, expert_type: str) -> bool:
        """
        Check if this optimizer is compatible with a specific expert type.
        
        Args:
            expert_type: String identifier for expert type
            
        Returns:
            Boolean indicating compatibility
        """
        pass
    
    @abstractmethod
    def suggest_default_parameters(self, expert_type: str) -> Dict[str, Any]:
        """
        Suggest default parameters for a specific expert type.
        
        Args:
            expert_type: String identifier for expert type
            
        Returns:
            Dictionary of suggested default parameters
        """
        pass


class OptimizerFactory:
    """
    Factory for creating optimizer instances.
    
    This class provides methods for creating optimizers appropriate
    for different expert types and problem domains.
    """
    
    @staticmethod
    def create_optimizer(
        optimizer_type: str,
        config: Dict[str, Any] = None
    ) -> Optimizer:
        """
        Create an optimizer of the specified type.
        
        Args:
            optimizer_type: Type of optimizer to create
            config: Optional configuration parameters
            
        Returns:
            Optimizer instance
            
        Raises:
            ValueError: If optimizer_type is not recognized
        """
        # This will be implemented by concrete factory classes
        raise NotImplementedError("OptimizerFactory is an interface and cannot be used directly")
    
    @staticmethod
    def create_optimizer_for_expert(
        expert_type: str,
        expert_domain: str,
        config: Dict[str, Any] = None
    ) -> ExpertSpecificOptimizer:
        """
        Create an optimizer appropriate for the specified expert type.
        
        Args:
            expert_type: Type of expert (e.g., "lstm", "random_forest")
            expert_domain: Domain of the expert (e.g., "physiological")
            config: Optional configuration parameters
            
        Returns:
            ExpertSpecificOptimizer instance
            
        Raises:
            ValueError: If no suitable optimizer is available
        """
        # This will be implemented by concrete factory classes
        raise NotImplementedError("OptimizerFactory is an interface and cannot be used directly")
