"""
Expert model interfaces for the Mixture of Experts (MoE) framework.

This module defines interfaces for expert models within the MoE system,
establishing a common contract that all expert implementations must follow.
"""

import logging
from abc import abstractmethod
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from .base import Configurable, Persistable

logger = logging.getLogger(__name__)


class ExpertModel(Configurable, Persistable):
    """
    Base interface for all expert models in the MoE framework.
    
    Expert models are specialized predictors that focus on specific
    aspects of the problem domain (e.g., physiological, behavioral,
    or environmental factors for migraine prediction).
    """
    
    @abstractmethod
    def predict(self, features: np.ndarray) -> np.ndarray:
        """
        Generate predictions from input features.
        
        Args:
            features: Input feature matrix of shape (n_samples, n_features)
            
        Returns:
            Predictions of shape (n_samples, n_outputs)
        """
        pass
    
    @abstractmethod
    def get_specialty(self) -> str:
        """
        Return the expert's specialty domain.
        
        Returns:
            String identifier for the expert's specialty area
            (e.g., 'physiological', 'behavioral', 'environmental')
        """
        pass
    
    @abstractmethod
    def get_confidence(self, features: np.ndarray) -> np.ndarray:
        """
        Return confidence estimates for predictions.
        
        Args:
            features: Input feature matrix of shape (n_samples, n_features)
            
        Returns:
            Confidence scores of shape (n_samples,)
        """
        pass
    
    @abstractmethod
    def train(self, X_train: np.ndarray, y_train: np.ndarray) -> Dict[str, Any]:
        """
        Train the expert model on the provided data.
        
        Args:
            X_train: Training feature matrix of shape (n_samples, n_features)
            y_train: Target values of shape (n_samples,)
            
        Returns:
            Dictionary containing training metrics
        """
        pass
    
    @abstractmethod
    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
        """
        Evaluate the expert model on test data.
        
        Args:
            X_test: Test feature matrix of shape (n_samples, n_features)
            y_test: Test target values of shape (n_samples,)
            
        Returns:
            Dictionary of evaluation metrics (e.g., accuracy, F1 score)
        """
        pass
    
    def get_feature_importance(self) -> Dict[str, float]:
        """
        Get feature importance scores if available.
        
        Returns:
            Dictionary mapping feature names to importance scores,
            or empty dict if not supported by the model
        """
        return {}


class ExpertRegistry:
    """
    Registry for managing expert models in the MoE framework.
    
    The registry maintains a collection of expert models and provides
    methods for registering, retrieving, and managing them.
    """
    
    def __init__(self):
        """Initialize an empty expert registry."""
        self.experts: Dict[str, ExpertModel] = {}
        
    def register(self, name: str, expert: ExpertModel) -> None:
        """
        Register an expert model with the specified name.
        
        Args:
            name: Unique identifier for the expert
            expert: Expert model instance
        """
        if name in self.experts:
            logger.warning(f"Expert '{name}' already exists and will be overwritten")
        self.experts[name] = expert
        logger.info(f"Registered expert '{name}' with specialty '{expert.get_specialty()}'")
        
    def get_expert(self, name: str) -> Optional[ExpertModel]:
        """
        Get an expert model by name.
        
        Args:
            name: Identifier of the expert to retrieve
            
        Returns:
            Expert model if found, None otherwise
        """
        expert = self.experts.get(name)
        if expert is None:
            logger.warning(f"Expert '{name}' not found in registry")
        return expert
        
    def get_all_experts(self) -> Dict[str, ExpertModel]:
        """
        Get all registered expert models.
        
        Returns:
            Dictionary mapping expert names to expert instances
        """
        return self.experts
        
    def get_experts_by_specialty(self, specialty: str) -> Dict[str, ExpertModel]:
        """
        Get all experts with the specified specialty.
        
        Args:
            specialty: Specialty to filter by
            
        Returns:
            Dictionary mapping expert names to expert instances with the specified specialty
        """
        return {
            name: expert for name, expert in self.experts.items() 
            if expert.get_specialty() == specialty
        }
        
    def remove_expert(self, name: str) -> bool:
        """
        Remove an expert from the registry.
        
        Args:
            name: Name of the expert to remove
            
        Returns:
            True if the expert was removed, False if not found
        """
        if name in self.experts:
            del self.experts[name]
            logger.info(f"Removed expert '{name}' from registry")
            return True
        logger.warning(f"Expert '{name}' not found in registry")
        return False
