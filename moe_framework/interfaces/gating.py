"""
Gating network interfaces for the Mixture of Experts (MoE) framework.

This module defines interfaces for gating networks that determine
the weighting of expert models for prediction integration.
"""

import logging
from abc import abstractmethod
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from .base import Configurable, Persistable
from .expert import ExpertModel

logger = logging.getLogger(__name__)


class GatingNetwork(Configurable, Persistable):
    """
    Base interface for gating networks in the MoE framework.
    
    Gating networks determine how to distribute weight among experts
    based on input features and context information.
    """
    
    @abstractmethod
    def predict_weights(
        self, 
        features: np.ndarray, 
        experts: List[ExpertModel],
        context: Dict[str, Any] = None
    ) -> Dict[str, float]:
        """
        Predict weights for each expert based on input features and context.
        
        Args:
            features: Input feature matrix of shape (n_samples, n_features)
            experts: List of available expert models
            context: Optional additional context information
            
        Returns:
            Dictionary mapping expert names to weights (summing to 1.0)
        """
        pass
    
    @abstractmethod
    def train(
        self,
        features: np.ndarray,
        expert_performances: Dict[str, np.ndarray],
        context: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        Train the gating network using expert performance data.
        
        Args:
            features: Input feature matrix used for predictions
            expert_performances: Performance metrics for each expert
            context: Optional additional context information
            
        Returns:
            Dictionary containing training metrics
        """
        pass
    
    @abstractmethod
    def update_weights(
        self,
        features: np.ndarray,
        performances: Dict[str, float],
        previous_weights: Dict[str, float]
    ) -> Dict[str, float]:
        """
        Update weights based on recent performance feedback.
        
        Args:
            features: Feature matrix that generated the predictions
            performances: Performance metrics for each expert
            previous_weights: Weights assigned in the previous iteration
            
        Returns:
            Updated weight dictionary
        """
        pass


class QualityAwareGating(GatingNetwork):
    """
    Extension of gating network that incorporates data quality information.
    
    This interface adds methods for working with data quality metrics
    to adjust expert weighting based on the quality of input data.
    """
    
    @abstractmethod
    def incorporate_quality_metrics(
        self,
        weights: Dict[str, float],
        quality_metrics: Dict[str, float]
    ) -> Dict[str, float]:
        """
        Adjust weights based on data quality metrics.
        
        Args:
            weights: Initial expert weights
            quality_metrics: Quality metrics for different data domains
            
        Returns:
            Adjusted weights incorporating quality information
        """
        pass
    
    @abstractmethod
    def get_quality_thresholds(self) -> Dict[str, float]:
        """
        Get the quality thresholds for different data domains.
        
        Returns:
            Dictionary mapping domain names to quality thresholds
        """
        pass
    
    @abstractmethod
    def set_quality_thresholds(self, thresholds: Dict[str, float]) -> None:
        """
        Set quality thresholds for different data domains.
        
        Args:
            thresholds: Dictionary mapping domain names to quality thresholds
        """
        pass


class DriftAwareGating(GatingNetwork):
    """
    Extension of gating network that incorporates drift detection.
    
    This interface adds methods for detecting and responding to
    concept drift in input data distributions.
    """
    
    @abstractmethod
    def detect_drift(
        self,
        recent_features: np.ndarray,
        historical_features: np.ndarray
    ) -> Dict[str, float]:
        """
        Detect drift between recent and historical feature distributions.
        
        Args:
            recent_features: Recent feature matrix
            historical_features: Historical feature matrix
            
        Returns:
            Dictionary of drift metrics for different domains
        """
        pass
    
    @abstractmethod
    def adjust_for_drift(
        self,
        weights: Dict[str, float],
        drift_metrics: Dict[str, float]
    ) -> Dict[str, float]:
        """
        Adjust weights based on detected drift.
        
        Args:
            weights: Initial expert weights
            drift_metrics: Metrics quantifying drift in different domains
            
        Returns:
            Adjusted weights accounting for drift
        """
        pass
