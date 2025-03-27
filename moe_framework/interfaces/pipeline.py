"""
Pipeline interfaces for the Mixture of Experts (MoE) framework.

This module defines interfaces for end-to-end pipelines that coordinate
data flow from input through preprocessing, expert execution, and result
integration.
"""

import logging
from abc import abstractmethod
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from .base import Configurable, Persistable, PatientContext, PredictionResult

logger = logging.getLogger(__name__)


class Pipeline(Configurable, Persistable):
    """
    Base interface for end-to-end processing pipelines in the MoE framework.
    
    Pipelines orchestrate the entire workflow from data input to final prediction
    output, coordinating preprocessing, expert execution, and result integration.
    """
    
    @abstractmethod
    def execute(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute the complete pipeline on input data.
        
        Args:
            input_data: Raw input data in dictionary format
            
        Returns:
            Dictionary containing results and metadata
        """
        pass
    
    @abstractmethod
    def preprocess(self, input_data: Dict[str, Any]) -> Dict[str, np.ndarray]:
        """
        Preprocess raw input data for expert models.
        
        Args:
            input_data: Raw input data
            
        Returns:
            Preprocessed features for different domains
        """
        pass
    
    @abstractmethod
    def execute_experts(
        self, 
        features: Dict[str, np.ndarray],
        weights: Dict[str, float]
    ) -> Dict[str, np.ndarray]:
        """
        Execute expert models with the given features and weights.
        
        Args:
            features: Preprocessed input features for each domain
            weights: Weight dictionary for each expert
            
        Returns:
            Dictionary mapping expert names to prediction outputs
        """
        pass
    
    @abstractmethod
    def integrate_predictions(
        self,
        predictions: Dict[str, np.ndarray],
        weights: Dict[str, float]
    ) -> np.ndarray:
        """
        Integrate predictions from multiple experts into a final output.
        
        Args:
            predictions: Dictionary of expert predictions
            weights: Weight dictionary for each expert
            
        Returns:
            Integrated prediction result
        """
        pass
    
    @abstractmethod
    def generate_result(
        self,
        integrated_prediction: np.ndarray,
        expert_predictions: Dict[str, np.ndarray],
        weights: Dict[str, float],
        context: PatientContext = None
    ) -> PredictionResult:
        """
        Generate a structured prediction result with metadata.
        
        Args:
            integrated_prediction: Final integrated prediction
            expert_predictions: Individual expert predictions
            weights: Expert weights used
            context: Optional patient context
            
        Returns:
            Structured prediction result
        """
        pass


class TrainingPipeline(Pipeline):
    """
    Extension of Pipeline for training workflows.
    
    This interface adds methods specific to training expert models
    and gating networks with training and validation data.
    """
    
    @abstractmethod
    def train(
        self,
        training_data: Dict[str, Any],
        validation_data: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Train all components of the MoE system.
        
        Args:
            training_data: Training dataset
            validation_data: Optional validation dataset
            
        Returns:
            Dictionary of training metrics
        """
        pass
    
    @abstractmethod
    def train_experts(
        self,
        training_data: Dict[str, Any]
    ) -> Dict[str, Dict[str, Any]]:
        """
        Train individual expert models.
        
        Args:
            training_data: Training dataset
            
        Returns:
            Dictionary mapping expert names to training metrics
        """
        pass
    
    @abstractmethod
    def train_gating_network(
        self,
        training_data: Dict[str, Any],
        expert_metrics: Dict[str, Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Train the gating network using expert performance data.
        
        Args:
            training_data: Training dataset
            expert_metrics: Performance metrics for each expert
            
        Returns:
            Dictionary of gating network training metrics
        """
        pass
    
    @abstractmethod
    def evaluate(
        self,
        test_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Evaluate the complete pipeline on test data.
        
        Args:
            test_data: Test dataset
            
        Returns:
            Dictionary of evaluation metrics
        """
        pass


class CheckpointingPipeline(Pipeline):
    """
    Extension of Pipeline with checkpointing capabilities.
    
    This interface adds methods for saving and restoring pipeline state
    to enable resumable processing and recovery from failures.
    """
    
    @abstractmethod
    def save_checkpoint(self, path: str) -> bool:
        """
        Save a checkpoint of the current pipeline state.
        
        Args:
            path: Directory path where to save the checkpoint
            
        Returns:
            Success status
        """
        pass
    
    @abstractmethod
    def load_checkpoint(self, path: str) -> bool:
        """
        Load a checkpoint to restore pipeline state.
        
        Args:
            path: Directory path from which to load the checkpoint
            
        Returns:
            Success status
        """
        pass
    
    @abstractmethod
    def list_checkpoints(self, base_dir: str) -> List[str]:
        """
        List available checkpoints in the specified directory.
        
        Args:
            base_dir: Base directory to search for checkpoints
            
        Returns:
            List of checkpoint paths
        """
        pass
    
    @abstractmethod
    def get_checkpoint_metadata(self, checkpoint_path: str) -> Dict[str, Any]:
        """
        Get metadata about a specific checkpoint.
        
        Args:
            checkpoint_path: Path to the checkpoint
            
        Returns:
            Dictionary containing checkpoint metadata
        """
        pass
