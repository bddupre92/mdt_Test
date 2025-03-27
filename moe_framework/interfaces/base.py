"""
Base interfaces for the Mixture of Experts (MoE) framework.

This module defines the core abstract base classes and interfaces that all
components of the MoE framework must implement. These interfaces ensure
consistent interaction between different parts of the system.
"""

import os
import json
import logging
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class Configurable(ABC):
    """Base interface for all configurable components."""
    
    @abstractmethod
    def get_config(self) -> Dict[str, Any]:
        """
        Get the current configuration of the component.
        
        Returns:
            Dict containing the configuration parameters
        """
        pass
        
    @abstractmethod
    def set_config(self, config: Dict[str, Any]) -> None:
        """
        Configure the component with the provided parameters.
        
        Args:
            config: Dictionary containing configuration parameters
        """
        pass


class Persistable(ABC):
    """Base interface for all components that can be persisted to disk."""
    
    @abstractmethod
    def save_state(self, path: str) -> bool:
        """
        Save the component state to the specified path.
        
        Args:
            path: Directory path where the state should be saved
            
        Returns:
            Success status
        """
        pass
        
    @abstractmethod
    def load_state(self, path: str) -> bool:
        """
        Load the component state from the specified path.
        
        Args:
            path: Directory path from which to load the state
            
        Returns:
            Success status
        """
        pass


class DataStructure:
    """Base class for all data structures used in the framework."""
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the data structure to a dictionary.
        
        Returns:
            Dictionary representation of the data structure
        """
        return {
            key: value for key, value in self.__dict__.items() 
            if not key.startswith('_')
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'DataStructure':
        """
        Create a data structure from a dictionary.
        
        Args:
            data: Dictionary containing the data structure fields
            
        Returns:
            New instance of the data structure
        """
        instance = cls()
        for key, value in data.items():
            if hasattr(instance, key):
                setattr(instance, key, value)
        return instance
    
    def to_json(self) -> str:
        """
        Convert the data structure to a JSON string.
        
        Returns:
            JSON string representation
        """
        return json.dumps(self.to_dict())
    
    @classmethod
    def from_json(cls, json_str: str) -> 'DataStructure':
        """
        Create a data structure from a JSON string.
        
        Args:
            json_str: JSON string containing the data structure
            
        Returns:
            New instance of the data structure
        """
        return cls.from_dict(json.loads(json_str))


# Core data structures for the MoE framework

class PatientContext(DataStructure):
    """Data structure for patient context information."""
    
    def __init__(self):
        self.patient_id: str = ""
        self.demographics: Dict[str, Any] = {}
        self.medical_history: List[Dict[str, Any]] = []
        self.preferences: Dict[str, Any] = {}
        self.device_info: Dict[str, Any] = {}
        self.timestamp: str = datetime.now().isoformat()


class PredictionResult(DataStructure):
    """Data structure for prediction results."""
    
    def __init__(self):
        self.prediction: Union[float, np.ndarray] = 0.0
        self.confidence: float = 0.0
        self.expert_contributions: Dict[str, float] = {}
        self.explanation: Dict[str, Any] = {}
        self.feature_importance: Dict[str, float] = {}
        self.timestamp: str = datetime.now().isoformat()


class SystemState(DataStructure):
    """Data structure for system state information."""
    
    def __init__(self):
        self.meta_learner_state: Dict[str, Any] = {}
        self.expert_states: Dict[str, Dict[str, Any]] = {}
        self.gating_network_state: Dict[str, Any] = {}
        self.optimizer_states: Dict[str, Dict[str, Any]] = {}
        self.performance_metrics: Dict[str, Any] = {
            "expert_benchmarks": {},      # Individual expert performance metrics
            "gating_evaluation": {},      # Gating network accuracy metrics
            "end_to_end_metrics": {},     # Overall system performance metrics
            "baseline_comparisons": {},    # Comparisons against baseline selectors
            "statistical_tests": {},       # Results of statistical significance tests
            "visualization_metadata": {},  # Metadata for visualization generation
            "temporal_analysis": {},       # Time-based performance analysis
            "experiment_id": "",          # Identifier for the evaluation experiment
            "data_config_id": ""          # Reference to the data configuration used
        }
        self.version_info: Dict[str, str] = {
            "timestamp": datetime.now().isoformat(),
            "version": "0.1.0"
        }
