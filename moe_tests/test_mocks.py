"""
Mock classes and utilities for testing the MoE framework.

This module provides mocks of core MoE framework components to facilitate
testing without requiring the actual implementations.
"""

from enum import Enum, auto
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Union


class MoEEventTypes(Enum):
    """Mock enum for MoE event types used in tests."""
    PIPELINE_INITIALIZED = auto()
    EXPERTS_INITIALIZED = auto()
    GATING_INITIALIZED = auto()
    INTEGRATION_INITIALIZED = auto()
    TRAINING_STARTED = auto()
    TRAINING_COMPLETED = auto()
    PREDICTION_STARTED = auto()
    PREDICTION_COMPLETED = auto()
    EXPERT_SELECTED = auto()
    EXPERT_PREDICTION = auto()
    CONFIDENCE_CALCULATED = auto()


class MockMoEPipeline:
    """
    Mock MoE pipeline for testing.
    """
    
    def __init__(self, config=None, verbose=False):
        """Initialize the mock pipeline."""
        self.config = config or {}
        self.verbose = verbose
        self.is_trained = False
        self.experts = {}
        self.gating_network = None
        
        # Initialize experts based on config or default
        if 'experts' in self.config and isinstance(self.config['experts'], dict):
            for expert_name, expert_config in self.config['experts'].items():
                self.experts[expert_name] = {"name": expert_name, "config": expert_config}
        else:
            # Default experts if not provided
            self.experts = {
                "expert1": {"name": "expert1", "config": {"model_type": "linear"}},
                "expert2": {"name": "expert2", "config": {"model_type": "tree"}},
                "expert3": {"name": "expert3", "config": {"model_type": "svm"}}
            }
    
    def train(self, X, y):
        """Mock training method."""
        self.is_trained = True
        self.X_train = X
        self.y_train = y
        return self
    
    def predict(self, X):
        """Mock prediction method."""
        if not self.is_trained:
            raise ValueError("Pipeline not trained")
        
        # Return mock predictions
        n_samples = len(X)
        return np.random.normal(self.y_train.mean(), self.y_train.std(), size=n_samples)
    
    def get_expert_weights_batch(self, X):
        """Mock method to get expert weights."""
        n_samples = len(X)
        weights = {}
        
        # Generate weights for each expert
        for expert_name in self.experts:
            weights[expert_name] = np.random.dirichlet(
                np.ones(3) * 5, size=n_samples
            )[:, 0].tolist()
            
        return weights
    
    def get_expert_predictions(self, X):
        """Mock method to get expert predictions."""
        n_samples = len(X)
        predictions = {}
        
        # Generate predictions for each expert
        for expert_name in self.experts:
            base = self.y_train.mean() if hasattr(self, 'y_train') else 0
            std = self.y_train.std() if hasattr(self, 'y_train') else 1
            predictions[expert_name] = base + np.random.normal(0, std * 0.2, size=n_samples)
            
        return predictions
    
    def get_prediction_confidence(self, X):
        """Mock method to get prediction confidence."""
        n_samples = len(X)
        return np.random.beta(5, 2, size=n_samples)
    
    def save(self, path):
        """Mock save method."""
        return True
    
    def load(self, path):
        """Mock load method."""
        self.is_trained = True
        return self


class MockMoEMetricsCalculator:
    """
    Mock metrics calculator for testing.
    """
    
    def __init__(self, config=None):
        """Initialize the mock metrics calculator."""
        self.config = config or {}
        
    def calculate_metrics(self, y_true, y_pred, expert_weights, expert_predictions, confidence_scores=None):
        """Mock method to calculate metrics."""
        return {
            'standard': {
                'rmse': 0.25,
                'mae': 0.2,
                'r2': 0.85,
                'mape': 5.0
            },
            'expert_contribution': {
                'normalized_entropy': 0.65,
                'expert_dominance_counts': {expert: 10 for expert in expert_weights},
                'max_weight_distribution': [0.8, 0.15, 0.05]
            },
            'confidence': {
                'mean_confidence': 0.75,
                'confidence_error_correlation': -0.6,
                'high_confidence_accuracy': 0.9
            },
            'temporal': {
                'drift_score': 0.1,
                'temporal_consistency': 0.85
            }
        }
    
    def visualize_metrics(self, output_dir, prefix, y_true, y_pred, expert_weights, 
                          expert_predictions, confidence_scores=None):
        """Mock method to visualize metrics."""
        # Create dummy files to simulate visualization outputs
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        file_paths = []
        for vis_type in ['expert_contributions', 'confidence_distribution', 'error_analysis']:
            file_path = os.path.join(output_dir, f"{prefix}_{vis_type}.png")
            with open(file_path, 'w') as f:
                f.write("Mock visualization file")
            file_paths.append(file_path)
            
        return file_paths
