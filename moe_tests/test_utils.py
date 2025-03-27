"""
Test utilities for MoE framework testing.

This module provides utility functions and mock classes to facilitate
testing of the MoE framework components.
"""

import numpy as np
import pandas as pd
from sklearn.datasets import make_regression
from typing import Tuple, Dict, List, Any, Optional


def generate_test_data(
    n_samples: int = 100,
    n_features: int = 5,
    n_clusters: int = 3, 
    noise: float = 0.1,
    random_state: Optional[int] = None
) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """
    Generate synthetic data for testing.
    
    Args:
        n_samples: Number of samples to generate
        n_features: Number of features
        n_clusters: Number of clusters for patient grouping
        noise: Noise level
        random_state: Random state for reproducibility
        
    Returns:
        X_train: Training features
        y_train: Training targets
        X_test: Testing features
        y_test: Testing targets
    """
    # Set random state for reproducibility
    if random_state is not None:
        np.random.seed(random_state)
        
    # Generate regression data
    X, y = make_regression(
        n_samples=n_samples * 2,  # Double for train/test split
        n_features=n_features,
        noise=noise,
        random_state=random_state
    )
    
    # Convert to DataFrame and Series
    features = [f'feature_{i}' for i in range(n_features)]
    X_df = pd.DataFrame(X, columns=features)
    y_series = pd.Series(y, name='target')
    
    # Split into train and test
    X_train = X_df.iloc[:n_samples]
    y_train = y_series.iloc[:n_samples]
    X_test = X_df.iloc[n_samples:]
    y_test = y_series.iloc[n_samples:]
    
    return X_train, y_train, X_test, y_test


class MockExpert:
    """
    Mock expert model for testing.
    """
    
    def __init__(self, name: str, error_level: float = 0.1):
        """
        Initialize the mock expert.
        
        Args:
            name: Name of the expert
            error_level: Error level for predictions
        """
        self.name = name
        self.error_level = error_level
        self.is_trained = False
        
    def train(self, X: pd.DataFrame, y: pd.Series) -> None:
        """
        Train the mock expert.
        
        Args:
            X: Training features
            y: Training targets
        """
        self.is_trained = True
        self.mean = y.mean()
        self.std = y.std()
        
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Generate predictions.
        
        Args:
            X: Features
            
        Returns:
            Array of predictions
        """
        if not self.is_trained:
            raise ValueError("Expert not trained")
            
        # Generate noisy predictions around the mean
        n_samples = len(X)
        return self.mean + np.random.normal(0, self.error_level * self.std, size=n_samples)
    
    def get_confidence(self, X: pd.DataFrame) -> np.ndarray:
        """
        Generate confidence scores.
        
        Args:
            X: Features
            
        Returns:
            Array of confidence scores
        """
        n_samples = len(X)
        return np.random.beta(5, 2, size=n_samples)


class MockGatingNetwork:
    """
    Mock gating network for testing.
    """
    
    def __init__(self, expert_names: List[str]):
        """
        Initialize the mock gating network.
        
        Args:
            expert_names: List of expert names to select from
        """
        self.expert_names = expert_names
        self.is_trained = False
        
    def train(self, X: pd.DataFrame, y: pd.Series, expert_errors: Dict[str, List[float]]) -> None:
        """
        Train the mock gating network.
        
        Args:
            X: Training features
            y: Training targets
            expert_errors: Dictionary mapping expert names to their errors
        """
        self.is_trained = True
        
    def predict_weights(self, X: pd.DataFrame) -> Dict[str, List[float]]:
        """
        Predict weights for each expert.
        
        Args:
            X: Features
            
        Returns:
            Dictionary mapping expert names to their weights
        """
        if not self.is_trained:
            raise ValueError("Gating network not trained")
            
        n_samples = len(X)
        n_experts = len(self.expert_names)
        
        # Generate random weights for each expert
        weights = {}
        for i, expert in enumerate(self.expert_names):
            # Each expert gets a different bias to create some variation
            bias = 1.0 - (i * 0.2)
            raw_weights = np.random.beta(2 * bias, 2, size=n_samples)
            weights[expert] = raw_weights.tolist()
            
        # Normalize weights so they sum to 1 for each sample
        for i in range(n_samples):
            sample_weights = [weights[expert][i] for expert in self.expert_names]
            total = sum(sample_weights)
            
            if total > 0:
                for expert in self.expert_names:
                    weights[expert][i] /= total
                    
        return weights
