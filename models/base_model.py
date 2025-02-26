"""
base_model.py
------------
Base class for all models
"""

from abc import ABC, abstractmethod
import numpy as np

class BaseModel(ABC):
    """Abstract base class for all models"""
    
    @abstractmethod
    def fit(self, X, y):
        """
        Fit the model to the data
        
        Args:
            X: Features
            y: Labels
            
        Returns:
            dict: Training history
        """
        pass
    
    @abstractmethod
    def predict(self, X):
        """
        Make predictions
        
        Args:
            X: Features
            
        Returns:
            array-like: Predictions
        """
        pass
    
    @abstractmethod
    def score(self, X, y):
        """
        Calculate score (accuracy for classification)
        
        Args:
            X: Features
            y: True labels
            
        Returns:
            float: Score
        """
        pass
