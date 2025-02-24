"""
base_model.py
-------------
Abstract base class for our ML models, ensuring a consistent interface.
"""

from abc import ABC, abstractmethod

class BaseModel(ABC):
    @abstractmethod
    def train(self, X, y):
        """
        Train the model on features X and labels y.
        """
        pass
    
    @abstractmethod
    def predict(self, X):
        """
        Predict labels or probabilities for X.
        """
        pass
