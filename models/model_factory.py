"""
model_factory.py
--------------
Factory for creating model instances with appropriate wrappers
"""

from typing import Dict, Any, Optional, List
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from .base_model import BaseModel

class SklearnModelWrapper(BaseModel):
    """Wrapper for scikit-learn models"""
    
    def __init__(self, model):
        self.model = model
        self.feature_importances_ = None
        self.confusion_matrix_ = None
        self.feature_names = None
    
    def fit(self, X, y, feature_names: Optional[List[str]] = None):
        """
        Train model and store feature importances
        
        Args:
            X: Features
            y: Labels
            feature_names: Optional list of feature names
        """
        self.model.fit(X, y)
        self.feature_names = feature_names
        
        # Store feature importances if available
        if hasattr(self.model, 'feature_importances_'):
            importances = self.model.feature_importances_
            if feature_names:
                self.feature_importances_ = dict(zip(feature_names, importances))
            else:
                self.feature_importances_ = dict(enumerate(importances))
        
        return self
    
    def predict(self, X):
        """Get predictions"""
        return self.model.predict(X)
    
    def predict_proba(self, X):
        """Get prediction probabilities"""
        return self.model.predict_proba(X)
    
    def score(self, X, y):
        """Get model score"""
        return self.model.score(X, y)
    
    def get_feature_importances(self) -> Optional[Dict]:
        """Get feature importance dictionary"""
        return self.feature_importances_

class ModelFactory:
    """Factory for creating model instances"""
    
    def create_model(self, config: Dict[str, Any]) -> BaseModel:
        """
        Create a model instance based on configuration
        
        Args:
            config: Model configuration with parameters
            
        Returns:
            Wrapped model instance
        """
        # For now, we only support RandomForest
        model = RandomForestClassifier(
            n_estimators=config.get('n_estimators', 100),
            max_depth=config.get('max_depth', None),
            min_samples_split=config.get('min_samples_split', 10),
            min_samples_leaf=config.get('min_samples_leaf', 4),
            max_features=config.get('max_features', 'sqrt'),
            random_state=42
        )
        
        return SklearnModelWrapper(model)
