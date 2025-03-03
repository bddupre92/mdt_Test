"""
model_factory.py
--------------
Factory for creating model instances with appropriate wrappers
"""

from typing import Dict, Any, Optional, List
import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
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
        """Get prediction probabilities if available"""
        if hasattr(self.model, 'predict_proba'):
            return self.model.predict_proba(X)
        else:
            # For regressors, return None
            return None
    
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
        # Validate and sanitize all parameters to ensure they're valid
        # n_estimators: must be a positive integer
        n_estimators = config.get('n_estimators', 100)
        if not isinstance(n_estimators, int) or n_estimators < 1:
            n_estimators = 100
        
        # max_depth: must be a positive integer or None
        max_depth = config.get('max_depth', None)
        if max_depth is not None:
            if not isinstance(max_depth, (int, float)) or max_depth < 1:
                max_depth = None
            else:
                max_depth = int(max_depth)
        
        # min_samples_split: must be an int >= 2 or a float in (0,1)
        min_samples_split = config.get('min_samples_split', 2)
        if isinstance(min_samples_split, (int, float)):
            if isinstance(min_samples_split, int) and min_samples_split < 2:
                min_samples_split = 2
            elif isinstance(min_samples_split, float) and (min_samples_split <= 0 or min_samples_split >= 1):
                min_samples_split = 2
        else:
            min_samples_split = 2
        
        # min_samples_leaf: must be a positive integer or a float in (0,0.5)
        min_samples_leaf = config.get('min_samples_leaf', 1)
        if isinstance(min_samples_leaf, (int, float)):
            if isinstance(min_samples_leaf, int) and min_samples_leaf < 1:
                min_samples_leaf = 1
            elif isinstance(min_samples_leaf, float) and (min_samples_leaf <= 0 or min_samples_leaf >= 0.5):
                min_samples_leaf = 1
        else:
            min_samples_leaf = 1
        
        # max_features: must be 'sqrt', 'log2', None, int, or float
        max_features = config.get('max_features', 'sqrt')
        valid_str_features = ['sqrt', 'log2', 'auto', None]
        if isinstance(max_features, str) and max_features not in valid_str_features:
            max_features = 'sqrt'
        elif isinstance(max_features, (int, float)) and max_features <= 0:
            max_features = 'sqrt'
            
        # Get additional classification parameters
        bootstrap = config.get('bootstrap', True)
        class_weight = config.get('class_weight', None)
        
        # Determine whether to use classifier or regressor
        task_type = config.get('task_type', 'regression')
        if task_type == 'classification':
            model = RandomForestClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                min_samples_split=min_samples_split,
                min_samples_leaf=min_samples_leaf,
                max_features=max_features,
                bootstrap=bootstrap,
                class_weight=class_weight,
                random_state=42
            )
        else:
            model = RandomForestRegressor(
                n_estimators=n_estimators,
                max_depth=max_depth,
                min_samples_split=min_samples_split,
                min_samples_leaf=min_samples_leaf,
                max_features=max_features,
                random_state=42
            )
        
        return SklearnModelWrapper(model)
