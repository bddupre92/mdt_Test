"""
model_factory.py
---------------
Factory class for creating different types of models
"""

from typing import Dict, Any
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from .base_model import BaseModel

class SklearnModelWrapper(BaseModel):
    """Wrapper for scikit-learn models to maintain consistent interface"""
    def __init__(self, model):
        self.model = model
        self.history = {'accuracy': [], 'loss': []}
        
    def fit(self, X, y):
        """Fit the model and track basic metrics"""
        self.model.fit(X, y)
        pred = self.model.predict(X)
        accuracy = np.mean(pred == y)
        self.history['accuracy'].append(accuracy)
        self.history['loss'].append(1 - accuracy)
        return self.history
        
    def predict(self, X):
        return self.model.predict(X)
        
    def predict_proba(self, X):
        """Get probability estimates for each class"""
        if hasattr(self.model, 'predict_proba'):
            return self.model.predict_proba(X)
        # For models without predict_proba, use decision_function if available
        elif hasattr(self.model, 'decision_function'):
            df = self.model.decision_function(X)
            if df.ndim == 1:
                return np.vstack([1-df, df]).T
            else:
                return df
        else:
            # Fallback to hard predictions
            pred = self.predict(X)
            return np.eye(2)[pred]
        
    def score(self, X, y):
        return self.model.score(X, y)

class ModelFactory:
    """Factory class for creating models based on configuration"""
    
    def __init__(self):
        self.model_types = {
            'random_forest': self._create_random_forest,
            'logistic_regression': self._create_logistic_regression,
            'svm': self._create_svm
        }
    
    def create_model(self, config: Dict[str, Any]) -> BaseModel:
        """
        Create a model based on configuration
        
        Args:
            config: Dictionary containing model type and parameters
                   Example: {'type': 'random_forest', 'params': {'n_estimators': 100}}
        
        Returns:
            BaseModel: Initialized model
        """
        model_type = config['type']
        if model_type not in self.model_types:
            raise ValueError(f"Unknown model type: {model_type}")
            
        params = config.get('params', {})
        return self.model_types[model_type](params)
        
    def _create_random_forest(self, params: Dict[str, Any]) -> BaseModel:
        """Create random forest classifier"""
        model = RandomForestClassifier(
            n_estimators=params.get('n_estimators', 100),
            max_depth=params.get('max_depth', None),
            random_state=42
        )
        return SklearnModelWrapper(model)
        
    def _create_logistic_regression(self, params: Dict[str, Any]) -> BaseModel:
        """Create logistic regression classifier"""
        model = LogisticRegression(
            C=params.get('C', 1.0),
            random_state=42
        )
        return SklearnModelWrapper(model)
        
    def _create_svm(self, params: Dict[str, Any]) -> BaseModel:
        """Create SVM classifier"""
        model = SVC(
            C=params.get('C', 1.0),
            kernel=params.get('kernel', 'rbf'),
            random_state=42
        )
        return SklearnModelWrapper(model)
