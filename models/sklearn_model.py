"""
Scikit-learn model wrapper.
"""
import numpy as np
from sklearn.base import BaseEstimator
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

class SklearnModel:
    """Wrapper for scikit-learn models."""
    
    def __init__(self, model: BaseEstimator):
        """Initialize model."""
        self.model = model
        self.feature_importance_ = None
    
    def train(self, X: np.ndarray, y: np.ndarray) -> None:
        """Train model (alias for fit)."""
        return self.fit(X, y)
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Fit model to data."""
        self.model.fit(X, y)
        if hasattr(self.model, 'feature_importances_'):
            self.feature_importance_ = self.model.feature_importances_
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions."""
        return self.model.predict(X)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Get prediction probabilities."""
        return self.model.predict_proba(X)
    
    def evaluate(self, X: np.ndarray, y: np.ndarray) -> dict:
        """Evaluate model performance."""
        y_pred = self.predict(X)
        return {
            'accuracy': accuracy_score(y, y_pred),
            'precision': precision_score(y, y_pred),
            'recall': recall_score(y, y_pred),
            'f1': f1_score(y, y_pred)
        }
    
    def get_feature_importance(self) -> np.ndarray:
        """Get feature importance scores."""
        if self.feature_importance_ is None:
            raise ValueError("Model does not support feature importance")
        return self.feature_importance_
    
    def save(self, path: str) -> None:
        """Save model to disk."""
        import joblib
        joblib.dump(self.model, path)
    
    @classmethod
    def load(cls, path: str) -> 'SklearnModel':
        """Load model from disk."""
        import joblib
        model = joblib.load(path)
        return cls(model)
