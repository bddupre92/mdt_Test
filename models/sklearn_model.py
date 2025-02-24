"""
sklearn_model.py
----------------
Wrapper for a scikit-learn model.
"""

from .base_model import BaseModel

class SklearnModel(BaseModel):
    def __init__(self, sk_model):
        """
        sk_model: an sklearn estimator (e.g. RandomForestClassifier, LogisticRegression)
        """
        self.sk_model = sk_model
    
    def train(self, X, y):
        self.sk_model.fit(X, y)
    
    def predict(self, X):
        # For classification, this might return predicted labels
        return self.sk_model.predict(X)
    
    def predict_proba(self, X):
        # if available
        if hasattr(self.sk_model, 'predict_proba'):
            return self.sk_model.predict_proba(X)
        else:
            # fallback
            return None
