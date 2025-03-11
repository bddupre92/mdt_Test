"""Pattern Classification Framework

This module implements various pattern classification methods for physiological signal analysis.
It provides specialized classifiers for different types of patterns:
- Binary classification (e.g., migraine vs. no migraine)
- Multi-class classification (e.g., different migraine types)
- Probabilistic classification (with confidence scores)
- Ensemble methods (combining multiple classifiers)

Each classifier implements the PatternClassifier interface defined in __init__.py.
"""

import numpy as np
from scipy import stats
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from typing import Dict, List, Optional, Union, Tuple, Any
from abc import ABC, abstractmethod

class PatternClassifier(ABC):
    """Abstract base class for pattern classifiers."""
    
    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray, **kwargs) -> None:
        """Train the classifier.
        
        Args:
            X: Input features of shape (n_samples, n_features)
            y: Target labels of shape (n_samples,)
            **kwargs: Additional training parameters
        """
        pass
    
    @abstractmethod
    def predict(self, X: np.ndarray, **kwargs) -> np.ndarray:
        """Predict class labels for samples in X.
        
        Args:
            X: Input features of shape (n_samples, n_features)
            **kwargs: Additional prediction parameters
            
        Returns:
            Predicted class labels of shape (n_samples,)
        """
        pass
    
    @abstractmethod
    def predict_proba(self, X: np.ndarray, **kwargs) -> np.ndarray:
        """Predict class probabilities for samples in X.
        
        Args:
            X: Input features of shape (n_samples, n_features)
            **kwargs: Additional prediction parameters
            
        Returns:
            Class probabilities of shape (n_samples, n_classes)
        """
        pass
    
    def evaluate(self, X: np.ndarray, y: np.ndarray, cv: int = 5) -> Dict[str, float]:
        """Evaluate classifier performance using cross-validation.
        
        Args:
            X: Input features of shape (n_samples, n_features)
            y: Target labels of shape (n_samples,)
            cv: Number of cross-validation folds
            
        Returns:
            Dictionary containing performance metrics
        """
        # Perform cross-validation
        cv_scores = cross_val_score(self.model, X, y, cv=cv)
        
        # Calculate metrics
        metrics = {
            'accuracy_mean': np.mean(cv_scores),
            'accuracy_std': np.std(cv_scores),
            'accuracy_min': np.min(cv_scores),
            'accuracy_max': np.max(cv_scores)
        }
        
        return metrics

class BinaryClassifier(PatternClassifier):
    """Binary classifier for migraine prediction."""
    
    def __init__(self, classifier_type: str = 'rf', **kwargs):
        """Initialize binary classifier.
        
        Args:
            classifier_type: Type of classifier ('rf', 'svm', 'mlp', 'gb')
            **kwargs: Additional classifier parameters
        """
        self.classifier_type = classifier_type.lower()
        self.kwargs = kwargs
        self.scaler = StandardScaler()
        
        # Initialize the classifier
        if self.classifier_type == 'rf':
            self.model = RandomForestClassifier(
                n_estimators=kwargs.get('n_estimators', 100),
                max_depth=kwargs.get('max_depth', None),
                random_state=kwargs.get('random_state', 42)
            )
        elif self.classifier_type == 'svm':
            self.model = SVC(
                kernel=kwargs.get('kernel', 'rbf'),
                C=kwargs.get('C', 1.0),
                probability=True,
                random_state=kwargs.get('random_state', 42)
            )
        elif self.classifier_type == 'mlp':
            self.model = MLPClassifier(
                hidden_layer_sizes=kwargs.get('hidden_layer_sizes', (100, 50)),
                max_iter=kwargs.get('max_iter', 1000),
                random_state=kwargs.get('random_state', 42)
            )
        elif self.classifier_type == 'gb':
            self.model = GradientBoostingClassifier(
                n_estimators=kwargs.get('n_estimators', 100),
                learning_rate=kwargs.get('learning_rate', 0.1),
                random_state=kwargs.get('random_state', 42)
            )
        else:
            raise ValueError(f"Unknown classifier type: {classifier_type}")
    
    def fit(self, X: np.ndarray, y: np.ndarray, **kwargs) -> None:
        """Train the binary classifier.
        
        Args:
            X: Input features of shape (n_samples, n_features)
            y: Binary target labels of shape (n_samples,)
            **kwargs: Additional training parameters
        """
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Train the model
        self.model.fit(X_scaled, y)
    
    def predict(self, X: np.ndarray, **kwargs) -> np.ndarray:
        """Predict binary class labels.
        
        Args:
            X: Input features of shape (n_samples, n_features)
            **kwargs: Additional prediction parameters
            
        Returns:
            Binary predictions of shape (n_samples,)
        """
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)
    
    def predict_proba(self, X: np.ndarray, **kwargs) -> np.ndarray:
        """Predict class probabilities.
        
        Args:
            X: Input features of shape (n_samples, n_features)
            **kwargs: Additional prediction parameters
            
        Returns:
            Class probabilities of shape (n_samples, 2)
        """
        X_scaled = self.scaler.transform(X)
        return self.model.predict_proba(X_scaled)

class EnsembleClassifier(PatternClassifier):
    """Ensemble classifier combining multiple base classifiers."""
    
    def __init__(self, base_classifiers: Optional[List[PatternClassifier]] = None,
                 weights: Optional[List[float]] = None):
        """Initialize ensemble classifier.
        
        Args:
            base_classifiers: List of base classifiers
            weights: List of classifier weights (must sum to 1)
        """
        self.base_classifiers = base_classifiers or [
            BinaryClassifier('rf'),
            BinaryClassifier('svm'),
            BinaryClassifier('mlp')
        ]
        
        if weights is None:
            # Equal weights by default
            self.weights = np.ones(len(self.base_classifiers)) / len(self.base_classifiers)
        else:
            if len(weights) != len(self.base_classifiers):
                raise ValueError("Number of weights must match number of classifiers")
            if not np.isclose(sum(weights), 1.0):
                raise ValueError("Weights must sum to 1")
            self.weights = np.array(weights)
    
    def fit(self, X: np.ndarray, y: np.ndarray, **kwargs) -> None:
        """Train all base classifiers.
        
        Args:
            X: Input features of shape (n_samples, n_features)
            y: Target labels of shape (n_samples,)
            **kwargs: Additional training parameters
        """
        for classifier in self.base_classifiers:
            classifier.fit(X, y, **kwargs)
    
    def predict(self, X: np.ndarray, **kwargs) -> np.ndarray:
        """Predict class labels using weighted majority voting.
        
        Args:
            X: Input features of shape (n_samples, n_features)
            **kwargs: Additional prediction parameters
            
        Returns:
            Predicted class labels of shape (n_samples,)
        """
        # Get predictions from all classifiers
        predictions = np.array([clf.predict(X) for clf in self.base_classifiers])
        
        # Weight the predictions
        weighted_pred = np.average(predictions, axis=0, weights=self.weights)
        
        # Convert to binary predictions
        return (weighted_pred > 0.5).astype(int)
    
    def predict_proba(self, X: np.ndarray, **kwargs) -> np.ndarray:
        """Predict class probabilities using weighted averaging.
        
        Args:
            X: Input features of shape (n_samples, n_features)
            **kwargs: Additional prediction parameters
            
        Returns:
            Class probabilities of shape (n_samples, n_classes)
        """
        # Get probabilities from all classifiers
        probas = np.array([clf.predict_proba(X) for clf in self.base_classifiers])
        
        # Weight the probabilities
        return np.average(probas, axis=0, weights=self.weights)

class ProbabilisticClassifier(PatternClassifier):
    """Probabilistic classifier with uncertainty estimation."""
    
    def __init__(self, base_classifier: Optional[PatternClassifier] = None,
                 n_bootstrap: int = 100):
        """Initialize probabilistic classifier.
        
        Args:
            base_classifier: Base classifier to use
            n_bootstrap: Number of bootstrap samples for uncertainty estimation
        """
        self.base_classifier = base_classifier or BinaryClassifier('rf')
        self.n_bootstrap = n_bootstrap
        self.bootstrap_models = []
    
    def fit(self, X: np.ndarray, y: np.ndarray, **kwargs) -> None:
        """Train the classifier using bootstrap aggregating.
        
        Args:
            X: Input features of shape (n_samples, n_features)
            y: Target labels of shape (n_samples,)
            **kwargs: Additional training parameters
        """
        n_samples = len(X)
        
        # Train base classifier on full dataset
        self.base_classifier.fit(X, y, **kwargs)
        
        # Train bootstrap models
        self.bootstrap_models = []
        for _ in range(self.n_bootstrap):
            # Create bootstrap sample
            indices = np.random.choice(n_samples, size=n_samples, replace=True)
            X_boot, y_boot = X[indices], y[indices]
            
            # Train model on bootstrap sample
            model = BinaryClassifier(self.base_classifier.classifier_type,
                                   **self.base_classifier.kwargs)
            model.fit(X_boot, y_boot, **kwargs)
            self.bootstrap_models.append(model)
    
    def predict(self, X: np.ndarray, **kwargs) -> np.ndarray:
        """Predict class labels using the base classifier.
        
        Args:
            X: Input features of shape (n_samples, n_features)
            **kwargs: Additional prediction parameters
            
        Returns:
            Predicted class labels of shape (n_samples,)
        """
        return self.base_classifier.predict(X)
    
    def predict_proba(self, X: np.ndarray, **kwargs) -> np.ndarray:
        """Predict class probabilities with uncertainty estimates.
        
        Args:
            X: Input features of shape (n_samples, n_features)
            **kwargs: Additional prediction parameters
            
        Returns:
            Class probabilities of shape (n_samples, 2)
        """
        # Get predictions from all bootstrap models
        boot_probas = np.array([model.predict_proba(X) for model in self.bootstrap_models])
        
        # Calculate mean and standard deviation of probabilities
        mean_proba = np.mean(boot_probas, axis=0)
        std_proba = np.std(boot_probas, axis=0)
        
        # Store uncertainty estimates as attributes
        self.uncertainty_ = std_proba
        
        return mean_proba
    
    def get_uncertainty(self) -> np.ndarray:
        """Get uncertainty estimates from the last prediction.
        
        Returns:
            Standard deviation of predicted probabilities
        """
        if not hasattr(self, 'uncertainty_'):
            raise RuntimeError("No uncertainty estimates available. Run predict_proba first.")
        return self.uncertainty_ 