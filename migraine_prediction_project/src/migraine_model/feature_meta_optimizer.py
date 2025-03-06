"""
Meta-Optimizer Based Feature Selection

This module provides functionality to use the meta-optimization framework
to select the best features for migraine prediction.
"""

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import cross_val_score
import logging

# Import meta-optimizer components
try:
    from meta.meta_learner import MetaLearner
    from meta.algorithm import Algorithm
    from meta.surrogate import Surrogate
    META_OPTIMIZER_AVAILABLE = True
except ImportError:
    META_OPTIMIZER_AVAILABLE = False
    logging.warning("Meta-optimizer package not available. Using standard feature selection.")

class MetaFeatureSelector:
    """
    Feature selector that uses meta-optimization to find the optimal feature subset.
    """
    
    def __init__(self, base_model, n_features=None, scoring='roc_auc',
                cv=5, meta_method='de', surrogate='rf', verbose=False):
        """
        Initialize the meta-feature selector.
        
        Args:
            base_model: Base estimator model to use for evaluation
            n_features: Maximum number of features to select (None for auto)
            scoring: Scoring metric for cross-validation
            cv: Number of cross-validation folds
            meta_method: Meta-optimization method ('de', 'ga', 'pso', etc.)
            surrogate: Surrogate model for meta-optimization ('rf', 'gp', etc.)
            verbose: Whether to display detailed logs
        """
        self.base_model = base_model
        self.n_features = n_features
        self.scoring = scoring
        self.cv = cv
        self.meta_method = meta_method
        self.surrogate = surrogate
        self.verbose = verbose
        self.selected_features_ = None
        self.feature_importance_ = None
        self.meta_learner = None
        
    def _evaluate_feature_subset(self, feature_mask, X, y, feature_names):
        """
        Evaluate a subset of features using cross-validation.
        
        Args:
            feature_mask: Binary mask indicating which features to use
            X: Feature matrix
            y: Target vector
            feature_names: Names of features
            
        Returns:
            Mean cross-validation score
        """
        # Select features using the binary mask
        selected_indices = np.where(feature_mask == 1)[0]
        
        if len(selected_indices) == 0:
            return 0.0  # No features selected, return lowest score
            
        # Get selected features
        X_selected = X[:, selected_indices]
        
        # Clone the model
        model = clone(self.base_model)
        
        try:
            # Evaluate model with selected features
            scores = cross_val_score(
                model, X_selected, y, 
                cv=self.cv, scoring=self.scoring
            )
            mean_score = np.mean(scores)
            
            # Add a small penalty for using many features
            penalty = 0.001 * len(selected_indices) / X.shape[1]
            
            return mean_score - penalty
        except Exception as e:
            if self.verbose:
                logging.warning(f"Error evaluating feature subset: {e}")
            return 0.0
            
    def _setup_meta_optimization(self, X, y, feature_names):
        """
        Set up the meta-optimization problem for feature selection.
        
        Args:
            X: Feature matrix
            y: Target vector
            feature_names: Names of features
            
        Returns:
            Configured meta_learner object
        """
        if not META_OPTIMIZER_AVAILABLE:
            raise ImportError("Meta-optimizer package not available")
            
        n_features = X.shape[1]
        
        # Define the optimization problem
        def objective_function(feature_mask):
            # Convert to binary
            binary_mask = np.round(feature_mask).astype(int)
            return -self._evaluate_feature_subset(binary_mask, X, y, feature_names)
            
        # Define search space - each feature can be 0 or 1
        search_space = [(0, 1)] * n_features
        
        # Initialize algorithms (using DE as default)
        algorithm = Algorithm(self.meta_method)
        
        # Initialize surrogate model
        surrogate_model = Surrogate(self.surrogate)
        
        # Initialize meta-learner
        meta_learner = MetaLearner(
            algorithm=algorithm,
            surrogate=surrogate_model,
            objective_function=objective_function,
            search_space=search_space,
            maximize=False,  # We're minimizing the negative score
            exploration_factor=0.3,
            history_weight=0.7
        )
        
        return meta_learner
        
    def fit(self, X, y, feature_names=None):
        """
        Fit the feature selector.
        
        Args:
            X: Feature matrix
            y: Target vector
            feature_names: Optional list of feature names
            
        Returns:
            Self
        """
        # Convert input to numpy arrays
        X_array = X.values if isinstance(X, pd.DataFrame) else X
        y_array = y.values if isinstance(y, pd.Series) else y
        
        # Get feature names
        if feature_names is None:
            if isinstance(X, pd.DataFrame):
                feature_names = X.columns.tolist()
            else:
                feature_names = [f"feature_{i}" for i in range(X_array.shape[1])]
                
        # Determine number of features to select
        n_features = self.n_features or max(1, X_array.shape[1] // 2)
        
        if META_OPTIMIZER_AVAILABLE:
            # Use meta-optimization
            self.meta_learner = self._setup_meta_optimization(X_array, y_array, feature_names)
            
            # Run meta-optimization
            n_iterations = min(30, X_array.shape[1] * 5)  # Scale iterations with feature count
            best_solution, _ = self.meta_learner.run(n_iterations=n_iterations)
            
            # Get binary feature mask from best solution
            best_mask = np.round(best_solution).astype(int)
            
            # Get selected features
            selected_indices = np.where(best_mask == 1)[0]
            self.selected_features_ = [feature_names[i] for i in selected_indices]
            
            # Feature importance from meta-learner
            features_history = self.meta_learner.get_feature_statistics()
            self.feature_importance_ = {
                feature_names[i]: features_history.get(i, {}).get('importance', 0)
                for i in range(len(feature_names))
            }
        else:
            # Use standard feature selection as fallback
            from sklearn.feature_selection import SelectKBest, f_classif
            selector = SelectKBest(f_classif, k=n_features)
            selector.fit(X_array, y_array)
            
            # Get selected features
            selected_indices = selector.get_support(indices=True)
            self.selected_features_ = [feature_names[i] for i in selected_indices]
            
            # Feature importance from F-scores
            self.feature_importance_ = {
                feature_names[i]: selector.scores_[i] for i in range(len(feature_names))
            }
            
        if self.verbose:
            logging.info(f"Selected {len(self.selected_features_)} features: {self.selected_features_}")
            
        return self
        
    def transform(self, X):
        """
        Transform the data to include only selected features.
        
        Args:
            X: Feature matrix
            
        Returns:
            Transformed feature matrix with only selected features
        """
        if self.selected_features_ is None:
            raise ValueError("Selector has not been fitted yet")
            
        if isinstance(X, pd.DataFrame):
            return X[self.selected_features_]
        else:
            # Handle numpy arrays
            feature_names = [f"feature_{i}" for i in range(X.shape[1])]
            selected_indices = [i for i, name in enumerate(feature_names) 
                              if name in self.selected_features_]
            return X[:, selected_indices]
            
    def fit_transform(self, X, y, feature_names=None):
        """
        Fit and transform in one step.
        
        Args:
            X: Feature matrix
            y: Target vector
            feature_names: Optional list of feature names
            
        Returns:
            Transformed feature matrix with only selected features
        """
        self.fit(X, y, feature_names)
        return self.transform(X)
        
    def get_feature_importance(self):
        """
        Get feature importance from the selector.
        
        Returns:
            Dictionary mapping feature names to importance scores
        """
        if self.feature_importance_ is None:
            raise ValueError("Selector has not been fitted yet")
            
        return self.feature_importance_
