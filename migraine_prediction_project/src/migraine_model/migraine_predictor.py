"""
Easy-to-use interface for migraine prediction using MetaOptimizer.
"""

import pandas as pd
import numpy as np
import os
import sys
import pickle
import time
import json
from typing import Dict, Any, List, Tuple, Optional, Union
from pathlib import Path
import logging
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.ensemble import RandomForestClassifier

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Try to import MetaOptimizer from the meta_optimizer package
try:
    from meta_optimizer.meta.meta_optimizer import MetaOptimizer
    from meta_optimizer.optimizers.optimizer_factory import (
        DifferentialEvolutionOptimizer, 
        EvolutionStrategyOptimizer,
        AntColonyOptimizer,
        GreyWolfOptimizer
    )
    OPTIMIZER_AVAILABLE = True
except ImportError:
    logger.warning("MetaOptimizer not available from original project. Using scikit-learn implementation.")
    OPTIMIZER_AVAILABLE = False

# Try to import explainability components from the meta_optimizer package
try:
    from meta_optimizer.explainability.explainer_factory import ExplainerFactory
    from meta_optimizer.explainability.base_explainer import BaseExplainer
    EXPLAINABILITY_AVAILABLE = True
except ImportError:
    logger.warning("Explainability components not available from original project.")
    EXPLAINABILITY_AVAILABLE = False

# Import ModelManager from the package
from .model import ModelManager


class MigrainePredictor:
    """
    Interface for training and using the migraine prediction model.
    """
    
    def __init__(self, model_dir: str = "models"):
        """
        Initialize the predictor.
        
        Args:
            model_dir: Directory to store models
        """
        self.model_dir = model_dir
        self.model_manager = ModelManager(model_dir)
        
        # Initialize model and related objects to None
        self.model = None
        self.scaler = None
        self.feature_columns = [
            'sleep_hours',
            'stress_level',
            'weather_pressure',
            'heart_rate',
            'hormonal_level'
        ]
        self.target_column = 'migraine_occurred'
        self.model_id = None
        self.model_metadata = {}
        
        # Create model directory if it doesn't exist
        os.makedirs(self.model_dir, exist_ok=True)
    
    def train(self, data: pd.DataFrame, model_name: str = "migraine_model", 
              description: str = "", make_default: bool = True) -> str:
        """
        Train a new model.
        
        Args:
            data: Training data
            model_name: Name for the model
            description: Model description
            make_default: Whether to make this model the default
            
        Returns:
            Model ID
        """
        global OPTIMIZER_AVAILABLE
        
        # Check if required columns are in the data
        missing_cols = [col for col in self.feature_columns + [self.target_column] 
                      if col not in data.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        # Prepare data for training
        X = data[self.feature_columns].values
        y = data[self.target_column].values
        
        # Normalize features
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)
        
        if OPTIMIZER_AVAILABLE:
            try:
                # Try to use the original MetaOptimizer implementation
                logger.info("Using original MetaOptimizer implementation...")
                
                # Define objective function for optimization
                def objective(params):
                    predictions = X_scaled.dot(params)
                    predictions = 1 / (1 + np.exp(-predictions))  # Sigmoid
                    binary_preds = (predictions > 0.5).astype(int)
                    
                    # Return negative accuracy as we want to maximize accuracy
                    return -accuracy_score(y, binary_preds)
                
                # Train model using MetaOptimizer
                bounds = [(-2, 2) for _ in range(len(self.feature_columns))]
                dim = len(bounds)
                
                # Use MetaOptimizer if available
                self.model = MetaOptimizer(
                    # Pass required parameters based on your specific implementation
                )
                
                # Run optimization
                logger.info("Training model using MetaOptimizer...")
                start_time = time.time()
                solution, value = self.model.optimize(
                    objective=objective,
                    max_iterations=100,
                    tolerance=1e-5
                )
                end_time = time.time()
                
                # Store parameters
                self.model_params = solution
                training_time = end_time - start_time
                
                # Calculate metrics
                train_predictions = self._predict_with_params(X_scaled, solution)
                train_binary = (train_predictions > 0.5).astype(int)
                train_accuracy = accuracy_score(y, train_binary)
                
                logger.info(f"Model trained with accuracy: {train_accuracy:.4f}")
                
                # Store model metadata
                self.model_metadata = {
                    'accuracy': train_accuracy,
                    'training_time': training_time,
                    'feature_columns': self.feature_columns,
                    'params': solution.tolist(),
                    'optimizer': 'MetaOptimizer'
                }
            
            except Exception as e:
                logger.warning(f"Error using MetaOptimizer: {e}. Falling back to scikit-learn.")
                OPTIMIZER_AVAILABLE = False
                # Fall through to scikit-learn implementation
        
        if not OPTIMIZER_AVAILABLE:
            # Use scikit-learn implementation
            logger.info("Using scikit-learn implementation...")
            
            # Create and train a RandomForest model (better explainability than LogisticRegression)
            self.model = RandomForestClassifier(n_estimators=100, random_state=42)
            
            # Training
            start_time = time.time()
            self.model.fit(X_scaled, y)
            end_time = time.time()
            training_time = end_time - start_time
            
            # Calculate metrics
            train_predictions = self.model.predict_proba(X_scaled)[:, 1]
            train_binary = (train_predictions > 0.5).astype(int)
            train_accuracy = accuracy_score(y, train_binary)
            
            logger.info(f"Model trained with accuracy: {train_accuracy:.4f}")
            
            # Store model metadata
            self.model_metadata = {
                'accuracy': train_accuracy,
                'training_time': training_time,
                'feature_columns': self.feature_columns,
                'feature_importances': self.model.feature_importances_.tolist(),
                'model_type': 'RandomForest'
            }
        
        # Save the model
        model_id = self.model_manager.save_model(
            model=self.model,
            name=model_name,
            description=description,
            make_default=make_default,
            metadata=self.model_metadata
        )
        
        self.model_id = model_id
        logger.info(f"Model saved with ID: {model_id}")
        
        return model_id
    
    def _predict_with_params(self, X: np.ndarray, params: np.ndarray) -> np.ndarray:
        """
        Make predictions using the given parameters.
        
        Args:
            X: Feature matrix
            params: Model parameters
            
        Returns:
            Array of prediction probabilities
        """
        if OPTIMIZER_AVAILABLE and hasattr(self, 'model_params'):
            # Linear combination of features and parameters
            raw_predictions = X.dot(params)
            
            # Apply sigmoid to get probabilities
            probabilities = 1 / (1 + np.exp(-raw_predictions))
            
            return probabilities
        else:
            # Use scikit-learn model
            return self.model.predict_proba(X)[:, 1]
    
    def predict(self, data: pd.DataFrame) -> List[int]:
        """
        Make predictions for all samples in the data.
        
        Args:
            data: DataFrame containing features
            
        Returns:
            List of binary predictions (0 or 1)
        """
        if self.model_id is None and not hasattr(self, 'model'):
            raise ValueError("No model loaded. Please train or load a model first.")
        
        # Extract features
        if set(self.feature_columns).issubset(data.columns):
            X = data[self.feature_columns].values
        else:
            raise ValueError(f"Missing required feature columns. Expected: {self.feature_columns}")
        
        # Scale features
        if self.scaler:
            X_scaled = self.scaler.transform(X)
        else:
            X_scaled = X
        
        # Make prediction based on whether we're using MetaOptimizer or scikit-learn
        if OPTIMIZER_AVAILABLE and hasattr(self, 'model_params'):
            # Get model parameters
            params = self.model_params
            
            # Make prediction
            probabilities = self._predict_with_params(X_scaled, params)
            predictions = [1 if p > 0.5 else 0 for p in probabilities]
        else:
            # Use scikit-learn model
            probabilities = self.model.predict_proba(X_scaled)[:, 1]
            predictions = [1 if p > 0.5 else 0 for p in probabilities]
        
        return predictions
    
    def predict_with_details(self, data: pd.DataFrame) -> List[Dict[str, Any]]:
        """
        Make predictions with detailed information for all samples.
        
        Args:
            data: DataFrame containing features
            
        Returns:
            List of dictionaries with prediction results
        """
        if self.model_id is None and not hasattr(self, 'model'):
            raise ValueError("No model loaded. Please train or load a model first.")
        
        # Extract features
        if set(self.feature_columns).issubset(data.columns):
            X = data[self.feature_columns].values
        else:
            raise ValueError(f"Missing required feature columns. Expected: {self.feature_columns}")
        
        # Scale features
        if self.scaler:
            X_scaled = self.scaler.transform(X)
        else:
            X_scaled = X
        
        # Make prediction based on whether we're using MetaOptimizer or scikit-learn
        if OPTIMIZER_AVAILABLE and hasattr(self, 'model_params'):
            # Get model parameters
            params = self.model_params
            
            # Make prediction
            probabilities = self._predict_with_params(X_scaled, params)
            predictions = [1 if p > 0.5 else 0 for p in probabilities]
        else:
            # Use scikit-learn model
            probabilities = self.model.predict_proba(X_scaled)[:, 1]
            predictions = [1 if p > 0.5 else 0 for p in probabilities]
        
        # Create detailed result for each sample
        results = []
        for i, (prob, pred) in enumerate(zip(probabilities, predictions)):
            results.append({
                'probability': float(prob),
                'prediction': int(pred),
                'features_used': self.feature_columns
            })
        
        return results
    
    def evaluate(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Evaluate the model on test data.
        
        Args:
            data: Test data with features and target
            
        Returns:
            Dictionary of evaluation metrics
        """
        if self.model is None:
            raise ValueError("No model loaded. Please train or load a model first.")
        
        # Check if data has the required columns
        missing_cols = [col for col in self.feature_columns + [self.target_column] 
                       if col not in data.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        # Make predictions
        y_true = data[self.target_column].values
        predictions = self.predict(data)
        
        # Calculate metrics
        metrics = {
            'accuracy': accuracy_score(y_true, predictions),
            'precision': precision_score(y_true, predictions, zero_division=0),
            'recall': recall_score(y_true, predictions, zero_division=0),
            'f1': f1_score(y_true, predictions, zero_division=0)
        }
        
        # Add feature importance if available
        if hasattr(self, 'model') and hasattr(self.model, 'feature_importances_'):
            feature_importance = {}
            for i, col in enumerate(self.feature_columns):
                feature_importance[col] = float(self.model.feature_importances_[i])
            metrics['feature_importance'] = feature_importance
        
        return metrics
    
    def load_model(self, model_id: Optional[str] = None) -> None:
        """
        Load a previously trained model.
        
        Args:
            model_id: ID of the model to load, or None for default
        """
        self.model, self.model_metadata = self.model_manager.load_model(model_id)
        
        # Set feature names and params from metadata
        self.feature_columns = self.model_metadata.get('feature_columns', self.feature_columns)
        self.target_column = self.model_metadata.get('target_column', self.target_column)
        self.model_id = self.model_metadata.get('id', 'Unknown')
        
        logger.info(f"Model loaded: {self.model_id}")
    
    @property
    def feature_names(self) -> List[str]:
        """
        Get the feature names for the current model.
        
        Returns:
            List of feature names
        """
        return self.feature_columns
    
    def get_model_metadata(self) -> Dict[str, Any]:
        """
        Get the metadata of the currently loaded model.
        
        Returns:
            Dictionary containing model metadata
        """
        return self.model_metadata if hasattr(self, 'model_metadata') else {}
    
    def optimize(self, data: pd.DataFrame, model_name: str = "optimized_model", 
                description: str = "Model trained with meta-optimization", 
                optimizer: str = "meta", max_evals: int = 500) -> str:
        """
        Train a model using meta-optimization techniques from the MDT framework.
        
        Args:
            data: DataFrame containing training data
            model_name: Name for the trained model
            description: Description for the model
            optimizer: Type of optimizer to use ('meta', 'de', 'es', 'gwo', 'aco')
            max_evals: Maximum number of function evaluations
            
        Returns:
            ID of the trained model
        """
        # Prepare data
        self._validate_data(data)
        X, y = self._preprocess_data(data)
        
        try:
            # Try to import optimizers from meta_optimizer package
            from meta_optimizer.meta.meta_optimizer import MetaOptimizer
            from meta_optimizer.optimizers.optimizer_factory import (
                DifferentialEvolutionOptimizer, 
                EvolutionStrategyOptimizer,
                AntColonyOptimizer,
                GreyWolfOptimizer
            )
            
            # Define dimensions and bounds
            n_features = X.shape[1]
            bounds = [(0, 1) for _ in range(n_features)]  # Feature weights bounds
            
            # Create optimizers based on selection
            optimizers = {}
            if optimizer in ["meta", "de"]:
                optimizers["DE"] = DifferentialEvolutionOptimizer(dim=n_features, bounds=bounds, name="DE")
            if optimizer in ["meta", "es"]:
                optimizers["ES"] = EvolutionStrategyOptimizer(dim=n_features, bounds=bounds, name="ES")
            if optimizer in ["meta", "gwo"]:
                optimizers["GWO"] = GreyWolfOptimizer(dim=n_features, bounds=bounds, name="GWO")
            if optimizer in ["meta", "aco"]:
                optimizers["ACO"] = AntColonyOptimizer(dim=n_features, bounds=bounds, name="ACO")
            
            # Define objective function
            def objective_function(weights):
                # Use weights for feature selection/importance
                weighted_X = X * np.array(weights)
                
                # Train a model on weighted features
                from sklearn.ensemble import RandomForestClassifier
                from sklearn.model_selection import cross_val_score
                
                model = RandomForestClassifier(n_estimators=100, random_state=42)
                scores = cross_val_score(model, weighted_X, y, cv=5, scoring='accuracy')
                
                # Return negative mean score (optimizers minimize)
                return -np.mean(scores)
            
            # Use MetaOptimizer or single optimizer based on selection
            if optimizer == "meta" and len(optimizers) > 1:
                meta_opt = MetaOptimizer(dim=n_features, bounds=bounds, optimizers=optimizers)
                result = meta_opt.run(objective_function, max_evals=max_evals)
                best_weights = result["best_parameters"]
                best_score = -result["best_score"]  # Convert back to positive score
                best_optimizer = result.get("best_optimizer", "Unknown")
                logger.info(f"Meta-optimization complete. Best optimizer: {best_optimizer}")
            else:
                # Use single optimizer
                opt_key = list(optimizers.keys())[0]
                opt = optimizers[opt_key]
                result = opt.run(objective_function, max_evals=max_evals)
                best_weights = result["best_parameters"]
                best_score = -result["best_score"]  # Convert back to positive score
                logger.info(f"Optimization with {opt_key} complete.")
            
            # Train final model with optimized weights
            weighted_X = X * np.array(best_weights)
            from sklearn.ensemble import RandomForestClassifier
            final_model = RandomForestClassifier(n_estimators=100, random_state=42)
            final_model.fit(weighted_X, y)
            
            # Calculate feature importance using the weights
            feature_importance = {}
            for i, feature in enumerate(self.feature_columns):
                feature_importance[feature] = best_weights[i]
            
            # Save model
            metadata = {
                'feature_weights': best_weights,
                'feature_importance': feature_importance,
                'optimization_score': best_score,
                'optimizer_used': optimizer,
                'max_evals': max_evals,
                'feature_columns': self.feature_columns,
                'target_column': self.target_column,
                'description': description
            }
            
            model_id = self.model_manager.save_model(
                final_model, model_name, description, make_default=True, metadata=metadata
            )
            
            # Update instance variables
            self.model = final_model
            self.model_id = model_id
            self.model_metadata = metadata
            
            logger.info(f"Model trained with optimal weights. Accuracy: {best_score:.4f}")
            return model_id
            
        except ImportError as e:
            logger.warning(f"Error importing optimization components: {e}. Falling back to standard training method.")
            return self.train(data, model_name, description)
    
    def save_as_pickle(self, file_path: str) -> None:
        """
        Save the model as a pickle file.
        
        Args:
            file_path: Path to save the model
        """
        if self.model is None:
            raise ValueError("No model loaded. Please train or load a model first.")
        
        # Create a dictionary to save
        model_package = {
            'model': self.model,
            'scaler': self.scaler,
            'feature_columns': self.feature_columns,
            'target_column': self.target_column,
            'model_id': self.model_id,
            'model_metadata': self.model_metadata
        }
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(os.path.abspath(file_path)), exist_ok=True)
        
        # Save to file
        with open(file_path, 'wb') as f:
            pickle.dump(model_package, f)
        
        logger.info(f"Model saved to {file_path}")
    
    @classmethod
    def load_from_pickle(cls, file_path: str) -> 'MigrainePredictor':
        """
        Load a model from a pickle file.
        
        Args:
            file_path: Path to the pickle file
            
        Returns:
            MigrainePredictor instance
        """
        # Load from file
        with open(file_path, 'rb') as f:
            model_package = pickle.load(f)
        
        # Create a new instance
        predictor = cls()
        
        # Update instance variables
        predictor.model = model_package['model']
        predictor.scaler = model_package['scaler']
        predictor.feature_columns = model_package['feature_columns']
        predictor.target_column = model_package['target_column']
        predictor.model_id = model_package['model_id']
        predictor.model_metadata = model_package['model_metadata']
        
        logger.info(f"Model loaded from {file_path}")
        return predictor
    
    def list_models(self) -> List[Dict[str, Any]]:
        """
        List all available models.
        
        Returns:
            List of model dictionaries
        """
        return self.model_manager.list_models()
    
    def get_feature_importance(self) -> Dict[str, float]:
        """
        Get feature importance for the current model.
        
        Returns:
            Dictionary mapping feature names to importance scores
        """
        if OPTIMIZER_AVAILABLE and hasattr(self, 'model_params'):
            # For linear models, feature importance is related to the magnitude of coefficients
            importances = np.abs(self.model_params)
            
            # Normalize to sum to 1
            importances = importances / np.sum(importances)
            
            # Create dictionary mapping feature names to importance scores
            return {feature: float(importance) for feature, importance in zip(self.feature_columns, importances)}
        elif hasattr(self, 'model') and hasattr(self.model, 'feature_importances_'):
            # For RandomForest, use built-in feature importances
            return {feature: float(importance) for feature, importance in 
                   zip(self.feature_columns, self.model.feature_importances_)}
        else:
            raise ValueError("No model loaded or model doesn't support feature importance")
    
    def get_model_metrics(self) -> Dict[str, float]:
        """
        Get model performance metrics.
        
        Returns:
            Dictionary of metrics
        """
        if self.model_metadata is None or 'metrics' not in self.model_metadata:
            return {}
        
        return self.model_metadata['metrics']
    
    def get_model_version(self) -> str:
        """
        Get model version.
        
        Returns:
            Model version string
        """
        if hasattr(self, 'model_metadata') and 'version' in self.model_metadata:
            return self.model_metadata['version']
        return "Unknown"
    
    def _validate_data(self, data: pd.DataFrame):
        """
        Validate that the data contains the required columns.
        
        Args:
            data: DataFrame to validate
            
        Raises:
            ValueError: If required columns are missing
        """
        # Check if data contains all feature columns
        if not set(self.feature_columns).issubset(data.columns):
            missing_cols = set(self.feature_columns) - set(data.columns)
            raise ValueError(f"Missing required feature columns: {missing_cols}")
        
        # Check if data contains target column for training
        if self.target_column not in data.columns:
            raise ValueError(f"Missing target column: {self.target_column}")
    
    def _preprocess_data(self, data: pd.DataFrame):
        """
        Preprocess data for training or prediction.
        
        Args:
            data: DataFrame to preprocess
            
        Returns:
            Tuple of (X, y) arrays for features and target
        """
        # Extract features and target
        X = data[self.feature_columns].values
        y = data[self.target_column].values
        
        # Create and fit scaler if not already created
        if self.scaler is None:
            self.scaler = StandardScaler()
            X_scaled = self.scaler.fit_transform(X)
        else:
            X_scaled = self.scaler.transform(X)
        
        return X_scaled, y
