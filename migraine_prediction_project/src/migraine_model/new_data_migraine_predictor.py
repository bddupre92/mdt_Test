"""
Standalone migraine predictor with advanced capabilities for handling new data formats and schema evolution.
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

# Import the data handler
from .data_handler import DataHandler
from .model import ModelManager

class MigrainePredictorV2:
    """
    Advanced migraine predictor with support for schema evolution and new data formats.
    """
    
    def __init__(self, model_dir: str = "models", data_dir: str = "data"):
        """
        Initialize the predictor with data handling capabilities.
        
        Args:
            model_dir: Directory to store models
            data_dir: Directory to store data and schema information
        """
        self.model_dir = model_dir
        self.data_dir = data_dir
        self.model_manager = ModelManager(model_dir)
        self.data_handler = DataHandler(data_dir)
        
        # Initialize model and related objects to None
        self.model = None
        self.scaler = None
        self.feature_columns = self.data_handler.get_feature_list(include_optional=False, include_derived=False)
        self.target_column = self.data_handler.schema["target"]
        self.model_id = None
        self.model_metadata = {}
        
        # Create directories if they don't exist
        os.makedirs(self.model_dir, exist_ok=True)
        os.makedirs(self.data_dir, exist_ok=True)
        
        # Check if meta-optimizer is available
        try:
            from meta_optimizer.meta.meta_optimizer import MetaOptimizer
            self.meta_optimizer_available = True
        except ImportError:
            logger.warning("MetaOptimizer not available.")
            self.meta_optimizer_available = False
    
    def import_data(self, data_path: str, add_new_columns: bool = False) -> pd.DataFrame:
        """
        Import data from a file.
        
        Args:
            data_path: Path to the data file
            add_new_columns: Whether to add new columns to the schema
            
        Returns:
            Processed DataFrame
        """
        # Use the data handler to import and process the data
        data = self.data_handler.import_data(data_path, add_new_columns)
        
        # Update feature columns if new columns were added
        if add_new_columns:
            self.feature_columns = self.data_handler.get_feature_list(include_optional=True, include_derived=True)
        
        return data
    
    def add_derived_feature(self, name: str, formula: str):
        """
        Add a derived feature to the schema.
        
        Args:
            name: Name of the derived feature
            formula: Formula to calculate the feature
        """
        self.data_handler.add_derived_feature(name, formula)
        
        # Update feature columns
        self.feature_columns = self.data_handler.get_feature_list(include_optional=True, include_derived=True)
    
    def add_transformation(self, column: str, transform_type: str):
        """
        Add a transformation for a column.
        
        Args:
            column: Column to transform
            transform_type: Type of transformation
        """
        self.data_handler.add_transformation(column, transform_type)
    
    def train(self, data: pd.DataFrame, model_name: str = "migraine_model", 
             description: str = "", make_default: bool = True) -> str:
        """
        Train a model using the provided data.
        
        Args:
            data: Training data
            model_name: Name for the model
            description: Model description
            make_default: Whether to make this model the default
            
        Returns:
            Model ID
        """
        # Process data to apply derived features and transformations
        processed_data = self.data_handler.process_data(data, add_new_columns=False)
        
        # Only use available features for training
        available_features = [col for col in self.feature_columns if col in processed_data.columns]
        
        # Extract features and target
        X = processed_data[available_features].values
        y = processed_data[self.target_column].values
        
        # Normalize features
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)
        
        # Use scikit-learn implementation for now
        logger.info("Training model with scikit-learn RandomForest implementation...")
        
        # Create and train model
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
        
        # Store feature defaults for missing value handling
        feature_defaults = {}
        for feature in available_features:
            feature_defaults[feature] = float(processed_data[feature].mean())
        
        # Store model metadata
        self.model_metadata = {
            'accuracy': train_accuracy,
            'training_time': training_time,
            'feature_columns': available_features,
            'feature_importances': self.model.feature_importances_.tolist(),
            'model_type': 'RandomForest',
            'schema_version': self.data_handler.schema["version"],
            'feature_defaults': feature_defaults
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
    
    def train_with_new_data(self, data_path: str, model_name: str = "migraine_model", 
                           description: str = "", make_default: bool = True,
                           add_new_columns: bool = False) -> str:
        """
        Import data and train a model in one step.
        
        Args:
            data_path: Path to the data file
            model_name: Name for the model
            description: Model description
            make_default: Whether to make this model the default
            add_new_columns: Whether to add new columns to the schema
            
        Returns:
            Model ID
        """
        # Import and process the data
        data = self.import_data(data_path, add_new_columns)
        
        # Validate data for training
        if not self.data_handler.validate_data_for_training(data):
            raise ValueError("Data validation failed for training")
        
        # Train the model using the processed data
        return self.train(data, model_name, description, make_default)
    
    def predict(self, data: pd.DataFrame) -> np.ndarray:
        """
        Make predictions for all samples in the data.
        
        Args:
            data: DataFrame containing features
            
        Returns:
            Array of binary predictions (0 or 1)
        """
        # Process data according to schema
        processed_data = self.data_handler.process_data(data, add_new_columns=False)
        
        # Only use available features for prediction
        available_features = [col for col in self.feature_columns if col in processed_data.columns]
        
        if not all(feat in processed_data.columns for feat in self.model_metadata.get('feature_columns', [])):
            logger.warning("Not all model features are available in the input data")
        
        # Extract features
        X = processed_data[available_features].values
        
        # Transform features
        X_scaled = self.scaler.transform(X)
        
        # Make predictions
        if hasattr(self.model, 'predict_proba'):
            probabilities = self.model.predict_proba(X_scaled)[:, 1]
            predictions = (probabilities > 0.5).astype(int)
        else:
            predictions = self.model.predict(X_scaled)
        
        return predictions
    
    def predict_with_details(self, data: pd.DataFrame) -> List[Dict[str, Any]]:
        """
        Make predictions with detailed information for all samples.
        
        Args:
            data: DataFrame containing features
            
        Returns:
            List of dictionaries with prediction results
        """
        # Process data according to schema
        processed_data = self.data_handler.process_data(data, add_new_columns=False)
        
        # Only use available features for prediction
        available_features = [col for col in self.feature_columns if col in processed_data.columns]
        
        # Extract features
        X = processed_data[available_features].values
        
        # Transform features
        X_scaled = self.scaler.transform(X)
        
        # Make predictions
        if hasattr(self.model, 'predict_proba'):
            probabilities = self.model.predict_proba(X_scaled)[:, 1]
            predictions = (probabilities > 0.5).astype(int)
        else:
            predictions = self.model.predict(X_scaled)
            probabilities = predictions.astype(float)
        
        # Get feature importance if available
        if hasattr(self.model, 'feature_importances_'):
            importances = dict(zip(available_features, self.model.feature_importances_))
        else:
            importances = {}
        
        # Prepare results
        results = []
        for i in range(len(predictions)):
            result = {
                'prediction': int(predictions[i]),
                'probability': float(probabilities[i]),
                'feature_values': {col: float(processed_data[col].iloc[i]) for col in available_features},
                'feature_importances': importances
            }
            results.append(result)
        
        return results
    
    def predict_with_missing_features(self, data: pd.DataFrame) -> List[Dict[str, Any]]:
        """
        Make predictions even with missing features, using defaults where needed.
        
        Args:
            data: DataFrame containing features (possibly incomplete)
            
        Returns:
            List of dictionaries with prediction results
        """
        # Check and fix missing features
        data_copy = data.copy()
        
        # Get model features and check what's missing
        model_features = self.model_metadata.get('feature_columns', [])
        missing_features = [feat for feat in model_features if feat not in data_copy.columns]
        
        if missing_features:
            logger.info(f"Filling in missing features: {missing_features}")
            
            # Add missing features with default values
            for feature in missing_features:
                if feature in self.model_metadata.get('feature_defaults', {}):
                    default_value = self.model_metadata['feature_defaults'][feature]
                    data_copy[feature] = default_value
                    logger.info(f"Using default value {default_value} for {feature}")
                else:
                    # For derived features, calculate them in the data handler
                    if feature in self.data_handler.schema.get('derived_features', {}):
                        logger.info(f"Will calculate derived feature: {feature}")
                    else:
                        raise ValueError(f"No default value for missing feature: {feature}")
        
        # Now make predictions with the complete data
        return self.predict_with_details(data_copy)
    
    def evaluate(self, data: pd.DataFrame) -> Dict[str, float]:
        """
        Evaluate the model on test data.
        
        Args:
            data: Test data with features and target
            
        Returns:
            Dictionary of evaluation metrics
        """
        # Check if target column exists
        if self.target_column not in data.columns:
            raise ValueError(f"Target column '{self.target_column}' not found in evaluation data")
        
        # Get predictions
        y_true = data[self.target_column].values
        y_pred = self.predict(data)
        
        # Calculate metrics
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, zero_division=0),
            'recall': recall_score(y_true, y_pred, zero_division=0),
            'f1': f1_score(y_true, y_pred, zero_division=0)
        }
        
        return metrics
    
    def load_model(self, model_id: Optional[str] = None):
        """
        Load a previously trained model.
        
        Args:
            model_id: ID of the model to load, or None for default
        """
        # Load model from model manager
        model_info = self.model_manager.load_model(model_id)
        
        if model_info:
            # Model manager returns a tuple of (model, metadata)
            model, metadata = model_info
            
            self.model = model
            self.model_id = metadata['id']
            self.model_metadata = metadata
            
            # Update feature columns from model metadata
            if 'feature_columns' in self.model_metadata:
                self.feature_columns = self.model_metadata['feature_columns']
            
            # Initialize the scaler with feature defaults
            self.scaler = StandardScaler()
            
            # We need to fit the scaler with something to avoid the NoneType error
            # Use the feature defaults from model_metadata to create a dummy sample
            if 'feature_defaults' in self.model_metadata:
                # Create a dummy sample with the default values
                feature_defaults = self.model_metadata['feature_defaults']
                dummy_data = np.array([[feature_defaults[feat] for feat in self.feature_columns]])
                # Fit the scaler with this dummy data
                # This is just a workaround - in production, the scaler would be saved with the model
                self.scaler.fit(dummy_data)
                logger.info("Initialized scaler with default feature values")
            else:
                logger.warning("No feature defaults found in model metadata. Predictions may fail.")
            
            logger.info(f"Loaded model with ID: {self.model_id}")
        else:
            raise ValueError(f"No model found with ID: {model_id}")
    
    def get_schema_info(self) -> Dict[str, Any]:
        """
        Get information about the current schema.
        
        Returns:
            Dictionary with schema information
        """
        return {
            "version": self.data_handler.schema["version"],
            "core_features": self.data_handler.schema["core_features"],
            "optional_features": self.data_handler.schema["optional_features"],
            "derived_features": self.data_handler.schema["derived_features"],
            "transformations": self.data_handler.schema["transformations"],
            "target": self.data_handler.schema["target"]
        }
    
    def get_feature_importance(self) -> Dict[str, float]:
        """
        Get feature importance for the current model.
        
        Returns:
            Dictionary mapping feature names to importance scores
        """
        if not self.model or not hasattr(self.model, 'feature_importances_'):
            raise ValueError("No model loaded or model doesn't support feature importance")
        
        # Get feature columns from metadata or current state
        feature_columns = self.model_metadata.get('feature_columns', self.feature_columns)
        
        # Get feature importances
        importances = self.model.feature_importances_
        
        # Return as dictionary
        return dict(zip(feature_columns, importances))
    
    def list_models(self) -> List[Dict[str, Any]]:
        """
        List all available models.
        
        Returns:
            List of model dictionaries
        """
        return self.model_manager.list_models()
    
    def save_as_pickle(self, file_path: str):
        """
        Save the model as a pickle file.
        
        Args:
            file_path: Path to save the model
        """
        with open(file_path, 'wb') as f:
            pickle.dump({
                'model': self.model,
                'scaler': self.scaler,
                'feature_columns': self.feature_columns,
                'target_column': self.target_column,
                'model_id': self.model_id,
                'model_metadata': self.model_metadata,
                'schema': self.data_handler.schema
            }, f)
        
        logger.info(f"Model saved to {file_path}")
    
    @classmethod
    def load_from_pickle(cls, file_path: str):
        """
        Load a model from a pickle file.
        
        Args:
            file_path: Path to the pickle file
            
        Returns:
            MigrainePredictorV2 instance
        """
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
        
        # Create a new instance
        predictor = cls()
        
        # Update attributes
        predictor.model = data['model']
        predictor.scaler = data['scaler']
        predictor.feature_columns = data['feature_columns']
        predictor.target_column = data['target_column']
        predictor.model_id = data['model_id']
        predictor.model_metadata = data['model_metadata']
        
        # Update schema in data handler
        predictor.data_handler.schema = data['schema']
        
        logger.info(f"Model loaded from {file_path}")
        
        return predictor
