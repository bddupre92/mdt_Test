"""
Extension to the MigrainePredictor class that handles new data formats and schema evolution.
"""

import pandas as pd
import numpy as np
import os
import sys
import logging
from typing import Dict, Any, List, Tuple, Optional, Union
from pathlib import Path

# Import original predictor and data handler
from .migraine_predictor import MigrainePredictor
from .data_handler import DataHandler

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class EnhancedMigrainePredictor(MigrainePredictor):
    """
    Enhanced migraine predictor with support for new data formats and schema evolution.
    """
    
    def __init__(self, model_dir: str = "models", data_dir: str = "data"):
        """
        Initialize the enhanced predictor.
        
        Args:
            model_dir: Directory to store models
            data_dir: Directory to store data files and schema information
        """
        # Initialize the data handler first
        self.data_handler = DataHandler(data_dir)
        
        # Initialize the parent class with only core features
        self.feature_columns = self.data_handler.schema["core_features"]
        self.target_column = self.data_handler.schema["target"]
        
        # Call parent init after setting features
        super().__init__(model_dir)
        
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
        
        # If new columns were added, update the feature columns
        if add_new_columns:
            self.feature_columns = self.data_handler.get_feature_list(include_optional=True, include_derived=True)
        
        return data
    
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
        
        # Update feature columns after importing data
        self.feature_columns = self.data_handler.get_feature_list(include_optional=True, include_derived=True)
        
        # Validate data for training
        if not self.data_handler.validate_data_for_training(data):
            raise ValueError("Data validation failed for training")
        
        # Train the model using the processed data
        return self.train(data, model_name, description, make_default)
    
    def add_derived_feature(self, name: str, formula: str):
        """
        Add a derived feature to the schema.
        
        Args:
            name: Name of the derived feature
            formula: Formula to calculate the feature
        """
        self.data_handler.add_derived_feature(name, formula)
        
        # Update feature columns to include the new derived feature
        self.feature_columns = self.data_handler.get_feature_list(include_optional=True, include_derived=True)
    
    def add_transformation(self, column: str, transform_type: str):
        """
        Add a transformation for a column.
        
        Args:
            column: Column to transform
            transform_type: Type of transformation
        """
        self.data_handler.add_transformation(column, transform_type)
    
    def _preprocess_data(self, data: pd.DataFrame):
        """
        Preprocess data for training or prediction.
        
        Args:
            data: DataFrame to preprocess
            
        Returns:
            Tuple of (X, y) arrays for features and target
        """
        # Process data according to schema
        processed_data = self.data_handler.process_data(data, add_new_columns=False)
        
        # Get feature columns - only use available columns
        available_features = [col for col in self.feature_columns if col in processed_data.columns]
        
        # Extract features
        X = processed_data[available_features].values
        
        # Extract target if available
        if self.target_column in processed_data.columns:
            y = processed_data[self.target_column].values
        else:
            # For prediction only
            y = None
        
        # Create and fit scaler if not already created
        if self.scaler is None:
            self.scaler = StandardScaler()
            X_scaled = self.scaler.fit_transform(X)
        else:
            X_scaled = self.scaler.transform(X)
        
        return X_scaled, y
    
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
    
    def save_model_with_schema(self, model_name: str = "migraine_model", description: str = "", make_default: bool = True) -> str:
        """
        Save the current model along with the schema.
        
        Args:
            model_name: Name for the model
            description: Model description
            make_default: Whether to make this model the default
            
        Returns:
            Model ID
        """
        # Create enhanced metadata with schema
        enhanced_metadata = self.model_metadata.copy() if hasattr(self, 'model_metadata') else {}
        
        # Add schema information
        enhanced_metadata.update({
            "schema_version": self.data_handler.schema["version"],
            "feature_columns": self.feature_columns,
            "core_features": self.data_handler.schema["core_features"],
            "optional_features": self.data_handler.schema["optional_features"],
            "derived_features": self.data_handler.schema["derived_features"],
            "transformations": self.data_handler.schema["transformations"]
        })
        
        # Save the model with enhanced metadata
        model_id = self.model_manager.save_model(
            model=self.model,
            name=model_name,
            description=description,
            make_default=make_default,
            metadata=enhanced_metadata
        )
        
        self.model_id = model_id
        logger.info(f"Model saved with ID and schema: {model_id}")
        
        return model_id
    
    def predict_with_missing_features(self, data: pd.DataFrame) -> List[Dict[str, Any]]:
        """
        Make predictions even with missing features, using defaults where needed.
        
        Args:
            data: DataFrame containing features (possibly incomplete)
            
        Returns:
            List of dictionaries with prediction results
        """
        # Create a copy of the data
        processed_data = data.copy()
        
        # Check for missing core features
        missing_core = set(self.data_handler.schema["core_features"]) - set(processed_data.columns)
        if missing_core:
            logger.warning(f"Missing core features: {missing_core}")
            
            # For each missing feature, add with default value (mean from training)
            for feature in missing_core:
                if feature in self.model_metadata.get("feature_defaults", {}):
                    default_value = self.model_metadata["feature_defaults"][feature]
                    logger.info(f"Using default value {default_value} for missing feature {feature}")
                    processed_data[feature] = default_value
                else:
                    logger.error(f"No default found for missing core feature: {feature}")
                    raise ValueError(f"Missing core feature with no default: {feature}")
        
        # Now proceed with normal prediction
        return self.predict_with_details(processed_data)
    
    def update_model_with_new_data(self, data_path: str, model_name: str = None, 
                                  description: str = None, make_default: bool = True,
                                  add_new_columns: bool = False) -> str:
        """
        Update an existing model with new data.
        
        Args:
            data_path: Path to the new data file
            model_name: Name for the updated model (defaults to current model name + '_updated')
            description: Model description
            make_default: Whether to make this model the default
            add_new_columns: Whether to add new columns to the schema
            
        Returns:
            Model ID of the updated model
        """
        # Import and process the new data
        new_data = self.import_data(data_path, add_new_columns)
        
        # If model_name not provided, use current model name + '_updated'
        if model_name is None:
            if self.model_id:
                model_name = f"{self.model_id}_updated"
            else:
                model_name = "updated_model"
        
        # If description not provided, create a default one
        if description is None:
            description = f"Model updated with new data from {data_path}"
        
        # Train a new model with the combined data
        return self.train(new_data, model_name, description, make_default)

    def train(self, data: pd.DataFrame, model_name: str = "migraine_model", 
          description: str = "", make_default: bool = True) -> str:
        """
        Train a new model with support for derived features.
        
        Args:
            data: Training data
            model_name: Name for the model
            description: Model description
            make_default: Whether to make this model the default
            
        Returns:
            Model ID
        """
        # Process data according to schema, including derived features
        processed_data = self.data_handler.process_data(data, add_new_columns=False)
        
        # Now call parent's train method with processed data
        return super().train(processed_data, model_name, description, make_default)
