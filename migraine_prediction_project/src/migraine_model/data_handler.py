"""
Data handling utilities for migraine prediction models.
Handles importing new data and adapting to schema changes.
"""

import pandas as pd
import numpy as np
import os
import logging
from typing import List, Dict, Any, Optional, Tuple, Union
from pathlib import Path
import json

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class DataHandler:
    """
    Handles data import, validation, and schema evolution for migraine prediction models.
    """
    
    def __init__(self, data_dir: str = "data"):
        """
        Initialize DataHandler.
        
        Args:
            data_dir: Directory to store data files and schema information
        """
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.schema_file = self.data_dir / "schema.json"
        
        # Initialize/load schema
        self._initialize_schema()
    
    def _initialize_schema(self):
        """Initialize or load the schema from disk."""
        if self.schema_file.exists():
            with open(self.schema_file, 'r') as f:
                self.schema = json.load(f)
        else:
            # Default schema with core features
            self.schema = {
                "version": 1,
                "core_features": [
                    "sleep_hours",
                    "stress_level",
                    "weather_pressure",
                    "heart_rate",
                    "hormonal_level"
                ],
                "target": "migraine_occurred",
                "optional_features": [],
                "derived_features": {},
                "transformations": {},
                "history": []
            }
            self._save_schema()
    
    def _save_schema(self):
        """Save the current schema to disk."""
        with open(self.schema_file, 'w') as f:
            json.dump(self.schema, f, indent=2)
    
    def import_data(self, data_path: str, add_new_columns: bool = False) -> pd.DataFrame:
        """
        Import data from CSV, Excel, or other supported formats.
        
        Args:
            data_path: Path to the data file
            add_new_columns: Whether to add new columns to the schema
            
        Returns:
            Processed DataFrame
        """
        # Determine file type and load data
        path = Path(data_path)
        if not path.exists():
            raise FileNotFoundError(f"Data file not found: {data_path}")
        
        if path.suffix.lower() == '.csv':
            data = pd.read_csv(data_path)
        elif path.suffix.lower() in ['.xlsx', '.xls']:
            data = pd.read_excel(data_path)
        elif path.suffix.lower() == '.json':
            data = pd.read_json(data_path)
        elif path.suffix.lower() == '.parquet':
            data = pd.read_parquet(data_path)
        else:
            raise ValueError(f"Unsupported file format: {path.suffix}")
        
        # Process the data according to the schema
        return self.process_data(data, add_new_columns)
    
    def process_data(self, data: pd.DataFrame, add_new_columns: bool = False) -> pd.DataFrame:
        """
        Process data according to the schema, handling new columns.
        
        Args:
            data: DataFrame to process
            add_new_columns: Whether to add new columns to the schema
            
        Returns:
            Processed DataFrame
        """
        # Create a copy of the data
        processed_data = data.copy()
        
        # Calculate derived features
        for feature, formula in self.schema.get("derived_features", {}).items():
            try:
                # Replace df with processed_data in the formula
                updated_formula = formula.replace("df[", "processed_data[")
                processed_data[feature] = eval(updated_formula)
                logger.debug(f"Calculated derived feature: {feature}")
            except Exception as e:
                logger.warning(f"Failed to calculate derived feature '{feature}': {str(e)}")
        
        # Check for required columns - now check after derived features are calculated
        missing_core = set(self.schema["core_features"]) - set(processed_data.columns)
        if missing_core and not hasattr(self, 'flexible_features') and not getattr(self, 'flexible_features', False):
            # Strict mode - require all core features
            raise ValueError(f"Missing required core features: {missing_core}")
        elif missing_core:
            # Flexible mode - warn about missing features but continue
            logger.warning(f"Missing some core features: {missing_core}. Continuing with available features.")
        
        if self.schema["target"] not in processed_data.columns:
            logger.warning(f"Target column '{self.schema['target']}' not found. This data may be for prediction only.")
        
        # Identify new columns
        current_features = set(self.schema["core_features"] + self.schema["optional_features"])
        new_columns = set(processed_data.columns) - current_features - {self.schema["target"]} - set(self.schema.get("derived_features", {}).keys())
        
        if new_columns and add_new_columns:
            # Add new columns to schema
            self.schema["optional_features"].extend(list(new_columns))
            logger.info(f"Added new columns to schema: {new_columns}")
            
            # Update schema version
            self.schema["version"] += 1
            
            # Record change in history
            self.schema["history"].append({
                "version": self.schema["version"],
                "action": "add_columns",
                "columns": list(new_columns),
                "timestamp": pd.Timestamp.now().isoformat()
            })
            
            # Save updated schema
            self._save_schema()
        elif new_columns:
            logger.warning(f"New columns found but not added to schema: {new_columns}")
        
        # Apply transformations from schema
        for column, transform in self.schema.get("transformations", {}).items():
            if column in processed_data.columns:
                try:
                    if transform == "log":
                        processed_data[column] = np.log1p(processed_data[column])
                    elif transform == "sqrt":
                        processed_data[column] = np.sqrt(processed_data[column])
                    elif transform == "standard":
                        mean = processed_data[column].mean()
                        std = processed_data[column].std()
                        processed_data[column] = (processed_data[column] - mean) / std
                except Exception as e:
                    logger.error(f"Error applying transformation to column '{column}': {e}")
        
        return processed_data
    
    def add_derived_feature(self, name: str, formula: str):
        """
        Add a derived feature to the schema.
        
        Args:
            name: Name of the derived feature
            formula: Formula to calculate the feature (using Python expressions)
        """
        self.schema["derived_features"][name] = formula
        
        # Update schema version
        self.schema["version"] += 1
        
        # Record change in history
        self.schema["history"].append({
            "version": self.schema["version"],
            "action": "add_derived_feature",
            "feature": name,
            "formula": formula,
            "timestamp": pd.Timestamp.now().isoformat()
        })
        
        # Save updated schema
        self._save_schema()
        logger.info(f"Added derived feature '{name}' with formula: {formula}")
    
    def add_transformation(self, column: str, transform_type: str):
        """
        Add a transformation for a column.
        
        Args:
            column: Column to transform
            transform_type: Type of transformation (log, sqrt, standard)
        """
        valid_transforms = ["log", "sqrt", "standard"]
        if transform_type not in valid_transforms:
            raise ValueError(f"Invalid transformation type. Must be one of {valid_transforms}")
        
        self.schema["transformations"][column] = transform_type
        
        # Update schema version
        self.schema["version"] += 1
        
        # Record change in history
        self.schema["history"].append({
            "version": self.schema["version"],
            "action": "add_transformation",
            "column": column,
            "transform": transform_type,
            "timestamp": pd.Timestamp.now().isoformat()
        })
        
        # Save updated schema
        self._save_schema()
        logger.info(f"Added {transform_type} transformation for column '{column}'")
    
    def get_feature_list(self, include_optional: bool = True, include_derived: bool = True) -> List[str]:
        """
        Get the list of features according to the current schema.
        
        Args:
            include_optional: Whether to include optional features
            include_derived: Whether to include derived features
            
        Returns:
            List of feature names
        """
        features = self.schema["core_features"].copy()
        
        if include_optional:
            features.extend(self.schema["optional_features"])
        
        if include_derived and self.schema.get("derived_features"):
            features.extend(list(self.schema["derived_features"].keys()))
        
        return features
    
    def export_schema(self, output_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Export the current schema.
        
        Args:
            output_path: Optional path to save the schema
            
        Returns:
            Schema dictionary
        """
        if output_path:
            with open(output_path, 'w') as f:
                json.dump(self.schema, f, indent=2)
        
        return self.schema
    
    def import_schema(self, schema_path: str):
        """
        Import a schema from a file.
        
        Args:
            schema_path: Path to the schema file
        """
        with open(schema_path, 'r') as f:
            new_schema = json.load(f)
        
        # Validate schema structure
        required_keys = ["version", "core_features", "target", "optional_features"]
        if not all(key in new_schema for key in required_keys):
            raise ValueError("Invalid schema: missing required keys")
        
        # Backup current schema
        old_schema_path = self.data_dir / f"schema_backup_v{self.schema['version']}.json"
        with open(old_schema_path, 'w') as f:
            json.dump(self.schema, f, indent=2)
        
        # Update schema
        self.schema = new_schema
        
        # Save updated schema
        self._save_schema()
        logger.info(f"Imported schema (version {new_schema['version']})")
    
    def validate_data_for_training(self, data: pd.DataFrame) -> bool:
        """
        Validate that data is suitable for training.
        
        Args:
            data: DataFrame to validate
            
        Returns:
            True if data is valid for training
        """
        # Check for required columns
        missing_core = set(self.schema["core_features"]) - set(data.columns)
        if missing_core:
            logger.error(f"Missing required core features: {missing_core}")
            return False
        
        if self.schema["target"] not in data.columns:
            logger.error(f"Missing target column '{self.schema['target']}'")
            return False
        
        # Validate data types and values
        for column in self.schema["core_features"]:
            # Check for non-numeric features
            if not pd.api.types.is_numeric_dtype(data[column]):
                logger.warning(f"Column '{column}' is not numeric")
            
            # Check for missing values
            if data[column].isna().any():
                logger.warning(f"Column '{column}' contains missing values")
        
        # Check target distribution for classification
        if self.schema["target"] in data.columns:
            target_counts = data[self.schema["target"]].value_counts()
            if len(target_counts) < 2:
                logger.error(f"Target column '{self.schema['target']}' has less than 2 classes")
                return False
            
            # Check for class imbalance
            min_class = target_counts.min()
            max_class = target_counts.max()
            if min_class / max_class < 0.1:
                logger.warning(f"Severe class imbalance detected: {target_counts.to_dict()}")
        
        return True
