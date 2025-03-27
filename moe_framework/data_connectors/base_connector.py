"""
Base Data Connector for MoE Framework

This module defines the base interface for all data connectors in the MoE framework.
Each connector implementation must adhere to this interface to ensure compatibility
with the framework's components, particularly the EC algorithms.
"""

import os
import pandas as pd
import numpy as np
import logging
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple, Union, Any

logger = logging.getLogger(__name__)

class BaseDataConnector(ABC):
    """
    Abstract base class for all data connectors in the MoE framework.
    
    This class defines the standard interface that all data connector implementations
    must follow to ensure compatibility with the MoE framework's components,
    particularly the EC algorithms.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None, verbose: bool = False):
        """
        Initialize the data connector.
        
        Args:
            config: Optional configuration dictionary for the connector
            verbose: Whether to display detailed logs during processing
        """
        self.config = config or {}
        self.verbose = verbose
        self.metadata = {}
        self._setup_logging()
        
    def _setup_logging(self):
        """Configure logging based on verbosity setting."""
        log_level = logging.DEBUG if self.verbose else logging.INFO
        logging.basicConfig(level=log_level, 
                           format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                           force=False)
    
    @abstractmethod
    def connect(self, connection_params: Dict[str, Any]) -> bool:
        """
        Establish connection to the data source.
        
        Args:
            connection_params: Parameters required to connect to the data source
            
        Returns:
            True if connection was successful, False otherwise
        """
        pass
    
    @abstractmethod
    def load_data(self, query_params: Dict[str, Any]) -> pd.DataFrame:
        """
        Load data from the connected data source.
        
        Args:
            query_params: Parameters to specify what data to load
            
        Returns:
            Pandas DataFrame with loaded data
        """
        pass
    
    @abstractmethod
    def get_schema(self) -> Dict[str, Any]:
        """
        Get the schema information for the loaded data.
        
        Returns:
            Dictionary with schema information
        """
        pass
    
    def get_metadata(self) -> Dict[str, Any]:
        """
        Get metadata about the loaded data that can inform EC algorithm selection.
        
        Returns:
            Dictionary with metadata information
        """
        return self.metadata
    
    def extract_features_for_ec(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Extract and prepare features in a format suitable for EC algorithms.
        
        This method ensures that the data structure is compatible with EC algorithms
        by handling categorical variables, missing values, and other preprocessing
        steps specific to EC requirements.
        
        Args:
            data: Input DataFrame with raw data
            
        Returns:
            DataFrame with features prepared for EC algorithms
        """
        if data.empty:
            logger.warning("Empty DataFrame provided to extract_features_for_ec")
            return data
            
        # Create a copy to avoid modifying the original
        ec_data = data.copy()
        
        # Handle categorical features - EC algorithms typically need numerical inputs
        cat_columns = ec_data.select_dtypes(include=['object', 'category']).columns
        for col in cat_columns:
            if col in ec_data.columns:
                # For low cardinality, use one-hot encoding
                if ec_data[col].nunique() < 10:
                    dummies = pd.get_dummies(ec_data[col], prefix=col, drop_first=True)
                    ec_data = pd.concat([ec_data.drop(col, axis=1), dummies], axis=1)
                else:
                    # For high cardinality, use label encoding
                    ec_data[col] = pd.factorize(ec_data[col])[0]
        
        # Handle missing values - EC algorithms typically can't handle NaNs
        numeric_columns = ec_data.select_dtypes(include=np.number).columns
        for col in numeric_columns:
            if ec_data[col].isnull().any():
                # Use median imputation as a default strategy
                ec_data[col].fillna(ec_data[col].median(), inplace=True)
        
        # Update metadata with EC-relevant information
        self.metadata.update({
            'feature_count': len(ec_data.columns),
            'numeric_feature_count': len(numeric_columns),
            'categorical_feature_count': len(cat_columns),
            'sample_count': len(ec_data),
            'missing_value_percentage': data.isnull().mean().mean() * 100
        })
        
        return ec_data
    
    def validate_ec_compatibility(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Validate that the data is compatible with EC algorithms.
        
        Args:
            data: DataFrame to validate
            
        Returns:
            Dictionary with validation results
        """
        validation_results = {
            'is_compatible': True,
            'issues': [],
            'warnings': []
        }
        
        # Check for minimum number of samples
        if len(data) < 10:
            validation_results['is_compatible'] = False
            validation_results['issues'].append(
                f"Insufficient samples: {len(data)} (minimum 10 required for EC algorithms)"
            )
        
        # Check for numeric features
        numeric_columns = data.select_dtypes(include=np.number).columns
        if len(numeric_columns) == 0:
            validation_results['is_compatible'] = False
            validation_results['issues'].append(
                "No numeric features found (required for EC algorithms)"
            )
        
        # Check for missing values
        missing_percentage = data.isnull().mean().mean() * 100
        if missing_percentage > 50:
            validation_results['is_compatible'] = False
            validation_results['issues'].append(
                f"Too many missing values: {missing_percentage:.1f}% (maximum 50% allowed)"
            )
        elif missing_percentage > 20:
            validation_results['warnings'].append(
                f"High percentage of missing values: {missing_percentage:.1f}% (may affect EC algorithm performance)"
            )
        
        # Check for high cardinality categorical features
        cat_columns = data.select_dtypes(include=['object', 'category']).columns
        high_cardinality_cols = []
        for col in cat_columns:
            if data[col].nunique() > 100:
                high_cardinality_cols.append(col)
        
        if high_cardinality_cols:
            validation_results['warnings'].append(
                f"High cardinality categorical features detected: {high_cardinality_cols} "
                "(may require special handling for EC algorithms)"
            )
        
        return validation_results
    
    def close(self) -> None:
        """Close the connection to the data source."""
        pass
