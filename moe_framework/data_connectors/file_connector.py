"""
File-based Data Connector for MoE Framework

This module provides a connector for loading data from file-based sources
(CSV, Excel, etc.) while preserving feature characteristics needed by EC algorithms.
"""

import os
import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Optional, Tuple, Union, Any
from pathlib import Path

from .base_connector import BaseDataConnector

logger = logging.getLogger(__name__)

class FileDataConnector(BaseDataConnector):
    """
    Connector for loading data from file-based sources (CSV, Excel, etc.).
    
    This connector provides functionality to load data from various file formats
    while preserving feature characteristics needed by EC algorithms.
    """
    
    SUPPORTED_FORMATS = {
        'csv': {'read_func': pd.read_csv, 'write_func': 'to_csv'},
        'excel': {'read_func': pd.read_excel, 'write_func': 'to_excel'},
        'json': {'read_func': pd.read_json, 'write_func': 'to_json'},
        'parquet': {'read_func': pd.read_parquet, 'write_func': 'to_parquet'},
        'pickle': {'read_func': pd.read_pickle, 'write_func': 'to_pickle'},
    }
    
    def __init__(self, config: Optional[Dict[str, Any]] = None, verbose: bool = False):
        """
        Initialize the file data connector.
        
        Args:
            config: Optional configuration dictionary with the following keys:
                - data_dir: Directory where data files are stored
                - default_format: Default file format to use when not specified
                - encoding: Default encoding to use for text files
            verbose: Whether to display detailed logs during processing
        """
        super().__init__(config, verbose)
        self.data_dir = self.config.get('data_dir', os.getcwd())
        self.default_format = self.config.get('default_format', 'csv')
        self.encoding = self.config.get('encoding', 'utf-8')
        self.data = None
        self.file_path = None
        self.file_format = None
        self.schema = None
        
        # Create data directory if it doesn't exist
        os.makedirs(self.data_dir, exist_ok=True)
        
        if self.verbose:
            logger.info(f"Initialized FileDataConnector with data_dir: {self.data_dir}")
    
    def connect(self, connection_params: Dict[str, Any]) -> bool:
        """
        Establish connection to the file data source.
        
        For file-based connectors, this mainly validates that the file exists
        and is accessible.
        
        Args:
            connection_params: Dictionary with the following keys:
                - file_path: Path to the data file (required)
                - format: File format (optional, inferred from extension if not provided)
                
        Returns:
            True if connection was successful, False otherwise
        """
        if 'file_path' not in connection_params:
            logger.error("file_path is required in connection_params")
            return False
        
        file_path = connection_params['file_path']
        
        # Handle relative paths
        if not os.path.isabs(file_path):
            file_path = os.path.join(self.data_dir, file_path)
        
        # Check if file exists
        if not os.path.exists(file_path):
            logger.error(f"File not found: {file_path}")
            return False
        
        # Determine file format
        if 'format' in connection_params:
            file_format = connection_params['format'].lower()
        else:
            # Infer format from file extension
            _, ext = os.path.splitext(file_path)
            file_format = ext.lstrip('.').lower()
            
            # Handle Excel extensions
            if file_format in ['xls', 'xlsx', 'xlsm']:
                file_format = 'excel'
        
        # Validate format is supported
        if file_format not in self.SUPPORTED_FORMATS:
            logger.error(f"Unsupported file format: {file_format}")
            return False
        
        self.file_path = file_path
        self.file_format = file_format
        
        if self.verbose:
            logger.info(f"Connected to file: {self.file_path} (format: {self.file_format})")
        
        return True
    
    def load_data(self, query_params: Dict[str, Any] = None) -> pd.DataFrame:
        """
        Load data from the connected file.
        
        Args:
            query_params: Dictionary with the following optional keys:
                - sheet_name: Sheet name for Excel files
                - usecols: Columns to load
                - nrows: Number of rows to load
                - skiprows: Number of rows to skip
                - Any other parameters accepted by the pandas read function
                
        Returns:
            Pandas DataFrame with loaded data
        """
        if not self.file_path:
            logger.error("No file connected. Call connect() first.")
            return pd.DataFrame()
        
        query_params = query_params or {}
        
        # Get the appropriate read function
        read_func = self.SUPPORTED_FORMATS[self.file_format]['read_func']
        
        # Add encoding for text-based formats
        if self.file_format in ['csv', 'json']:
            query_params.setdefault('encoding', self.encoding)
        
        try:
            # Load the data
            self.data = read_func(self.file_path, **query_params)
            
            if self.verbose:
                logger.info(f"Loaded data from {self.file_path}: {self.data.shape}")
            
            # Update metadata
            self._update_metadata()
            
            return self.data
            
        except Exception as e:
            logger.error(f"Error loading data from {self.file_path}: {str(e)}")
            return pd.DataFrame()
    
    def _update_metadata(self) -> None:
        """Update metadata based on loaded data."""
        if self.data is None or self.data.empty:
            return
            
        # Basic metadata
        self.metadata.update({
            'file_path': self.file_path,
            'file_format': self.file_format,
            'row_count': len(self.data),
            'column_count': len(self.data.columns),
            'column_names': list(self.data.columns),
            'dtypes': {col: str(dtype) for col, dtype in self.data.dtypes.items()},
            'missing_values': self.data.isnull().sum().to_dict(),
            'memory_usage': self.data.memory_usage(deep=True).sum() / (1024 * 1024)  # MB
        })
        
        # Add EC-relevant metadata
        numeric_columns = self.data.select_dtypes(include=np.number).columns
        categorical_columns = self.data.select_dtypes(include=['object', 'category']).columns
        
        self.metadata.update({
            'numeric_columns': list(numeric_columns),
            'categorical_columns': list(categorical_columns),
            'numeric_column_count': len(numeric_columns),
            'categorical_column_count': len(categorical_columns),
            'missing_value_percentage': self.data.isnull().mean().mean() * 100
        })
        
        # Add feature distribution information for EC algorithm selection
        if len(numeric_columns) > 0:
            # Calculate basic statistics for numeric columns
            stats = self.data[numeric_columns].describe().transpose()
            
            # Calculate additional statistics relevant for EC algorithms
            feature_stats = {}
            for col in numeric_columns:
                col_data = self.data[col].dropna()
                if len(col_data) > 0:
                    feature_stats[col] = {
                        'mean': stats.loc[col, 'mean'],
                        'std': stats.loc[col, 'std'],
                        'min': stats.loc[col, 'min'],
                        'max': stats.loc[col, 'max'],
                        'skewness': col_data.skew(),
                        'kurtosis': col_data.kurtosis(),
                        'unique_ratio': col_data.nunique() / len(col_data)
                    }
            
            self.metadata['feature_statistics'] = feature_stats
    
    def get_schema(self) -> Dict[str, Any]:
        """
        Get the schema information for the loaded data.
        
        Returns:
            Dictionary with schema information
        """
        if self.data is None:
            logger.error("No data loaded. Call load_data() first.")
            return {}
            
        if self.schema is None:
            self._detect_schema()
            
        return self.schema
    
    def _detect_schema(self) -> None:
        """Detect schema from loaded data."""
        if self.data is None or self.data.empty:
            self.schema = {}
            return
            
        # Known patterns for target columns
        target_patterns = ['migraine', 'headache', 'target', 'label', 'outcome', 'class']
        
        # Known patterns for date columns
        date_patterns = ['date', 'time', 'timestamp', 'datetime', 'day']
        
        # Known patterns for ID columns
        id_patterns = ['id', 'subject', 'patient', 'participant']
        
        # Initialize schema
        schema = {
            'target_column': None,
            'date_column': None,
            'id_column': None,
            'feature_columns': [],
            'categorical_columns': [],
            'numeric_columns': []
        }
        
        # Categorize columns
        for col in self.data.columns:
            col_lower = col.lower()
            
            # Check for target column
            if schema['target_column'] is None:
                if any(pattern in col_lower for pattern in target_patterns):
                    schema['target_column'] = col
                    continue
            
            # Check for date column
            if schema['date_column'] is None:
                if any(pattern in col_lower for pattern in date_patterns):
                    schema['date_column'] = col
                    continue
            
            # Check for ID column
            if schema['id_column'] is None:
                if any(pattern in col_lower for pattern in id_patterns):
                    schema['id_column'] = col
                    continue
            
            # Categorize remaining columns
            if col in self.metadata.get('numeric_columns', []):
                schema['numeric_columns'].append(col)
                schema['feature_columns'].append(col)
            elif col in self.metadata.get('categorical_columns', []):
                schema['categorical_columns'].append(col)
                schema['feature_columns'].append(col)
        
        # If no target column found, look for binary columns
        if schema['target_column'] is None:
            for col in schema['numeric_columns']:
                unique_values = self.data[col].dropna().unique()
                if len(unique_values) == 2 and set(unique_values).issubset({0, 1}):
                    schema['target_column'] = col
                    schema['numeric_columns'].remove(col)
                    schema['feature_columns'].remove(col)
                    break
        
        self.schema = schema
        
        if self.verbose:
            logger.info(f"Detected schema: {self.schema}")
    
    def save_data(self, data: pd.DataFrame, file_path: str, format_type: str = None, **kwargs) -> bool:
        """
        Save data to a file.
        
        Args:
            data: DataFrame to save
            file_path: Path to save the file
            format_type: File format (inferred from extension if not provided)
            kwargs: Additional arguments to pass to the pandas write function
            
        Returns:
            True if save was successful, False otherwise
        """
        # Handle relative paths
        if not os.path.isabs(file_path):
            file_path = os.path.join(self.data_dir, file_path)
        
        # Determine file format
        if format_type:
            file_format = format_type.lower()
        else:
            # Infer format from file extension
            _, ext = os.path.splitext(file_path)
            file_format = ext.lstrip('.').lower()
            
            # Handle Excel extensions
            if file_format in ['xls', 'xlsx', 'xlsm']:
                file_format = 'excel'
        
        # Validate format is supported
        if file_format not in self.SUPPORTED_FORMATS:
            logger.error(f"Unsupported file format for saving: {file_format}")
            return False
        
        try:
            # Get the appropriate write function
            write_func_name = self.SUPPORTED_FORMATS[file_format]['write_func']
            write_func = getattr(data, write_func_name)
            
            # Add encoding for text-based formats
            if file_format in ['csv', 'json']:
                kwargs.setdefault('encoding', self.encoding)
            
            # Save the data
            write_func(file_path, **kwargs)
            
            if self.verbose:
                logger.info(f"Saved data to {file_path}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error saving data to {file_path}: {str(e)}")
            return False
    
    def close(self) -> None:
        """Close the connection to the file data source."""
        self.file_path = None
        self.file_format = None
        self.data = None
        self.schema = None
        self.metadata = {}
        
        if self.verbose:
            logger.info("Closed file connection")
