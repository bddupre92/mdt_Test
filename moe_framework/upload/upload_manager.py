"""
Upload Manager for MoE Framework

This module provides a simple interface for uploading and validating data files
for use with the MoE framework, ensuring compatibility with EC implementations.
"""

import os
import logging
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any, BinaryIO
import pandas as pd

from ..data_connectors.file_connector import FileDataConnector
from ..data_connectors.data_quality import DataQualityAssessment

logger = logging.getLogger(__name__)

class UploadManager:
    """
    Manages the upload and validation of data files for the MoE framework.
    
    This class provides a simple interface for uploading data files, validating
    their structure and quality, and preparing them for use with EC algorithms.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None, verbose: bool = False):
        """
        Initialize the upload manager.
        
        Args:
            config: Optional configuration dictionary with the following keys:
                - upload_dir: Directory where uploaded files will be stored
                - allowed_formats: List of allowed file formats
                - max_file_size: Maximum allowed file size in bytes
                - environment: Current environment (dev, test, prod)
            verbose: Whether to display detailed logs during processing
        """
        self.config = config or {}
        self.verbose = verbose
        
        # Set up configuration with defaults
        self.upload_dir = self.config.get('upload_dir', os.path.join(os.getcwd(), 'uploads'))
        self.allowed_formats = self.config.get('allowed_formats', ['csv', 'excel', 'json', 'parquet'])
        self.max_file_size = self.config.get('max_file_size', 100 * 1024 * 1024)  # 100 MB default
        self.environment = self.config.get('environment', 'dev')
        
        # Create upload directory if it doesn't exist
        os.makedirs(self.upload_dir, exist_ok=True)
        
        # Create environment-specific subdirectory
        self.env_upload_dir = os.path.join(self.upload_dir, self.environment)
        os.makedirs(self.env_upload_dir, exist_ok=True)
        
        # Initialize components
        self.file_connector = FileDataConnector(
            config={'data_dir': self.env_upload_dir},
            verbose=self.verbose
        )
        self.quality_assessment = DataQualityAssessment(
            verbose=self.verbose
        )
        
        # Upload history
        self.upload_history = []
        
        if self.verbose:
            logger.info(f"Initialized UploadManager with upload_dir: {self.upload_dir}")
            logger.info(f"Environment: {self.environment}")
            logger.info(f"Environment-specific upload directory: {self.env_upload_dir}")
    
    def validate_file(self, file_path: str) -> Dict[str, Any]:
        """
        Validate a file before upload.
        
        Args:
            file_path: Path to the file to validate
            
        Returns:
            Dictionary with validation results
        """
        validation_results = {
            'is_valid': True,
            'issues': [],
            'warnings': [],
            'file_info': {
                'name': os.path.basename(file_path),
                'size': 0,
                'format': ''
            }
        }
        
        # Check if file exists
        if not os.path.exists(file_path):
            validation_results['is_valid'] = False
            validation_results['issues'].append(f"File not found: {file_path}")
            return validation_results
        
        # Get file info
        file_size = os.path.getsize(file_path)
        file_name = os.path.basename(file_path)
        file_ext = os.path.splitext(file_name)[1].lstrip('.').lower()
        
        # Map file extension to format
        format_map = {
            'csv': 'csv',
            'xls': 'excel',
            'xlsx': 'excel',
            'xlsm': 'excel',
            'json': 'json',
            'parquet': 'parquet'
        }
        file_format = format_map.get(file_ext, file_ext)
        
        validation_results['file_info'] = {
            'name': file_name,
            'size': file_size,
            'format': file_format
        }
        
        # Check file size
        if file_size > self.max_file_size:
            validation_results['is_valid'] = False
            validation_results['issues'].append(
                f"File size ({file_size / (1024 * 1024):.2f} MB) exceeds maximum allowed size "
                f"({self.max_file_size / (1024 * 1024):.2f} MB)"
            )
        
        # Check file format
        if file_format not in self.allowed_formats:
            validation_results['is_valid'] = False
            validation_results['issues'].append(
                f"File format '{file_format}' is not allowed. "
                f"Allowed formats: {', '.join(self.allowed_formats)}"
            )
        
        return validation_results
    
    def upload_file(self, file_path: str, target_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Upload a file to the upload directory.
        
        Args:
            file_path: Path to the file to upload
            target_name: Optional name for the uploaded file (if None, original name is used)
            
        Returns:
            Dictionary with upload results
        """
        # Validate file
        validation_results = self.validate_file(file_path)
        if not validation_results['is_valid']:
            return {
                'success': False,
                'validation': validation_results,
                'message': "File validation failed"
            }
        
        file_name = os.path.basename(file_path)
        file_format = validation_results['file_info']['format']
        
        # Determine target file name
        if target_name:
            # Ensure target name has the correct extension
            target_ext = os.path.splitext(target_name)[1]
            if not target_ext:
                # Add extension based on format
                ext_map = {
                    'csv': '.csv',
                    'excel': '.xlsx',
                    'json': '.json',
                    'parquet': '.parquet'
                }
                target_name = f"{target_name}{ext_map.get(file_format, '.dat')}"
        else:
            target_name = file_name
        
        # Create target path
        target_path = os.path.join(self.env_upload_dir, target_name)
        
        try:
            # Copy file to upload directory
            import shutil
            shutil.copy2(file_path, target_path)
            
            # Record upload in history
            upload_record = {
                'original_file': file_path,
                'uploaded_file': target_path,
                'file_name': target_name,
                'file_format': file_format,
                'file_size': validation_results['file_info']['size'],
                'timestamp': pd.Timestamp.now().isoformat()
            }
            self.upload_history.append(upload_record)
            
            if self.verbose:
                logger.info(f"File uploaded successfully: {target_path}")
            
            return {
                'success': True,
                'validation': validation_results,
                'upload_path': target_path,
                'message': "File uploaded successfully"
            }
            
        except Exception as e:
            logger.error(f"Error uploading file: {str(e)}")
            return {
                'success': False,
                'validation': validation_results,
                'message': f"Error uploading file: {str(e)}"
            }
    
    def process_uploaded_file(self, file_path: str, target_column: Optional[str] = None) -> Dict[str, Any]:
        """
        Process an uploaded file, including data quality assessment.
        
        Args:
            file_path: Path to the uploaded file
            target_column: Optional name of the target column
            
        Returns:
            Dictionary with processing results
        """
        # Connect to the file
        connection_params = {'file_path': file_path}
        if not self.file_connector.connect(connection_params):
            return {
                'success': False,
                'message': f"Failed to connect to file: {file_path}"
            }
        
        # Load the data
        data = self.file_connector.load_data()
        if data.empty:
            return {
                'success': False,
                'message': f"Failed to load data from file: {file_path}"
            }
        
        # Get schema information
        schema = self.file_connector.get_schema()
        
        # If target column is not specified, use the one from schema
        if target_column is None and schema.get('target_column'):
            target_column = schema['target_column']
        
        # Assess data quality
        quality_results = self.quality_assessment.assess_quality(data, target_column)
        
        # Check EC compatibility
        ec_compatibility = self.file_connector.validate_ec_compatibility(data)
        
        # Extract features for EC
        ec_data = self.file_connector.extract_features_for_ec(data)
        
        # Prepare result
        result = {
            'success': True,
            'file_path': file_path,
            'file_name': os.path.basename(file_path),
            'data_shape': data.shape,
            'schema': schema,
            'quality_assessment': quality_results,
            'ec_compatibility': ec_compatibility,
            'ec_data_shape': ec_data.shape,
            'metadata': self.file_connector.get_metadata()
        }
        
        if self.verbose:
            logger.info(f"Processed uploaded file: {file_path}")
            logger.info(f"Data shape: {data.shape}")
            logger.info(f"Quality score: {quality_results.get('quality_score', 0.0):.2f}")
            logger.info(f"EC compatibility: {ec_compatibility.get('is_compatible', False)}")
        
        return result
    
    def get_upload_history(self) -> List[Dict[str, Any]]:
        """
        Get the upload history.
        
        Returns:
            List of upload records
        """
        return self.upload_history
    
    def save_upload_history(self, file_path: Optional[str] = None) -> bool:
        """
        Save the upload history to a file.
        
        Args:
            file_path: Path to save the history (if None, a default path is used)
            
        Returns:
            True if save was successful, False otherwise
        """
        if not file_path:
            file_path = os.path.join(self.upload_dir, f"upload_history_{self.environment}.json")
        
        try:
            with open(file_path, 'w') as f:
                json.dump(self.upload_history, f, indent=2)
            
            if self.verbose:
                logger.info(f"Upload history saved to {file_path}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error saving upload history: {str(e)}")
            return False
    
    def load_upload_history(self, file_path: Optional[str] = None) -> bool:
        """
        Load the upload history from a file.
        
        Args:
            file_path: Path to load the history from (if None, a default path is used)
            
        Returns:
            True if load was successful, False otherwise
        """
        if not file_path:
            file_path = os.path.join(self.upload_dir, f"upload_history_{self.environment}.json")
        
        if not os.path.exists(file_path):
            if self.verbose:
                logger.warning(f"Upload history file not found: {file_path}")
            return False
        
        try:
            with open(file_path, 'r') as f:
                self.upload_history = json.load(f)
            
            if self.verbose:
                logger.info(f"Upload history loaded from {file_path}")
                logger.info(f"Loaded {len(self.upload_history)} upload records")
            
            return True
            
        except Exception as e:
            logger.error(f"Error loading upload history: {str(e)}")
            return False
