"""
Execution Pipeline for MoE Framework

This module provides a streamlined execution workflow for the MoE framework,
enabling one-click execution of the complete pipeline with EC algorithms.
"""

import os
import logging
import json
import time
from typing import Dict, List, Optional, Tuple, Union, Any
import pandas as pd
import numpy as np

from ..data_connectors.file_connector import FileDataConnector
from ..data_connectors.data_quality import DataQualityAssessment
from ..upload.upload_manager import UploadManager

logger = logging.getLogger(__name__)

class ExecutionPipeline:
    """
    Manages the end-to-end execution of the MoE framework pipeline.
    
    This class provides a streamlined workflow for executing the complete
    MoE framework pipeline, from data loading to model evaluation, with a
    focus on preserving EC algorithm functionality.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None, verbose: bool = False):
        """
        Initialize the execution pipeline.
        
        Args:
            config: Optional configuration dictionary with settings for all components
            verbose: Whether to display detailed logs during processing
        """
        self.config = config or {}
        self.verbose = verbose
        
        # Set up configuration with defaults
        self.output_dir = self.config.get('output_dir', os.path.join(os.getcwd(), 'results'))
        self.environment = self.config.get('environment', 'dev')
        
        # Create output directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Create environment-specific subdirectory
        self.env_output_dir = os.path.join(self.output_dir, self.environment)
        os.makedirs(self.env_output_dir, exist_ok=True)
        
        # Initialize components
        upload_config = self.config.get('upload', {})
        self.upload_manager = UploadManager(
            config=upload_config,
            verbose=self.verbose
        )
        
        # Initialize execution state
        self.execution_state = {
            'status': 'initialized',
            'start_time': None,
            'end_time': None,
            'current_step': None,
            'steps_completed': [],
            'errors': [],
            'warnings': [],
            'results': {}
        }
        
        if self.verbose:
            logger.info(f"Initialized ExecutionPipeline with output_dir: {self.output_dir}")
            logger.info(f"Environment: {self.environment}")
    
    def execute(self, data_path: str, target_column: Optional[str] = None, 
                config_override: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Execute the complete MoE framework pipeline.
        
        Args:
            data_path: Path to the input data file
            target_column: Optional name of the target column
            config_override: Optional configuration overrides for this execution
            
        Returns:
            Dictionary with execution results
        """
        # Update configuration if overrides provided
        if config_override:
            self.config.update(config_override)
        
        # Initialize execution state
        self.execution_state = {
            'status': 'running',
            'start_time': time.time(),
            'end_time': None,
            'current_step': 'initialization',
            'steps_completed': [],
            'errors': [],
            'warnings': [],
            'results': {},
            'config': self.config
        }
        
        try:
            # Step 1: Data Loading
            self._update_execution_state('data_loading')
            
            # Process the data file through the upload manager
            upload_result = self.upload_manager.process_uploaded_file(data_path, target_column)
            
            if not upload_result.get('success', False):
                raise Exception(f"Data loading failed: {upload_result.get('message', 'Unknown error')}")
            
            self.execution_state['results']['data_loading'] = upload_result
            self._complete_step('data_loading')
            
            # Step 2: Data Quality Assessment
            self._update_execution_state('data_quality_assessment')
            
            # Quality assessment results are already in the upload_result
            quality_results = upload_result.get('quality_assessment', {})
            quality_score = quality_results.get('quality_score', 0.0)
            
            # Log quality assessment results
            if self.verbose:
                logger.info(f"Data quality assessment completed with score: {quality_score:.2f}")
                
                # Log specific quality metrics
                for metric, value in quality_results.items():
                    if metric != 'quality_score':
                        logger.info(f"  {metric}: {value}")
            
            self.execution_state['results']['data_quality_assessment'] = quality_results
            self._complete_step('data_quality_assessment')
            
            # Step 3: EC Compatibility Check
            self._update_execution_state('ec_compatibility_check')
            
            # EC compatibility results are already in the upload_result
            ec_compatibility = upload_result.get('ec_compatibility', {})
            is_compatible = ec_compatibility.get('is_compatible', False)
            
            if not is_compatible:
                self.execution_state['warnings'].append(
                    f"Data is not fully compatible with EC algorithms: {ec_compatibility.get('message', 'Unknown issue')}"
                )
                
                # Log compatibility issues
                if self.verbose:
                    logger.warning(f"EC compatibility issues detected:")
                    for issue in ec_compatibility.get('issues', []):
                        logger.warning(f"  {issue}")
            
            self.execution_state['results']['ec_compatibility_check'] = ec_compatibility
            self._complete_step('ec_compatibility_check')
            
            # Step 4: Prepare Execution Summary
            self._update_execution_state('prepare_summary')
            
            # Create a unique execution ID
            execution_id = f"exec_{int(time.time())}_{os.getpid()}"
            
            # Prepare summary
            summary = {
                'execution_id': execution_id,
                'data_file': os.path.basename(data_path),
                'data_shape': upload_result.get('data_shape'),
                'quality_score': quality_score,
                'ec_compatible': is_compatible,
                'timestamp': pd.Timestamp.now().isoformat(),
                'environment': self.environment
            }
            
            self.execution_state['results']['summary'] = summary
            self._complete_step('prepare_summary')
            
            # Step 5: Save Results
            self._update_execution_state('save_results')
            
            # Create a results directory for this execution
            results_dir = os.path.join(self.env_output_dir, execution_id)
            os.makedirs(results_dir, exist_ok=True)
            
            # Save execution state
            with open(os.path.join(results_dir, 'execution_state.json'), 'w') as f:
                # Create a copy of the execution state with the end time set
                final_state = self.execution_state.copy()
                final_state['end_time'] = time.time()
                final_state['status'] = 'completed'
                
                json.dump(final_state, f, indent=2)
            
            # Save summary
            with open(os.path.join(results_dir, 'summary.json'), 'w') as f:
                json.dump(summary, f, indent=2)
            
            # Update execution state with results path
            self.execution_state['results']['results_path'] = results_dir
            self._complete_step('save_results')
            
            # Complete execution
            self.execution_state['status'] = 'completed'
            self.execution_state['end_time'] = time.time()
            
            if self.verbose:
                logger.info(f"Execution completed successfully")
                logger.info(f"Results saved to: {results_dir}")
            
            return {
                'success': True,
                'execution_id': execution_id,
                'results_path': results_dir,
                'summary': summary,
                'execution_state': self.execution_state
            }
            
        except Exception as e:
            logger.error(f"Execution failed: {str(e)}")
            
            # Update execution state
            self.execution_state['status'] = 'failed'
            self.execution_state['end_time'] = time.time()
            self.execution_state['errors'].append(str(e))
            
            return {
                'success': False,
                'message': f"Execution failed: {str(e)}",
                'execution_state': self.execution_state
            }
    
    def _update_execution_state(self, step: str) -> None:
        """
        Update the execution state with the current step.
        
        Args:
            step: Name of the current step
        """
        self.execution_state['current_step'] = step
        
        if self.verbose:
            logger.info(f"Starting step: {step}")
    
    def _complete_step(self, step: str) -> None:
        """
        Mark a step as completed in the execution state.
        
        Args:
            step: Name of the completed step
        """
        self.execution_state['steps_completed'].append(step)
        
        if self.verbose:
            logger.info(f"Completed step: {step}")
    
    def get_execution_state(self) -> Dict[str, Any]:
        """
        Get the current execution state.
        
        Returns:
            Dictionary with the current execution state
        """
        return self.execution_state
    
    def load_execution_results(self, execution_id: str) -> Dict[str, Any]:
        """
        Load the results of a previous execution.
        
        Args:
            execution_id: ID of the execution to load
            
        Returns:
            Dictionary with the execution results
        """
        results_dir = os.path.join(self.env_output_dir, execution_id)
        
        if not os.path.exists(results_dir):
            return {
                'success': False,
                'message': f"Execution results not found: {execution_id}"
            }
        
        try:
            # Load execution state
            with open(os.path.join(results_dir, 'execution_state.json'), 'r') as f:
                execution_state = json.load(f)
            
            # Load summary
            with open(os.path.join(results_dir, 'summary.json'), 'r') as f:
                summary = json.load(f)
            
            return {
                'success': True,
                'execution_id': execution_id,
                'results_path': results_dir,
                'summary': summary,
                'execution_state': execution_state
            }
            
        except Exception as e:
            logger.error(f"Error loading execution results: {str(e)}")
            return {
                'success': False,
                'message': f"Error loading execution results: {str(e)}"
            }
