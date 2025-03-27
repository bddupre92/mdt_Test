"""
Physiological Expert Training Workflow Module

This module provides a specialized training workflow for physiological expert models,
optimizing the training process for physiological data characteristics.
"""

import logging
import pandas as pd
from typing import Dict, List, Any, Optional

from .base_workflow import ExpertTrainingWorkflow
from ...experts.physiological_expert import PhysiologicalExpert
from ...experts.optimizer_adapters import DifferentialEvolutionAdapter

# Configure logging
logger = logging.getLogger(__name__)

class PhysiologicalTrainingWorkflow(ExpertTrainingWorkflow):
    """
    Specialized training workflow for physiological expert models.
    
    This workflow handles the unique aspects of physiological data, including
    vital sign preprocessing, variability feature extraction, and Differential
    Evolution optimization.
    """
    
    def train(self, expert: PhysiologicalExpert, data: pd.DataFrame, target_column: str,
              optimize_hyperparams: bool = True, extract_variability: bool = True,
              normalize_vitals: bool = True, **kwargs) -> Dict[str, Any]:
        """
        Train a physiological expert model with specialized optimizations.
        
        Args:
            expert: The physiological expert model to train
            data: The input data
            target_column: The target column name
            optimize_hyperparams: Whether to optimize hyperparameters
            extract_variability: Whether to extract variability features
            normalize_vitals: Whether to normalize vital signs
            **kwargs: Additional keyword arguments
            
        Returns:
            Dictionary with training results
        """
        if self.verbose:
            logger.info("Starting physiological expert training workflow")
            
        # Store configuration
        config = {
            'optimize_hyperparams': optimize_hyperparams,
            'extract_variability': extract_variability,
            'normalize_vitals': normalize_vitals
        }
        
        # Extract features that the expert can use
        usable_features = self._get_usable_features(expert, data)
        
        # Split data for training and validation
        if 'X_train' in kwargs and 'y_train' in kwargs:
            X_train = kwargs['X_train']
            y_train = kwargs['y_train']
        else:
            X_train = data[usable_features]
            y_train = data[target_column]
            
        # Apply physiological-specific preprocessing
        if normalize_vitals:
            vital_columns = [col for col in X_train.columns if 
                           any(term in col.lower() for term in 
                             ['heart_rate', 'bp', 'pressure', 'temp', 'gsr', 
                              'eeg', 'emg', 'ecg', 'spo2', 'respiratory'])]
            
            if self.verbose and vital_columns:
                logger.info(f"Normalizing {len(vital_columns)} vital sign columns")
                
            # Normalization happens inside the expert.fit method in our implementation
        
        # Extract variability features if requested
        if extract_variability and hasattr(expert, 'extract_variability_features'):
            if self.verbose:
                logger.info("Extracting variability features from physiological signals")
                
            # Adjust X_train to include extracted features
            # This is a simplified example - actual implementation would be more sophisticated
            try:
                X_train = expert.extract_variability_features(X_train)
            except Exception as e:
                logger.error(f"Error extracting variability features: {str(e)}")
        
        # Optimize hyperparameters if requested
        if optimize_hyperparams:
            if self.verbose:
                logger.info("Optimizing hyperparameters with Differential Evolution")
                
            try:
                # Initialize the optimizer with the expert's hyperparameter space
                optimizer = DifferentialEvolutionAdapter(
                    param_space=expert.get_hyperparameter_space(),
                    eval_func=lambda params: expert.evaluate_hyperparameters(params, X_train, y_train),
                    max_iterations=kwargs.get('max_optimizer_iterations', 30),
                    population_size=kwargs.get('optimizer_population_size', 15),
                    verbose=self.verbose
                )
                
                # Run optimization
                best_params = optimizer.optimize()
                
                # Apply best parameters to the expert
                expert.set_hyperparameters(best_params)
                
                if self.verbose:
                    logger.info(f"Optimized hyperparameters: {best_params}")
            except Exception as e:
                logger.error(f"Hyperparameter optimization failed: {str(e)}")
        
        try:
            # Train the expert with optimized settings
            expert.fit(X_train, y_train, **kwargs)
            
            # Perform cross-validation if requested
            cv_results = None
            if kwargs.get('perform_cv', False):
                cv_results = self._cross_validate(
                    expert, X_train, y_train, 
                    n_splits=kwargs.get('cv_splits', 5)
                )
            
            # Return results
            return {
                'success': True,
                'expert_type': 'physiological',
                'config': config,
                'cross_validation': cv_results
            }
        except Exception as e:
            logger.error(f"Error training physiological expert: {str(e)}")
            return {
                'success': False,
                'message': f'Training failed: {str(e)}'
            }
            
    def _get_usable_features(self, expert: PhysiologicalExpert, data: pd.DataFrame) -> List[str]:
        """
        Get features usable by the physiological expert.
        
        Args:
            expert: The physiological expert model
            data: The input data
            
        Returns:
            List of feature column names
        """
        # Get default usable features from parent class
        all_features = super()._get_usable_features(expert, data)
        
        # Filter for physiological data columns - preferably this would use the expert's own
        # logic for identifying relevant features, but this is a simplified example
        physiological_features = [col for col in all_features if 
                                  any(term in col.lower() for term in 
                                      ['heart_rate', 'bp', 'blood_pressure', 'temperature',
                                       'gsr', 'ecg', 'eeg', 'emg', 'spo2', 'respiratory',
                                       'pulse', 'oxygen', 'hrv', 'galvanic', 'conductance',
                                       'sweating', 'vitals', 'physiological'])]
        
        # If no physiological features found, warn and return all features
        if not physiological_features and self.verbose:
            logger.warning("No physiological features found in data. Using all available features.")
            return all_features
            
        return physiological_features
