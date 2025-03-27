"""
Physiological Expert Training Workflow Module

This module provides a specialized training workflow for physiological expert models,
optimizing the training process for physiological data characteristics.
"""

import logging
import pandas as pd
from typing import Dict, List, Any, Optional
import numpy as np
from sklearn.model_selection import cross_val_score

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
                param_space = expert.get_hyperparameter_space()
                
                # Convert hyperparameter space to bounds format expected by adapter
                bounds = []
                for param_name, param_config in param_space.items():
                    if param_config['type'] in ['float', 'int']:
                        for bound in param_config['bounds']:
                            bounds.append(bound)
                
                # Create fitness function closure
                def fitness_function(params):
                    try:
                        # Convert parameters to dictionary format
                        param_index = 0
                        param_dict = {}
                        for param_name, param_config in param_space.items():
                            if param_config['type'] in ['float', 'int']:
                                param_count = len(param_config['bounds'])
                                if param_count == 1:
                                    param_dict[param_name] = params[param_index]
                                    param_index += 1
                                else:
                                    param_dict[param_name] = [params[param_index + i] for i in range(param_count)]
                                    param_index += param_count
                        
                        # Safety checks for neural network parameters
                        if 'hidden_layers' in param_dict:
                            # Ensure hidden layers are never zero or negative
                            hidden_layers = [max(1, int(layer)) for layer in param_dict['hidden_layers']]
                            param_dict['hidden_layers'] = hidden_layers
                            
                        # Create model with these parameters
                        if expert.model_type == 'neural_network':
                            # For MLPRegressor, convert hidden_layers to hidden_layer_sizes tuple
                            if 'hidden_layers' in param_dict:
                                param_dict['hidden_layer_sizes'] = tuple(param_dict.pop('hidden_layers'))
                            model = expert.model.__class__(**param_dict)
                        else:
                            model = expert.model.__class__(**param_dict)
                        
                        # Evaluate using cross-validation with error handling
                        try:
                            cv_scores = cross_val_score(model, X_train, y_train, cv=3, scoring='neg_mean_squared_error', error_score=float('nan'))
                            valid_scores = [s for s in cv_scores if not np.isnan(s)]
                            if valid_scores:
                                return -np.mean(valid_scores)
                            else:
                                logger.warning("All CV scores were NaN, returning worst possible score")
                                # Capture the warning for later reporting
                                if not hasattr(expert, 'training_history'):
                                    expert.training_history = {}
                                if 'warnings' not in expert.training_history:
                                    expert.training_history['warnings'] = []
                                expert.training_history['warnings'].append("All cross-validation scores were NaN, possible data quality issues")
                                return float('inf')
                        except Exception as cv_e:
                            logger.error(f"Cross-validation error: {str(cv_e)}")
                            # Capture the error for later reporting
                            if not hasattr(expert, 'training_history'):
                                expert.training_history = {}
                            if 'errors' not in expert.training_history:
                                expert.training_history['errors'] = []
                            expert.training_history['errors'].append(f"Cross-validation error: {str(cv_e)}")
                            return float('inf')  # Return worst possible score
                    except Exception as e:
                        logger.error(f"Error in fitness function: {str(e)}")
                        return float('inf')  # Return worst possible score
                
                # Initialize optimizer with proper parameters
                optimizer = DifferentialEvolutionAdapter(
                    fitness_function=fitness_function,
                    bounds=bounds,
                    population_size=kwargs.get('optimizer_population_size', 15),
                    max_iterations=kwargs.get('max_optimizer_iterations', 30),
                    crossover_probability=0.7,
                    differential_weight=0.8,
                    random_seed=42
                )
                
                # Run optimization
                best_params, best_fitness = optimizer.optimize()
                
                # Convert optimized parameters back to dictionary format
                param_index = 0
                best_params_dict = {}
                for param_name, param_config in param_space.items():
                    if param_config['type'] in ['float', 'int']:
                        param_count = len(param_config['bounds'])
                        if param_count == 1:
                            value = best_params[param_index]
                            # Convert to int if needed
                            if param_config['type'] == 'int':
                                value = int(value)
                            best_params_dict[param_name] = value
                            param_index += 1
                        else:
                            values = [best_params[param_index + i] for i in range(param_count)]
                            # Convert to int if needed
                            if param_config['type'] == 'int':
                                values = [int(v) for v in values]
                            # Ensure positive values for hidden layers
                            if param_name == 'hidden_layers':
                                values = [max(4, v) for v in values]
                            best_params_dict[param_name] = values
                            param_index += param_count
                
                # Log the optimized parameters
                logger.info(f"Optimized parameters: {best_params_dict}")
                
                # Special handling for neural network model parameters
                if expert.model_type == 'neural_network' and 'hidden_layers' in best_params_dict:
                    # Store the optimized hidden layers
                    expert.hidden_layers = best_params_dict['hidden_layers']
                    # For MLP models, we need to convert to tuple for hidden_layer_sizes
                    if hasattr(expert.model, 'hidden_layer_sizes'):
                        expert.model.hidden_layer_sizes = tuple(expert.hidden_layers)
                
                # Apply best parameters to the expert
                for param_name, value in best_params_dict.items():
                    setattr(expert, param_name, value)
                    
                # Reinitialize model with optimized parameters
                expert._initialize_model()
                
                if self.verbose:
                    logger.info(f"Optimized hyperparameters: {best_params_dict}")
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
