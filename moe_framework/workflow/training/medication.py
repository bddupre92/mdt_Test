"""
Medication Expert Training Workflow Module

This module provides a specialized training workflow for medication history expert models,
optimizing the training process for medication data characteristics.
"""

import logging
import pandas as pd
from typing import Dict, List, Any, Optional
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import cross_val_score, train_test_split

from .base_workflow import ExpertTrainingWorkflow
from ...experts.medication_history_expert import MedicationHistoryExpert
from ...experts.optimizer_adapters import HybridEvolutionaryAdapter

# Configure logging
logger = logging.getLogger(__name__)

class MedicationTrainingWorkflow(ExpertTrainingWorkflow):
    """
    Specialized training workflow for medication history expert models.
    
    This workflow handles the unique aspects of medication data, including
    dosage, frequency, and drug interaction analysis, along with
    Hybrid Evolutionary Optimization.
    """
    
    def train(self, expert: MedicationHistoryExpert, data: pd.DataFrame, target_column: str,
              optimize_hyperparams: bool = True, impute_missing: bool = True, 
              include_frequency: bool = True, include_interactions: bool = True, **kwargs) -> Dict[str, Any]:
        """
        Train the medication expert on the provided data.
        
        Args:
            expert: The medication expert model instance
            data: The training data
            target_column: The column containing the target variable
            optimize_hyperparams: Whether to optimize hyperparameters
            impute_missing: Whether to impute missing values
            include_frequency: Whether to include medication frequency features
            include_interactions: Whether to include medication interaction features
            **kwargs: Additional keyword arguments
            
        Returns:
            Dictionary with training results
        """
        try:
            # Set verbosity based on kwargs
            self.verbose = kwargs.get('verbose', False)
            
            # Initialize configuration
            config = {
                'optimize_hyperparams': optimize_hyperparams,
                'impute_missing': impute_missing,
                'include_frequency': include_frequency,
                'include_interactions': include_interactions
            }
            
            # Process blood pressure columns (handle "120/80" format)
            if 'blood_pressure' in data.columns:
                # Function to extract systolic/diastolic values
                def extract_bp(bp_value):
                    if isinstance(bp_value, str) and '/' in bp_value:
                        try:
                            systolic, diastolic = bp_value.split('/')
                            return float(systolic), float(diastolic)
                        except:
                            return None, None
                    return bp_value, None
                
                # Create new blood pressure columns
                bp_values = data['blood_pressure'].apply(extract_bp)
                data['blood_pressure_systolic'] = bp_values.str[0]
                data['blood_pressure_diastolic'] = bp_values.str[1]
                
                # Drop the original column to avoid confusion
                data = data.drop('blood_pressure', axis=1)
            
            # Update expert configuration
            expert.include_frequency = include_frequency
            expert.include_interactions = include_interactions
            
            # Save original target column name
            expert.target_column = target_column
            
            # Identify usable features
            usable_features = self._get_usable_features(expert, data)
            
            if len(usable_features) == 0:
                logger.error("No valid features found for medication expert")
                return {
                    'success': False,
                    'message': 'No valid features found for medication expert'
                }
            
            # Extract features and target
            X = data[usable_features].copy()
            y = data[target_column].copy()
            
            # Check for sufficient data
            if len(X) < 10:
                logger.error("Insufficient data for medication expert training")
                return {
                    'success': False,
                    'message': 'Insufficient data for training (minimum 10 samples required)'
                }
                
            # Preprocess data
            X = expert.preprocess_data(X)
            
            # Split the data for training/validation
            X_train, X_val, y_train, y_val = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
            
            # Feature extraction
            X_train_features, feature_cols = expert.extract_features(X_train)
            
            # Keep track of columns with string values that need encoding
            string_columns = [col for col in feature_cols if X_train_features[col].dtype == 'object']
            
            # Apply hybrid optimization if requested
            if optimize_hyperparams:
                if self.verbose:
                    logger.info("Optimizing with Hybrid Evolutionary Approach")
                    
                try:
                    # Define parameter space
                    param_space = expert.get_hyperparameter_space()
                    param_bounds = []
                    
                    # Convert parameter dictionary to list of (min, max) bounds
                    for param_name, param_config in param_space.items():
                        if param_config['type'] in ['float', 'int']:
                            param_bounds.extend(param_config['bounds'])
                    
                    # Set bounds for optimization        
                    bounds = param_bounds
                    
                    # Define fitness function (cross-validation score)
                    def fitness_function(params):
                        try:
                            # Convert parameters to dictionary format
                            param_index = 0
                            param_dict = {}
                            for param_name, param_config in param_space.items():
                                if param_config['type'] in ['float', 'int']:
                                    param_count = len(param_config['bounds'])
                                    if param_count == 1:
                                        # Convert to int if needed
                                        value = params[param_index]
                                        if param_config['type'] == 'int':
                                            value = int(value)
                                        param_dict[param_name] = value
                                        param_index += 1
                                    else:
                                        values = [params[param_index + i] for i in range(param_count)]
                                        # Convert to int if needed
                                        if param_config['type'] == 'int':
                                            values = [int(v) for v in values]
                                        param_dict[param_name] = values
                                        param_index += param_count
                        
                            # Create a model with safe parameters for HistGradientBoostingRegressor
                            model_params = {
                                'max_iter': max(10, int(param_dict.get('max_iter', 100))),
                                'learning_rate': max(0.01, min(0.3, param_dict.get('learning_rate', 0.1))),
                                'max_depth': max(1, int(param_dict.get('max_depth', 5))),
                                'min_samples_leaf': max(1, int(param_dict.get('min_samples_leaf', 20))),
                                'l2_regularization': max(0.0, float(param_dict.get('l2_regularization', 0.0))),
                                'random_state': 42
                            }
                            
                            # Create the model with safe parameters
                            model = HistGradientBoostingRegressor(**model_params)
                            
                            # Prepare data safely
                            X_safe = X_train.copy()
                            y_safe = y_train.copy()
                            
                            # Handle NaN values
                            if X_safe.isnull().any().any():
                                X_safe = X_safe.fillna(X_safe.mean())
                            if y_safe.isnull().any():
                                y_safe = y_safe.fillna(y_safe.mean())
                            
                            # Handle string/object columns by checking data types first
                            for col in X_safe.columns:
                                if X_safe[col].dtype == 'object':
                                    # Convert strings to category codes
                                    X_safe[col] = X_safe[col].astype('category').cat.codes
                            
                            # Make sure all data is numeric
                            X_safe = X_safe.apply(pd.to_numeric, errors='coerce')
                            X_safe = X_safe.fillna(X_safe.mean())
                            
                            # Use cross_val_score with error handling
                            try:
                                cv_scores = cross_val_score(
                                    model, X_safe, y_safe, 
                                    cv=3, 
                                    scoring='neg_mean_squared_error',
                                    error_score=float('nan')  # Return NaN for errors
                                )
                                # Filter out any NaNs
                                valid_scores = [s for s in cv_scores if not np.isnan(s)]
                                
                                if valid_scores:
                                    return -np.mean(valid_scores)  # Return negative MSE for minimization
                                else:
                                    logger.warning("All cross-validation scores were NaN, returning worst possible score")
                                    return float('inf')
                            except Exception as inner_e:
                                # Convert any error to string safely
                                error_msg = str(inner_e)
                                logger.error(f"Cross-validation error: {error_msg}")
                                return float('inf')
                        except Exception as e:
                            logger.error(f"Error in fitness function: {str(e)}")
                            return float('inf')  # Return worst possible score
                    
                    # Initialize optimizer
                    optimizer = HybridEvolutionaryAdapter(
                        fitness_function=fitness_function,
                        bounds=bounds,
                        population_size=kwargs.get('optimizer_population_size', 10),
                        max_iterations=kwargs.get('max_optimizer_iterations', 20),
                        crossover_rate=0.7,
                        mutation_rate=0.1,
                        local_search_iterations=3,
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
                                best_params_dict[param_name] = values
                                param_index += param_count
                    
                    # Update expert with optimized parameters
                    if hasattr(expert, 'model') and hasattr(expert.model, '__class__'):
                        # Create a new model with the optimized parameters
                        model_params = {}
                        if 'max_iter' in best_params_dict:
                            model_params['max_iter'] = max(10, min(1000, int(best_params_dict.get('max_iter', 100))))
                        if 'learning_rate' in best_params_dict:
                            model_params['learning_rate'] = max(0.001, min(0.5, float(best_params_dict.get('learning_rate', 0.1))))
                        if 'max_depth' in best_params_dict:
                            model_params['max_depth'] = max(1, min(50, int(best_params_dict.get('max_depth', 5))))
                        if 'min_samples_leaf' in best_params_dict:
                            model_params['min_samples_leaf'] = max(1, min(100, int(best_params_dict.get('min_samples_leaf', 20))))
                        if 'l2_regularization' in best_params_dict:
                            model_params['l2_regularization'] = max(0.0, min(20.0, float(best_params_dict.get('l2_regularization', 0.0))))
                            
                        expert.model = HistGradientBoostingRegressor(**model_params, random_state=42)
                    
                    if self.verbose:
                        logger.info(f"Optimized hyperparameters: {best_params_dict}")
                        logger.info(f"Best fitness score: {best_fitness}")
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
                    'expert_type': 'medication',
                    'config': config,
                    'cross_validation': cv_results
                }
            except Exception as e:
                logger.error(f"Error training medication expert: {str(e)}")
                return {
                    'success': False,
                    'message': f'Training failed: {str(e)}'
                }
        except Exception as e:
            logger.error(f"Error in medication training workflow: {str(e)}")
            return {
                'success': False,
                'message': f'Training failed: {str(e)}'
            }
            
    def _get_usable_features(self, expert: MedicationHistoryExpert, data: pd.DataFrame) -> List[str]:
        """
        Get features usable by the medication expert.
        
        Args:
            expert: The medication expert model
            data: The input data
            
        Returns:
            List of feature column names
        """
        # Get default usable features from parent class
        all_features = super()._get_usable_features(expert, data)
        
        # Filter for medication data columns - preferably this would use the expert's own
        # logic for identifying relevant features, but this is a simplified example
        medication_terms = ['medication', 'drug', 'pill', 'dose', 'dosage', 'treatment',
                          'prescription', 'pharma', 'medicine', 'supplement', 'therapy',
                          'remedy', 'regimen', 'frequency', 'administration']
        
        medication_features = [col for col in all_features if 
                             any(term in col.lower() for term in medication_terms)]
        
        # If no medication features found, warn and return all features
        if not medication_features and self.verbose:
            logger.warning("No medication features found in data. Using all available features.")
            return all_features
            
        return medication_features
