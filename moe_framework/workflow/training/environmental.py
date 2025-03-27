"""
Environmental Expert Training Workflow Module

This module provides a specialized training workflow for environmental expert models,
optimizing the training process for environmental data characteristics.
"""

import logging
import pandas as pd
from typing import Dict, List, Any, Optional
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import cross_val_score

from .base_workflow import ExpertTrainingWorkflow
from ...experts.environmental_expert import EnvironmentalExpert
from ...experts.optimizer_adapters import EvolutionStrategyAdapter

# Configure logging
logger = logging.getLogger(__name__)

class EnvironmentalTrainingWorkflow(ExpertTrainingWorkflow):
    """
    Specialized training workflow for environmental expert models.
    
    This workflow handles the unique aspects of environmental data, including
    weather, air quality, and location-based factors, along with Evolution
    Strategy for hyperparameter optimization.
    """
    
    def train(self, expert: EnvironmentalExpert, data: pd.DataFrame, target_column: str,
              optimize_hyperparams: bool = True, include_weather: bool = True,
              include_pollution: bool = True, **kwargs) -> Dict[str, Any]:
        """
        Train an environmental expert model with specialized optimizations.
        
        Args:
            expert: The environmental expert model to train
            data: The input data
            target_column: The target column name
            optimize_hyperparams: Whether to optimize hyperparameters with ES
            include_weather: Whether to include weather-related features
            include_pollution: Whether to include pollution-related features
            **kwargs: Additional keyword arguments
            
        Returns:
            Dictionary with training results
        """
        if self.verbose:
            logger.info("Starting environmental expert training workflow")
            
        # Suppress numpy warnings during training
        import warnings
        # Temporarily suppress numpy warnings
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=RuntimeWarning, message='Mean of empty slice')
            
            # Store configuration
            config = {
                'optimize_hyperparams': optimize_hyperparams,
                'include_weather': include_weather,
                'include_pollution': include_pollution
            }
            
            # Get features based on configuration
            usable_features = self._get_usable_features(expert, data)
            filtered_features = []
            
            # Filter weather-related features
            if include_weather:
                weather_features = [col for col in usable_features if 
                                   any(term in col.lower() for term in 
                                      ['weather', 'temperature', 'humidity', 'pressure', 
                                       'precipitation', 'rain', 'snow', 'wind', 'storm',
                                       'cloud', 'sun', 'atmospheric'])]
                filtered_features.extend(weather_features)
                
            # Filter pollution-related features
            if include_pollution:
                pollution_features = [col for col in usable_features if 
                                     any(term in col.lower() for term in 
                                        ['pollution', 'air_quality', 'aqi', 'pm2.5', 'pm10', 
                                         'ozone', 'co2', 'no2', 'so2', 'voc', 'allergen'])]
                filtered_features.extend(pollution_features)
                
            # If no filtered features, use all usable features
            if not filtered_features:
                filtered_features = usable_features
                if self.verbose:
                    logger.warning("No matching environmental features found. Using all available features.")
            
            # Split data for training and validation
            if 'X_train' in kwargs and 'y_train' in kwargs:
                X_train = kwargs['X_train'][filtered_features]
                y_train = kwargs['y_train']
            else:
                X_train = data[filtered_features]
                y_train = data[target_column]
            
            # Apply hyperparameter optimization if requested
            if optimize_hyperparams:
                if self.verbose:
                    logger.info("Optimizing hyperparameters with Evolution Strategy")
                    
                try:
                    # Get hyperparameter space from expert
                    param_space = expert.get_hyperparameter_space()
                    
                    # Convert hyperparameter space to bounds format expected by adapter
                    bounds = []
                    for param_name, param_config in param_space.items():
                        if param_config['type'] in ['float', 'int']:
                            for bound in param_config['bounds']:
                                bounds.append(bound)
                    
                    # Define fitness function for hyperparameter optimization
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
                        
                            # Use safe parameter values to ensure model can be created
                            safe_params = {
                                'n_estimators': max(10, int(param_dict.get('n_estimators', 100))),
                                'learning_rate': max(0.01, min(0.3, param_dict.get('learning_rate', 0.1))),
                                'max_depth': max(1, int(param_dict.get('max_depth', 3))),
                                'min_samples_split': max(2, int(param_dict.get('min_samples_split', 2))),
                                'random_state': 42  # Always use fixed random state for deterministic results
                            }
                            
                            # Create a model with these parameters
                            model = GradientBoostingRegressor(**safe_params)
                            
                            # Prepare data
                            X_safe = X_train.copy()
                            y_safe = y_train.copy()
                            
                            # Handle NaN values
                            if X_safe.isnull().any().any():
                                X_safe = X_safe.fillna(X_safe.mean())
                            if y_safe.isnull().any():
                                y_safe = y_safe.fillna(y_safe.mean())
                            
                            # Set error scoring to a fixed value instead of raising exceptions
                            try:
                                cv_scores = cross_val_score(
                                    model, X_safe, y_safe, 
                                    cv=3, 
                                    scoring='neg_mean_squared_error',
                                    error_score=float('nan')  # Use NaN for errors
                                )
                                # Filter out any NaN values and return the mean
                                valid_scores = [s for s in cv_scores if not np.isnan(s)]
                                if valid_scores:
                                    return -np.mean(valid_scores)
                                else:
                                    logger.warning("All CV scores were NaN, returning worst possible score")
                                    return float('inf')
                            except Exception as inner_e:
                                logger.error(f"Cross-validation error: {str(inner_e)}")
                                return float('inf')
                        except Exception as e:
                            logger.error(f"Error in fitness function: {str(e)}")
                            return float('inf')  # Return worst possible score
                    
                    # Initialize optimizer with proper parameters
                    optimizer = EvolutionStrategyAdapter(
                        fitness_function=fitness_function,
                        bounds=bounds,
                        population_size=kwargs.get('optimizer_population_size', 20),
                        max_iterations=kwargs.get('max_optimizer_iterations', 30),
                        initial_step_size=0.3,
                        adaptation_rate=0.2,
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
                    
                    # Apply best parameters to the expert's model
                    expert.model = GradientBoostingRegressor(**best_params_dict, random_state=42)
                    
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
                    'expert_type': 'environmental',
                    'config': config,
                    'cross_validation': cv_results
                }
            except Exception as e:
                logger.error(f"Error training environmental expert: {str(e)}")
                return {
                    'success': False,
                    'message': f'Training failed: {str(e)}'
                }
            
    def _get_usable_features(self, expert: EnvironmentalExpert, data: pd.DataFrame) -> List[str]:
        """
        Get features usable by the environmental expert.
        
        Args:
            expert: The environmental expert model
            data: The input data
            
        Returns:
            List of feature column names
        """
        # Get default usable features from parent class
        all_features = super()._get_usable_features(expert, data)
        
        # Filter for environmental data columns - preferably this would use the expert's own
        # logic for identifying relevant features, but this is a simplified example
        environmental_features = [col for col in all_features if 
                                  any(term in col.lower() for term in 
                                     ['weather', 'temperature', 'humidity', 'pressure', 
                                      'pollution', 'air_quality', 'aqi', 'pm2.5', 'pm10',
                                      'environmental', 'climate', 'season', 'location',
                                      'altitude', 'barometric', 'pollen', 'allergen'])]
        
        # If no environmental features found, warn and return all features
        if not environmental_features and self.verbose:
            logger.warning("No environmental features found in data. Using all available features.")
            return all_features
            
        return environmental_features
