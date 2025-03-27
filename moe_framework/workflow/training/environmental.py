"""
Environmental Expert Training Workflow Module

This module provides a specialized training workflow for environmental expert models,
optimizing the training process for environmental data characteristics.
"""

import logging
import pandas as pd
from typing import Dict, List, Any, Optional

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
                # Create the ES optimizer
                optimizer = EvolutionStrategyAdapter(
                    param_space=expert.get_hyperparameter_space(),
                    eval_func=lambda params: expert.evaluate_hyperparameters(params, X_train, y_train),
                    max_iterations=kwargs.get('max_optimizer_iterations', 30),
                    population_size=kwargs.get('optimizer_population_size', 10),
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
