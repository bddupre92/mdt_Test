"""
Behavioral Expert Training Workflow Module

This module provides a specialized training workflow for behavioral expert models,
optimizing the training process for behavioral data characteristics.
"""

import logging
import pandas as pd
from typing import Dict, List, Any, Optional
import numpy as np
import warnings

from .base_workflow import ExpertTrainingWorkflow
from ...experts.behavioral_expert import BehavioralExpert
from ...experts.optimizer_adapters import AntColonyAdapter

# Configure logging
logger = logging.getLogger(__name__)

class BehavioralTrainingWorkflow(ExpertTrainingWorkflow):
    """
    Specialized training workflow for behavioral expert models.
    
    This workflow handles the unique aspects of behavioral data, including
    sleep, activity, and stress pattern analysis, along with Ant Colony
    Optimization for feature selection.
    """
    
    def train(self, expert: BehavioralExpert, data: pd.DataFrame, target_column: str,
              optimize_feature_selection: bool = True, include_sleep: bool = True,
              include_activity: bool = True, include_stress: bool = True, **kwargs) -> Dict[str, Any]:
        """
        Train a behavioral expert model with specialized optimizations.
        
        Args:
            expert: The behavioral expert model to train
            data: The input data
            target_column: The target column name
            optimize_feature_selection: Whether to use ACO for feature selection
            include_sleep: Whether to include sleep-related features
            include_activity: Whether to include activity-related features
            include_stress: Whether to include stress-related features
            **kwargs: Additional keyword arguments
            
        Returns:
            Dictionary with training results
        """
        if self.verbose:
            logger.info("Starting behavioral expert training workflow")
            
        # Suppress numpy warnings during training
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=RuntimeWarning, message='Mean of empty slice')
            
            # Store configuration
            config = {
                'optimize_feature_selection': optimize_feature_selection,
                'include_sleep': include_sleep,
                'include_activity': include_activity,
                'include_stress': include_stress
            }
            
            # Get features based on configuration
            usable_features = self._get_usable_features(expert, data)
            filtered_features = []
            
            # Filter sleep-related features
            if include_sleep:
                sleep_features = [col for col in usable_features if 
                                  any(term in col.lower() for term in 
                                     ['sleep', 'insomnia', 'rem', 'nrem', 'wakeup', 'bed'])]
                filtered_features.extend(sleep_features)
                
            # Filter activity-related features
            if include_activity:
                activity_features = [col for col in usable_features if 
                                    any(term in col.lower() for term in 
                                       ['activity', 'exercise', 'steps', 'movement', 'walk', 
                                        'run', 'sedentary', 'calories', 'workout'])]
                filtered_features.extend(activity_features)
                
            # Filter stress-related features
            if include_stress:
                stress_features = [col for col in usable_features if 
                                  any(term in col.lower() for term in 
                                     ['stress', 'anxiety', 'mood', 'emotion', 'mental', 
                                      'relaxation', 'mindfulness', 'tension'])]
                filtered_features.extend(stress_features)
                
            # If no filtered features, use all usable features
            if not filtered_features:
                filtered_features = usable_features
                if self.verbose:
                    logger.warning("No matching behavioral features found. Using all available features.")
            
            # Split data for training and validation
            if 'X_train' in kwargs and 'y_train' in kwargs:
                X_train = kwargs['X_train'][filtered_features]
                y_train = kwargs['y_train']
            else:
                X_train = data[filtered_features]
                y_train = data[target_column]
            
            # Apply feature selection if requested
            if optimize_feature_selection:
                if self.verbose:
                    logger.info("Optimizing feature selection with Ant Colony Optimization")
                    
                try:
                    # Get all features that could be used
                    all_features = self._get_usable_features(expert, data)
                    
                    # Define bounds for binary feature selection (0 = exclude, 1 = include)
                    bounds = [(0, 1) for _ in range(len(all_features))]
                    
                    # Define fitness function for feature selection
                    def fitness_function(params):
                        try:
                            # Convert binary feature selection to list of selected features
                            selected_indices = [i for i, include in enumerate(params) if include > 0.5]
                            
                            if not selected_indices:  # If no features selected, penalize heavily
                                return float('inf')
                                
                            selected_features = [all_features[i] for i in selected_indices]
                            
                            # Create copy of data with only selected features
                            X_selected = X_train[selected_features]
                            
                            # Apply preprocessing if available
                            if hasattr(expert, 'preprocess_data'):
                                X_processed = expert.preprocess_data(X_selected)
                            else:
                                X_processed = X_selected
                            
                            # Evaluate fitness using cross-validation
                            from sklearn.model_selection import cross_val_score
                            scores = cross_val_score(
                                expert.model, X_processed, y_train, 
                                cv=3, scoring='neg_mean_squared_error'
                            )
                            
                            # Calculate base score (negative MSE)
                            base_score = np.mean(scores)
                            
                            # Add penalty for number of features (to encourage simplicity)
                            feature_penalty = 0.01 * len(selected_indices) / len(all_features)
                            
                            # Return negative of score (lower is better for ACO)
                            return -base_score + feature_penalty
                            
                        except Exception as e:
                            logger.error(f"Error in fitness function: {str(e)}")
                            return float('inf')  # Return worst possible score
                    
                    # Initialize optimizer
                    optimizer = AntColonyAdapter(
                        fitness_function=fitness_function,
                        bounds=bounds,
                        population_size=20,
                        max_iterations=30,
                        alpha=1.0,  # Pheromone importance
                        beta=2.0,   # Heuristic importance
                        evaporation_rate=0.1,
                        random_seed=42
                    )
                    
                    # Run optimization
                    best_params, best_fitness = optimizer.optimize()
                    
                    # Extract selected features
                    selected_indices = [i for i, include in enumerate(best_params) if include > 0.5]
                    selected_features = [all_features[i] for i in selected_indices]
                    
                    # Update expert with selected features
                    if hasattr(expert, 'feature_columns'):
                        expert.feature_columns = selected_features
                    
                    if self.verbose:
                        logger.info(f"Selected {len(selected_features)} features: {selected_features}")
                except Exception as e:
                    logger.error(f"Feature selection optimization failed: {str(e)}")
            
            try:
                # Train the expert with the selected features
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
                    'expert_type': 'behavioral',
                    'config': config,
                    'cross_validation': cv_results
                }
            except Exception as e:
                logger.error(f"Error training behavioral expert: {str(e)}")
                return {
                    'success': False,
                    'message': f'Training failed: {str(e)}'
                }
            
    def _get_usable_features(self, expert: BehavioralExpert, data: pd.DataFrame) -> List[str]:
        """
        Get features usable by the behavioral expert.
        
        Args:
            expert: The behavioral expert model
            data: The input data
            
        Returns:
            List of feature column names
        """
        # Get default usable features from parent class
        all_features = super()._get_usable_features(expert, data)
        
        # Filter for behavioral data columns - preferably this would use the expert's own
        # logic for identifying relevant features, but this is a simplified example
        behavioral_features = [col for col in all_features if 
                               any(term in col.lower() for term in 
                                  ['sleep', 'activity', 'exercise', 'steps', 'stress', 
                                   'mood', 'anxiety', 'emotion', 'mental', 'behavioral',
                                   'lifestyle', 'habit', 'routine', 'schedule'])]
        
        # If no behavioral features found, warn and return all features
        if not behavioral_features and self.verbose:
            logger.warning("No behavioral features found in data. Using all available features.")
            return all_features
            
        return behavioral_features
