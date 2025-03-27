"""
Behavioral Expert Training Workflow Module

This module provides a specialized training workflow for behavioral expert models,
optimizing the training process for behavioral data characteristics.
"""

import logging
import pandas as pd
from typing import Dict, List, Any, Optional

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
                # Create the feature selection optimizer
                optimizer = AntColonyAdapter(
                    feature_names=filtered_features,
                    eval_func=lambda selected_features: expert.evaluate_feature_set(
                        X_train[selected_features], y_train
                    ),
                    max_iterations=kwargs.get('max_optimizer_iterations', 20),
                    colony_size=kwargs.get('optimizer_colony_size', 15),
                    verbose=self.verbose
                )
                
                # Run optimization
                best_features = optimizer.optimize()
                
                if self.verbose:
                    logger.info(f"Selected {len(best_features)} features via ACO")
                
                # Update the expert with selected features
                expert.set_selected_features(best_features)
                
                # Update X_train to only include selected features
                X_train = X_train[best_features]
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
