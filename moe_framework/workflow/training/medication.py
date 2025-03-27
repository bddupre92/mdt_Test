"""
Medication Expert Training Workflow Module

This module provides a specialized training workflow for medication history expert models,
optimizing the training process for medication data characteristics.
"""

import logging
import pandas as pd
from typing import Dict, List, Any, Optional

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
              optimize_hyperparams: bool = True, include_dosage: bool = True,
              include_frequency: bool = True, include_interactions: bool = True, **kwargs) -> Dict[str, Any]:
        """
        Train a medication expert model with specialized optimizations.
        
        Args:
            expert: The medication expert model to train
            data: The input data
            target_column: The target column name
            optimize_hyperparams: Whether to optimize hyperparameters
            include_dosage: Whether to include dosage information
            include_frequency: Whether to include medication frequency
            include_interactions: Whether to include drug interaction features
            **kwargs: Additional keyword arguments
            
        Returns:
            Dictionary with training results
        """
        if self.verbose:
            logger.info("Starting medication history expert training workflow")
            
        # Store configuration
        config = {
            'optimize_hyperparams': optimize_hyperparams,
            'include_dosage': include_dosage,
            'include_frequency': include_frequency,
            'include_interactions': include_interactions
        }
        
        # Get features based on configuration
        usable_features = self._get_usable_features(expert, data)
        filtered_features = []
        
        # Filter dosage-related features
        if include_dosage:
            dosage_features = [col for col in usable_features if 
                              any(term in col.lower() for term in 
                                 ['dose', 'dosage', 'mg', 'ml', 'concentration',
                                  'strength', 'amount'])]
            filtered_features.extend(dosage_features)
            
        # Filter frequency-related features
        if include_frequency:
            frequency_features = [col for col in usable_features if 
                                 any(term in col.lower() for term in 
                                    ['frequency', 'daily', 'weekly', 'monthly', 'hourly',
                                     'times_per_day', 'schedule', 'regimen', 'routine'])]
            filtered_features.extend(frequency_features)
            
        # Filter interaction-related features
        if include_interactions:
            interaction_features = [col for col in usable_features if 
                                   any(term in col.lower() for term in 
                                      ['interaction', 'combination', 'cocktail', 'mixture',
                                       'polypharmacy', 'contraindication', 'synergy'])]
            filtered_features.extend(interaction_features)
            
        # If no filtered features, use all usable features
        if not filtered_features:
            filtered_features = usable_features
            if self.verbose:
                logger.warning("No matching medication features found. Using all available features.")
        
        # Split data for training and validation
        if 'X_train' in kwargs and 'y_train' in kwargs:
            X_train = kwargs['X_train'][filtered_features]
            y_train = kwargs['y_train']
        else:
            X_train = data[filtered_features]
            y_train = data[target_column]
            
        # Apply hybrid optimization if requested
        if optimize_hyperparams:
            if self.verbose:
                logger.info("Optimizing with Hybrid Evolutionary Approach")
                
            try:
                # Create the hybrid optimizer
                optimizer = HybridEvolutionaryAdapter(
                    param_space=expert.get_hyperparameter_space(),
                    feature_names=filtered_features,
                    eval_func=lambda params, features: expert.evaluate_configuration(
                        params, X_train[features], y_train
                    ),
                    max_iterations=kwargs.get('max_optimizer_iterations', 25),
                    population_size=kwargs.get('optimizer_population_size', 12),
                    verbose=self.verbose
                )
                
                # Run optimization
                best_config = optimizer.optimize()
                
                # Apply best configuration to the expert
                if 'params' in best_config:
                    expert.set_hyperparameters(best_config['params'])
                    
                if 'features' in best_config:
                    expert.set_selected_features(best_config['features'])
                    # Update X_train to only include selected features
                    X_train = X_train[best_config['features']]
                    
                if self.verbose:
                    logger.info(f"Optimized configuration: {best_config}")
            except Exception as e:
                logger.error(f"Hybrid optimization failed: {str(e)}")
        
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
        medication_features = [col for col in all_features if 
                              any(term in col.lower() for term in 
                                 ['medication', 'drug', 'pill', 'dose', 'dosage', 'treatment',
                                  'prescription', 'pharma', 'medicine', 'supplement', 'therapy',
                                  'remedy', 'regimen', 'frequency', 'administration'])]
        
        # If no medication features found, warn and return all features
        if not medication_features and self.verbose:
            logger.warning("No medication features found in data. Using all available features.")
            return all_features
            
        return medication_features
