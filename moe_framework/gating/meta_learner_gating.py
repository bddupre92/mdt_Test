"""
Meta Learner Gating Integration Module

This module provides integration between the Meta_Learner and the GatingNetwork,
enabling adaptive weighting of expert models based on context and performance metrics.
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Any, Optional, Tuple, Union, Callable
import os
import pickle

from meta.meta_learner import MetaLearner
from meta_optimizer.optimizers.optimizer_factory import OptimizerFactory
from moe_framework.experts.base_expert import BaseExpert

# Configure logging
logger = logging.getLogger(__name__)


class MetaLearnerGating:
    """
    Integration between Meta_Learner and GatingNetwork for adaptive expert weighting.
    
    This class provides methods to leverage the Meta_Learner for adaptive weighting
    of expert models based on context and performance metrics.
    
    Attributes:
        meta_learner (MetaLearner): Meta learner instance for adaptive weighting
        optimizer_factory (OptimizerFactory): Factory for creating optimizers
        context_features (List[str]): List of context features to use
        performance_metrics (List[str]): List of performance metrics to track
        expert_registry (Dict[str, Dict[str, Any]]): Registry of expert metadata
    """
    
    def __init__(self, 
                 meta_learner: Optional[MetaLearner] = None,
                 optimizer_factory: Optional[OptimizerFactory] = None,
                 context_features: List[str] = None,
                 performance_metrics: List[str] = None):
        """
        Initialize the Meta_Learner gating integration.
        
        Args:
            meta_learner: Meta learner instance (will create a new one if None)
            optimizer_factory: Factory for creating optimizers (will create a new one if None)
            context_features: List of context features to use
            performance_metrics: List of performance metrics to track
        """
        self.meta_learner = meta_learner or MetaLearner()
        self.optimizer_factory = optimizer_factory or OptimizerFactory()
        self.context_features = context_features or []
        self.performance_metrics = performance_metrics or ['rmse', 'mae', 'r2']
        self.expert_registry = {}
        self.experts = {}
        self.is_fitted = False
        
        logger.info("Initialized Meta_Learner gating integration")
    
    def register_expert(self, expert_id: str, expert: BaseExpert, metadata: Dict[str, Any] = None) -> None:
        """
        Register an expert with the Meta_Learner.
        
        Args:
            expert_id: Unique identifier for the expert
            expert: Expert model instance
            metadata: Additional metadata about the expert
        """
        # Register with Meta_Learner
        self.meta_learner.register_expert(expert_id, expert)
        
        # Store expert metadata
        self.expert_registry[expert_id] = {
            'expert': expert,
            'type': type(expert).__name__,
            'metadata': metadata or {},
            'performance_history': []
        }
        
        # Also store in experts dictionary for compatibility with GatingNetwork
        self.experts[expert_id] = expert
        
        logger.info(f"Registered expert {expert_id} with Meta_Learner gating")
    
    def extract_context(self, X: pd.DataFrame) -> Dict[str, Any]:
        """
        Extract context features from the input data.
        
        Args:
            X: Feature data
            
        Returns:
            Dictionary of context features
        """
        context = {}
        
        # Extract specified context features if available
        for feature in self.context_features:
            if feature in X.columns:
                context[feature] = X[feature].mean()
        
        # Add data quality metrics
        context['data_size'] = len(X)
        context['missing_ratio'] = X.isnull().mean().mean()
        context['feature_count'] = len(X.columns)
        
        # Add statistical properties
        numeric_cols = X.select_dtypes(include=['number']).columns
        if len(numeric_cols) > 0:
            numeric_data = X[numeric_cols]
            context['mean_values'] = numeric_data.mean().mean()
            context['std_values'] = numeric_data.std().mean()
            context['skewness'] = numeric_data.skew().mean()
            
            # Calculate correlation structure
            corr_matrix = numeric_data.corr().values
            context['mean_correlation'] = np.nanmean(np.abs(corr_matrix))
        
        logger.debug(f"Extracted context: {context}")
        return context
    
    def predict_weights(self, X: pd.DataFrame) -> Dict[str, np.ndarray]:
        """
        Predict weights for expert models based on context.
        
        Args:
            X: Feature data
            
        Returns:
            Dictionary mapping expert IDs to weight arrays
        """
        # Extract context
        context = self.extract_context(X)
        
        # Get weights from Meta_Learner
        meta_weights = self.meta_learner.predict_weights(context)
        
        # Convert to numpy arrays with same length as X
        n_samples = len(X)
        weights = {
            expert_id: np.ones(n_samples) * weight
            for expert_id, weight in meta_weights.items()
        }
        
        logger.debug(f"Predicted weights: {weights}")
        return weights
    
    def update_performance(self, 
                          expert_id: str, 
                          performance_metrics: Dict[str, float],
                          context: Dict[str, Any] = None) -> None:
        """
        Update performance metrics for an expert.
        
        Args:
            expert_id: Expert identifier
            performance_metrics: Dictionary of performance metrics
            context: Optional context information
        """
        if expert_id not in self.expert_registry:
            logger.warning(f"Expert {expert_id} not registered, cannot update performance")
            return
        
        # Update performance history
        self.expert_registry[expert_id]['performance_history'].append({
            'metrics': performance_metrics,
            'context': context
        })
        
        # Update Meta_Learner
        self.meta_learner.update_expert_performance(expert_id, performance_metrics)
        
        logger.debug(f"Updated performance for expert {expert_id}: {performance_metrics}")
    
    def select_optimizer(self, problem_characteristics: Dict[str, Any] = None) -> str:
        """
        Select an appropriate optimizer based on problem characteristics.
        
        Args:
            problem_characteristics: Dictionary of problem characteristics
            
        Returns:
            Optimizer type name
        """
        # Use Meta_Learner to select algorithm
        algorithm = self.meta_learner.select_algorithm(problem_characteristics or {})
        
        # Map algorithm names to valid optimizer types
        algorithm_to_optimizer = {
            'bayesian': 'evolution_strategy',
            'random_search': 'differential_evolution',
            'grid_search': 'grey_wolf',
            # Default fallback
            'default': 'evolution_strategy'
        }
        
        # Get the algorithm name if it's an object
        if hasattr(algorithm, 'name'):
            algorithm_name = algorithm.name
        else:
            algorithm_name = algorithm
            
        # Map to a valid optimizer type
        optimizer_type = algorithm_to_optimizer.get(algorithm_name, algorithm_to_optimizer['default'])
        
        logger.info(f"Selected optimizer: {optimizer_type} (from algorithm: {algorithm_name})")
        return optimizer_type
    
    def create_optimizer(self, 
                        optimizer_type: str = None, 
                        problem_characteristics: Dict[str, Any] = None,
                        **kwargs) -> Any:
        """
        Create an optimizer instance.
        
        Args:
            optimizer_type: Type of optimizer to create (will select automatically if None)
            problem_characteristics: Dictionary of problem characteristics
            **kwargs: Additional parameters for the optimizer
            
        Returns:
            Optimizer instance
        """
        # Select optimizer type if not specified
        if optimizer_type is None:
            optimizer_type = self.select_optimizer(problem_characteristics)
        
        # Set default parameters if not provided
        if optimizer_type == 'grey_wolf' and ('dim' not in kwargs or 'bounds' not in kwargs):
            # Default dimension and bounds for weight optimization
            n_experts = len(self.experts)
            if n_experts == 0:
                n_experts = 3  # Default if no experts registered yet
                
            if 'dim' not in kwargs:
                kwargs['dim'] = n_experts
                
            if 'bounds' not in kwargs:
                # Default bounds for weights: between 0 and 1
                kwargs['bounds'] = [(0.0, 1.0)] * n_experts
        
        # Create optimizer with updated parameters
        optimizer = self.optimizer_factory.create_optimizer(optimizer_type, **kwargs)
        
        logger.info(f"Created {optimizer_type} optimizer with parameters: {kwargs}")
        return optimizer
    
    def fit(self, X: pd.DataFrame, y: np.ndarray = None, context: pd.DataFrame = None, 
             expert_predictions: Dict[str, np.ndarray] = None, 
             expert_targets: Dict[str, np.ndarray] = None, 
             meta_context: Dict[str, Any] = None) -> None:
        """
        Fit the Meta_Learner to the data for adaptive expert weighting.
        
        Args:
            X: Feature data
            y: Target values (optional)
            context: Context features (optional)
            expert_predictions: Dictionary mapping expert IDs to predictions (optional)
            expert_targets: Dictionary mapping expert IDs to target weights (optional)
            meta_context: Additional contextual information for Meta_Learner (optional)
        """
        # Extract context from the data
        extracted_context = self.extract_context(X)
        
        # Add context features if provided
        if context is not None:
            if hasattr(context, 'empty') and not context.empty:
                # Handle DataFrame context
                for col in context.columns:
                    extracted_context[col] = context[col].mean()
            elif isinstance(context, dict):
                # Handle dictionary context
                extracted_context.update(context)
        
        # Add additional context if provided
        if meta_context:
            extracted_context.update(meta_context)
        
        # If expert_targets is not provided but expert_predictions and y are, calculate targets
        if expert_targets is None and expert_predictions is not None and y is not None:
            expert_targets = {}
            for expert_id, predictions in expert_predictions.items():
                # Calculate a simple accuracy-based weight
                error = np.mean(np.abs(predictions - y))
                # Convert error to a weight (lower error = higher weight)
                if error > 0:
                    expert_targets[expert_id] = 1.0 / error
                else:
                    expert_targets[expert_id] = 1.0
            
            # Normalize weights to sum to 1
            total_weight = sum(expert_targets.values())
            if total_weight > 0:
                for expert_id in expert_targets:
                    expert_targets[expert_id] /= total_weight
        
        # Calculate performance metrics for each expert if available
        for expert_id, expert_info in self.expert_registry.items():
            expert = expert_info['expert']
            
            # Skip if expert is not fitted
            if not hasattr(expert, 'is_fitted') or not expert.is_fitted:
                continue
            
            # Get expert predictions
            try:
                predictions = expert_predictions.get(expert_id) if expert_predictions else expert.predict(X)
                
                # Calculate performance metrics
                # If we have targets and y, use y as the target
                # If we have expert_targets but no y, use expert_targets
                # If we have neither, create dummy metrics
                metrics = {}
                
                if y is not None and expert_id in expert_predictions:
                    # Use actual target values if available
                    target = y
                    if 'rmse' in self.performance_metrics:
                        metrics['rmse'] = np.sqrt(np.mean((target - predictions) ** 2))
                    if 'mae' in self.performance_metrics:
                        metrics['mae'] = np.mean(np.abs(target - predictions))
                    if 'r2' in self.performance_metrics:
                        # Calculate R² score
                        ss_tot = np.sum((target - np.mean(target)) ** 2)
                        ss_res = np.sum((target - predictions) ** 2)
                        metrics['r2'] = 1 - (ss_res / ss_tot if ss_tot > 0 else 0)
                elif expert_id in expert_targets:
                    # Use expert targets if available
                    target = expert_targets[expert_id]
                    if 'rmse' in self.performance_metrics:
                        metrics['rmse'] = np.sqrt(np.mean((target - predictions) ** 2))
                    if 'mae' in self.performance_metrics:
                        metrics['mae'] = np.mean(np.abs(target - predictions))
                    if 'r2' in self.performance_metrics:
                        # Calculate R² score
                        ss_tot = np.sum((target - np.mean(target)) ** 2)
                        ss_res = np.sum((target - predictions) ** 2)
                        metrics['r2'] = 1 - (ss_res / ss_tot if ss_tot > 0 else 0)
                else:
                    # Create dummy metrics if no targets available
                    # This ensures performance_history is populated
                    metrics = {'dummy_metric': 1.0}
                
                # Update performance metrics
                self.update_performance(expert_id, metrics, extracted_context)
            except Exception as e:
                logger.warning(f"Error calculating performance for expert {expert_id}: {str(e)}")
                # Even on error, add a dummy entry to performance history
                self.expert_registry[expert_id]['performance_history'].append({
                    'metrics': {'error': str(e)},
                    'context': extracted_context
                })
        
        # Convert context dictionary to features array for Meta_Learner
        context_features = np.array(list(extracted_context.values())).reshape(1, -1)
        
        # Mark as fitted
        self.is_fitted = True
        
        logger.info("Meta_Learner gating fitted successfully")
        
        # Convert expert_targets to a suitable format for Meta_Learner
        # Use the mean of expert predictions as the target for the meta-learner
        if expert_targets:
            meta_targets = np.mean(list(expert_targets.values()), axis=0)
        else:
            # Fallback if no expert targets
            meta_targets = np.zeros(context_features.shape[0])
            
        # Fit the Meta_Learner with the prepared data
        self.meta_learner.fit(context_features, meta_targets)
        
        logger.info("Meta_Learner fitted for adaptive expert weighting")
    
    def optimize_weights(self, 
                        X: pd.DataFrame, 
                        expert_targets: Dict[str, np.ndarray],
                        meta_context: Dict[str, Any] = None,
                        optimizer_config: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Optimize expert weights using the Meta_Learner and selected optimizer.
        
        Args:
            X: Feature data
            expert_targets: Dictionary mapping expert IDs to target weights
            meta_context: Additional contextual information for Meta_Learner
            optimizer_config: Configuration for the optimizer
            
        Returns:
            Dictionary with optimization results
        """
        # Extract context
        context = self.extract_context(X)
        
        # Add additional context if provided
        if meta_context:
            context.update(meta_context)
        
        # Determine problem characteristics for optimizer selection
        problem_characteristics = {
            'dimensions': len(self.expert_registry),
            'data_size': len(X),
            'feature_count': len(X.columns),
            'expert_count': len(self.expert_registry),
            'context': context
        }
        
        # Select and create optimizer
        optimizer_type = optimizer_config.get('optimizer_type') if optimizer_config else None
        optimizer = self.create_optimizer(
            optimizer_type=optimizer_type,
            problem_characteristics=problem_characteristics,
            **(optimizer_config or {})
        )
        
        # Prepare evaluation function for the optimizer
        def evaluate_weights(weights):
            # Apply weights to expert predictions
            expert_predictions = {}
            for expert_id, expert_info in self.expert_registry.items():
                expert = expert_info['expert']
                expert_predictions[expert_id] = expert.predict(X)
            
            # Create weight dictionary
            expert_ids = list(self.expert_registry.keys())
            weight_dict = {expert_id: weights[i] for i, expert_id in enumerate(expert_ids)}
            
            # Calculate combined prediction
            combined = np.zeros(len(X))
            for expert_id, expert_weight in weight_dict.items():
                combined += expert_weight * expert_predictions[expert_id]
            
            # Calculate error (e.g., MSE)
            y_true = np.array(list(expert_targets.values())).mean(axis=0)  # Simple average as target
            mse = np.mean((y_true - combined) ** 2)
            
            return mse
        
        # Run optimization
        result = optimizer.run(evaluate_weights)
        
        # Update Meta_Learner with optimization results
        if 'best_weights' in result:
            expert_ids = list(self.expert_registry.keys())
            optimized_weights = {
                expert_id: result['best_weights'][i]
                for i, expert_id in enumerate(expert_ids)
            }
            
            # Update Meta_Learner with optimized weights
            self.meta_learner.update_weights(context, optimized_weights)
            
            logger.info(f"Optimization completed with score: {result.get('best_score')}")
        else:
            logger.warning("Optimization did not produce valid weights")
        
        return result
    
    def save(self, filepath: str) -> None:
        """
        Save the Meta_Learner gating integration to disk.
        
        Args:
            filepath: Path to save the model
        """
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Prepare the object for serialization
        # Note: We don't save the expert models themselves, just their IDs
        expert_ids = list(self.expert_registry.keys())
        
        save_dict = {
            'context_features': self.context_features,
            'performance_metrics': self.performance_metrics,
            'expert_ids': expert_ids,
            'expert_metadata': {
                expert_id: {
                    'type': info['type'],
                    'metadata': info['metadata'],
                    'performance_history': info['performance_history']
                }
                for expert_id, info in self.expert_registry.items()
            }
        }
        
        # Save Meta_Learner separately
        meta_learner_path = os.path.join(os.path.dirname(filepath), 'meta_learner.pkl')
        self.meta_learner.save(meta_learner_path)
        
        # Save to disk
        with open(filepath, 'wb') as f:
            pickle.dump(save_dict, f)
        
        logger.info(f"Meta_Learner gating integration saved to {filepath}")
    
    @classmethod
    def load(cls, 
            filepath: str, 
            experts: Dict[str, BaseExpert] = None,
            meta_learner_path: str = None) -> 'MetaLearnerGating':
        """
        Load a Meta_Learner gating integration from disk.
        
        Args:
            filepath: Path to load the model from
            experts: Dictionary of expert models to register
            meta_learner_path: Path to load the Meta_Learner from (if None, will look in same directory)
            
        Returns:
            Loaded Meta_Learner gating integration
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model file not found: {filepath}")
        
        # Load from disk
        with open(filepath, 'rb') as f:
            save_dict = pickle.load(f)
        
        # Load Meta_Learner
        if meta_learner_path is None:
            meta_learner_path = os.path.join(os.path.dirname(filepath), 'meta_learner.pkl')
        
        meta_learner = MetaLearner.load(meta_learner_path) if os.path.exists(meta_learner_path) else None
        
        # Create a new instance
        instance = cls(
            meta_learner=meta_learner,
            context_features=save_dict['context_features'],
            performance_metrics=save_dict['performance_metrics']
        )
        
        # Register experts if provided
        if experts:
            for expert_id, expert in experts.items():
                if expert_id in save_dict['expert_ids']:
                    metadata = save_dict['expert_metadata'][expert_id]['metadata']
                    instance.register_expert(expert_id, expert, metadata)
                    
                    # Restore performance history
                    instance.expert_registry[expert_id]['performance_history'] = \
                        save_dict['expert_metadata'][expert_id]['performance_history']
        
        logger.info(f"Loaded Meta_Learner gating integration from {filepath}")
        return instance
