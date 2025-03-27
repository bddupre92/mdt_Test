"""
Gating Network Optimizer Module

This module provides optimization capabilities for the gating network using
evolutionary algorithms, particularly the Grey Wolf Optimizer (GWO).
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Any, Optional, Tuple, Union, Callable
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

from meta_optimizer.optimizers.gwo import GreyWolfOptimizer
from moe_framework.gating.gating_network import GatingNetwork
from moe_framework.experts.base_expert import BaseExpert

# Configure logging
logger = logging.getLogger(__name__)


class GatingOptimizer:
    """
    Optimizer for the gating network using evolutionary algorithms.
    
    This class provides methods to optimize the gating network weights using
    the Grey Wolf Optimizer (GWO) algorithm.
    
    Attributes:
        gating_network (GatingNetwork): The gating network to optimize
        optimizer_type (str): Type of optimizer to use (default: 'gwo')
        fitness_metric (str): Metric to use for fitness evaluation (default: 'rmse')
        constraints (Dict[str, float]): Constraints for weight optimization
    """
    
    def __init__(self, 
                 gating_network: GatingNetwork,
                 optimizer_type: str = 'gwo',
                 fitness_metric: str = 'rmse',
                 constraints: Dict[str, float] = None):
        """
        Initialize the gating optimizer.
        
        Args:
            gating_network: The gating network to optimize
            optimizer_type: Type of optimizer to use (default: 'gwo')
            fitness_metric: Metric to use for fitness evaluation (default: 'rmse')
            constraints: Constraints for weight optimization
        """
        self.gating_network = gating_network
        self.optimizer_type = optimizer_type
        self.fitness_metric = fitness_metric
        self.constraints = constraints or {
            'min_weight': 0.0,
            'max_weight': 1.0,
            'sum_weights': 1.0
        }
        
        # Validate gating network
        if not self.gating_network.experts:
            raise ValueError("Gating network must have registered experts")
        
        logger.info(f"Initialized gating optimizer with {optimizer_type} optimizer")
    
    def optimize(self, 
                X: pd.DataFrame, 
                y: pd.Series,
                population_size: int = 30,
                max_iterations: int = 50,
                validation_split: float = 0.2,
                early_stopping: int = 10,
                verbose: bool = True) -> Dict[str, Any]:
        """
        Optimize the gating network weights using evolutionary algorithms.
        
        Args:
            X: Feature data
            y: Target data
            population_size: Size of the optimizer population
            max_iterations: Maximum number of iterations
            validation_split: Fraction of data to use for validation
            early_stopping: Number of iterations with no improvement before stopping
            verbose: Whether to display progress information
            
        Returns:
            Dictionary with optimization results
        """
        # Split data into train and validation sets
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=validation_split, random_state=42
        )
        
        # Get expert predictions for train and validation sets
        train_predictions = self._get_expert_predictions(X_train)
        val_predictions = self._get_expert_predictions(X_val)
        
        # Define the optimization problem
        n_experts = len(self.gating_network.experts)
        bounds = [(self.constraints['min_weight'], self.constraints['max_weight'])] * n_experts
        
        # Create the optimizer
        if self.optimizer_type == 'gwo':
            optimizer = GreyWolfOptimizer(
                dim=n_experts,
                bounds=bounds,
                population_size=population_size,
                max_evals=max_iterations * population_size,
                adaptive=True,
                verbose=verbose
            )
        else:
            raise ValueError(f"Unsupported optimizer type: {self.optimizer_type}")
        
        # Define the fitness function
        def fitness_function(weights):
            # Normalize weights to sum to 1
            weights = np.array(weights)
            weights = weights / np.sum(weights)
            
            # Combine expert predictions using these weights
            combined_predictions = self._combine_with_weights(train_predictions, weights)
            
            # Calculate error
            if self.fitness_metric == 'rmse':
                error = np.sqrt(mean_squared_error(y_train, combined_predictions))
            elif self.fitness_metric == 'mse':
                error = mean_squared_error(y_train, combined_predictions)
            else:
                raise ValueError(f"Unsupported fitness metric: {self.fitness_metric}")
            
            # Return negative error (since optimizers maximize fitness)
            return -error
        
        # Run the optimization
        logger.info(f"Starting {self.optimizer_type} optimization with {population_size} population size")
        result = optimizer.optimize(fitness_function)
        
        # Extract the best weights
        best_weights = np.array(result['best_position'])
        best_weights = best_weights / np.sum(best_weights)
        
        # Update the gating network model with the optimized weights
        self._update_gating_model(X, best_weights)
        
        # Evaluate on validation set
        val_combined = self._combine_with_weights(val_predictions, best_weights)
        val_error = np.sqrt(mean_squared_error(y_val, val_combined))
        
        # Prepare results
        expert_ids = list(self.gating_network.experts.keys())
        weight_dict = {expert_id: weight for expert_id, weight in zip(expert_ids, best_weights)}
        
        optimization_results = {
            'best_weights': weight_dict,
            'best_fitness': -result['best_fitness'],  # Convert back to error
            'validation_error': val_error,
            'iterations': result['iterations'],
            'evaluations': result['evaluations']
        }
        
        logger.info(f"Optimization completed with validation error: {val_error:.4f}")
        return optimization_results
    
    def _get_expert_predictions(self, X: pd.DataFrame) -> Dict[str, np.ndarray]:
        """
        Get predictions from all expert models.
        
        Args:
            X: Feature data
            
        Returns:
            Dictionary mapping expert IDs to prediction arrays
        """
        predictions = {}
        for expert_id, expert in self.gating_network.experts.items():
            predictions[expert_id] = expert.predict(X)
        return predictions
    
    def _combine_with_weights(self, 
                             predictions: Dict[str, np.ndarray], 
                             weights: np.ndarray) -> np.ndarray:
        """
        Combine expert predictions using specified weights.
        
        Args:
            predictions: Dictionary mapping expert IDs to prediction arrays
            weights: Weight array for each expert
            
        Returns:
            Combined predictions
        """
        expert_ids = list(self.gating_network.experts.keys())
        combined = np.zeros(len(next(iter(predictions.values()))))
        
        for i, expert_id in enumerate(expert_ids):
            if expert_id in predictions:
                combined += weights[i] * predictions[expert_id]
        
        return combined
    
    def _update_gating_model(self, X: pd.DataFrame, weights: np.ndarray) -> None:
        """
        Update the gating network model with the optimized weights.
        
        Args:
            X: Feature data
            weights: Optimized weight array
        """
        # This is a simplified implementation - in practice, you might want to
        # use the optimized weights to train a more sophisticated model
        
        # For now, we'll just update the gating network's weight constraints
        # to reflect the optimized weights
        expert_ids = list(self.gating_network.experts.keys())
        
        # Create target weights for each sample (same weights for all samples)
        n_samples = len(X)
        expert_targets = {
            expert_id: np.ones(n_samples) * weights[i]
            for i, expert_id in enumerate(expert_ids)
        }
        
        # Fit the gating network with these targets
        self.gating_network.fit(X, expert_targets)
        
        logger.info("Updated gating network model with optimized weights")
