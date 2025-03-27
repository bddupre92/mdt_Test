"""
Expert-Optimizer Integration Module

This module provides integration between expert models and specialized optimizers
in the MoE framework. It handles the connection between expert types and their
optimized hyperparameters, evaluation functions, and early stopping criteria.
"""

import logging
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Union, Callable
import warnings

from moe_framework.experts.base_expert import BaseExpert
from meta_optimizer.optimizers.optimizer_factory import OptimizerFactory
from meta_optimizer.optimizers.base_optimizer import BaseOptimizer
from meta_optimizer.evaluation.expert_evaluation_functions import create_evaluation_function

# Configure logging
logger = logging.getLogger(__name__)

class ExpertOptimizerIntegration:
    """
    Class to integrate experts with their specialized optimizers.
    
    This class serves as a bridge between expert models and optimization algorithms,
    configuring each optimizer according to the specific needs of each expert domain.
    
    Attributes:
        expert: The expert model to be optimized
        optimizer_factory: Factory for creating optimizers
        optimizer: The optimizer instance for this expert
        evaluation_function: Function to evaluate optimization results
    """
    
    def __init__(self, expert: BaseExpert, optimizer_factory: Optional[OptimizerFactory] = None):
        """
        Initialize the expert-optimizer integration.
        
        Args:
            expert: The expert model to be optimized
            optimizer_factory: Factory for creating optimizers (created if None)
        """
        self.expert = expert
        self.optimizer_factory = optimizer_factory or OptimizerFactory()
        self.optimizer = None
        self.evaluation_function = None
        self.early_stopping_config = {}
        
        # Determine expert type from class name
        expert_type = self._get_expert_type()
        
        # Set up optimizer and evaluation function for this expert type
        self._setup_optimizer(expert_type)
        self._setup_evaluation_function(expert_type)
    
    def _get_expert_type(self) -> str:
        """
        Determine the expert type from the expert class name.
        
        Returns:
            String representing the expert type
        """
        class_name = self.expert.__class__.__name__.lower()
        
        if 'physiological' in class_name:
            return 'physiological'
        elif 'environmental' in class_name:
            return 'environmental'
        elif 'behavioral' in class_name:
            return 'behavioral'
        elif 'medication' in class_name:
            return 'medication_history'
        else:
            warnings.warn(f"Unknown expert type: {class_name}. Using default configuration.")
            return 'physiological'  # Default to a sensible choice if unknown
    
    def _setup_optimizer(self, expert_type: str) -> None:
        """
        Set up the optimizer for this expert type.
        
        Args:
            expert_type: Type of expert
        """
        try:
            # Get problem characteristics for this expert type
            problem_chars = self.optimizer_factory.get_problem_characterization(expert_type)
            
            # Get default hyperparameter space for this expert type to determine dimensions and bounds
            default_params = self.optimizer_factory._get_default_parameters(expert_type)
            
            # Extract hyperparameter space information from the expert if available
            hyperparameter_space = getattr(self.expert, 'hyperparameter_space', None)
            
            # Create extra params dict with required dim and bounds parameters
            extra_params = {}
            
            # Set appropriate 'dim' and 'bounds' based on the expert type
            if expert_type == 'physiological':
                # Default values if not available from the expert
                extra_params['dim'] = 4
                extra_params['bounds'] = [(50, 200), (5, 20), (2, 20), (1, 10)]
            elif expert_type == 'environmental':
                extra_params['dim'] = 4
                extra_params['bounds'] = [(50, 200), (0.01, 0.3), (3, 10), (2, 20)]
            elif expert_type == 'behavioral':
                extra_params['dim'] = 4
                extra_params['bounds'] = [(50, 200), (1, 10), (2, 20), (1, 10)]
            elif expert_type == 'medication_history':
                extra_params['dim'] = 5
                extra_params['bounds'] = [(50, 200), (0.01, 0.3), (3, 15), (1, 20), (0, 10)]
            
            # Override with values from the expert if available
            if hyperparameter_space and hasattr(hyperparameter_space, 'bounds'):
                extra_params['bounds'] = hyperparameter_space.bounds
                extra_params['dim'] = len(hyperparameter_space.bounds)
            
            # Attempt to create evaluation function for this expert type
            try:
                self.evaluation_function = create_evaluation_function(expert_type)
                logger.info(f"Using evaluation function '{self.evaluation_function.__name__}' for {expert_type} expert")
                
                # Add fitness_function to the parameters for optimizers that need it
                if expert_type == 'medication_history':
                    extra_params['fitness_function'] = self.evaluation_function
                    
                # Create optimizer with characteristics-based configuration
                self.optimizer = self.optimizer_factory.create_expert_optimizer(expert_type, **extra_params)
            except ValueError as e:
                # Use a fallback approach if the specialized evaluation function is not available
                logger.warning(f"Error setting up specialized optimizer: {str(e)}. Using default optimizer.")
                
                # Make sure to include essential parameters
                default_params = {
                    'dim': extra_params.get('dim', 4),  # Default dimension if not already specified
                    'bounds': extra_params.get('bounds', [(0, 1)] * 4),  # Default bounds if not already specified
                }
                
                # Merge with any optimizer-specific parameters
                default_params.update(extra_params)
                
                # Use differential evolution as a fallback optimizer
                self.optimizer = self.optimizer_factory.create_optimizer('differential_evolution', **default_params)
            
            # Extract early stopping configuration
            for key, value in default_params.items():
                if key.startswith('early_stopping_'):
                    self.early_stopping_config[key.replace('early_stopping_', '')] = value
            
            logger.info(f"Created optimizer for {expert_type} expert: {self.optimizer.__class__.__name__}")
            
        except (ValueError, KeyError) as e:
            # Fall back to a default optimizer if there's an error
            logger.warning(f"Error setting up specialized optimizer: {e}. Using default optimizer.")
            self.optimizer = self.optimizer_factory.create_optimizer('differential_evolution')
    
    def _setup_evaluation_function(self, expert_type: str) -> None:
        """
        Set up the evaluation function for this expert type.
        
        Args:
            expert_type: Type of expert
        """
        try:
            # Get recommended evaluation function for this expert type
            eval_func_type = self.optimizer_factory.get_evaluation_function_type(expert_type)
            
            # Create the evaluation function
            self.evaluation_function = create_evaluation_function(eval_func_type)
            
            logger.info(f"Using evaluation function '{eval_func_type}' for {expert_type} expert")
            
        except (ValueError, KeyError) as e:
            # Fall back to MSE if there's an error
            logger.warning(f"Error setting up specialized evaluation function: {e}. Using default MSE.")
            self.evaluation_function = lambda y_true, y_pred: np.mean((y_true - y_pred) ** 2)
    
    def optimize_hyperparameters(self, X: np.ndarray, y: np.ndarray, 
                                param_bounds: Dict[str, Tuple[float, float]],
                                param_types: Optional[Dict[str, str]] = None,
                                max_iterations: int = 30,
                                cv_folds: int = 5) -> Dict[str, Any]:
        """
        Optimize hyperparameters for the expert model.
        
        Args:
            X: Training data features
            y: Training data targets
            param_bounds: Dictionary mapping parameter names to (min, max) bounds
            param_types: Dictionary mapping parameter names to types ('float', 'int', 'categorical')
            max_iterations: Maximum number of optimization iterations
            cv_folds: Number of cross-validation folds
            
        Returns:
            Dictionary of optimized hyperparameters
        """
        from sklearn.model_selection import KFold
        
        # Prepare param_types if not provided
        if param_types is None:
            param_types = {param: 'float' for param in param_bounds}
        
        # Set up bounds for optimizer
        bounds = []
        param_names = []
        for param, (min_val, max_val) in param_bounds.items():
            bounds.append((min_val, max_val))
            param_names.append(param)
        
        # Define fitness function using cross-validation
        def fitness_function(params):
            # Map the params array back to a dictionary
            param_dict = {}
            for i, param in enumerate(param_names):
                value = params[i]
                # Convert to the appropriate type
                if param_types[param] == 'int':
                    value = int(value)
                param_dict[param] = value
            
            # Apply hyperparameters to the expert's model
            for param, value in param_dict.items():
                if hasattr(self.expert.model, param):
                    # Special handling for boolean parameters that may be represented as integers
                    if param in ['bootstrap', 'warm_start', 'verbose'] and param_types[param] == 'int':
                        # Convert integer to boolean (0 -> False, non-zero -> True)
                        value = bool(value)
                    
                    # Handle specific parameter constraints
                    if param == 'max_features' and isinstance(value, float):
                        # Ensure max_features is in the valid range (0.0, 1.0]
                        value = min(1.0, max(0.01, value))
                    elif param == 'min_samples_split':
                        # Ensure min_samples_split is at least 2
                        value = max(2, int(value))
                    elif param == 'min_samples_leaf':
                        # Ensure min_samples_leaf is at least 1
                        value = max(1, int(value))
                    
                    setattr(self.expert.model, param, value)
            
            # Use k-fold cross-validation to evaluate
            kf = KFold(n_splits=cv_folds, shuffle=True, random_state=42)
            scores = []
            
            for train_idx, val_idx in kf.split(X):
                X_train, X_val = X[train_idx], X[val_idx]
                y_train, y_val = y[train_idx], y[val_idx]
                
                # Fit the model
                self.expert.fit(X_train, y_train)
                
                # Predict and evaluate
                y_pred = self.expert.predict(X_val)
                score = self.evaluation_function(y_val, y_pred)
                scores.append(score)
            
            # Return mean score (lower is better for optimization)
            return np.mean(scores)
        
        # Different optimizer classes have different interfaces for configuring the fitness function and bounds
        # Handle different optimizer types
        optimizer_class_name = self.optimizer.__class__.__name__
        
        # Check if the optimizer has set_fitness_function and set_bounds methods
        if hasattr(self.optimizer, 'set_fitness_function') and hasattr(self.optimizer, 'set_bounds'):
            # Use set_fitness_function and set_bounds for optimizers that support it
            self.optimizer.set_fitness_function(fitness_function)
            self.optimizer.set_bounds(bounds)
            best_params, best_fitness = self.optimizer.optimize(max_iterations=max_iterations)
        elif optimizer_class_name == 'DifferentialEvolutionOptimizer':
            # For DifferentialEvolutionOptimizer, the fitness function is provided directly to optimize method
            best_params, best_fitness = self.optimizer.optimize(fitness_function, max_evals=max_iterations * 100)
        elif optimizer_class_name == 'HybridEvolutionaryOptimizer':
            # For HybridEvolutionaryOptimizer, fitness function is already set in constructor
            best_params, best_fitness = self.optimizer.optimize(max_iterations=max_iterations)
        else:
            # Generic fallback - try to find a way to optimize
            logger.warning(f"Using fallback interface for optimizer type: {optimizer_class_name}")
            try:
                # First try: assume optimize takes both the fitness function and max_iterations
                best_params, best_fitness = self.optimizer.optimize(fitness_function, max_iterations=max_iterations)
            except TypeError:
                try:
                    # Second try: assume optimizer has fitness_function set in constructor and uses max_iterations
                    best_params, best_fitness = self.optimizer.optimize(max_iterations=max_iterations)
                except TypeError:
                    try:
                        # Third try: assume it uses max_evals instead of max_iterations
                        best_params, best_fitness = self.optimizer.optimize(fitness_function, max_evals=max_iterations * 100)
                    except TypeError:
                        # Last resort: assume no parameters needed
                        logger.warning(f"All fallback attempts failed, trying with no parameters for {optimizer_class_name}")
                        best_params, best_fitness = self.optimizer.optimize()
        
        # Map best params back to dictionary
        optimized_params = {}
        for i, param in enumerate(param_names):
            value = best_params[i]
            # Convert to the appropriate type
            if param_types[param] == 'int':
                value = int(value)
            # Special handling for boolean parameters that are represented as integers
            if param in ['bootstrap', 'warm_start', 'verbose'] and param_types[param] == 'int':
                # Convert 0/1 to False/True
                value = bool(value)
            optimized_params[param] = value
        
        logger.info(f"Optimized hyperparameters: {optimized_params}, fitness: {best_fitness}")
        return optimized_params
    
    def apply_early_stopping(self, monitor_callback: Callable, 
                            patience: Optional[int] = None, 
                            min_delta: Optional[float] = None) -> Callable:
        """
        Create an early stopping function based on the expert-specific configuration.
        
        Args:
            monitor_callback: Function that returns the current value to monitor
            patience: Number of iterations to wait (uses config if None)
            min_delta: Minimum change to count as improvement (uses config if None)
            
        Returns:
            Early stopping function that returns True when training should stop
        """
        # Use provided values or fall back to configuration
        patience = patience or self.early_stopping_config.get('patience', 5)
        min_delta = min_delta or self.early_stopping_config.get('min_delta', 0.001)
        
        # Initialize tracking variables in closure
        best_value = float('inf')
        counter = 0
        
        def early_stopping_function() -> bool:
            """Check if training should stop based on monitored value"""
            nonlocal best_value, counter
            
            current = monitor_callback()
            
            # Check if improved
            if best_value - current > min_delta:
                best_value = current
                counter = 0
                return False
            else:
                counter += 1
                if counter >= patience:
                    logger.info(f"Early stopping triggered after {counter} iterations without improvement")
                    return True
                return False
        
        return early_stopping_function
