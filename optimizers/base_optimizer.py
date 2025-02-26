"""
base_optimizer.py
-----------------
Base class for optimization algorithms with common functionality
and adaptive parameter management.
"""

import numpy as np
import time
import pandas as pd
from typing import Tuple, List, Dict, Any, Optional, Callable
from abc import ABC, abstractmethod

from meta.optimizer_state import OptimizerState

class BaseOptimizer(ABC):
    def __init__(self,
                 dim: int,
                 bounds: List[Tuple[float, float]],
                 population_size: int = 50,
                 max_evals: int = 10000,
                 adaptive: bool = True,
                 **kwargs):
        """
        Initialize base optimizer.
        
        Args:
            dim: Problem dimensionality
            bounds: Parameter bounds
            population_size: Population size
            max_evals: Maximum function evaluations
            adaptive: Whether to use parameter adaptation
        """
        self.dim = dim
        self.bounds = bounds
        self.population_size = population_size
        self.max_evals = max_evals
        self.adaptive = adaptive
        
        # Initialize state
        self.evaluations = 0
        self.best_solution = None
        self.best_score = float('inf')
        self.convergence_curve = []
        self.diversity_history = []
        self.success_history = np.zeros(20)  # Track last 20 iterations
        self.success_idx = 0
        self.param_history = {'diversity': []}
        self.start_time = None
        self.end_time = None
        self.history = []
        self.name = self.__class__.__name__  # Add name attribute
        
        # Performance tracking
        self.performance_history = pd.DataFrame(columns=['iteration', 'score'])
        self._current_iteration = 0
        
    def reset(self):
        """Reset optimizer state"""
        self.evaluations = 0
        self.best_solution = None
        self.best_score = float('inf')
        self.convergence_curve = []
        self.diversity_history = []
        self.success_history = np.zeros(20)  # Track last 20 iterations
        self.success_idx = 0
        self.param_history = {'diversity': []}
        self.start_time = None
        self.end_time = None
        self.history = []
        self.performance_history = pd.DataFrame(columns=['iteration', 'score'])
        self._current_iteration = 0
        
    def _random_solution(self) -> np.ndarray:
        """Generate random solution within bounds"""
        return np.array([
            np.random.uniform(low, high)
            for low, high in self.bounds
        ])
    
    def _clip_to_bounds(self, x: np.ndarray) -> np.ndarray:
        """Clip solution to bounds"""
        return np.clip(x, 
                      [b[0] for b in self.bounds],
                      [b[1] for b in self.bounds])
    
    def _check_convergence(self) -> bool:
        """Check if optimization should stop"""
        return (self.evaluations >= self.max_evals or
                self.best_score <= 1e-8)
    
    def _update_parameters(self):
        """Update optimizer parameters based on performance"""
        pass  # Implemented by adaptive optimizers
    
    def _evaluate(self, solution: np.ndarray, objective_func: Callable) -> float:
        """Evaluate solution and update state"""
        solution = np.asarray(solution)
        score = objective_func(solution)
        self.evaluations += 1
        
        # Update best solution
        if score < self.best_score:
            self.best_score = score
            self.best_solution = solution.copy()
            self.success_history[self.success_idx] = 1.0
        else:
            self.success_history[self.success_idx] = 0.0
            
        self.success_idx = (self.success_idx + 1) % len(self.success_history)
        
        # Record convergence
        self.convergence_curve.append(self.best_score)
        
        # Update performance history
        new_row = pd.DataFrame({
            'iteration': [self._current_iteration],
            'score': [score]
        })
        self.performance_history = pd.concat([
            self.performance_history,
            new_row
        ], ignore_index=True)
        self._current_iteration += 1
        
        return score
    
    def _update_history(self, score: float):
        """Update performance history.
        
        Args:
            score: Current objective function value
        """
        new_row = pd.DataFrame({
            'iteration': [self._current_iteration],
            'score': [score]
        })
        self.performance_history = pd.concat([
            self.performance_history,
            new_row
        ], ignore_index=True)
        self._current_iteration += 1
    
    def _calculate_diversity(self) -> float:
        """Calculate population diversity"""
        if not hasattr(self, 'population'):
            return 0.0
            
        # Calculate mean distance from centroid
        centroid = np.mean(self.population, axis=0)
        distances = np.sqrt(np.sum((self.population - centroid)**2, axis=1))
        return np.mean(distances)
    
    def _update_diversity(self):
        """Calculate and store population diversity."""
        if self.population is None or len(self.population) < 2:
            diversity = 0.0
        else:
            # Calculate mean pairwise distance
            distances = []
            for i in range(len(self.population)):
                for j in range(i + 1, len(self.population)):
                    dist = np.linalg.norm(self.population[i] - self.population[j])
                    distances.append(dist)
            diversity = np.mean(distances)
        
        self.diversity_history.append(diversity)
        self.param_history['diversity'].append(diversity)
    
    def _init_population(self) -> np.ndarray:
        """Initialize random population within bounds"""
        population = np.zeros((self.population_size, self.dim))
        for i, (lower, upper) in enumerate(self.bounds):
            population[:, i] = np.random.uniform(lower, upper, self.population_size)
        return population
    
    def _bound_solution(self, x: np.ndarray) -> np.ndarray:
        """Ensure solution stays within bounds"""
        x = np.asarray(x)
        # Use numpy's clip function for vectorized bounds checking
        x = np.clip(x, 
                   [b[0] for b in self.bounds],
                   [b[1] for b in self.bounds])
        return x
    
    def get_convergence_curve(self) -> List[float]:
        """Get convergence curve"""
        return self.convergence_curve
    
    def get_state(self) -> OptimizerState:
        """Get current optimizer state"""
        return OptimizerState(
            evaluations=self.evaluations,
            runtime=(self.end_time - self.start_time) if self.end_time else 0,
            history=list(enumerate(self.convergence_curve)),
            success_rate=np.mean(self.success_history),
            diversity_history=self.diversity_history
        )
    
    def get_parameter_history(self) -> Dict[str, List[float]]:
        """Get history of parameter values"""
        return self.param_history
    
    def get_parameters(self) -> Dict[str, Any]:
        """Get current optimizer parameters"""
        params = {
            'population_size': self.population_size,
            'max_evals': self.max_evals,
            'adaptive': self.adaptive,
            'success_rate': np.mean(self.success_history)
        }
        
        # Add any additional parameters from param_history
        for param_name, history in self.param_history.items():
            if history:  # Only add if there's history
                params[param_name] = history[-1]  # Get most recent value
                
        return params
    
    def get_performance_history(self) -> pd.DataFrame:
        """Get performance history"""
        return self.performance_history
    
    def optimize(self,
                objective_func: Callable,
                max_evals: Optional[int] = None,
                record_history: bool = True,
                context: Optional[Dict[str, Any]] = None) -> Tuple[np.ndarray, float]:
        """
        Run optimization process.
        
        Args:
            objective_func: Function to minimize
            max_evals: Maximum number of function evaluations (overrides init value)
            record_history: Whether to record convergence and parameter history
            context: Optional problem context
            
        Returns:
            Tuple of (best solution found as numpy array, best score)
        """
        # Update max_evals if provided
        if max_evals is not None:
            self.max_evals = max_evals
            
        # Start timing
        self.start_time = time.time()
        
        # Run optimization (implemented by subclasses)
        solution, score = self._optimize(objective_func, context)
        
        # End timing
        self.end_time = time.time()
        
        return solution, score
    
    @abstractmethod
    def _optimize(self,
                 objective_func: Callable,
                 context: Optional[Dict[str, Any]] = None) -> Tuple[np.ndarray, float]:
        """
        Internal optimization method to be implemented by subclasses.
        
        Args:
            objective_func: Function to minimize
            context: Optional problem context
            
        Returns:
            Tuple of (best solution found as numpy array, best score)
        """
        pass
