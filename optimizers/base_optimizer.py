"""
base_optimizer.py
-----------------
Base class for optimization algorithms with common functionality
and adaptive parameter management.
"""

from abc import ABC, abstractmethod
from typing import Tuple, List, Dict, Any, Optional, Callable
import numpy as np
import time

class OptimizerState:
    """Track optimizer state during optimization"""
    def __init__(self):
        self.best_solution = None
        self.best_score = float('inf')
        self.generation = 0
        self.evaluations = 0
        self.runtime = 0.0
        self.history = []

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
        self.state = OptimizerState()
        
        # Success tracking (last 20 iterations)
        self.success_history = np.zeros(20)
        self.success_idx = 0
        
        # Parameter history for adaptive variants
        self.param_history = {}
    
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
        return (self.state.evaluations >= self.max_evals or
                self.state.best_score <= 1e-8)
    
    def _update_state(self, solution: np.ndarray, score: float) -> None:
        """Update optimizer state with new solution"""
        improved = score < self.state.best_score
        
        if improved:
            self.state.best_solution = solution.copy()
            self.state.best_score = score
            self.state.history.append((self.state.evaluations, score))
            
            # Update success history
            self.success_history[self.success_idx] = 1.0
        else:
            self.success_history[self.success_idx] = 0.0
            
        self.success_idx = (self.success_idx + 1) % len(self.success_history)
    
    def get_state(self) -> OptimizerState:
        """Return current optimization state"""
        return self.state
    
    def get_parameter_history(self) -> Dict[str, List[float]]:
        """Return history of parameter adaptations"""
        return self.param_history
    
    def get_convergence_curve(self) -> List[Tuple[int, float]]:
        """Return the convergence history"""
        return self.state.history
    
    @abstractmethod
    def optimize(self,
                objective_func: Callable,
                context: Optional[Dict[str, Any]] = None) -> Tuple[np.ndarray, float]:
        """
        Run optimization process.
        
        Args:
            objective_func: Function to minimize
            context: Optional problem context
            
        Returns:
            Tuple of (best_solution, best_score)
        """
        pass
