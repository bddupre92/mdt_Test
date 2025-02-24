"""
base_optimizer.py
-----------------
Defines a base abstract class for custom optimization algorithms.
"""

from abc import ABC, abstractmethod
from typing import List, Tuple, Callable
import numpy as np

class BaseOptimizer(ABC):
    """
    Abstract base class for an optimization algorithm.
    Subclasses should implement optimize(objective_func, ...).
    """
    def __init__(self, dim: int, bounds: List[Tuple[float, float]]):
        """
        Initialize the optimizer with problem dimensions and bounds.
        
        :param dim: Number of dimensions in the optimization problem
        :param bounds: List of (lower, upper) bounds for each dimension
        """
        self.dim = dim
        self.bounds = bounds
        if len(bounds) != dim:
            raise ValueError(f"Expected {dim} bounds, got {len(bounds)}")
    
    def _clip_to_bounds(self, x: np.ndarray) -> np.ndarray:
        """Clip solution to bounds"""
        return np.clip(x, 
                      [b[0] for b in self.bounds],
                      [b[1] for b in self.bounds])
    
    def _random_solution(self) -> np.ndarray:
        """Generate a random solution within bounds"""
        return np.array([
            np.random.uniform(low, high)
            for low, high in self.bounds
        ])
    
    @abstractmethod
    def optimize(self, objective_func: Callable) -> Tuple[np.ndarray, float]:
        """
        Runs the optimization process on a given objective function.
        
        :param objective_func: A callable f(x) -> float, to minimize
        :return: (best_solution, best_score)
        """
        pass
