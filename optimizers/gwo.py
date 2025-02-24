"""
gwo.py
-------
Grey Wolf Optimizer implementation.
"""

import numpy as np
from typing import Tuple, Callable, List
from .base_optimizer import BaseOptimizer

class GreyWolfOptimizer(BaseOptimizer):
    def __init__(self, dim: int, bounds: List[Tuple[float, float]], 
                 population_size: int = 50, num_generations: int = 100):
        """
        Initialize GWO optimizer.
        
        :param dim: Number of dimensions
        :param bounds: List of (lower, upper) bounds for each dimension
        :param population_size: Number of wolves in the pack
        :param num_generations: Number of iterations/generations
        """
        super().__init__(dim, bounds)
        self.num_wolves = population_size
        self.max_iter = num_generations
    
    def optimize(self, objective_func: Callable) -> Tuple[np.ndarray, float]:
        """
        Run GWO optimization.
        
        :param objective_func: Function to minimize
        :return: (best_solution, best_score)
        """
        # Initialize population
        wolves = np.array([self._random_solution() for _ in range(self.num_wolves)])
        scores = np.array([objective_func(wolf) for wolf in wolves])
        
        # Get initial alpha, beta, and delta wolves
        sorted_indices = np.argsort(scores)
        alpha = wolves[sorted_indices[0]].copy()
        beta = wolves[sorted_indices[1]].copy()
        delta = wolves[sorted_indices[2]].copy()
        alpha_score = scores[sorted_indices[0]]
        
        for iteration in range(self.max_iter):
            # Update a (linearly decreased from 2 to 0)
            a = 2 * (1 - iteration / self.max_iter)
            
            for i in range(self.num_wolves):
                for j in range(self.dim):
                    # Calculate hunting parameters
                    r1, r2 = np.random.random(2)
                    A1 = 2 * a * r1 - a
                    C1 = 2 * r2
                    
                    r1, r2 = np.random.random(2)
                    A2 = 2 * a * r1 - a
                    C2 = 2 * r2
                    
                    r1, r2 = np.random.random(2)
                    A3 = 2 * a * r1 - a
                    C3 = 2 * r2
                    
                    # Calculate positions relative to alpha, beta, delta
                    D_alpha = abs(C1 * alpha[j] - wolves[i, j])
                    D_beta = abs(C2 * beta[j] - wolves[i, j])
                    D_delta = abs(C3 * delta[j] - wolves[i, j])
                    
                    X1 = alpha[j] - A1 * D_alpha
                    X2 = beta[j] - A2 * D_beta
                    X3 = delta[j] - A3 * D_delta
                    
                    # Update wolf position
                    wolves[i, j] = (X1 + X2 + X3) / 3
            
            # Clip solutions to bounds
            wolves = np.array([self._clip_to_bounds(wolf) for wolf in wolves])
            
            # Update scores
            scores = np.array([objective_func(wolf) for wolf in wolves])
            
            # Update alpha, beta, and delta
            sorted_indices = np.argsort(scores)
            if scores[sorted_indices[0]] < alpha_score:
                alpha = wolves[sorted_indices[0]].copy()
                alpha_score = scores[sorted_indices[0]]
            beta = wolves[sorted_indices[1]].copy()
            delta = wolves[sorted_indices[2]].copy()
        
        return alpha, alpha_score
