"""
gwo.py
-------
Grey Wolf Optimizer with adaptive parameters.
"""

import numpy as np
from typing import Dict, Any, Optional, List, Tuple, Callable
from .base_optimizer import BaseOptimizer
import time

class GreyWolfOptimizer(BaseOptimizer):
    def __init__(self,
                 dim: int,
                 bounds: List[Tuple[float, float]],
                 population_size: int = 50,
                 max_evals: int = 10000,
                 adaptive: bool = True,
                 a_init: float = 2.0,
                 **kwargs):
        """
        Initialize GWO optimizer.
        
        Args:
            dim: Problem dimensionality
            bounds: Parameter bounds
            population_size: Population size
            max_evals: Maximum function evaluations
            adaptive: Whether to use parameter adaptation
            a_init: Initial value of a parameter
        """
        super().__init__(dim=dim, bounds=bounds, population_size=population_size,
                        max_evals=max_evals, adaptive=adaptive)
        
        # GWO parameters
        self.a_init = a_init
        self.a = a_init
        
        # Initialize population
        self.population = self._init_population()
        self.population_scores = np.full(population_size, np.inf)
        
        # Initialize wolf hierarchy
        self.alpha = None
        self.beta = None
        self.delta = None
        self.alpha_score = np.inf
        self.beta_score = np.inf
        self.delta_score = np.inf
        
        # Success tracking
        self.success_history = np.zeros(20)  # Track last 20 iterations
        self.success_idx = 0
        
        # Initialize parameter history
        if adaptive:
            self.param_history = {
                'a': [a_init],
                'success_rate': [],
                'diversity': []
            }
        else:
            self.param_history = {
                'diversity': []
            }
    
    def _calculate_diversity(self) -> float:
        """Calculate population diversity"""
        centroid = np.mean(self.population, axis=0)
        distances = np.sqrt(np.sum((self.population - centroid)**2, axis=1))
        return np.mean(distances)
    
    def _update_wolves(self, scores: np.ndarray):
        """Update alpha, beta, and delta wolves"""
        sorted_indices = np.argsort(scores)
        
        # Update alpha
        if scores[sorted_indices[0]] < self.alpha_score:
            self.alpha = self.population[sorted_indices[0]].copy()
            self.alpha_score = scores[sorted_indices[0]]
            self.success_history[self.success_idx] = 1
        else:
            self.success_history[self.success_idx] = 0
        
        # Update beta
        if scores[sorted_indices[1]] < self.beta_score:
            self.beta = self.population[sorted_indices[1]].copy()
            self.beta_score = scores[sorted_indices[1]]
        
        # Update delta
        if scores[sorted_indices[2]] < self.delta_score:
            self.delta = self.population[sorted_indices[2]].copy()
            self.delta_score = scores[sorted_indices[2]]
        
        self.success_idx = (self.success_idx + 1) % len(self.success_history)
    
    def _update_parameters(self):
        """Update optimizer parameters based on performance"""
        if not self.adaptive:
            return
            
        # Calculate success rate
        success_rate = np.mean(self.success_history)
        
        # Update a parameter based on success rate
        if success_rate > 0.5:
            self.a *= 0.9  # Decrease a to focus on exploitation
        else:
            self.a *= 1.1  # Increase a to encourage exploration
            
        # Keep a within reasonable bounds
        self.a = np.clip(self.a, 0.1, self.a_init)
        
        # Record parameter values
        self.param_history['a'].append(self.a)
        self.param_history['success_rate'].append(success_rate)
    
    def _update_diversity(self):
        """Track diversity"""
        diversity = self._calculate_diversity()
        self.param_history['diversity'].append(diversity)
    
    def _calculate_position_update(self, wolf: np.ndarray, leader: np.ndarray) -> np.ndarray:
        """Calculate position update towards a leader"""
        r1 = np.random.rand(self.dim)
        r2 = np.random.rand(self.dim)
        A = 2 * self.a * r1 - self.a
        C = 2 * r2
        
        d_leader = np.abs(C * leader - wolf)
        x_leader = leader - A * d_leader
        return x_leader
    
    def reset(self):
        """Reset optimizer state"""
        super().reset()
        
        # Reset parameters
        self.a = self.a_init
        
        # Initialize population
        self.population = self._init_population()
        self.population_scores = np.full(self.population_size, np.inf)
        
        # Initialize wolf hierarchy
        self.alpha = None
        self.beta = None
        self.delta = None
        self.alpha_score = np.inf
        self.beta_score = np.inf
        self.delta_score = np.inf
        
        # Initialize parameter history
        if self.adaptive:
            self.param_history.update({
                'a': [self.a_init],
                'success_rate': []
            })
        else:
            self.param_history = {
                'diversity': []
            }
            
    def _optimize(self, objective_func: Callable, context: Optional[Dict[str, Any]] = None) -> np.ndarray:
        """Run GWO optimization"""
        # Evaluate initial population
        for i in range(self.population_size):
            self.population_scores[i] = self._evaluate(self.population[i], objective_func)
        
        # Track initial diversity
        self._update_diversity()
        
        # Initialize alpha, beta, and delta wolves
        self._update_wolves(self.population_scores)
        
        while not self._check_convergence():
            # Update a parameter
            if self.adaptive:
                self._update_parameters()
            else:
                self.a = self.a_init * (1 - self.evaluations / self.max_evals)
            
            # Update each wolf's position
            for i in range(self.population_size):
                # Calculate position updates towards alpha, beta, and delta
                X1 = self._calculate_position_update(self.population[i], self.alpha)
                X2 = self._calculate_position_update(self.population[i], self.beta)
                X3 = self._calculate_position_update(self.population[i], self.delta)
                
                # Update position
                self.population[i] = (X1 + X2 + X3) / 3.0
                self.population[i] = self._bound_solution(self.population[i])
                
                # Evaluate new position
                self.population_scores[i] = self._evaluate(self.population[i], objective_func)
            
            # Update wolf hierarchy
            self._update_wolves(self.population_scores)
            
            # Track diversity
            self._update_diversity()
        
        return self.alpha
