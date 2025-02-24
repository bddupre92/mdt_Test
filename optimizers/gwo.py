"""
gwo.py
-------
Enhanced Grey Wolf Optimizer with adaptive parameters and
improved exploration/exploitation balance.
"""

from typing import Tuple, List, Callable, Dict, Any, Optional
import numpy as np
from .base_optimizer import BaseOptimizer
import time

class GreyWolfOptimizer(BaseOptimizer):
    def __init__(self,
                 dim: int,
                 bounds: List[Tuple[float, float]],
                 population_size: int = 30,
                 max_evals: int = 10000,
                 adaptive: bool = True,
                 **kwargs):
        """
        Initialize GWO optimizer with adaptive parameters.
        
        Args:
            dim: Problem dimensionality
            bounds: Parameter bounds
            population_size: Pack size
            max_evals: Maximum function evaluations
            adaptive: Whether to use parameter adaptation
        """
        super().__init__(dim, bounds, population_size, max_evals=max_evals,
                        adaptive=adaptive, **kwargs)
        
        # GWO-specific parameters
        self.a_max = 2.0  # Maximum value of a
        self.a_min = 0.0  # Minimum value of a
        self.a = self.a_max
        
        # Initialize tracking variables
        self.diversity_history = []
        self.success_history = np.zeros(20)  # Track last 20 iterations
        self.success_idx = 0
        
        # Adaptive parameters
        if adaptive:
            self.param_history['a'] = [self.a_max]
            self.stagnation_counter = 0
            self.min_diversity = 0.1
            
            # Learning rates for parameter adaptation
            self.lr_a = 0.1
            self.lr_diversity = 0.2
    
    def _calculate_diversity(self, population: np.ndarray) -> float:
        """Calculate population diversity"""
        centroid = np.mean(population, axis=0)
        distances = np.sqrt(np.sum((population - centroid)**2, axis=1))
        return np.mean(distances) / np.sqrt(self.dim)
    
    def _adapt_parameters(self, evaluations: int) -> None:
        """Adapt GWO parameters based on search progress"""
        if not self.adaptive:
            return
            
        success_rate = np.mean(self.success_history)
        
        # Update a parameter
        if success_rate < 0.2:
            # Increase exploration
            self.a = min(self.a_max, self.a * (1 + self.lr_a))
        else:
            # Increase exploitation
            self.a = max(self.a_min, self.a * (1 - self.lr_a))
        
        # Store parameter history
        self.param_history['a'].append(self.a)
    
    def optimize(self,
                objective_func: Callable,
                context: Optional[Dict[str, Any]] = None) -> Tuple[np.ndarray, float]:
        """
        Run the GWO optimization process.
        
        Args:
            objective_func: Function to minimize
            context: Optional problem context
            
        Returns:
            Tuple of (best_solution, best_score)
        """
        start_time = time.time()
        
        # Initialize population
        population = np.array([
            self._random_solution()
            for _ in range(self.population_size)
        ])
        
        # Evaluate initial population
        scores = np.array([
            objective_func(sol) for sol in population
        ])
        self.state.evaluations += self.population_size
        
        # Initialize alpha, beta, and delta wolves
        sorted_indices = np.argsort(scores)
        alpha_pos = population[sorted_indices[0]].copy()
        beta_pos = population[sorted_indices[1]].copy()
        delta_pos = population[sorted_indices[2]].copy()
        
        alpha_score = scores[sorted_indices[0]]
        self._update_state(alpha_pos, alpha_score)
        
        # Main optimization loop
        while not self._check_convergence():
            self.state.generation += 1
            
            # Calculate diversity
            diversity = self._calculate_diversity(population)
            self.diversity_history.append(diversity)
            
            # Update a linearly from a_max to a_min
            if not self.adaptive:
                self.a = self.a_max - (self.state.evaluations / self.max_evals) * (self.a_max - self.a_min)
            
            # Update each wolf's position
            new_population = np.zeros_like(population)
            for i in range(self.population_size):
                # Calculate A and C vectors
                r1 = np.random.random(self.dim)
                r2 = np.random.random(self.dim)
                
                A1 = 2 * self.a * r1 - self.a
                C1 = 2 * r2
                
                A2 = 2 * self.a * r1 - self.a
                C2 = 2 * r2
                
                A3 = 2 * self.a * r1 - self.a
                C3 = 2 * r2
                
                # Calculate new position
                D_alpha = abs(C1 * alpha_pos - population[i])
                X1 = alpha_pos - A1 * D_alpha
                
                D_beta = abs(C2 * beta_pos - population[i])
                X2 = beta_pos - A2 * D_beta
                
                D_delta = abs(C3 * delta_pos - population[i])
                X3 = delta_pos - A3 * D_delta
                
                new_population[i] = self._clip_to_bounds(
                    (X1 + X2 + X3) / 3.0
                )
            
            # Evaluate new population
            new_scores = np.array([
                objective_func(sol) for sol in new_population
            ])
            self.state.evaluations += self.population_size
            
            # Update success history
            improvement = np.any(new_scores < scores)
            self.success_history[self.success_idx] = float(improvement)
            self.success_idx = (self.success_idx + 1) % len(self.success_history)
            
            # Update population
            population = new_population
            scores = new_scores
            
            # Update alpha, beta, and delta wolves
            sorted_indices = np.argsort(scores)
            if scores[sorted_indices[0]] < self.state.best_score:
                alpha_pos = population[sorted_indices[0]].copy()
                beta_pos = population[sorted_indices[1]].copy()
                delta_pos = population[sorted_indices[2]].copy()
                self._update_state(alpha_pos, scores[sorted_indices[0]])
            
            # Adapt parameters
            if self.adaptive:
                self._adapt_parameters(self.state.evaluations)
        
        self.state.runtime = time.time() - start_time
        return self.state.best_solution, self.state.best_score
