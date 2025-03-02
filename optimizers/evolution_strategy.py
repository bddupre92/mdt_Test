"""Evolution Strategy optimizer implementation."""
from typing import Dict, List, Tuple, Optional, Any, Callable
import numpy as np
import logging
from .base_optimizer import BaseOptimizer


class EvolutionStrategyOptimizer(BaseOptimizer):
    """Evolution Strategy (μ + λ) optimizer."""
    
    def __init__(self, dim: int, bounds: List[Tuple[float, float]], 
                 population_size: Optional[int] = None,
                 mu: Optional[int] = None,
                 sigma: float = 0.1,
                 adaptive: bool = True):
        """
        Initialize ES optimizer.
        
        Args:
            dim: Number of dimensions
            bounds: List of (min, max) bounds for each dimension
            population_size: Optional population size (λ)
            mu: Parent population size (μ)
            sigma: Initial step size
            adaptive: Whether to use adaptive parameters
        """
        super().__init__(dim, bounds, population_size, adaptive)
        
        # ES-specific parameters
        self.mu = mu or self.population_size // 4
        self.sigma = sigma
        self.sigma_history = []
        
        # Success-based step size adaptation
        self.success_threshold = 0.2
        self.adaptation_speed = 0.2
        
        # Log ES-specific parameters
        self.logger.debug(f"mu: {self.mu}, sigma: {self.sigma}")
        
    def _iterate(self):
        """Perform one iteration of ES."""
        # Generate offspring
        offspring = np.zeros((self.population_size, self.dim))
        offspring_fitness = np.zeros(self.population_size)
        
        for i in range(self.population_size):
            # Select random parent
            parent_idx = np.random.randint(self.mu)
            parent = self.population[parent_idx]
            
            # Generate offspring with Gaussian mutation
            child = parent + self.sigma * np.random.randn(self.dim)
            
            # Ensure within bounds
            for j in range(self.dim):
                if child[j] < self.bounds[j][0]:
                    child[j] = self.bounds[j][0]
                elif child[j] > self.bounds[j][1]:
                    child[j] = self.bounds[j][1]
                    
            # Evaluate offspring
            offspring[i] = child
            offspring_fitness[i] = self.objective_func(child)
            self.evaluations += 1
            
            # Update best solution
            if offspring_fitness[i] < self.best_score:
                self.best_score = offspring_fitness[i]
                self.best_solution = child.copy()
                self.success_history.append(True)
            else:
                self.success_history.append(False)
        
        # Combine parents and offspring
        combined = np.vstack([self.population[:self.mu], offspring])
        combined_fitness = np.concatenate([
            [self.objective_func(x) for x in self.population[:self.mu]],
            offspring_fitness
        ])
        self.evaluations += self.mu
        
        # Select best mu individuals as new parents
        indices = np.argsort(combined_fitness)[:self.mu]
        self.population[:self.mu] = combined[indices]
        
        # Update convergence curve
        self.convergence_curve.append(self.best_score)
        
        # Adaptive parameter update
        if self.adaptive:
            self._update_parameters()
            
        self._current_iteration += 1
        
    def _update_parameters(self):
        """Update ES control parameters."""
        if len(self.success_history) < 10:
            return
            
        # Calculate success rate over last 10 iterations
        recent_success = np.mean(self.success_history[-10:])
        
        # Update step size based on success rate
        if recent_success < self.success_threshold:
            self.sigma *= (1 - self.adaptation_speed)  # Decrease step size
        else:
            self.sigma *= (1 + self.adaptation_speed)  # Increase step size
            
        # Keep sigma in reasonable bounds relative to search space
        max_range = max(b[1] - b[0] for b in self.bounds)
        self.sigma = np.clip(self.sigma, 1e-8 * max_range, 0.5 * max_range)
        
        # Record parameter history
        self.sigma_history.append(self.sigma)
