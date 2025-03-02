"""Differential Evolution optimizer implementation."""
from typing import Dict, List, Tuple, Optional, Any, Callable
import numpy as np
import logging
from .base_optimizer import BaseOptimizer


class DifferentialEvolutionOptimizer(BaseOptimizer):
    """Differential Evolution optimizer."""
    
    def __init__(self, dim: int, bounds: List[Tuple[float, float]], 
                 population_size: Optional[int] = None,
                 F: float = 0.8,
                 CR: float = 0.5,
                 adaptive: bool = True):
        """
        Initialize DE optimizer.
        
        Args:
            dim: Number of dimensions
            bounds: List of (min, max) bounds for each dimension
            population_size: Optional population size
            F: Mutation factor
            CR: Crossover rate
            adaptive: Whether to use adaptive parameters
        """
        super().__init__(dim, bounds, population_size, adaptive)
        
        # DE-specific parameters
        self.F = F
        self.CR = CR
        self.F_history = []
        self.CR_history = []
        
        # Log DE-specific parameters
        self.logger = logging.getLogger(__name__)
        self.logger.debug(f"F: {self.F}, CR: {self.CR}")
        
    def _iterate(self):
        """Perform one iteration of DE."""
        for i in range(self.population_size):
            # Select random indices for mutation
            idxs = [idx for idx in range(self.population_size) if idx != i]
            a, b, c = np.random.choice(idxs, 3, replace=False)
            
            # Create mutant vector
            mutant = self.population[a] + self.F * (self.population[b] - self.population[c])
            
            # Ensure mutant is within bounds
            for j in range(self.dim):
                if mutant[j] < self.bounds[j][0]:
                    mutant[j] = self.bounds[j][0]
                elif mutant[j] > self.bounds[j][1]:
                    mutant[j] = self.bounds[j][1]
            
            # Crossover
            trial = np.copy(self.population[i])
            j_rand = np.random.randint(self.dim)
            for j in range(self.dim):
                if np.random.random() < self.CR or j == j_rand:
                    trial[j] = mutant[j]
            
            # Selection
            f_trial = self.objective_func(trial)
            f_target = self.objective_func(self.population[i])
            
            self.evaluations += 2
            
            if f_trial < f_target:
                self.population[i] = trial
                if f_trial < self.best_score:
                    self.best_score = f_trial
                    self.best_solution = trial.copy()
                    self.success_history.append(True)
                else:
                    self.success_history.append(False)
            else:
                self.success_history.append(False)
                
        # Update convergence curve
        self.convergence_curve.append(self.best_score)
        
        # Adaptive parameter update
        if self.adaptive:
            self._update_parameters()
            
        self._current_iteration += 1
        
    def _update_parameters(self):
        """Update DE control parameters."""
        if len(self.success_history) < 10:
            return
            
        # Calculate success rate over last 10 iterations
        recent_success = np.mean(self.success_history[-10:])
        
        # Adjust F based on success rate
        if recent_success < 0.2:
            self.F *= 0.9  # Reduce step size
        elif recent_success > 0.8:
            self.F *= 1.1  # Increase step size
            
        # Keep F in reasonable bounds
        self.F = np.clip(self.F, 0.1, 2.0)
        
        # Adjust CR based on diversity
        if len(self.diversity_history) > 1:
            if self.diversity_history[-1] < 0.1:
                self.CR = min(0.9, self.CR * 1.1)  # Increase mixing
            elif self.diversity_history[-1] > 0.5:
                self.CR = max(0.1, self.CR * 0.9)  # Reduce mixing
                
        # Record parameter history
        self.F_history.append(self.F)
        self.CR_history.append(self.CR)
