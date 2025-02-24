"""
de.py
-----
Differential Evolution optimizer with adaptive parameters.
"""

import numpy as np
from typing import Dict, Any, Optional, List, Tuple, Callable
from .base_optimizer import BaseOptimizer
import time

class DifferentialEvolutionOptimizer(BaseOptimizer):
    def __init__(self,
                 dim: int,
                 bounds: List[Tuple[float, float]],
                 population_size: int = 50,
                 max_evals: int = 10000,
                 F: float = 0.8,
                 CR: float = 0.5,
                 adaptive: bool = True,
                 **kwargs):
        """
        Initialize DE optimizer.
        
        Args:
            dim: Problem dimensionality
            bounds: Parameter bounds
            population_size: Population size
            max_evals: Maximum function evaluations
            F: Mutation factor
            CR: Crossover rate
            adaptive: Whether to use parameter adaptation
        """
        super().__init__(dim=dim, bounds=bounds, population_size=population_size,
                        max_evals=max_evals, adaptive=adaptive)
        
        # DE parameters
        self.F_init = F
        self.CR_init = CR
        self.F = F
        self.CR = CR
        
        # Initialize population
        self.population = self._init_population()
        self.population_scores = np.full(population_size, np.inf)
        
        # Initialize parameter history
        if adaptive:
            self.param_history = {
                'F': [F],
                'CR': [CR],
                'success_rate': []
            }
        else:
            self.param_history = {}
        
        # Success tracking
        self.success_history = np.zeros(20)  # Track last 20 iterations
        self.success_idx = 0
        
        # Diversity tracking
        self.diversity_history = []
    
    def _mutate(self, target_idx: int) -> np.ndarray:
        """Generate mutant vector using DE mutation"""
        # Select three random distinct vectors
        idxs = [i for i in range(self.population_size) if i != target_idx]
        r1, r2, r3 = np.random.choice(idxs, size=3, replace=False)
        
        # Get best solution index
        best_idx = np.argmin(self.population_scores)
        
        # Generate mutant using current-to-best/1 strategy
        mutant = (self.population[target_idx] + 
                 self.F * (self.population[best_idx] - self.population[target_idx]) +
                 self.F * (self.population[r1] - self.population[r2]))
        
        return self._bound_solution(mutant)
    
    def _crossover(self, target: np.ndarray, mutant: np.ndarray) -> np.ndarray:
        """Perform binomial crossover"""
        mask = np.random.rand(self.dim) <= self.CR
        # Ensure at least one component is taken from mutant
        if not np.any(mask):
            mask[np.random.randint(0, self.dim)] = True
        return np.where(mask, mutant, target)
    
    def _update_parameters(self):
        """Update optimizer parameters based on performance"""
        if not self.adaptive:
            return
            
        # Calculate success rate
        success_rate = np.mean(self.success_history)
        
        # Update F and CR based on success rate
        if success_rate > 0.5:
            self.F *= 0.9  # Decrease F to focus on exploitation
            self.CR *= 0.9  # Decrease CR to focus on exploitation
        else:
            self.F *= 1.1  # Increase F to encourage exploration
            self.CR *= 1.1  # Increase CR to encourage exploration
            
        # Keep parameters within reasonable bounds
        self.F = np.clip(self.F, 0.1, 2.0)
        self.CR = np.clip(self.CR, 0.1, 0.9)
        
        # Record parameter values
        self.param_history['F'].append(self.F)
        self.param_history['CR'].append(self.CR)
        self.param_history['success_rate'].append(success_rate)
    
    def _update_diversity(self):
        """Track population diversity"""
        diversity = np.mean(np.std(self.population, axis=0))
        self.diversity_history.append(diversity)
    
    def reset(self):
        """Reset optimizer state"""
        super().reset()
        
        # Reset parameters
        self.F = self.F_init
        self.CR = self.CR_init
        
        # Reset parameter history
        if self.adaptive:
            self.param_history.update({
                'F': [self.F_init],
                'CR': [self.CR_init],
                'success_rate': []
            })
    
    def _optimize(self, objective_func: Callable, context: Optional[Dict[str, Any]] = None) -> np.ndarray:
        """Run DE optimization"""
        # Evaluate initial population
        for i in range(self.population_size):
            self.population_scores[i] = self._evaluate(self.population[i], objective_func)
        
        # Track initial diversity
        self._update_diversity()
        
        while not self._check_convergence():
            # Generate and evaluate trial solutions
            for i in range(self.population_size):
                # Generate trial solution
                mutant = self._mutate(i)
                trial = self._crossover(self.population[i], mutant)
                
                # Evaluate trial solution
                trial_score = self._evaluate(trial, objective_func)
                
                # Selection
                if trial_score <= self.population_scores[i]:
                    self.population[i] = trial
                    self.population_scores[i] = trial_score
            
            # Update parameters
            if self.adaptive:
                self._update_parameters()
            
            # Track diversity
            self._update_diversity()
        
        return self.best_solution
