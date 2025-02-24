"""
es.py
------
Evolution Strategy optimizer with adaptive parameters.
"""

import numpy as np
from typing import Dict, Any, Optional, List, Tuple, Callable
from .base_optimizer import BaseOptimizer
import time

class EvolutionStrategyOptimizer(BaseOptimizer):
    def __init__(self,
                 dim: int,
                 bounds: List[Tuple[float, float]],
                 population_size: int = 50,
                 max_evals: int = 10000,
                 adaptive: bool = True,
                 offspring_size: int = None,
                 **kwargs):
        """
        Initialize ES optimizer.
        
        Args:
            dim: Problem dimensionality
            bounds: Parameter bounds
            population_size: Population size
            max_evals: Maximum function evaluations
            adaptive: Whether to use parameter adaptation
            offspring_size: Number of offspring (default: population_size)
        """
        super().__init__(dim=dim, bounds=bounds, population_size=population_size,
                        max_evals=max_evals, adaptive=adaptive)
        
        # ES parameters
        self.offspring_size = offspring_size or population_size
        self.strategy_params = np.ones(population_size) * 0.1  # Initial step sizes
        
        # Initialize population
        self.population = self._init_population()
        self.population_scores = np.full(population_size, np.inf)
        
        # Initialize parameter history
        if adaptive:
            self.param_history = {
                'step_size': [0.1],  # Initial mean step size
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
    
    def _generate_offspring(self) -> np.ndarray:
        """Generate offspring using mutation"""
        # Select random parent
        parent_idx = np.random.randint(self.population_size)
        parent = self.population[parent_idx]
        
        # Generate offspring using Gaussian mutation
        mutation = np.random.normal(0, self.strategy_params[parent_idx])
        offspring = parent + mutation
        
        # Bound offspring to constraints
        return self._bound_solution(offspring)
    
    def _update_parameters(self):
        """Update optimizer parameters based on performance"""
        if not self.adaptive:
            return
            
        # Calculate success rate
        success_rate = np.mean(self.success_history)
        
        # Update step sizes based on success rate
        if success_rate > 0.2:
            self.strategy_params *= 1.1  # Increase step sizes
        else:
            self.strategy_params *= 0.9  # Decrease step sizes
            
        # Keep step sizes within reasonable bounds
        self.strategy_params = np.clip(self.strategy_params, 0.01, 1.0)
        
        # Record parameter values
        self.param_history['step_size'].append(np.mean(self.strategy_params))
        self.param_history['success_rate'].append(success_rate)
    
    def _update_diversity(self):
        diversity = self._calculate_diversity()
        if 'diversity' not in self.param_history:
            self.param_history['diversity'] = []
        self.param_history['diversity'].append(diversity)
    
    def reset(self):
        """Reset optimizer state"""
        super().reset()
        
        # Reset parameters
        self.strategy_params = np.ones(self.population_size) * 0.1
        
        # Initialize population
        self.population = self._init_population()
        self.population_scores = np.full(self.population_size, np.inf)
        
        # Initialize parameter history
        if self.adaptive:
            self.param_history = {
                'step_size': [0.1],
                'success_rate': [],
                'diversity': []
            }
        else:
            self.param_history = {
                'diversity': []
            }
    
    def _optimize(self, objective_func: Callable, context: Optional[Dict[str, Any]] = None) -> np.ndarray:
        """Run ES optimization"""
        # Evaluate initial population
        for i in range(self.population_size):
            self.population_scores[i] = self._evaluate(self.population[i], objective_func)
        
        # Track initial diversity
        self._update_diversity()
        
        while not self._check_convergence():
            # Generate offspring
            offspring = np.zeros((self.offspring_size, self.dim))
            offspring_scores = np.zeros(self.offspring_size)
            
            for i in range(self.offspring_size):
                # Generate offspring
                offspring[i] = self._generate_offspring()
                
                # Evaluate offspring
                offspring_scores[i] = self._evaluate(offspring[i], objective_func)
            
            # Selection
            combined_pop = np.vstack((self.population, offspring))
            combined_scores = np.concatenate((self.population_scores, offspring_scores))
            
            # Select best individuals
            best_indices = np.argsort(combined_scores)[:self.population_size]
            self.population = combined_pop[best_indices]
            self.population_scores = combined_scores[best_indices]
            
            # Update parameters
            if self.adaptive:
                self._update_parameters()
            
            # Track diversity
            self._update_diversity()
        
        return self.population[0]
    
    def optimize(self, objective_func, context: Optional[Dict[str, Any]] = None) -> np.ndarray:
        """
        Run ES optimization.
        
        Args:
            objective_func: Function to minimize
            context: Optional problem context
            
        Returns:
            Best solution found
        """
        self.start_time = time.time()
        
        # Initialize population scores
        for i in range(self.population_size):
            self.population_scores[i] = self._evaluate(self.population[i], objective_func)
        
        # Sort initial population
        sort_idx = np.argsort(self.population_scores)
        self.population = self.population[sort_idx]
        self.population_scores = self.population_scores[sort_idx]
        
        # Track initial diversity
        self._update_diversity()
        
        # Track best solution
        self.best_solution = self.population[0].copy()
        self.best_score = self.population_scores[0]
        
        while not self._check_convergence():
            # Generate and evaluate offspring
            offspring = np.zeros((self.offspring_size, self.dim))
            offspring_scores = np.zeros(self.offspring_size)
            
            for i in range(self.offspring_size):
                # Generate offspring
                offspring[i] = self._generate_offspring()
                
                # Evaluate offspring
                offspring_scores[i] = self._evaluate(offspring[i], objective_func)
            
            # Combine parents and offspring
            combined_pop = np.vstack((self.population, offspring))
            combined_scores = np.concatenate((self.population_scores, offspring_scores))
            
            # Sort combined population
            sort_idx = np.argsort(combined_scores)
            self.population = combined_pop[sort_idx]
            self.population_scores = combined_scores[sort_idx]
            
            # Update success history
            if self.population_scores[0] < self.best_score:
                self.success_history[self.success_idx] = 1
                self.best_solution = self.population[0].copy()
                self.best_score = self.population_scores[0]
            else:
                self.success_history[self.success_idx] = 0
            
            self.success_idx = (self.success_idx + 1) % len(self.success_history)
            
            # Update parameters
            if self.adaptive:
                self._update_parameters()
            
            # Track diversity
            self._update_diversity()
        
        self.end_time = time.time()
        return self.best_solution
