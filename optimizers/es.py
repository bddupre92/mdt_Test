"""
es.py
-----
Enhanced Evolution Strategy optimizer with self-adaptive parameters
and improved mutation control.
"""

from typing import Tuple, List, Callable, Dict, Any, Optional
import numpy as np
from .base_optimizer import BaseOptimizer
import time

class EvolutionStrategyOptimizer(BaseOptimizer):
    def __init__(self,
                 dim: int,
                 bounds: List[Tuple[float, float]],
                 population_size: int = 100,
                 max_evals: int = 10000,
                 adaptive: bool = True,
                 min_sigma: float = 1e-8,
                 max_sigma: float = 1.0,
                 **kwargs):
        """
        Initialize ES optimizer with self-adaptive parameters.
        
        Args:
            dim: Problem dimensionality
            bounds: Parameter bounds
            population_size: Population size (lambda)
            max_evals: Maximum function evaluations
            adaptive: Whether to use parameter adaptation
            min_sigma: Minimum step size
            max_sigma: Maximum step size
        """
        super().__init__(dim, bounds, population_size, max_evals=max_evals,
                        adaptive=adaptive, **kwargs)
        
        # ES-specific parameters
        self.mu = self.population_size // 4  # Parent population size
        self.sigma0 = 0.3  # Initial step size
        self.tau = 1.0 / np.sqrt(2.0 * self.dim)  # Learning rate for sigma
        self.tau_prime = 1.0 / np.sqrt(2.0 * np.sqrt(self.dim))  # Overall learning rate
        self.min_sigma = min_sigma
        self.max_sigma = max_sigma
        
        # Initialize parent population
        self.parents = np.array([
            self._random_solution()
            for _ in range(self.mu)
        ])
        self.parent_sigmas = np.full((self.mu, self.dim), self.sigma0)
        
        # Adaptive parameters
        if adaptive:
            self.param_history['sigma'] = [self.sigma0]
            self.param_history['success_rate'] = []
    
    def _mutate(self, parent: np.ndarray, sigma: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Mutate parent solution using uncorrelated mutation with n step sizes.
        
        Args:
            parent: Parent solution
            sigma: Step sizes for each dimension
            
        Returns:
            Tuple of (mutated_solution, new_sigmas)
        """
        # Update step sizes
        r = np.random.normal(0, 1)
        new_sigmas = sigma * np.exp(self.tau_prime * r + self.tau * np.random.normal(0, 1, self.dim))
        new_sigmas = np.clip(new_sigmas, self.min_sigma, self.max_sigma)
        
        # Create offspring
        offspring = parent + new_sigmas * np.random.normal(0, 1, self.dim)
        offspring = self._clip_to_bounds(offspring)
        
        return offspring, new_sigmas
    
    def _recombine(self, parents: np.ndarray, parent_sigmas: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Perform intermediate recombination of parents.
        
        Args:
            parents: Parent solutions
            parent_sigmas: Parent step sizes
            
        Returns:
            Tuple of (recombined_solution, recombined_sigmas)
        """
        # Select random parents
        indices = np.random.choice(len(parents), size=2, replace=False)
        p1, p2 = parents[indices]
        s1, s2 = parent_sigmas[indices]
        
        # Intermediate recombination
        child = (p1 + p2) / 2.0
        sigma = (s1 + s2) / 2.0
        
        return child, sigma
    
    def optimize(self,
                objective_func: Callable,
                context: Optional[Dict[str, Any]] = None) -> Tuple[np.ndarray, float]:
        """
        Run the ES optimization process.
        
        Args:
            objective_func: Function to minimize
            context: Optional problem context
            
        Returns:
            Tuple of (best_solution, best_score)
        """
        start_time = time.time()
        
        # Evaluate initial parent population
        parent_scores = np.array([
            objective_func(p) for p in self.parents
        ])
        self.state.evaluations += self.mu
        
        # Track best solution
        best_idx = np.argmin(parent_scores)
        self._update_state(
            self.parents[best_idx],
            parent_scores[best_idx]
        )
        
        # Main optimization loop
        while not self._check_convergence():
            self.state.generation += 1
            
            # Create and evaluate offspring
            offspring = []
            offspring_sigmas = []
            offspring_scores = []
            
            for _ in range(self.population_size):
                # Recombine parents
                child, sigma = self._recombine(self.parents, self.parent_sigmas)
                
                # Mutate child
                mutant, new_sigma = self._mutate(child, sigma)
                
                # Evaluate
                score = objective_func(mutant)
                self.state.evaluations += 1
                
                offspring.append(mutant)
                offspring_sigmas.append(new_sigma)
                offspring_scores.append(score)
            
            # Convert to arrays
            offspring = np.array(offspring)
            offspring_sigmas = np.array(offspring_sigmas)
            offspring_scores = np.array(offspring_scores)
            
            # Update best solution
            best_offspring_idx = np.argmin(offspring_scores)
            if offspring_scores[best_offspring_idx] < self.state.best_score:
                self._update_state(
                    offspring[best_offspring_idx],
                    offspring_scores[best_offspring_idx]
                )
            
            # Select new parent population (mu,lambda)
            selected = np.argsort(offspring_scores)[:self.mu]
            self.parents = offspring[selected]
            self.parent_sigmas = offspring_sigmas[selected]
            parent_scores = offspring_scores[selected]
            
            # Adapt parameters
            if self.adaptive:
                success_rate = np.mean(self.success_history)
                self.param_history['success_rate'].append(success_rate)
                
                # Adapt global step size based on 1/5 success rule
                if success_rate < 0.2:
                    self.sigma0 = max(self.min_sigma, self.sigma0 * 0.85)
                elif success_rate > 0.2:
                    self.sigma0 = min(self.max_sigma, self.sigma0 * 1.15)
                    
                self.param_history['sigma'].append(self.sigma0)
        
        self.state.runtime = time.time() - start_time
        return self.state.best_solution, self.state.best_score
