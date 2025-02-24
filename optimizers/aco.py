"""
aco.py
-------
Enhanced Ant Colony Optimization with adaptive parameters
and improved pheromone management for continuous optimization.
"""

from typing import Tuple, List, Callable, Dict, Any, Optional
import numpy as np
from .base_optimizer import BaseOptimizer
import time

class AntColonyOptimizer(BaseOptimizer):
    def __init__(self,
                 dim: int,
                 bounds: List[Tuple[float, float]],
                 population_size: int = 50,
                 archive_size: int = 10,
                 q: float = 0.1,
                 xi: float = 0.1,
                 adaptive: bool = True,
                 **kwargs):
        """
        Initialize ACO optimizer with adaptive parameters.
        
        Args:
            dim: Problem dimensionality
            bounds: Parameter bounds
            population_size: Number of ants
            archive_size: Size of solution archive
            q: Initial search localization factor
            xi: Pheromone evaporation rate
            adaptive: Whether to use parameter adaptation
        """
        super().__init__(dim, bounds, population_size, adaptive=adaptive, **kwargs)
        
        # ACO-specific parameters
        self.archive_size = archive_size
        self.q = q  # Search localization (diversity control)
        self.xi = xi  # Pheromone evaporation rate
        
        # Initialize solution archive
        self.archive = np.array([
            self._random_solution()
            for _ in range(self.archive_size)
        ])
        self.archive_scores = np.full(self.archive_size, np.inf)
        
        # Adaptive parameters
        if adaptive:
            self.param_history['q'] = [q]
            self.param_history['xi'] = [xi]
            
            # Learning rates
            self.lr_q = 0.1
            self.lr_xi = 0.05
            
            # Diversity thresholds
            self.min_diversity = 0.1
            self.max_diversity = 0.5
    
    def _calculate_weights(self) -> np.ndarray:
        """Calculate selection weights for archive solutions"""
        ranks = np.argsort(np.argsort(self.archive_scores))
        return 1.0 / (self.q * self.archive_size * np.sqrt(2 * np.pi)) * \
               np.exp(-0.5 * ((ranks / (self.q * self.archive_size)) ** 2))
    
    def _sample_gaussian(self, weights: np.ndarray) -> np.ndarray:
        """Generate new solution using Gaussian sampling"""
        # Select solution using weights
        selected_idx = np.random.choice(
            self.archive_size,
            p=weights / np.sum(weights)
        )
        selected = self.archive[selected_idx]
        
        # Calculate standard deviations
        dists = np.abs(self.archive - selected)
        sigma = self.xi * np.sum(dists * weights.reshape(-1, 1), axis=0) / np.sum(weights)
        
        # Generate new solution
        return self._clip_to_bounds(
            selected + sigma * np.random.normal(0, 1, self.dim)
        )
    
    def _calculate_diversity(self) -> float:
        """Calculate diversity of the solution archive"""
        centroid = np.mean(self.archive, axis=0)
        distances = np.sqrt(np.sum((self.archive - centroid)**2, axis=1))
        return np.mean(distances) / np.sqrt(self.dim)
    
    def _adapt_parameters(self) -> None:
        """
        Adapt q and xi parameters based on search progress
        and archive diversity.
        """
        if not self.adaptive:
            return
            
        if len(self.success_history) >= 10:
            success_rate = sum(self.success_history[-10:]) / 10
            diversity = self._calculate_diversity()
            
            # Adapt search localization (q)
            if diversity < self.min_diversity:
                # Increase exploration
                self.q = min(0.5, self.q * (1 + self.lr_q))
            elif diversity > self.max_diversity:
                # Increase exploitation
                self.q = max(0.01, self.q * (1 - self.lr_q))
            
            # Adapt pheromone evaporation (xi)
            if success_rate < 0.2:
                # Increase exploration
                self.xi = min(0.3, self.xi * (1 + self.lr_xi))
            else:
                # Increase exploitation
                self.xi = max(0.05, self.xi * (1 - self.lr_xi))
            
            # Store parameter history
            self.param_history['q'].append(self.q)
            self.param_history['xi'].append(self.xi)
    
    def optimize(self,
                objective_func: Callable,
                context: Optional[Dict[str, Any]] = None) -> Tuple[np.ndarray, float]:
        """
        Run the ACO optimization process.
        
        Args:
            objective_func: Function to minimize
            context: Optional problem context
            
        Returns:
            Tuple of (best_solution, best_score)
        """
        start_time = time.time()
        
        # Initialize archive
        self.archive_scores = np.array([
            objective_func(sol) for sol in self.archive
        ])
        self.state.evaluations += self.archive_size
        
        # Track best solution
        best_idx = np.argmin(self.archive_scores)
        self._update_state(
            self.archive[best_idx],
            self.archive_scores[best_idx]
        )
        
        # Main optimization loop
        while not self._check_convergence():
            self.state.generation += 1
            
            # Generate new solutions
            weights = self._calculate_weights()
            new_solutions = np.array([
                self._sample_gaussian(weights)
                for _ in range(self.population_size)
            ])
            
            # Evaluate new solutions
            new_scores = np.array([
                objective_func(sol) for sol in new_solutions
            ])
            self.state.evaluations += self.population_size
            
            # Update archive
            combined = np.vstack((self.archive, new_solutions))
            combined_scores = np.concatenate((self.archive_scores, new_scores))
            
            # Select best solutions for archive
            selected = np.argsort(combined_scores)[:self.archive_size]
            self.archive = combined[selected]
            self.archive_scores = combined_scores[selected]
            
            # Update best solution
            if self.archive_scores[0] < self.state.best_score:
                self._update_state(
                    self.archive[0],
                    self.archive_scores[0]
                )
            
            # Adapt parameters
            if self.adaptive:
                self._adapt_parameters()
        
        self.state.runtime = time.time() - start_time
        return self.state.best_solution, self.state.best_score
