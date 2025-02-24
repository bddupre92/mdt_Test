"""
de.py
-----
Differential Evolution optimizer implementation.
"""

import numpy as np
from typing import Tuple, Callable, List
from .base_optimizer import BaseOptimizer

class DifferentialEvolutionOptimizer(BaseOptimizer):
    def __init__(self, dim: int, bounds: List[Tuple[float, float]], 
                 population_size: int = 50, num_generations: int = 100,
                 F: float = 0.8, CR: float = 0.7):
        """
        Initialize DE optimizer.
        
        :param dim: Number of dimensions
        :param bounds: List of (lower, upper) bounds for each dimension
        :param population_size: Size of population
        :param num_generations: Number of generations
        :param F: Mutation factor (typically in [0.5, 1.0])
        :param CR: Crossover rate (typically in [0.5, 1.0])
        """
        super().__init__(dim, bounds)
        self.population_size = population_size
        self.num_generations = num_generations
        self.F = F  # mutation factor
        self.CR = CR  # crossover rate
    
    def _mutation(self, population: np.ndarray, best_idx: int) -> np.ndarray:
        """Apply DE mutation"""
        mutants = np.zeros_like(population)
        
        for i in range(self.population_size):
            # Select three random vectors, different from current
            idxs = [idx for idx in range(self.population_size) if idx != i]
            a, b, c = population[np.random.choice(idxs, 3, replace=False)]
            
            # Create mutant
            mutant = a + self.F * (b - c)
            
            # Clip to bounds
            mutants[i] = self._clip_to_bounds(mutant)
        
        return mutants
    
    def _crossover(self, population: np.ndarray, mutants: np.ndarray) -> np.ndarray:
        """Apply DE crossover"""
        trials = np.zeros_like(population)
        
        for i in range(self.population_size):
            # Ensure at least one dimension is crossed
            j_rand = np.random.randint(self.dim)
            
            # Generate crossover mask
            mask = np.random.random(self.dim) < self.CR
            mask[j_rand] = True
            
            # Create trial vector
            trial = np.where(mask, mutants[i], population[i])
            trials[i] = self._clip_to_bounds(trial)
        
        return trials
    
    def optimize(self, objective_func: Callable) -> Tuple[np.ndarray, float]:
        """
        Run DE optimization.
        
        :param objective_func: Function to minimize
        :return: (best_solution, best_score)
        """
        # Initialize population
        population = np.array([self._random_solution() for _ in range(self.population_size)])
        scores = np.array([objective_func(ind) for ind in population])
        
        best_idx = np.argmin(scores)
        best_solution = population[best_idx].copy()
        best_score = scores[best_idx]
        
        for generation in range(self.num_generations):
            # Create mutants
            mutants = self._mutation(population, best_idx)
            
            # Create trial vectors through crossover
            trials = self._crossover(population, mutants)
            
            # Evaluate trials
            trial_scores = np.array([objective_func(trial) for trial in trials])
            
            # Selection
            improvements = trial_scores < scores
            population[improvements] = trials[improvements]
            scores[improvements] = trial_scores[improvements]
            
            # Update best solution
            gen_best_idx = np.argmin(scores)
            if scores[gen_best_idx] < best_score:
                best_score = scores[gen_best_idx]
                best_solution = population[gen_best_idx].copy()
        
        return best_solution, best_score
