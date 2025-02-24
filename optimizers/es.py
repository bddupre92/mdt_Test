"""
es.py
-----
Evolution Strategy optimizer implementation.
"""

import numpy as np
from typing import Tuple, Callable, List
from .base_optimizer import BaseOptimizer

class EvolutionStrategy(BaseOptimizer):
    def __init__(self, dim: int, bounds: List[Tuple[float, float]], 
                 population_size: int = 50, num_generations: int = 100,
                 mutation_rate: float = 0.1, crossover_rate: float = 0.7):
        """
        Initialize ES optimizer.
        
        :param dim: Number of dimensions
        :param bounds: List of (lower, upper) bounds for each dimension
        :param population_size: Size of population
        :param num_generations: Number of generations to run
        :param mutation_rate: Probability of mutation
        :param crossover_rate: Probability of crossover
        """
        super().__init__(dim, bounds)
        self.population_size = population_size
        self.num_generations = num_generations
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        
        # Strategy parameters
        self.step_size = np.array([(high - low) / 10 for low, high in bounds])
    
    def _mutate(self, individual: np.ndarray) -> np.ndarray:
        """Apply mutation to an individual"""
        mutation = np.random.normal(0, self.step_size, self.dim)
        mutation *= np.random.random(self.dim) < self.mutation_rate
        return self._clip_to_bounds(individual + mutation)
    
    def _crossover(self, parent1: np.ndarray, parent2: np.ndarray) -> np.ndarray:
        """Apply crossover between two parents"""
        if np.random.random() < self.crossover_rate:
            # Uniform crossover
            mask = np.random.random(self.dim) < 0.5
            child = np.where(mask, parent1, parent2)
            return self._clip_to_bounds(child)
        return parent1.copy()
    
    def optimize(self, objective_func: Callable) -> Tuple[np.ndarray, float]:
        """
        Run ES optimization.
        
        :param objective_func: Function to minimize
        :return: (best_solution, best_score)
        """
        # Initialize population
        population = np.array([self._random_solution() for _ in range(self.population_size)])
        scores = np.array([objective_func(ind) for ind in population])
        
        best_solution = population[np.argmin(scores)].copy()
        best_score = np.min(scores)
        
        for generation in range(self.num_generations):
            # Selection (tournament selection)
            offspring = []
            for _ in range(self.population_size):
                # Tournament selection
                idx1, idx2 = np.random.randint(0, self.population_size, 2)
                parent1 = population[idx1] if scores[idx1] < scores[idx2] else population[idx2]
                
                idx1, idx2 = np.random.randint(0, self.population_size, 2)
                parent2 = population[idx1] if scores[idx1] < scores[idx2] else population[idx2]
                
                # Create offspring through crossover and mutation
                child = self._crossover(parent1, parent2)
                child = self._mutate(child)
                offspring.append(child)
            
            # Replace population with offspring
            population = np.array(offspring)
            scores = np.array([objective_func(ind) for ind in population])
            
            # Update best solution
            min_idx = np.argmin(scores)
            if scores[min_idx] < best_score:
                best_score = scores[min_idx]
                best_solution = population[min_idx].copy()
            
            # Adapt step sizes using 1/5th rule
            success_rate = np.mean(scores < np.mean(scores))
            if success_rate > 0.2:
                self.step_size *= 1.1
            elif success_rate < 0.2:
                self.step_size *= 0.9
        
        return best_solution, best_score
