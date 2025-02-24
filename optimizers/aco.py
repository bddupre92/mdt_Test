"""
aco.py
------
Ant Colony Optimization for continuous domains.
"""

import numpy as np
from typing import Tuple, Callable, List
from .base_optimizer import BaseOptimizer

class AntColonyOptimizer(BaseOptimizer):
    def __init__(self, dim: int, bounds: List[Tuple[float, float]], 
                 population_size: int = 50, num_generations: int = 100,
                 alpha: float = 1.0, beta: float = 2.0, rho: float = 0.1,
                 Q: float = 1.0):
        """
        Initialize ACO optimizer.
        
        :param dim: Number of dimensions
        :param bounds: List of (lower, upper) bounds for each dimension
        :param population_size: Number of ants
        :param num_generations: Number of iterations
        :param alpha: Pheromone importance factor
        :param beta: Heuristic importance factor
        :param rho: Pheromone evaporation rate
        :param Q: Pheromone deposit factor
        """
        super().__init__(dim, bounds)
        self.num_ants = population_size
        self.max_iter = num_generations
        self.alpha = alpha
        self.beta = beta
        self.rho = rho
        self.Q = Q
        
        # Initialize pheromone trails
        self.n_points = 100  # Discretization points per dimension
        self.pheromone = np.ones((self.dim, self.n_points)) / self.n_points
        self.points = np.array([
            np.linspace(low, high, self.n_points)
            for low, high in bounds
        ])
    
    def _continuous_to_discrete(self, x: np.ndarray) -> np.ndarray:
        """Convert continuous solution to discrete indices"""
        indices = []
        for i, (low, high) in enumerate(self.bounds):
            idx = int((x[i] - low) / (high - low) * (self.n_points - 1))
            idx = np.clip(idx, 0, self.n_points - 1)
            indices.append(idx)
        return np.array(indices)
    
    def _discrete_to_continuous(self, indices: np.ndarray) -> np.ndarray:
        """Convert discrete indices to continuous solution"""
        return np.array([
            self.points[i, idx]
            for i, idx in enumerate(indices)
        ])
    
    def optimize(self, objective_func: Callable) -> Tuple[np.ndarray, float]:
        """
        Run ACO optimization.
        
        :param objective_func: Function to minimize
        :return: (best_solution, best_score)
        """
        best_solution = None
        best_score = float('inf')
        
        for iteration in range(self.max_iter):
            # Generate solutions for all ants
            solutions = []
            scores = []
            
            for ant in range(self.num_ants):
                # Build solution using pheromone trails
                solution_indices = np.zeros(self.dim, dtype=int)
                for d in range(self.dim):
                    # Calculate probabilities
                    p = self.pheromone[d] ** self.alpha
                    p /= np.sum(p)
                    
                    # Select point
                    solution_indices[d] = np.random.choice(self.n_points, p=p)
                
                # Convert to continuous and evaluate
                solution = self._discrete_to_continuous(solution_indices)
                score = objective_func(solution)
                
                solutions.append(solution)
                scores.append(score)
                
                # Update best solution
                if score < best_score:
                    best_score = score
                    best_solution = solution.copy()
            
            # Update pheromone trails
            self.pheromone *= (1 - self.rho)  # Evaporation
            
            # Add new pheromone
            for solution, score in zip(solutions, scores):
                indices = self._continuous_to_discrete(solution)
                delta = self.Q / (score + 1e-10)  # Avoid division by zero
                for d, idx in enumerate(indices):
                    self.pheromone[d, idx] += delta
        
        return best_solution, best_score
