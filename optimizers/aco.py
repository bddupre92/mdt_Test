"""
aco.py
------
Ant Colony Optimization (ACO) for continuous optimization.
"""

import numpy as np
from typing import Dict, Any, Optional, List, Tuple, Callable
from .base_optimizer import BaseOptimizer
import time

class AntColonyOptimizer(BaseOptimizer):
    def __init__(self,
                 dim: int,
                 bounds: List[Tuple[float, float]],
                 population_size: int = 50,
                 max_evals: int = 10000,
                 adaptive: bool = True,
                 evaporation_rate: float = 0.1,
                 intensification: float = 2.0,
                 alpha: float = 1.0,
                 beta: float = 2.0,
                 grid_points: int = 10,
                 tau_init: float = 1.0,
                 rho: float = 0.1,
                 q0: float = 0.5,
                 **kwargs):
        """
        Initialize ACO optimizer.
        
        Args:
            dim: Problem dimensionality
            bounds: List of (lower, upper) bounds for each dimension
            population_size: Number of ants
            max_evals: Maximum function evaluations
            adaptive: Whether to use parameter adaptation
            evaporation_rate: Pheromone evaporation rate
            intensification: Pheromone intensification factor
            alpha: Pheromone influence
            beta: Heuristic influence
            grid_points: Number of discrete regions per dimension
            tau_init: Initial pheromone value
            rho: Pheromone evaporation rate
            q0: Exploration-exploitation trade-off
        """
        super().__init__(dim=dim, bounds=bounds, population_size=population_size,
                        max_evals=max_evals, adaptive=adaptive)
        
        # ACO parameters
        self.evaporation_rate = evaporation_rate
        self.intensification = intensification
        self.alpha = alpha
        self.beta = beta
        self.grid_points = grid_points
        self.tau_init = tau_init
        self.rho = rho
        self.q0 = q0
        
        # Initialize pheromone matrix
        self.pheromone = np.full((dim, grid_points), tau_init)
        
        # Initialize population
        self.population = self._init_population()
        self.population_scores = np.full(population_size, np.inf)
        
        # Success tracking
        self.success_history = np.zeros(20)  # Track last 20 iterations
        self.success_idx = 0
        
        # Adaptive parameters
        if adaptive:
            self.param_history = {
                'rho': [rho],
                'q0': [q0],
                'success_rate': [],
                'diversity': []
            }
        else:
            self.param_history = {
                'diversity': []
            }
    
    def reset(self):
        """Reset optimizer state"""
        super().reset()
        
        # Initialize population
        self.population = self._init_population()
        self.population_scores = np.full(self.population_size, np.inf)
        
        # Initialize pheromone matrix
        self.pheromone = np.full((self.dim, self.grid_points), self.tau_init)
        
        # Initialize adaptive parameters
        if self.adaptive:
            self.param_history = {
                'rho': [self.rho],
                'q0': [self.q0],
                'success_rate': [],
                'diversity': []
            }
        else:
            self.param_history = {
                'diversity': []
            }
    
    def _discretize(self, x: np.ndarray) -> np.ndarray:
        """Convert continuous solution to discrete regions"""
        regions = np.zeros(self.dim, dtype=int)
        for i, (lower, upper) in enumerate(self.bounds):
            regions[i] = int((x[i] - lower) / (upper - lower) * self.grid_points)
            regions[i] = np.clip(regions[i], 0, self.grid_points - 1)
        return regions
    
    def _continuous_value(self, region: int, dim: int) -> float:
        """Convert discrete region to continuous value"""
        lower, upper = self.bounds[dim]
        region_size = (upper - lower) / self.grid_points
        return lower + (region + 0.5) * region_size
    
    def _update_pheromone(self):
        """Update pheromone levels based on solution quality"""
        # Evaporation
        self.pheromone *= (1 - self.rho)
        
        # Intensification
        for i in range(self.population_size):
            regions = self._discretize(self.population[i])
            delta = self.intensification / (self.population_scores[i] + 1e-10)
            for j, region in enumerate(regions):
                self.pheromone[j, region] += delta
    
    def _select_region(self, dim: int) -> int:
        """Select region using pheromone levels"""
        p = self.pheromone[dim] ** self.alpha
        p /= np.sum(p)
        return np.random.choice(self.grid_points, p=p)
    
    def _construct_solution(self) -> np.ndarray:
        """Construct new solution using pheromone information"""
        solution = np.zeros(self.dim)
        for i in range(self.dim):
            region = self._select_region(i)
            solution[i] = self._continuous_value(region, i)
            # Add small random perturbation
            lower, upper = self.bounds[i]
            perturbation = np.random.normal(0, (upper - lower) / (4 * self.grid_points))
            solution[i] = np.clip(solution[i] + perturbation, lower, upper)
        return solution
    
    def _update_parameters(self):
        """Update optimizer parameters based on performance"""
        if not self.adaptive:
            return
            
        # Calculate success rate
        success_rate = np.mean(self.success_history)
        
        # Update rho based on success rate
        if success_rate > 0.5:
            self.rho *= 0.9  # Decrease evaporation to focus on exploitation
        else:
            self.rho *= 1.1  # Increase evaporation to encourage exploration
            
        # Update q0 based on success rate
        if success_rate > 0.5:
            self.q0 = min(0.9, self.q0 * 1.1)  # Increase exploitation
        else:
            self.q0 = max(0.1, self.q0 * 0.9)  # Increase exploration
            
        # Keep parameters within reasonable bounds
        self.rho = np.clip(self.rho, 0.01, 0.5)
        
        # Record parameter values
        self.param_history['rho'].append(self.rho)
        self.param_history['q0'].append(self.q0)
        self.param_history['success_rate'].append(success_rate)
    
    def _optimize(self, objective_func: Callable, context: Optional[Dict[str, Any]] = None) -> np.ndarray:
        """Run ACO optimization"""
        # Evaluate initial population
        for i in range(self.population_size):
            self.population_scores[i] = self._evaluate(self.population[i], objective_func)
        
        # Track initial diversity
        self._update_diversity()
        
        while not self._check_convergence():
            # Generate new solutions
            new_solutions = np.zeros((self.population_size, self.dim))
            new_scores = np.zeros(self.population_size)
            
            for i in range(self.population_size):
                # Construct solution using pheromone information
                solution = self._construct_solution()
                new_solutions[i] = solution
                
                # Evaluate solution
                new_scores[i] = self._evaluate(solution, objective_func)
            
            # Update population
            combined_solutions = np.vstack((self.population, new_solutions))
            combined_scores = np.concatenate((self.population_scores, new_scores))
            
            # Select best solutions
            best_indices = np.argsort(combined_scores)[:self.population_size]
            self.population = combined_solutions[best_indices]
            self.population_scores = combined_scores[best_indices]
            
            # Update pheromone trails
            self._update_pheromone()
            
            # Update parameters
            if self.adaptive:
                self._update_parameters()
            
            # Track diversity
            self._update_diversity()
        
        return self.best_solution
    
    def optimize(self, objective_func: Callable,
                max_evals: Optional[int] = None,
                record_history: bool = True,
                context: Optional[Dict[str, Any]] = None) -> np.ndarray:
        """
        Run ACO optimization.
        
        Args:
            objective_func: Function to minimize
            max_evals: Maximum number of function evaluations (overrides init value)
            record_history: Whether to record convergence and parameter history
            context: Optional problem context
            
        Returns:
            Best solution found
        """
        # Update max_evals if provided
        if max_evals is not None:
            self.max_evals = max_evals
            
        self.start_time = time.time()
        
        # Initialize pheromone matrix
        self.pheromone = np.ones((self.dim, self.grid_points)) * self.tau_init
        
        while not self._check_convergence():
            # Generate solutions
            solutions = []
            scores = []
            
            for ant in range(self.population_size):
                # Construct solution
                solution = self._construct_solution()
                
                # Evaluate solution
                score = self._evaluate(solution, objective_func)
                
                solutions.append(solution)
                scores.append(score)
                
            # Convert to numpy arrays
            solutions = np.array(solutions)
            scores = np.array(scores)
            
            # Update population
            combined_solutions = np.vstack((self.population, solutions))
            combined_scores = np.concatenate((self.population_scores, scores))
            
            # Select best solutions
            best_indices = np.argsort(combined_scores)[:self.population_size]
            self.population = combined_solutions[best_indices]
            self.population_scores = combined_scores[best_indices]
            
            # Update pheromone trails
            self._update_pheromone()
            
            # Update parameters
            if self.adaptive:
                self._update_parameters()
            
            # Track diversity
            self._update_diversity()
        
        self.end_time = time.time()
        return self.best_solution
