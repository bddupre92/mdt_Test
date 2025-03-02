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
                 evaporation_rate: float = 0.05,
                 intensification: float = 2.5,
                 alpha: float = 1.0,
                 beta: float = 2.0,
                 num_points: int = 20,
                 initial_pheromone: float = 1.0,
                 tau_min: float = 0.01,
                 tau_max: float = 20.0,
                 rho: float = 0.05,
                 q0: float = 0.7,
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
            num_points: Number of discrete regions per dimension
            initial_pheromone: Initial pheromone value
            tau_min: Minimum pheromone value
            tau_max: Maximum pheromone value
            rho: Pheromone evaporation rate
            q0: Exploration-exploitation trade-off
        """
        # Only pass parameters that BaseOptimizer accepts
        super().__init__(dim=dim, bounds=bounds, population_size=population_size, adaptive=adaptive)
        
        # Store max_evals as instance variable
        self.max_evals = max_evals
        
        # ACO parameters
        self.evaporation_rate = evaporation_rate
        self.intensification = intensification
        self.alpha = alpha
        self.beta = beta
        self.num_points = num_points
        self.initial_pheromone = initial_pheromone
        self.tau_min = tau_min
        self.tau_max = tau_max
        self.rho = rho
        self.q0 = q0
        
        # Initialize pheromone matrix
        self.pheromone = np.full((dim, num_points), initial_pheromone)
        
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
        self.pheromone = np.full((self.dim, self.num_points), self.initial_pheromone)
        
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
            regions[i] = int((x[i] - lower) / (upper - lower) * self.num_points)
            regions[i] = np.clip(regions[i], 0, self.num_points - 1)
        return regions
    
    def _value_from_index(self, dim: int, index: int) -> float:
        """Convert discrete region to continuous value"""
        lower, upper = self.bounds[dim]
        region_size = (upper - lower) / self.num_points
        return lower + (index + 0.5) * region_size
    
    def _update_pheromones(self, solutions: List[np.ndarray], scores: List[float]):
        """Update pheromone levels based on solution quality"""
        # Evaporation
        self.pheromone *= (1 - self.rho)
        
        # Intensification
        for i in range(self.population_size):
            regions = self._discretize(solutions[i])
            delta = self.intensification / (scores[i] + 1e-10)
            for j, region in enumerate(regions):
                self.pheromone[j, region] += delta
    
    def _select_region(self, dim: int) -> int:
        """Select region using pheromone levels"""
        pheromone = self.pheromone[dim]
        heuristic = 1.0 / (1.0 + np.abs(np.linspace(self.bounds[dim][0], self.bounds[dim][1], self.num_points)))
        
        # Combine pheromone and heuristic information
        probs = (pheromone ** self.alpha) * (heuristic ** self.beta)
        probs = probs / np.sum(probs)
        
        return np.random.choice(self.num_points, p=probs)
    
    def _construct_solution(self) -> np.ndarray:
        """Construct new solution using pheromone information"""
        solution = np.zeros(self.dim)
        for i in range(self.dim):
            region = self._select_region(i)
            solution[i] = self._value_from_index(i, region)
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
    
    def _optimize(self, objective_func: Callable, context: Optional[Dict[str, Any]] = None) -> Tuple[np.ndarray, float]:
        """Run ACO optimization"""
        # Store objective function for local search
        self.objective_func = objective_func
        
        # Initialize pheromone matrix
        self.pheromone = np.ones((self.dim, self.num_points)) * self.initial_pheromone
        
        # Initialize best solution
        self.best_solution = None
        self.best_score = float('inf')
        
        while not self._check_convergence():
            # Generate solutions for each ant
            solutions = []
            scores = []
            for ant in range(self.population_size):
                # Construct solution
                solution = np.zeros(self.dim)
                for d in range(self.dim):
                    # Calculate probabilities
                    pheromone = self.pheromone[d]
                    heuristic = 1.0 / (1.0 + np.abs(np.linspace(self.bounds[d][0], self.bounds[d][1], self.num_points)))
                    
                    # Combine pheromone and heuristic information
                    probs = (pheromone ** self.alpha) * (heuristic ** self.beta)
                    probs = probs / np.sum(probs)
                    
                    # Select value
                    point_idx = np.random.choice(self.num_points, p=probs)
                    solution[d] = self._value_from_index(d, point_idx)
                    
                    # Local search with probability q0
                    if np.random.random() < self.q0:
                        # Generate small perturbation
                        delta = np.random.normal(0, 0.1)
                        new_val = solution[d] + delta
                        new_val = np.clip(new_val, self.bounds[d][0], self.bounds[d][1])
                        
                        # Keep better solution
                        temp_sol = solution.copy()
                        temp_sol[d] = new_val
                        if objective_func(temp_sol) < objective_func(solution):
                            solution[d] = new_val
                
                # Evaluate solution
                score = self._evaluate(solution, objective_func)
                solutions.append(solution)
                scores.append(score)
                
                # Update best solution
                if score < self.best_score:
                    self.best_solution = solution.copy()
                    self.best_score = score
            
            # Update pheromone trails
            self._update_pheromones(solutions, scores)
            
            # Apply evaporation
            self.pheromone *= (1 - self.evaporation_rate)
            
            # Enforce pheromone bounds
            self.pheromone = np.clip(self.pheromone, self.tau_min, self.tau_max)
            
            # Update parameters if adaptive
            if self.adaptive:
                self._update_parameters()
            
            # Track diversity
            self._update_diversity()
        
        # Clean up
        del self.objective_func
        
        return self.best_solution, self.best_score
    
    def optimize(self, objective_func: Callable,
                max_evals: Optional[int] = None,
                record_history: bool = True,
                context: Optional[Dict[str, Any]] = None) -> Tuple[np.ndarray, float]:
        """
        Run ACO optimization.
        
        Args:
            objective_func: Function to minimize
            max_evals: Maximum number of function evaluations (overrides init value)
            record_history: Whether to record convergence and parameter history
            context: Optional problem context
            
        Returns:
            Best solution found and its score
        """
        # Update max_evals if provided
        if max_evals is not None:
            self.max_evals = max_evals
            
        self.start_time = time.time()
        
        # Initialize pheromone matrix
        self.pheromone = np.ones((self.dim, self.num_points)) * self.initial_pheromone
        
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
            self._update_pheromones(solutions, scores)
            
            # Update parameters
            if self.adaptive:
                self._update_parameters()
            
            # Track diversity
            self._update_diversity()
        
        self.end_time = time.time()
        return self.best_solution, self.best_score
