"""
aco.py
------
Ant Colony Optimization (ACO) for continuous optimization.
"""

import numpy as np
from typing import Dict, Any, Optional, List, Tuple, Callable
from .base_optimizer import BaseOptimizer
import time
from tqdm.auto import tqdm

class AntColonyOptimizer(BaseOptimizer):
    def __init__(self,
                 dim: int,
                 bounds: List[Tuple[float, float]],
                 population_size: int = 30,
                 max_evals: int = 10000,
                 adaptive: bool = False,
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
                 verbose: bool = False,
                 timeout: float = 60,
                 iteration_timeout: float = 10,
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
            rho: Pheromone evaporation coefficient
            q0: Probability of choosing best path
            verbose: Whether to print progress
            timeout: Maximum time in seconds
            iteration_timeout: Maximum time per iteration in seconds
        """
        super().__init__(
            dim=dim,
            bounds=bounds,
            population_size=population_size,
            adaptive=adaptive,
            timeout=timeout,
            iteration_timeout=iteration_timeout,
            verbose=verbose
        )
        
        # Store ACO-specific parameters
        self.max_evals = max_evals
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
        
        # Initialize pheromone matrix - shape should be (dim, num_points)
        self.pheromone = np.ones((dim, num_points)) * self.initial_pheromone
        
        # Create grid of points for each dimension
        self.grids = []
        for i in range(dim):
            lower, upper = bounds[i]
            self.grids.append(np.linspace(lower, upper, num_points))
        
        # Initialize solutions
        self.solutions = np.zeros((population_size, dim))
        self.scores = np.zeros(population_size)
        
        # Best solution found so far
        self.best_solution = None
        self.best_score = float('inf')
        
        # Initialize history
        self.history = []
        
        # Initialize iterations and evaluations
        self.iterations = 0
        self.evaluations = 0
        
        # Initialize time tracking
        self.start_time = None
        self.end_time = None
        
    def reset(self):
        """Reset the optimizer to its initial state."""
        # Reset pheromone matrix - shape should be (dim, num_points)
        self.pheromone = np.ones((self.dim, self.num_points)) * self.initial_pheromone
        
        # Reset solutions
        self.solutions = np.zeros((self.population_size, self.dim))
        self.scores = np.zeros(self.population_size)
        
        # Reset best solution
        self.best_solution = None
        self.best_score = float('inf')
        
        # Reset history
        self.history = []
        
        # Reset iterations and evaluations
        self.iterations = 0
        self.evaluations = 0
        
        # Reset time tracking
        self.start_time = None
        self.end_time = None
        
    def _select_region(self, ant_idx, dim_idx):
        """
        Select a region for ant to explore based on pheromone and heuristic.
        
        Args:
            ant_idx: Index of the ant
            dim_idx: Index of the dimension
            
        Returns:
            region_idx: Index of the selected region
        """
        # Get available regions
        regions = np.arange(self.num_points)
        
        # If using best-so-far solution as guide, select with probability q0
        if self.best_solution is not None and np.random.random() < self.q0:
            # Find closest region to best solution
            grid = self.grids[dim_idx]
            best_val = self.best_solution[dim_idx]
            region_idx = np.abs(grid - best_val).argmin()
            return region_idx
        
        # Calculate selection probabilities
        pheromone = self.pheromone[dim_idx, :]
        
        # Calculate heuristic information (optional)
        if self.beta > 0:
            # Use distance to best solution as heuristic (if available)
            if self.best_solution is not None:
                best_val = self.best_solution[dim_idx]
                dists = np.abs(self.grids[dim_idx] - best_val)
                heuristic = 1.0 / (1.0 + dists)  # Inverse distance (closer is better)
            else:
                # If no best solution yet, use uniform heuristic
                heuristic = np.ones(self.num_points)
        else:
            heuristic = np.ones(self.num_points)
        
        # Calculate probabilities
        prob = (pheromone ** self.alpha) * (heuristic ** self.beta)
        prob = prob / np.sum(prob)
        
        # Select region based on probabilities
        region_idx = np.random.choice(regions, p=prob)
        
        return region_idx
        
    def _update_pheromones(self):
        """Update pheromone levels based on ant performance."""
        # Evaporate pheromones
        self.pheromone = (1 - self.rho) * self.pheromone
        
        # Add new pheromones based on solution quality
        for i in range(self.population_size):
            solution = self.solutions[i]
            score = self.scores[i]
            
            # Only update if solution is valid (not inf or nan)
            if np.isfinite(score):
                # Calculate pheromone deposit amount (higher for better solutions)
                deposit = self.intensification / (1 + score)
                
                # Find closest region indices for this solution
                for dim in range(self.dim):
                    # Find closest region
                    val = solution[dim]
                    region_idx = np.abs(self.grids[dim] - val).argmin()
                    
                    # Update pheromone
                    self.pheromone[dim, region_idx] += deposit
        
        # Enforce pheromone limits if specified
        if self.tau_min is not None and self.tau_max is not None:
            self.pheromone = np.clip(self.pheromone, self.tau_min, self.tau_max)
            
    def _adapt_parameters(self):
        """Adapt parameters based on search progress."""
        # Example: Increase exploitation as search progresses
        progress = min(1.0, self.evaluations / self.max_evals)
        
        # Increase q0 (exploitation) over time
        self.q0 = 0.5 + 0.4 * progress
        
        # Decrease alpha (reduce pheromone influence) over time
        self.alpha = max(0.5, 1.0 - 0.5 * progress)
        
        # Increase beta (increase heuristic influence) over time
        self.beta = 1.0 + 1.0 * progress
        
    def _iterate(self, objective_func):
        """
        Perform one iteration of the ACO algorithm.
        
        Args:
            objective_func: Function to optimize
            
        Returns:
            Tuple of (best solution, best score)
        """
        # Generate solutions for each ant
        for i in range(self.population_size):
            # Generate new solution
            for j in range(self.dim):
                # Select region
                region_idx = self._select_region(i, j)
                
                # Select point within region (add some randomness)
                grid_val = self.grids[j][region_idx]
                
                # Add random variation within grid cell
                if region_idx < self.num_points - 1:
                    next_val = self.grids[j][region_idx + 1]
                    cell_width = next_val - grid_val
                else:
                    prev_val = self.grids[j][region_idx - 1]
                    cell_width = grid_val - prev_val
                
                # Random value within grid cell vicinity
                rand_offset = (np.random.random() - 0.5) * cell_width
                self.solutions[i, j] = grid_val + rand_offset
            
            # Enforce bounds
            for j in range(self.dim):
                lower, upper = self.bounds[j]
                self.solutions[i, j] = max(lower, min(upper, self.solutions[i, j]))
            
            # Evaluate solution
            self.scores[i] = objective_func(self.solutions[i])
            self.evaluations += 1
            
            # Update best solution
            if self.scores[i] < self.best_score:
                self.best_score = self.scores[i]
                self.best_solution = self.solutions[i].copy()
                self.history.append(self.best_score)
        
        # Update pheromones
        self._update_pheromones()
        
        # Adaptive parameter updates (optional)
        if self.adaptive:
            self._adapt_parameters()
        
        # Increment iteration counter
        self.iterations += 1
        
        return self.best_solution, self.best_score
        
    def optimize(self, objective_func, callback=None):
        """
        Perform ant colony optimization.
        
        Args:
            objective_func: Function to optimize
            callback: Optional callback function called after each iteration
            
        Returns:
            best_solution: Best solution found
            best_score: Best score achieved
        """
        # Record start time
        self.start_time = time.time()
        
        # Initialize pheromone matrix - shape should be (dim, num_points)
        self.pheromone = np.ones((self.dim, self.num_points)) * self.initial_pheromone
        
        # Main optimization loop
        with tqdm(total=self.max_evals, disable=not self.verbose) as pbar:
            while self.evaluations < self.max_evals:
                # Check timeout
                if self.timeout and time.time() - self.start_time > self.timeout:
                    if self.verbose:
                        print(f"Timeout reached after {self.evaluations} evaluations")
                    break
                    
                # Start iteration timer
                iteration_start = time.time()
                
                # Perform one iteration
                best_solution, best_score = self._iterate(objective_func)
                
                # Update best solution
                if best_score < self.best_score:
                    self.best_score = best_score
                    self.best_solution = best_solution.copy()
                    self.history.append(self.best_score)
                    
                    # Call callback if provided
                    if callback:
                        callback(self.best_solution, self.best_score, self.evaluations)
                
                # Update progress bar
                pbar.update(1)
                pbar.set_description(f"Best: {self.best_score:.6f}")
                
                # Check if max evaluations reached
                if self.evaluations >= self.max_evals:
                    break
                
                # Check iteration timeout
                if self.iteration_timeout and time.time() - iteration_start > self.iteration_timeout:
                    if self.verbose:
                        print(f"Iteration timeout reached after {time.time() - iteration_start:.2f}s")
                    break
        
        # Record end time
        self.end_time = time.time()
        
        # Return best solution and score
        return self.best_solution, self.best_score
        
    def get_info(self) -> Dict[str, Any]:
        """Get information about the optimizer state."""
        info = super().get_info()
        info.update({
            'iterations': self.iterations,
            'evaluations': self.evaluations,
            'best_score': self.best_score,
            'best_solution': self.best_solution.tolist() if self.best_solution is not None else None,
            'history': self.history,
            'runtime': self.end_time - self.start_time if self.end_time is not None else None,
            'parameters': {
                'alpha': self.alpha,
                'beta': self.beta,
                'rho': self.rho,
                'q0': self.q0,
                'num_points': self.num_points,
                'population_size': self.population_size
            }
        })
        return info
