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
            rho: Pheromone evaporation rate
            q0: Exploration-exploitation trade-off
            verbose: Whether to show progress bars
            timeout: Optimization timeout
            iteration_timeout: Iteration timeout
        """
        # Call parent constructor with timeout parameters
        super().__init__(dim=dim, bounds=bounds, population_size=population_size, 
                        adaptive=adaptive, timeout=timeout, 
                        iteration_timeout=iteration_timeout,
                        verbose=verbose)
        
        # Initialize best solution tracking
        self.best_score = float('inf')
        self.best_solution = None
        
        # Store parameters
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
        self.verbose = verbose
        
        # Store bounds
        self.lower_bounds = np.array([b[0] for b in bounds])
        self.upper_bounds = np.array([b[1] for b in bounds])
        
        # Initialize pheromone matrix with proper initialization
        self.pheromone = np.ones((dim, population_size)) * self.initial_pheromone
        
        # Initialize population with proper initialization
        self.population = self._init_population()
        self.scores = np.full(population_size, float('inf'))
        
        # Initialize success tracking
        self.success_window = 20  # Fixed window size for success tracking
        self.success_history = np.zeros(self.success_window, dtype=int)
        self.success_idx = 0
        
        # Initialize convergence history
        self.convergence_curve = []
        self.eval_curve = []
        self.time_curve = []
        self.start_time = None
        
        # Initialize parameter history
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
        # Call parent reset method
        super().reset()
        
        # Reset pheromone matrix
        self.pheromone = np.ones((self.dim, self.population_size)) * self.initial_pheromone
        
        # Reset population with proper initialization
        self.population = self._init_population()
        self.scores = np.full(self.population_size, float('inf'))
        
        # Initialize best solution tracking
        self.best_score = float('inf')
        self.best_solution = np.random.uniform(self.lower_bounds, self.upper_bounds, size=self.dim)
        
        # Reset success tracking
        self.success_history = np.zeros(self.success_window, dtype=int)
        self.success_idx = 0
        
        # Initialize parameter history
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
        for i in range(len(solutions)):
            regions = self._discretize(solutions[i])
            delta = self.intensification / (scores[i] + 1e-10)
            for j, region in enumerate(regions):
                self.pheromone[j, region] += delta
    
    def _select_region(self, dim: int) -> int:
        """Select region using pheromone levels"""
        # Ensure valid pheromone values
        pheromone = np.clip(self.pheromone[dim], self.tau_min, self.tau_max)
        
        # Create heuristic values
        bounds_range = np.linspace(self.bounds[dim][0], self.bounds[dim][1], self.num_points)
        heuristic = 1.0 / (1.0 + np.abs(bounds_range))
        
        # Combine pheromone and heuristic information
        probs = (pheromone ** self.alpha) * (heuristic ** self.beta)
        
        # Handle case where all probabilities are zero
        sum_probs = np.sum(probs)
        if sum_probs <= 0:
            return np.random.randint(0, self.num_points)
        
        probs = probs / sum_probs
        
        # Ensure valid probabilities
        if np.isnan(probs).any() or np.isinf(probs).any():
            return np.random.randint(0, self.num_points)
        
        # Use roulette wheel selection
        try:
            return np.random.choice(self.num_points, p=probs)
        except ValueError:
            # Fallback to random selection if probabilities don't sum to 1
            return np.random.randint(0, self.num_points)
    
    def _construct_solution(self) -> np.ndarray:
        """Construct new solution using pheromone information"""
        # Initialize solution array
        solution = np.zeros(self.dim)
        
        # Ensure pheromone matrix is initialized
        if not hasattr(self, 'pheromone') or self.pheromone is None:
            self.pheromone = np.ones((self.dim, self.population_size)) * self.initial_pheromone
        
        # Construct solution dimension by dimension
        for i in range(self.dim):
            # Select region using pheromone information
            region = self._select_region(i)
            
            # Convert region to continuous value
            solution[i] = self._value_from_index(i, region)
            
            # Ensure solution is within bounds
            solution[i] = min(max(solution[i], self.lower_bounds[i]), self.upper_bounds[i])
        
        return solution
    
    def _iterate(self, objective_func: Callable) -> Tuple[np.ndarray, float]:
        """
        Perform one iteration of the optimization algorithm.
        
        Args:
            objective_func: Objective function to minimize
            
        Returns:
            Tuple of (best solution, best score)
        """
        iter_start_time = time.time()
        
        # Ensure best solution and score are properly initialized
        if not hasattr(self, 'best_score') or self.best_score is None:
            self.best_score = float('inf')
            
        if not hasattr(self, 'best_solution') or self.best_solution is None:
            self.best_solution = np.random.uniform(self.lower_bounds, self.upper_bounds, size=self.dim)
            
        if not hasattr(self, '_current_iteration'):
            self._current_iteration = 0
            
        # Check if pheromone matrix is initialized
        if not hasattr(self, 'pheromone') or self.pheromone is None:
            self.pheromone = np.ones((self.dim, self.population_size)) * self.initial_pheromone
        
        # Generate solutions for each ant
        solutions = []
        scores = []
        
        for ant in range(self.population_size):
            # Check stop criteria
            if hasattr(self, '_check_stop_criteria') and self._check_stop_criteria():
                break
                
            # Check iteration timeout
            if time.time() - iter_start_time > self.iteration_timeout:
                self.logger.warning("Iteration timeout reached")
                break
                
            # Construct solution
            solution = self._construct_solution()
            
            # Evaluate solution
            score = self._evaluate(solution, objective_func)
            
            solutions.append(solution)
            scores.append(score)
            
            # Update best solution if needed
            if score < self.best_score:
                self.best_score = score
                self.best_solution = solution.copy()
                self.last_improvement_iter = self._current_iteration
                self.success_history[self.success_idx] = 1
            else:
                self.success_history[self.success_idx] = 0
                
            self.success_idx = (self.success_idx + 1) % len(self.success_history)
        
        # Update pheromone trails
        if solutions:  # Only update if we have solutions
            self._update_pheromones(solutions, scores)
            
            # Update parameters if adaptive
            if self.adaptive:
                self._update_parameters()
                
            # Track diversity
            self._update_diversity()
        
        # Increment iteration counter
        self._current_iteration += 1
        
        return self.best_solution.copy(), float(self.best_score)
    
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
    
    def optimize(self, objective_func: Callable,
                max_evals: Optional[int] = None,
                record_history: bool = True,
                context: Optional[Dict[str, Any]] = None) -> Tuple[np.ndarray, float]:
        """
        Run optimization.
        
        Args:
            objective_func: Objective function to minimize
            max_evals: Maximum function evaluations (if None, use self.max_evals)
            record_history: Whether to record optimization history
            context: Additional optimization context
            
        Returns:
            Tuple of (best solution, best score)
        """
        # Set parameters
        if max_evals is not None:
            self.max_evals = max_evals
        
        # Early initialization of best_score and best_solution
        if not hasattr(self, 'best_score') or self.best_score is None:
            self.best_score = float('inf')
        
        if not hasattr(self, 'best_solution') or self.best_solution is None:
            self.best_solution = np.random.uniform(self.lower_bounds, self.upper_bounds, size=self.dim)
        
        # Reset evaluation counter
        self.evaluations = 0
        self._current_iteration = 0
        self.start_time = time.time()
        
        # Try initializing the pheromone matrix and population
        if not hasattr(self, 'pheromone') or self.pheromone is None:
            self.pheromone = np.ones((self.dim, self.population_size)) * self.initial_pheromone
        
        if not hasattr(self, 'population') or self.population is None:
            self.population = self._init_population()
            self.scores = np.full(self.population_size, float('inf'))
        
        # Initialize progress bar
        if self.verbose:
            pbar = tqdm(total=self.max_evals, desc=f"ACO Optimization")
            pbar.n = self.evaluations
            pbar.refresh()
        else:
            pbar = None
        
        # Main loop
        try:
            while self.evaluations < self.max_evals:
                # Check stop criteria
                if hasattr(self, '_check_stop_criteria') and self._check_stop_criteria():
                    break
                
                # Iterate once
                solution, score = self._iterate(objective_func)
                
                # Update best solution if better
                if score < self.best_score:
                    self.best_score = score
                    self.best_solution = solution.copy()
                
                # Record history
                if record_history:
                    self._record_history()
                
                # Update progress bar
                if pbar is not None:
                    pbar.n = self.evaluations
                    pbar.refresh()
                
                # Increment iteration counter
                self._current_iteration += 1
        except Exception as e:
            if pbar is not None:
                pbar.close()
            self.logger.error(f"Error during optimization: {e}")
            import traceback
            traceback.print_exc()
            # Ensure we still return something valid
            if not hasattr(self, 'best_solution') or self.best_solution is None:
                self.best_solution = np.random.uniform(self.lower_bounds, self.upper_bounds, size=self.dim)
            if not hasattr(self, 'best_score') or self.best_score is None:
                self.best_score = float('inf')
        finally:
            if pbar is not None:
                pbar.close()
        
        self.end_time = time.time()
        
        # Double-check we're returning valid values
        if isinstance(self.best_solution, np.ndarray) and not np.isnan(self.best_score):
            return self.best_solution.copy(), float(self.best_score)
        else:
            # Return fallback values if something went wrong
            fallback_solution = np.random.uniform(self.lower_bounds, self.upper_bounds, size=self.dim)
            return fallback_solution, float('inf')

    def _optimize(self, objective_func: Callable, context: Optional[Dict[str, Any]] = None) -> Tuple[np.ndarray, float]:
        """Run ACO optimization - deprecated, use optimize instead"""
        return self.optimize(objective_func, context=context)
    
    def _update_diversity(self):
        """Track diversity"""
        diversity = self._calculate_diversity()
        self.param_history['diversity'].append(diversity)
        
    def _calculate_diversity(self) -> float:
        """
        Calculate population diversity as mean distance from centroid.
        
        Returns:
            Diversity measure (mean distance from centroid)
        """
        if len(self.population) == 0:
            return 0.0
            
        # Calculate centroid
        centroid = np.mean(self.population, axis=0)
        
        # Calculate distances from centroid
        distances = np.sqrt(np.sum((self.population - centroid) ** 2, axis=1))
        
        # Return mean distance
        return np.mean(distances)
        
    def _evaluate(self, solution: np.ndarray, objective_func: Callable) -> float:
        """
        Evaluate a solution using the objective function and track evaluations.
        
        Args:
            solution: Solution to evaluate
            objective_func: Objective function to evaluate
            
        Returns:
            Objective function value
        """
        # Initialize attributes if needed
        if not hasattr(self, 'best_score') or self.best_score is None:
            self.best_score = float('inf')
            
        if not hasattr(self, 'best_solution') or self.best_solution is None:
            self.best_solution = np.zeros(self.dim)
            
        if not hasattr(self, 'evaluations'):
            self.evaluations = 0
        
        # Increment evaluation counter
        self.evaluations += 1
        
        # Evaluate solution
        score = objective_func(solution)
        
        # Update best solution if better
        if score < self.best_score:
            self.best_score = score
            self.best_solution = solution.copy()
            
        return score
        
    def get_parameters(self) -> Dict[str, Any]:
        """
        Get optimizer parameters.
        
        Returns:
            Dictionary of optimizer parameters
        """
        params = {
            'name': 'ACO',
            'dim': self.dim,
            'population_size': self.population_size,
            'adaptive': self.adaptive,
            'evaporation_rate': self.evaporation_rate,
            'intensification': self.intensification,
            'alpha': self.alpha,
            'beta': self.beta,
            'num_points': self.num_points,
            'initial_pheromone': self.initial_pheromone,
            'tau_min': self.tau_min,
            'tau_max': self.tau_max,
            'rho': self.rho,
            'q0': self.q0,
            'verbose': self.verbose
        }
        return params

    def _record_history(self):
        """
        Record optimization history.
        """
        # Record best score
        self.convergence_curve.append(self.best_score)
        
        # Record time
        self.time_curve.append(time.time() - self.start_time)
        
        # Record evaluations
        self.eval_curve.append(self.evaluations)

    def _init_population(self) -> np.ndarray:
        """Initialize population with random solutions."""
        # Generate initial population
        population = np.random.uniform(
            low=self.lower_bounds,
            high=self.upper_bounds,
            size=(self.population_size, self.dim)
        )
        
        # Initialize best solution
        self.best_score = float('inf')
        self.best_solution = np.random.uniform(self.lower_bounds, self.upper_bounds, size=self.dim)
        
        # Initialize scores
        self.scores = np.full(self.population_size, float('inf'))
        
        return population
