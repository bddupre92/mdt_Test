"""
gwo.py
-------
Grey Wolf Optimizer with adaptive parameters.
"""

import numpy as np
from typing import Dict, Any, Optional, List, Tuple, Callable
from .base_optimizer import BaseOptimizer
import time
from tqdm.auto import tqdm
import logging

class GreyWolfOptimizer(BaseOptimizer):
    def __init__(self,
                 dim: int,
                 bounds: List[Tuple[float, float]],
                 population_size: int = 30,
                 max_evals: int = 10000,
                 adaptive: bool = False,
                 a_init: float = 2.0,
                 verbose: bool = False,
                 timeout: float = 60,
                 iteration_timeout: float = 10,
                 **kwargs):
        """Initialize Grey Wolf Optimizer."""
        super().__init__(dim=dim, bounds=bounds, population_size=population_size,
                        adaptive=adaptive, timeout=timeout,
                        iteration_timeout=iteration_timeout,
                        verbose=verbose)
        
        # Initialize best solution tracking
        self.best_score = float('inf')
        self.best_solution = None
        
        # Store parameters
        self.max_evals = max_evals
        self.verbose = verbose
        self.a_init = a_init
        self.a = a_init
        
        # Store bounds
        self.lower_bounds = np.array([b[0] for b in bounds])
        self.upper_bounds = np.array([b[1] for b in bounds])
        
        # Initialize wolf hierarchy with proper initialization
        self.alpha_pos = np.zeros(dim)
        self.alpha_score = float('inf')
        self.beta_pos = np.zeros(dim)
        self.beta_score = float('inf')
        self.delta_pos = np.zeros(dim)
        self.delta_score = float('inf')
        
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
        if self.adaptive:
            self.param_history = {
                'a': [self.a_init],
                'success_rate': [],
                'diversity': []
            }
        else:
            self.param_history = {
                'diversity': []
            }
        
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
    
    def _update_diversity(self):
        """
        Track population diversity.
        """
        diversity = self._calculate_diversity()
        self.param_history['diversity'].append(diversity)
    
    def _update_wolves(self, positions: np.ndarray, scores: np.ndarray):
        """
        Update alpha, beta, and delta wolves based on fitness scores.
        
        Args:
            positions: Population of wolves (shape: population_size x dim)
            scores: Fitness scores for each wolf (shape: population_size)
        """
        # Ensure scores are valid for comparison
        valid_scores = ~np.isnan(scores) & ~np.isinf(scores)
        
        if np.any(valid_scores):
            # Get indices of wolves sorted by fitness (lowest to highest)
            sorted_indices = np.argsort(scores)
            
            # Update alpha wolf (best solution)
            if scores[sorted_indices[0]] < self.alpha_score:
                self.alpha_pos = positions[sorted_indices[0]].copy()
                self.alpha_score = float(scores[sorted_indices[0]])
                
                # Update best solution tracking
                if not hasattr(self, 'best_solution') or self.best_solution is None:
                    self.best_solution = self.alpha_pos.copy()
                    self.best_score = self.alpha_score
                elif self.alpha_score < self.best_score:
                    self.best_solution = self.alpha_pos.copy()
                    self.best_score = self.alpha_score
                    
                self.success_history[self.success_idx] = 1
            else:
                self.success_history[self.success_idx] = 0
            
            # Update beta wolf (second-best solution) if there are at least 2 wolves
            if len(sorted_indices) > 1:
                if scores[sorted_indices[1]] < self.beta_score:
                    self.beta_pos = positions[sorted_indices[1]].copy()
                    self.beta_score = float(scores[sorted_indices[1]])
            
            # Update delta wolf (third-best solution) if there are at least 3 wolves
            if len(sorted_indices) > 2:
                if scores[sorted_indices[2]] < self.delta_score:
                    self.delta_pos = positions[sorted_indices[2]].copy()
                    self.delta_score = float(scores[sorted_indices[2]])
            
            # Increment success index
            self.success_idx = (self.success_idx + 1) % len(self.success_history)
    
    def _update_parameters(self):
        """Update optimizer parameters based on performance"""
        if not self.adaptive:
            return
            
        # Calculate success rate
        success_rate = np.mean(self.success_history)
        
        # Update a parameter based on success rate
        if success_rate > 0.5:
            self.a *= 0.9  # Decrease a to focus on exploitation
        else:
            self.a *= 1.1  # Increase a to encourage exploration
            
        # Keep a within reasonable bounds
        self.a = np.clip(self.a, 0.1, self.a_init)
        
        # Record parameter values
        self.param_history['a'].append(self.a)
        self.param_history['success_rate'].append(success_rate)
    
    def _calculate_position_update(self, wolf: np.ndarray, leader: np.ndarray) -> np.ndarray:
        """
        Calculate position update towards a leader wolf.
        
        Args:
            wolf: Current wolf position
            leader: Leader wolf position (alpha, beta, or delta)
            
        Returns:
            Updated position vector
        """
        # Generate random vectors between 0 and 1
        r1 = np.random.random(self.dim)
        r2 = np.random.random(self.dim)
        
        # Calculate vector coefficients
        A = 2 * self.a * r1 - self.a  # Exploration/exploitation control
        C = 2 * r2  # Random weight
        
        # Calculate distance vector
        D = np.abs(C * leader - wolf)
        
        # Calculate position update
        X = leader - A * D
        
        # Apply bounds
        X = np.maximum(np.minimum(X, self.upper_bounds), self.lower_bounds)
        
        return X
    
    def reset(self):
        """Reset optimizer state."""
        # Call parent reset method
        super().reset()
        
        # Reset parameters
        self.a = self.a_init
        
        # Reset wolf hierarchy with proper initialization
        self.alpha_pos = np.random.uniform(self.lower_bounds, self.upper_bounds, size=self.dim)
        self.alpha_score = float('inf')
        self.beta_pos = np.random.uniform(self.lower_bounds, self.upper_bounds, size=self.dim)
        self.beta_score = float('inf')
        self.delta_pos = np.random.uniform(self.lower_bounds, self.upper_bounds, size=self.dim)
        self.delta_score = float('inf')
        
        # Initialize best solution tracking
        self.best_score = float('inf')
        self.best_solution = np.random.uniform(self.lower_bounds, self.upper_bounds, size=self.dim)
        
        # Reset population with proper initialization
        self.population = self._init_population()
        self.scores = np.full(self.population_size, float('inf'))
        
        # Reset success tracking
        self.success_history = np.zeros(self.success_window, dtype=int)
        self.success_idx = 0
        
        # Initialize parameter history
        if self.adaptive:
            self.param_history = {
                'a': [self.a_init],
                'success_rate': [],
                'diversity': []
            }
        else:
            self.param_history = {
                'diversity': []
            }
            
    def _init_population(self) -> np.ndarray:
        """Initialize population with random solutions."""
        population = np.random.uniform(
            low=self.lower_bounds,
            high=self.upper_bounds,
            size=(self.population_size, self.dim)
        )
        
        # Initialize wolf positions with random solutions
        self.alpha_pos = np.random.uniform(self.lower_bounds, self.upper_bounds, size=self.dim)
        self.beta_pos = np.random.uniform(self.lower_bounds, self.upper_bounds, size=self.dim)
        self.delta_pos = np.random.uniform(self.lower_bounds, self.upper_bounds, size=self.dim)
        
        # Initialize wolf scores to infinity
        self.alpha_score = float('inf')
        self.beta_score = float('inf')
        self.delta_score = float('inf')
        
        # Initialize best solution tracking
        self.best_score = float('inf')
        self.best_solution = self.alpha_pos.copy()
        
        return population
    
    def _iterate(self, objective_func: Callable) -> Tuple[np.ndarray, float]:
        """
        Perform one iteration of the GWO algorithm.
        
        Args:
            objective_func: Objective function to minimize
            
        Returns:
            Tuple of (best solution, best score)
        """
        iter_start_time = time.time()
        
        # Ensure proper initialization
        if not hasattr(self, 'best_score') or self.best_score is None:
            self.best_score = float('inf')
        
        if not hasattr(self, 'best_solution') or self.best_solution is None:
            self.best_solution = np.random.uniform(self.lower_bounds, self.upper_bounds, size=self.dim)
        
        # Initialize wolf positions if needed
        if not hasattr(self, 'alpha_pos') or self.alpha_pos is None:
            self.alpha_pos = np.random.uniform(self.lower_bounds, self.upper_bounds, size=self.dim)
            self.alpha_score = float('inf')
        
        if not hasattr(self, 'beta_pos') or self.beta_pos is None:
            self.beta_pos = np.random.uniform(self.lower_bounds, self.upper_bounds, size=self.dim)
            self.beta_score = float('inf')
        
        if not hasattr(self, 'delta_pos') or self.delta_pos is None:
            self.delta_pos = np.random.uniform(self.lower_bounds, self.upper_bounds, size=self.dim)
            self.delta_score = float('inf')
        
        # Update parameter a (decreases linearly from a_init to 0)
        if not hasattr(self, '_current_iteration'):
            self._current_iteration = 0
        
        max_iterations = self.max_evals // self.population_size
        self.a = self.a_init - self.a_init * (self._current_iteration / max_iterations)
        
        # New population and scores
        new_positions = np.zeros((self.population_size, self.dim))
        new_scores = np.zeros(self.population_size)
        
        # For each wolf in the pack
        for i in range(self.population_size):
            # Check stop criteria
            if hasattr(self, '_check_stop_criteria') and self._check_stop_criteria():
                break
            
            # Check iteration timeout
            if time.time() - iter_start_time > self.iteration_timeout:
                self.logger.warning("Iteration timeout reached")
                break
            
            # Current wolf position
            current_wolf = self.population[i].copy()
            
            # Calculate position updates based on alpha, beta, and delta wolves
            pos1 = self._calculate_position_update(current_wolf, self.alpha_pos)
            pos2 = self._calculate_position_update(current_wolf, self.beta_pos)
            pos3 = self._calculate_position_update(current_wolf, self.delta_pos)
            
            # Average the positions
            new_pos = (pos1 + pos2 + pos3) / 3.0
            
            # Apply bounds
            new_pos = np.maximum(np.minimum(new_pos, self.upper_bounds), self.lower_bounds)
            
            # Store new position
            new_positions[i] = new_pos
            
            # Evaluate new position
            new_scores[i] = self._evaluate(new_pos, objective_func)
        
        # Update wolf hierarchy
        self._update_wolves(new_positions, new_scores)
        
        # Update population
        self.population = new_positions.copy()
        self.scores = new_scores.copy()
        
        # Update parameters for adaptive version
        if self.adaptive:
            self._update_parameters()
        
        # Update diversity metrics
        self._update_diversity()
        
        # Increment iteration counter
        self._current_iteration += 1
        
        # Return best solution found
        return self.best_solution.copy(), float(self.best_score)
    
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
    
    def optimize(self, objective_func: Callable,
                max_evals: Optional[int] = None,
                record_history: bool = True,
                context: Optional[Dict[str, Any]] = None) -> Tuple[np.ndarray, float]:
        """
        Run Grey Wolf Optimization.
        
        Args:
            objective_func: Objective function to minimize
            max_evals: Maximum function evaluations
            record_history: Whether to record optimization history
            context: Additional optimization context
            
        Returns:
            Tuple of (best solution, best score)
        """
        # Set parameters
        if max_evals is not None:
            self.max_evals = max_evals
        
        # Initialize tracking variables
        self.evaluations = 0
        self._current_iteration = 0
        self.start_time = time.time()
        
        # Ensure proper initialization
        if not hasattr(self, 'best_score') or self.best_score is None:
            self.best_score = float('inf')
        
        if not hasattr(self, 'best_solution') or self.best_solution is None:
            self.best_solution = np.random.uniform(self.lower_bounds, self.upper_bounds, size=self.dim)
        
        # Initialize wolf hierarchy
        if not hasattr(self, 'alpha_pos') or self.alpha_pos is None:
            self.alpha_pos = np.random.uniform(self.lower_bounds, self.upper_bounds, size=self.dim)
            self.alpha_score = float('inf')
        
        if not hasattr(self, 'beta_pos') or self.beta_pos is None:
            self.beta_pos = np.random.uniform(self.lower_bounds, self.upper_bounds, size=self.dim)
            self.beta_score = float('inf')
        
        if not hasattr(self, 'delta_pos') or self.delta_pos is None:
            self.delta_pos = np.random.uniform(self.lower_bounds, self.upper_bounds, size=self.dim)
            self.delta_score = float('inf')
        
        # Initialize population if needed
        if not hasattr(self, 'population') or self.population is None:
            self.population = self._init_population()
            self.scores = np.full(self.population_size, float('inf'))
        
        # Initialize progress bar if verbose
        if self.verbose:
            pbar = tqdm(total=self.max_evals, desc="GWO Optimization")
            pbar.n = self.evaluations
            pbar.refresh()
        else:
            pbar = None
        
        # Main optimization loop
        try:
            while self.evaluations < self.max_evals:
                # Check stop criteria
                if hasattr(self, '_check_stop_criteria') and self._check_stop_criteria():
                    break
                
                # Iterate once
                solution, score = self._iterate(objective_func)
                
                # Record history
                if record_history:
                    self._record_history()
                
                # Update progress bar
                if pbar is not None:
                    pbar.n = self.evaluations
                    pbar.refresh()
        except Exception as e:
            self.logger.error(f"Error during optimization: {e}")
            import traceback
            traceback.print_exc()
        finally:
            if pbar is not None:
                pbar.close()
        
        self.end_time = time.time()
        
        # Ensure we return valid values
        if isinstance(self.best_solution, np.ndarray) and not np.isnan(self.best_score):
            return self.best_solution.copy(), float(self.best_score)
        else:
            # Return fallback values if something went wrong
            fallback_solution = np.random.uniform(self.lower_bounds, self.upper_bounds, size=self.dim)
            return fallback_solution, float('inf')

    def get_parameters(self) -> Dict[str, Any]:
        """
        Get optimizer parameters.
        
        Returns:
            Dictionary of optimizer parameters
        """
        params = {
            'name': 'GWO',
            'dim': self.dim,
            'population_size': self.population_size,
            'adaptive': self.adaptive,
            'a_init': self.a_init,
            'verbose': self.verbose
        }
        return params

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
            
        if not hasattr(self, 'alpha_score') or self.alpha_score is None:
            self.alpha_score = float('inf')
            
        if not hasattr(self, 'beta_score') or self.beta_score is None:
            self.beta_score = float('inf')
            
        if not hasattr(self, 'delta_score') or self.delta_score is None:
            self.delta_score = float('inf')
        
        # Increment evaluation counter
        self.evaluations += 1
        
        # Evaluate solution
        score = objective_func(solution)
        
        # Update best solution if better
        if score < self.best_score:
            self.best_score = score
            self.best_solution = solution.copy()
            
        return score
