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

class GreyWolfOptimizer(BaseOptimizer):
    def __init__(self,
                 dim: int,
                 bounds: List[Tuple[float, float]],
                 population_size: int = 50,
                 max_evals: int = 10000,
                 adaptive: bool = True,
                 a_init: float = 2.0,
                 verbose: bool = False,
                 **kwargs):
        """
        Initialize GWO optimizer.
        
        Args:
            dim: Problem dimensionality
            bounds: Parameter bounds
            population_size: Population size
            max_evals: Maximum function evaluations
            adaptive: Whether to use parameter adaptation
            a_init: Initial value of a parameter
            verbose: Whether to show progress bars
        """
        # Only pass parameters that BaseOptimizer accepts
        super().__init__(dim=dim, bounds=bounds, population_size=population_size, adaptive=adaptive)
        
        # Store max_evals as instance variable
        self.max_evals = max_evals
        self.verbose = verbose
        
        # Store bounds
        self.bounds = bounds
        self.lower_bounds = np.array([b[0] for b in bounds])
        self.upper_bounds = np.array([b[1] for b in bounds])
        
        # GWO parameters
        self.a_init = a_init
        self.a = a_init
        
        # Initialize population as a (population_size, dim) array
        self.population = np.zeros((self.population_size, self.dim))
        for i, (lower, upper) in enumerate(self.bounds):
            self.population[:, i] = np.random.uniform(lower, upper, self.population_size)
        
        # Initialize scores
        self.population_scores = np.full(self.population_size, np.inf)
        
        # Initialize best solution
        self.best_solution = np.zeros(dim)
        self.best_score = np.inf
        
        # Initialize tracking variables
        self.evaluations = 0
        self.start_time = 0
        self.end_time = 0
        
        # Initialize history tracking
        self.convergence_curve = []
        self.time_curve = []
        self.eval_curve = []
        
        # Initialize wolf hierarchy
        self.alpha_wolf = None
        self.beta_wolf = None
        self.delta_wolf = None
        self.alpha_score = np.inf
        self.beta_score = np.inf
        self.delta_score = np.inf
        
        # Success tracking
        self.success_history = np.zeros(20)  # Track last 20 iterations
        self.success_idx = 0
        
        # Initialize parameter history
        if adaptive:
            self.param_history = {
                'a': [a_init],
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
    
    def _update_wolves(self, scores: np.ndarray):
        """Update alpha, beta, and delta wolves"""
        sorted_indices = np.argsort(scores)
        
        # Update alpha
        if scores[sorted_indices[0]] < self.alpha_score:
            self.alpha_wolf = self.population[sorted_indices[0]].copy()
            self.alpha_score = scores[sorted_indices[0]]
            self.success_history[self.success_idx] = 1
        else:
            self.success_history[self.success_idx] = 0
        
        # Update beta
        if scores[sorted_indices[1]] < self.beta_score:
            self.beta_wolf = self.population[sorted_indices[1]].copy()
            self.beta_score = scores[sorted_indices[1]]
        
        # Update delta
        if scores[sorted_indices[2]] < self.delta_score:
            self.delta_wolf = self.population[sorted_indices[2]].copy()
            self.delta_score = scores[sorted_indices[2]]
        
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
        """Calculate position update towards a leader"""
        r1 = np.random.rand(self.dim)
        r2 = np.random.rand(self.dim)
        A = 2 * self.a * r1 - self.a
        C = 2 * r2
        
        d_leader = np.abs(C * leader - wolf)
        x_leader = leader - A * d_leader
        return x_leader
    
    def reset(self):
        """Reset optimizer state"""
        super().reset()
        
        # Reset parameters
        self.a = self.a_init
        
        # Initialize population
        self.population = np.zeros((self.population_size, self.dim))
        for i, (lower, upper) in enumerate(self.bounds):
            self.population[:, i] = np.random.uniform(lower, upper, self.population_size)
        self.population_scores = np.full(self.population_size, np.inf)
        
        # Initialize wolf hierarchy
        self.alpha_wolf = None
        self.beta_wolf = None
        self.delta_wolf = None
        self.alpha_score = np.inf
        self.beta_score = np.inf
        self.delta_score = np.inf
        
        # Initialize parameter history
        if self.adaptive:
            self.param_history.update({
                'a': [self.a_init],
                'success_rate': []
            })
        else:
            self.param_history = {
                'diversity': []
            }
            
    def _iterate(self, objective_func: Callable, context: Optional[Dict[str, Any]] = None) -> None:
        """
        Perform one iteration of the Grey Wolf Optimizer.
        
        Args:
            objective_func: Function to minimize
            context: Optional problem context
        """
        # Update a parameter
        self.a = self.a_init - self.a_init * (self.evaluations / self.max_evals)
        
        # For each wolf in the pack
        for i in range(self.population_size):
            # Get current position
            X = self.population[i].copy()
            
            # Calculate A and C vectors
            A1 = 2 * self.a * np.random.random(self.dim) - self.a
            C1 = 2 * np.random.random(self.dim)
            A2 = 2 * self.a * np.random.random(self.dim) - self.a
            C2 = 2 * np.random.random(self.dim)
            A3 = 2 * self.a * np.random.random(self.dim) - self.a
            C3 = 2 * np.random.random(self.dim)
            
            # Calculate new position based on alpha, beta, and delta wolves
            D_alpha = np.abs(C1 * self.alpha_wolf - X)
            X1 = self.alpha_wolf - A1 * D_alpha
            
            D_beta = np.abs(C2 * self.beta_wolf - X)
            X2 = self.beta_wolf - A2 * D_beta
            
            D_delta = np.abs(C3 * self.delta_wolf - X)
            X3 = self.delta_wolf - A3 * D_delta
            
            # Average the positions
            X_new = (X1 + X2 + X3) / 3.0
            
            # Apply bounds
            X_new = np.clip(X_new, self.lower_bounds, self.upper_bounds)
            
            # Evaluate new position
            score_new = objective_func(X_new)
            self.evaluations += 1
            
            # Update position if better
            if score_new < self.population_scores[i]:
                self.population[i] = X_new
                self.population_scores[i] = score_new
                
                # Update alpha, beta, delta wolves if needed
                if score_new < self.best_score:
                    # Update delta to beta, beta to alpha, and alpha to new best
                    self.delta_wolf = self.beta_wolf.copy()
                    self.delta_score = self.beta_score
                    
                    self.beta_wolf = self.alpha_wolf.copy()
                    self.beta_score = self.alpha_score
                    
                    self.alpha_wolf = X_new.copy()
                    self.alpha_score = score_new
                    
                    self.best_solution = X_new.copy()
                    self.best_score = score_new
                elif score_new < self.beta_score:
                    # Update delta to beta and beta to new
                    self.delta_wolf = self.beta_wolf.copy()
                    self.delta_score = self.beta_score
                    
                    self.beta_wolf = X_new.copy()
                    self.beta_score = score_new
                elif score_new < self.delta_score:
                    # Update delta to new
                    self.delta_wolf = X_new.copy()
                    self.delta_score = score_new
    
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
        Run Grey Wolf optimization.
        
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
        
        # Initialize progress bar if verbose is enabled
        pbar = None
        if self.verbose:
            pbar = tqdm(total=self.max_evals, desc="GWO Optimization")
            pbar.update(0)
            
        # Initialize population and evaluate
        self.population = self._init_population()
        self.population_scores = np.zeros(self.population_size)
        
        for i in range(self.population_size):
            self.population_scores[i] = objective_func(self.population[i])
            self.evaluations += 1
            
        # Sort population by scores
        indices = np.argsort(self.population_scores)
        self.population = self.population[indices]
        self.population_scores = self.population_scores[indices]
        
        # Initialize alpha, beta, and delta wolves
        self.alpha_wolf = self.population[0].copy()
        self.alpha_score = self.population_scores[0]
        
        self.beta_wolf = self.population[1].copy()
        self.beta_score = self.population_scores[1]
        
        self.delta_wolf = self.population[2].copy()
        self.delta_score = self.population_scores[2]
        
        # Initialize best solution
        self.best_solution = self.alpha_wolf.copy()
        self.best_score = self.alpha_score
        
        # Main optimization loop
        while not self._check_convergence():
            self._iterate(objective_func, context)
            
            # Record history if requested
            if record_history:
                self._record_history()
            
            # Update progress bar
            if pbar:
                pbar.update(self.population_size)
                pbar.set_postfix({"best_score": f"{self.best_score:.6f}"})
        
        self.end_time = time.time()
        return self.best_solution, self.best_score

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
