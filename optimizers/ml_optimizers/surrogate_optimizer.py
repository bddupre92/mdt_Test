"""
surrogate_optimizer.py
--------------------
Surrogate Model-Based Optimizer using Gaussian Process regression.

This optimizer uses a Gaussian Process to model the objective function landscape
and guide the search process. It balances exploration and exploitation using
Expected Improvement (EI) acquisition function.
"""

import numpy as np
from typing import Callable, List, Tuple, Optional, Dict, Any
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from scipy.stats import norm
from ..base_optimizer import BaseOptimizer

class SurrogateOptimizer(BaseOptimizer):
    """
    Surrogate Model-Based Optimizer using Gaussian Process regression.
    
    The optimizer works by:
    1. Initially sampling points using Latin Hypercube Sampling
    2. Fitting a GP model to the observed points
    3. Using Expected Improvement to select next points
    4. Updating the model with new observations
    
    Args:
        dim: Problem dimensionality
        bounds: List of (lower, upper) bounds for each dimension
        pop_size: Population size for each iteration
        n_initial: Number of initial points to sample
        noise: Assumed noise level in observations
        length_scale: Length scale for the RBF kernel
        exploitation_ratio: Balance between exploration and exploitation (0-1)
    """
    
    def __init__(self,
                 dim: int,
                 bounds: List[Tuple[float, float]],
                 pop_size: int = 50,
                 n_initial: int = 20,
                 noise: float = 1e-6,
                 length_scale: float = 1.0,
                 exploitation_ratio: float = 0.5):
        super().__init__(dim, bounds)
        
        self.pop_size = pop_size
        self.n_initial = n_initial
        self.noise = noise
        self.length_scale = length_scale
        self.exploitation_ratio = exploitation_ratio
        
        # Initialize GP model with better configuration
        kernel = C(1.0) * RBF(
            length_scale=[length_scale] * dim,
            length_scale_bounds=(1e-2, 1e2)
        )
        self.model = GaussianProcessRegressor(
            kernel=kernel,
            alpha=noise,
            normalize_y=True,
            n_restarts_optimizer=2,
            random_state=42
        )
        
        # Storage for observations
        self.X_observed = None  # Observed points
        self.y_observed = None  # Observed values
        
        # Initialize history storage
        self.history = []
        self.diversity_history = []
        self.param_history = {
            'exploitation_ratio': [],
            'length_scale': []
        }
        
        # For normalizing objective values
        self.y_mean = 0
        self.y_std = 1
        
    def scale_point(self, x: np.ndarray) -> np.ndarray:
        """Scale point to [0, 1] range"""
        x_scaled = np.zeros_like(x)
        for i in range(self.dim):
            x_scaled[i] = (x[i] - self.bounds[i][0]) / (self.bounds[i][1] - self.bounds[i][0])
        return x_scaled
    
    def unscale_point(self, x: np.ndarray) -> np.ndarray:
        """Unscale point from [0, 1] range"""
        x_unscaled = np.zeros_like(x)
        for i in range(self.dim):
            x_unscaled[i] = x[i] * (self.bounds[i][1] - self.bounds[i][0]) + self.bounds[i][0]
        return x_unscaled
    
    def scale_points(self, X: np.ndarray) -> np.ndarray:
        """Scale multiple points to [0, 1] range"""
        return np.array([self.scale_point(x) for x in X])
    
    def unscale_points(self, X: np.ndarray) -> np.ndarray:
        """Unscale multiple points from [0, 1] range"""
        return np.array([self.unscale_point(x) for x in X])
    
    def latin_hypercube_sampling(self, n_samples: int) -> np.ndarray:
        """Generate initial points using Latin Hypercube Sampling"""
        # Generate the intervals
        cut_points = np.linspace(0, 1, n_samples + 1)
        
        # Create the sampling points
        points = np.zeros((n_samples, self.dim))
        
        for i in range(self.dim):
            # Generate points for each dimension
            points[:, i] = cut_points[:-1] + np.random.rand(n_samples) * (cut_points[1] - cut_points[0])
            # Shuffle the points
            np.random.shuffle(points[:, i])
        
        # Scale points to bounds
        for i in range(self.dim):
            points[:, i] = points[:, i] * (self.bounds[i][1] - self.bounds[i][0]) + self.bounds[i][0]
            
        return points
    
    def expected_improvement(self, X: np.ndarray) -> np.ndarray:
        """
        Calculate expected improvement
        
        Args:
            X: Points to evaluate
            
        Returns:
            Expected improvement values
        """
        # Scale points
        X_scaled = self.scale_points(X)
        
        # Get mean and std predictions
        mu, std = self.model.predict(X_scaled, return_std=True)
        
        # Current best (normalized)
        y_best = np.min(self.normalize_y(self.y_observed))
        
        # Handle numerical stability
        std = np.maximum(std, 1e-10)
        
        # Calculate improvement
        z = (y_best - mu) / std
        
        # Calculate EI using more stable implementation
        ei = std * (z * norm.cdf(z) + norm.pdf(z))
        ei[std < 1e-10] = 0.0  # No improvement if uncertainty is too small
        
        return ei
    
    def upper_confidence_bound(self, X: np.ndarray, beta: float = 2.0) -> np.ndarray:
        """
        Calculate upper confidence bound
        
        Args:
            X: Points to evaluate
            beta: Exploration parameter
            
        Returns:
            UCB values
        """
        # Scale points
        X_scaled = self.scale_points(X)
        
        # Get predictions with uncertainty
        mu, std = self.model.predict(X_scaled, return_std=True)
        
        # Handle numerical stability
        std = np.maximum(std, 1e-10)
        
        # Calculate UCB (negative because we're minimizing)
        return mu - beta * std
    
    def select_next_points(self, n_points: int) -> np.ndarray:
        """
        Select next points to evaluate
        
        Args:
            n_points: Number of points to select
            
        Returns:
            Selected points
        """
        # Generate candidates using LHS
        n_candidates = self.pop_size * 2
        candidates = self.latin_hypercube_sampling(n_candidates)
        
        # Calculate acquisition function values
        if np.random.random() < self.exploitation_ratio:
            scores = -self.expected_improvement(candidates)  # Negative because we want to minimize
        else:
            scores = self.upper_confidence_bound(candidates)
        
        # Select best points, ensuring some diversity
        selected_points = []
        remaining_candidates = candidates.copy()
        remaining_scores = scores.copy()
        
        for _ in range(n_points):
            if len(remaining_candidates) == 0:
                # If we run out of candidates, generate new ones
                remaining_candidates = self.latin_hypercube_sampling(n_candidates)
                if np.random.random() < self.exploitation_ratio:
                    remaining_scores = -self.expected_improvement(remaining_candidates)
                else:
                    remaining_scores = self.upper_confidence_bound(remaining_candidates)
            
            # Select best point
            best_idx = np.argmin(remaining_scores)
            selected_points.append(remaining_candidates[best_idx])
            
            # Remove points too close to the selected point
            distances = np.linalg.norm(remaining_candidates - remaining_candidates[best_idx], axis=1)
            mask = distances > 0.1  # Minimum distance threshold
            remaining_candidates = remaining_candidates[mask]
            remaining_scores = remaining_scores[mask]
        
        return np.array(selected_points)
    
    def normalize_y(self, y: np.ndarray) -> np.ndarray:
        """Normalize objective values"""
        if len(y) == 0:
            return y
        self.y_mean = np.mean(y)
        self.y_std = np.std(y) + 1e-8
        return (y - self.y_mean) / self.y_std
    
    def denormalize_y(self, y: np.ndarray) -> np.ndarray:
        """Denormalize objective values"""
        return y * self.y_std + self.y_mean
    
    def _optimize(self,
                objective_func: Callable,
                max_evals: Optional[int] = None,
                record_history: bool = True,
                context: Optional[Dict[str, Any]] = None) -> Tuple[np.ndarray, float]:
        """Internal optimization method"""
        if max_evals is None:
            max_evals = 1000
            
        # Initialize storage
        self.X_observed = np.zeros((0, self.dim))
        self.y_observed = np.zeros(0)
        
        # Initial sampling
        X_initial = self.latin_hypercube_sampling(self.n_initial)
        y_initial = np.array([objective_func(x) for x in X_initial])
        
        self.X_observed = np.vstack([self.X_observed, X_initial])
        self.y_observed = np.append(self.y_observed, y_initial)
        
        # Normalize objective values
        y_norm = self.normalize_y(self.y_observed)
        
        n_evals = self.n_initial
        best_score = np.min(self.y_observed)
        best_solution = self.X_observed[np.argmin(self.y_observed)]
        
        # Main optimization loop
        while n_evals < max_evals:
            # Fit GP model
            self.model.fit(self.scale_points(self.X_observed), y_norm)
            
            # Select next points
            n_remaining = min(self.pop_size, max_evals - n_evals)
            X_next = self.select_next_points(n_remaining)
            
            # Evaluate points
            y_next = np.array([objective_func(x) for x in X_next])
            
            # Update observations
            self.X_observed = np.vstack([self.X_observed, X_next])
            self.y_observed = np.append(self.y_observed, y_next)
            
            # Normalize all objective values
            y_norm = self.normalize_y(self.y_observed)
            
            # Update best solution
            if np.min(y_next) < best_score:
                best_score = np.min(y_next)
                best_solution = X_next[np.argmin(y_next)]
            
            # Record history
            if record_history:
                self.history.append({
                    'iteration': len(self.history),
                    'best_score': best_score,
                    'population': X_next.copy(),
                    'scores': y_next.copy(),
                    'best_solution': best_solution.copy()
                })
                
                # Calculate and record diversity
                self.diversity_history.append(
                    np.mean([np.linalg.norm(x - best_solution) for x in X_next])
                )
                
                # Record parameters
                self.param_history['exploitation_ratio'].append(self.exploitation_ratio)
                self.param_history['length_scale'].append(
                    self.model.kernel_.k2.length_scale.mean()  # Access RBF kernel length_scale
                )
            
            n_evals += n_remaining
            
        return best_solution, best_score
    
    def optimize(self,
                objective_func: Callable,
                max_evals: Optional[int] = None,
                record_history: bool = True,
                context: Optional[Dict[str, Any]] = None) -> Tuple[np.ndarray, float]:
        """
        Run the optimization process
        
        Args:
            objective_func: Function to minimize
            max_evals: Maximum number of function evaluations
            record_history: Whether to record optimization history
            context: Additional context for the objective function
            
        Returns:
            Tuple of (best_solution, best_score)
        """
        return self._optimize(objective_func, max_evals, record_history, context)
