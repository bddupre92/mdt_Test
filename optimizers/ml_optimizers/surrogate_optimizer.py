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
import pandas as pd
import warnings

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
        max_gp_size: Maximum points to use in GP model
    """
    
    def __init__(self,
                 dim: int,
                 bounds: List[Tuple[float, float]],
                 pop_size: int = 50,
                 n_initial: int = 20,
                 noise: float = 1e-6,
                 length_scale: float = 1.0,
                 exploitation_ratio: float = 0.5,
                 max_gp_size: int = 100):
        super().__init__(dim, bounds)
        
        self.pop_size = pop_size
        self.n_initial = n_initial
        self.noise = noise
        self.length_scale = length_scale
        self.exploitation_ratio = exploitation_ratio
        self.max_gp_size = max_gp_size
        
        # Initialize GP model
        kernel = RBF(length_scale=length_scale)
        self.model = GaussianProcessRegressor(
            kernel=kernel,
            alpha=noise,
            normalize_y=True,
            n_restarts_optimizer=1
        )
        
        # Storage for observations
        self.X_observed = np.zeros((0, dim))
        self.y_observed = np.zeros(0)
        self.y_mean = 0
        self.y_std = 1
        
        # Performance history
        self._performance_history = pd.DataFrame(columns=['iteration', 'score'])
        self.history = []
        
    def get_performance_history(self) -> pd.DataFrame:
        """Get optimizer performance history"""
        return self._performance_history
    
    def reset(self):
        """Reset optimizer state"""
        self.X_observed = np.zeros((0, self.dim))
        self.y_observed = np.zeros(0)
        self.history = []
        self._performance_history = pd.DataFrame(columns=['iteration', 'score'])
        
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
    
    def latin_hypercube_sampling(self, n_points: int) -> np.ndarray:
        """Generate points using Latin Hypercube Sampling"""
        # Generate Latin Hypercube samples
        points = np.zeros((n_points, self.dim))
        
        for i in range(self.dim):
            points[:, i] = np.random.permutation(n_points)
        
        # Scale to [0, 1] and then to bounds
        points = (points + np.random.uniform(0, 1, points.shape)) / n_points
        
        # Scale to bounds
        for i in range(self.dim):
            points[:, i] = points[:, i] * (self.bounds[i][1] - self.bounds[i][0]) + self.bounds[i][0]
            
        return points
    
    def select_next_points(self, n_points: int) -> np.ndarray:
        """Select next points to evaluate using acquisition function"""
        # Generate candidates using Latin Hypercube
        n_candidates = min(100, 10 * n_points)  # Reduced candidate pool
        X_candidates = self.latin_hypercube_sampling(n_candidates)
        
        # Scale candidates
        X_candidates_scaled = self.scale_points(X_candidates)
        
        # Get predictions and uncertainties
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            y_mean, y_std = self.model.predict(X_candidates_scaled, return_std=True)
        
        # Compute acquisition function values
        acq_values = self.acquisition_function(y_mean, y_std)
        
        # Select points balancing exploration and exploitation
        n_exploit = int(self.exploitation_ratio * n_points)
        n_explore = n_points - n_exploit
        
        # Get indices for both strategies
        exploit_indices = np.argsort(y_mean)[:n_exploit]
        explore_indices = np.argsort(-y_std)[:n_explore]
        
        # Combine indices
        selected_indices = np.concatenate([exploit_indices, explore_indices])
        
        return X_candidates[selected_indices]
        
    def acquisition_function(self, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
        """Expected Improvement acquisition function"""
        # Current best value
        y_best = np.min(self.y_observed) if len(self.y_observed) > 0 else 0
        
        # Compute z-score
        with np.errstate(divide='ignore', invalid='ignore'):
            z = (y_best - mean) / std
            
        # Compute EI
        ei = std * (z * norm.cdf(z) + norm.pdf(z))
        ei[std == 0] = 0  # Handle zero uncertainty
        
        return ei
    
    def _optimize(self,
                objective_func: Callable,
                max_evals: Optional[int] = None,
                record_history: bool = True,
                context: Optional[Dict[str, Any]] = None) -> Tuple[np.ndarray, float]:
        """Internal optimization method"""
        if max_evals is None:
            max_evals = 100  # Reduce default max evaluations
            
        # Reset storage
        self.X_observed = np.zeros((0, self.dim))
        self.y_observed = np.zeros(0)
        self._performance_history = pd.DataFrame(columns=['iteration', 'score'])
        self.history = []
        
        # Initial sampling
        X_initial = self.latin_hypercube_sampling(self.n_initial)
        y_initial = np.array([objective_func(x) for x in X_initial])
        
        # Store initial points
        self.X_observed = X_initial
        self.y_observed = y_initial
        
        # Update performance history
        for i, y in enumerate(y_initial):
            self._performance_history = pd.concat([
                self._performance_history,
                pd.DataFrame([{'iteration': i, 'score': y}])
            ], ignore_index=True)
        
        # Normalize objective values
        y_norm = self.normalize_y(self.y_observed)
        
        n_evals = self.n_initial
        best_score = np.min(self.y_observed)
        best_solution = self.X_observed[np.argmin(self.y_observed)]
        
        # Main optimization loop
        while n_evals < max_evals:
            # Early stopping if we've found a good solution
            if best_score < 1e-4:  # Relaxed threshold
                break
                
            # Select subset of points for GP if we exceed max_gp_size
            if len(self.y_observed) > self.max_gp_size:
                # Keep best points and some random points
                n_best = self.max_gp_size // 2
                best_indices = np.argsort(self.y_observed)[:n_best]
                random_indices = np.random.choice(
                    np.setdiff1d(np.arange(len(self.y_observed)), best_indices),
                    size=self.max_gp_size - n_best,
                    replace=False
                )
                indices = np.concatenate([best_indices, random_indices])
                X_train = self.X_observed[indices]
                y_train = y_norm[indices]
            else:
                X_train = self.X_observed
                y_train = y_norm
            
            # Fit GP model
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                self.model.fit(self.scale_points(X_train), y_train)
            
            # Select next points (reduced batch size)
            n_remaining = min(5, max_evals - n_evals)  # Process in very small batches
            X_next = self.select_next_points(n_remaining)
            
            # Evaluate points
            y_next = np.array([objective_func(x) for x in X_next])
            
            # Update observations
            self.X_observed = np.vstack([self.X_observed, X_next])
            self.y_observed = np.append(self.y_observed, y_next)
            
            # Update normalization
            y_norm = self.normalize_y(self.y_observed)
            
            # Update performance history
            for i, y in enumerate(y_next):
                self._performance_history = pd.concat([
                    self._performance_history,
                    pd.DataFrame([{'iteration': n_evals + i, 'score': y}])
                ], ignore_index=True)
            
            # Update best solution
            if np.min(y_next) < best_score:
                best_score = np.min(y_next)
                best_solution = X_next[np.argmin(y_next)]
            
            # Record history (minimal)
            if record_history:
                self.history.append({
                    'iteration': len(self.history),
                    'best_score': best_score
                })
            
            n_evals += n_remaining
            
            # Aggressive exploitation after initial exploration
            if n_evals > max_evals // 3:
                self.exploitation_ratio = 0.9
            
        return best_solution, best_score
    
    def optimize(self,
                objective_func: Callable,
                max_evals: Optional[int] = None,
                record_history: bool = True,
                context: Optional[Dict[str, Any]] = None) -> np.ndarray:
        """
        Optimize using Gaussian Process surrogate model
        
        Args:
            objective_func: Function to optimize
            max_evals: Maximum number of function evaluations
            record_history: Whether to record optimization history
            
        Returns:
            Best solution found
        """
        if max_evals is None:
            max_evals = 1000
            
        self.n_evals = 0
        best_solution = None
        best_score = float('inf')
        self.history = []  # Reset history
        
        # Initial random sampling
        X = np.random.uniform(
            low=[b[0] for b in self.bounds],
            high=[b[1] for b in self.bounds],
            size=(self.n_initial, self.dim)
        )
        y = np.array([objective_func(x) for x in X])
        self.n_evals += self.n_initial
        
        if record_history:
            for score in y:
                self.history.append({'iteration': self.n_evals, 'score': float(score)})
        
        # Find best solution from initial samples
        best_idx = np.argmin(y)
        best_solution = X[best_idx].copy()
        best_score = y[best_idx]
        
        # Main optimization loop
        while self.n_evals < max_evals:
            # Update GP model
            if len(X) > self.max_gp_size:
                # Subsample points for GP fitting
                indices = np.random.choice(len(X), self.max_gp_size, replace=False)
                X_train = X[indices]
                y_train = y[indices]
            else:
                X_train = X
                y_train = y
                
            try:
                self.model.fit(X_train, y_train)
            except Exception as e:
                warnings.warn(f"GP fit failed: {str(e)}")
                break
                
            # Generate candidates
            candidates = np.random.uniform(
                low=[b[0] for b in self.bounds],
                high=[b[1] for b in self.bounds],
                size=(self.pop_size, self.dim)
            )
            
            # Predict values and uncertainties
            try:
                mu, sigma = self.model.predict(candidates, return_std=True)
            except Exception as e:
                warnings.warn(f"GP prediction failed: {str(e)}")
                break
                
            # Calculate acquisition function (Expected Improvement)
            gamma = (np.min(y) - mu) / (sigma + 1e-9)
            ei = sigma * (gamma * norm.cdf(gamma) + norm.pdf(gamma))
            
            # Select next point (mix of best EI and random)
            n_exploit = int(self.pop_size * self.exploitation_ratio)
            n_explore = self.pop_size - n_exploit
            
            # Best EI points
            if n_exploit > 0:
                ei_indices = np.argsort(ei)[-n_exploit:]
                next_points = candidates[ei_indices]
                
                # Evaluate points
                scores = np.array([objective_func(x) for x in next_points])
                self.n_evals += n_exploit
                
                # Update dataset
                X = np.vstack([X, next_points])
                y = np.append(y, scores)
                
                if record_history:
                    for score in scores:
                        self.history.append({'iteration': self.n_evals, 'score': float(score)})
                
                # Update best solution
                min_idx = np.argmin(scores)
                if scores[min_idx] < best_score:
                    best_score = scores[min_idx]
                    best_solution = next_points[min_idx].copy()
            
            # Random points
            if n_explore > 0:
                explore_points = np.random.uniform(
                    low=[b[0] for b in self.bounds],
                    high=[b[1] for b in self.bounds],
                    size=(n_explore, self.dim)
                )
                scores = np.array([objective_func(x) for x in explore_points])
                self.n_evals += n_explore
                
                # Update dataset
                X = np.vstack([X, explore_points])
                y = np.append(y, scores)
                
                if record_history:
                    for score in scores:
                        self.history.append({'iteration': self.n_evals, 'score': float(score)})
                
                # Update best solution
                min_idx = np.argmin(scores)
                if scores[min_idx] < best_score:
                    best_score = scores[min_idx]
                    best_solution = explore_points[min_idx].copy()
            
            # Early stopping
            if best_score < 1e-4:
                break
        
        return best_solution

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
