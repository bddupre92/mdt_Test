"""
Problem Feature Analysis Module

This module provides tools for analyzing optimization problems and extracting
relevant features that can be used for algorithm selection.
"""

import numpy as np
import logging
from scipy.stats import skew, kurtosis
from scipy.optimize import differential_evolution
import multiprocessing
from functools import partial

class ProblemAnalyzer:
    """
    Analyzer for extracting features from optimization problems.
    
    This class provides methods to analyze optimization problems and extract
    features that can be used for algorithm selection in meta-learning.
    
    Features extracted:
    - Convexity estimation
    - Multimodality estimation
    - Separability estimation 
    - Smoothness estimation
    - Gradient estimation
    - Statistical moments (mean, variance, skewness, kurtosis)
    - Response surface characteristics
    """
    
    def __init__(self, bounds, dim, n_jobs=None):
        """
        Initialize the problem analyzer.
        
        Parameters:
        -----------
        bounds : list of tuples
            List of (min, max) tuples for each dimension
        dim : int
            Number of dimensions in the problem
        n_jobs : int, optional
            Number of parallel jobs for feature computation
        """
        self.bounds = bounds
        self.dim = dim
        self.logger = logging.getLogger("ProblemAnalyzer")
        
        # Set number of parallel jobs
        if n_jobs is None:
            self.n_jobs = max(1, multiprocessing.cpu_count() - 1)
        else:
            self.n_jobs = n_jobs
    
    def analyze_features(self, objective_func, n_samples=1000, detailed=True):
        """
        Extract features from an optimization problem.
        
        Parameters:
        -----------
        objective_func : callable
            The objective function to analyze f(x) -> float
        n_samples : int, optional
            Number of samples to use for feature estimation
        detailed : bool, optional
            Whether to compute detailed features (more expensive)
            
        Returns:
        --------
        dict
            Dictionary of extracted features
        """
        self.logger.info(f"Analyzing problem features with {n_samples} samples...")
        
        # Generate random samples from the search space
        X = self._generate_random_samples(n_samples)
        
        # Evaluate objective function at sample points
        y = np.array([objective_func(x) for x in X])
        
        # Extract basic statistical features
        features = self._extract_statistical_features(X, y)
        
        # Extract landscape features
        landscape_features = self._extract_landscape_features(X, y, objective_func)
        features.update(landscape_features)
        
        if detailed:
            # Extract more detailed features (may be computationally expensive)
            detailed_features = self._extract_detailed_features(objective_func)
            features.update(detailed_features)
        
        self.logger.info(f"Feature extraction completed: {len(features)} features extracted")
        return features
    
    def _generate_random_samples(self, n_samples):
        """Generate random samples from the search space."""
        X = np.zeros((n_samples, self.dim))
        for i in range(self.dim):
            X[:, i] = np.random.uniform(
                self.bounds[i][0], self.bounds[i][1], n_samples
            )
        return X
    
    def _extract_statistical_features(self, X, y):
        """Extract basic statistical features from function evaluations."""
        features = {}
        
        # Basic statistics
        features['y_mean'] = np.mean(y)
        features['y_std'] = np.std(y)
        features['y_min'] = np.min(y)
        features['y_max'] = np.max(y)
        features['y_range'] = features['y_max'] - features['y_min']
        
        # Distribution characteristics
        features['y_skewness'] = skew(y)
        features['y_kurtosis'] = kurtosis(y)
        
        # Normality test (approximate)
        z_scores = (y - features['y_mean']) / (features['y_std'] + 1e-10)
        features['y_normality'] = np.mean(np.abs(z_scores) < 2)
        
        return features
    
    def _extract_landscape_features(self, X, y, objective_func):
        """Extract features related to the optimization landscape."""
        features = {}
        
        # Estimate landscape ruggedness
        # Calculate gradient approximations using finite differences
        h = 1e-5
        gradients = []
        hessians = []
        
        # Sample a subset for gradient computation to avoid excessive evaluations
        subset_size = min(100, len(X))
        indices = np.random.choice(len(X), subset_size, replace=False)
        X_subset = X[indices]
        
        for x in X_subset:
            grad = np.zeros(self.dim)
            hess = np.zeros((self.dim, self.dim))
            
            # Compute gradient
            fx = objective_func(x)
            for i in range(self.dim):
                x_h = x.copy()
                x_h[i] += h
                fxh = objective_func(x_h)
                grad[i] = (fxh - fx) / h
            
            # Compute diagonal of Hessian
            for i in range(self.dim):
                x_h = x.copy()
                x_h[i] += h
                fxh = objective_func(x_h)
                
                x_2h = x.copy()
                x_2h[i] += 2*h
                fx2h = objective_func(x_2h)
                
                hess[i, i] = (fx2h - 2*fxh + fx) / (h*h)
            
            gradients.append(grad)
            hessians.append(np.diag(hess))  # Just use diagonal elements for simplicity
        
        gradients = np.array(gradients)
        hessians = np.array(hessians)
        
        # Gradient-based features
        features['grad_mean_magnitude'] = np.mean(np.linalg.norm(gradients, axis=1))
        features['grad_std_magnitude'] = np.std(np.linalg.norm(gradients, axis=1))
        
        # Hessian-based features (convexity estimation)
        features['hessian_mean'] = np.mean(hessians)
        features['convexity_ratio'] = np.mean(hessians > 0)  # Proportion of positive second derivatives
        
        # Estimate multimodality
        # Count number of local minima approximation
        points_sorted = X[np.argsort(y)]
        y_sorted = y[np.argsort(y)]
        
        # Consider a point a local minimum if it's better than its neighbors in the sample
        local_minima_count = 0
        for i in range(1, len(points_sorted)-1):
            x_cur = points_sorted[i]
            y_cur = y_sorted[i]
            
            # Find nearest neighbors
            distances = np.linalg.norm(X - x_cur, axis=1)
            nearest_indices = np.argsort(distances)[1:6]  # Get 5 nearest neighbors
            
            # Check if current point is better than all neighbors
            if np.all(y_cur <= y[nearest_indices]):
                local_minima_count += 1
        
        features['estimated_local_minima'] = local_minima_count
        features['multimodality_estimate'] = local_minima_count / (len(X) * 0.01)  # Normalized
        
        # Estimate ruggedness (variation in nearby points)
        ruggedness = 0
        for i in range(len(X_subset)):
            x = X_subset[i]
            y_val = objective_func(x)
            
            # Compute small perturbations
            perturbations = []
            for j in range(5):  # 5 random perturbations
                perturb = x + np.random.normal(0, 0.01, self.dim)
                # Clip to bounds
                for k in range(self.dim):
                    perturb[k] = max(min(perturb[k], self.bounds[k][1]), self.bounds[k][0])
                perturbations.append(perturb)
            
            # Evaluate perturbations
            perturb_vals = [objective_func(p) for p in perturbations]
            
            # Calculate variation
            ruggedness += np.std(perturb_vals) / (np.mean(np.abs(perturb_vals)) + 1e-10)
        
        features['ruggedness'] = ruggedness / len(X_subset)
        
        # Function response characteristics
        response_ratio = features['y_range'] / self.dim
        features['response_ratio'] = response_ratio
        
        return features
    
    def _extract_detailed_features(self, objective_func):
        """Extract more detailed, computationally expensive features."""
        features = {}
        
        # Estimate separability
        # A function is separable if optimizing each dimension independently
        # gives the same result as optimizing all dimensions together
        
        # Global optimization to find approximation of global minimum
        try:
            result = differential_evolution(
                objective_func, 
                self.bounds,
                maxiter=20,  # Limited iterations for speed
                popsize=10,
                tol=1e-2
            )
            global_min_estimate = result.fun
            global_min_point = result.x
            
            # Optimize each dimension separately
            separate_results = []
            for i in range(self.dim):
                # Fix all dimensions except i
                def f_fixed(xi):
                    x = global_min_point.copy()
                    x[i] = xi
                    return objective_func(x)
                
                # Optimize dimension i
                bounds_i = [self.bounds[i]]
                result_i = differential_evolution(
                    f_fixed, 
                    bounds_i,
                    maxiter=20,
                    popsize=10,
                    tol=1e-2
                )
                separate_results.append(result_i.fun)
            
            # Calculate separability metric
            separate_min = sum(separate_results) / len(separate_results)
            features['separability'] = np.abs(separate_min - global_min_estimate) / (np.abs(global_min_estimate) + 1e-10)
            features['separability'] = min(features['separability'], 1.0)  # Cap at 1.0
            
            # Check for plateaus in the objective function
            plateau_test_points = self._generate_random_samples(20)
            plateau_results = [objective_func(x) for x in plateau_test_points]
            features['plateau_ratio'] = np.mean(np.abs(np.diff(plateau_results)) < 1e-4)
            
        except Exception as e:
            self.logger.warning(f"Error in detailed feature extraction: {str(e)}")
            features['separability'] = 0.5  # Default value
            features['plateau_ratio'] = 0.0
        
        return features
