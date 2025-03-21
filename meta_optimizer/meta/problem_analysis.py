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
        
        # Create a shape-safe wrapper for the objective function
        def shape_safe_objective(x):
            # Ensure x is a 1D array with proper shape
            if hasattr(x, 'ndim') and x.ndim > 1:
                x = x.reshape(-1)  # Flatten to 1D
            return objective_func(x)
        
        try:
            # Generate random samples from the search space
            X = self._generate_random_samples(n_samples)
            
            # Evaluate objective function at sample points
            y = np.array([shape_safe_objective(x) for x in X])
            
            # Handle NaN or infinite values
            if np.any(np.isnan(y)) or np.any(np.isinf(y)):
                self.logger.warning("NaN or infinite values detected in function evaluations. Using filtered values.")
                valid_mask = ~(np.isnan(y) | np.isinf(y))
                if np.sum(valid_mask) < 10:  # Need at least 10 valid points
                    self.logger.error("Too few valid function evaluations. Using default values.")
                    return self._get_default_features()
                y = y[valid_mask]
                X = X[valid_mask]
            
            # Extract basic statistical features
            features = self._extract_statistical_features(X, y)
            
            # Extract landscape features with y_range
            landscape_features = self._extract_landscape_features(X, y, shape_safe_objective, y_range=features['y_range'])
            features.update(landscape_features)
            
            if detailed:
                # Extract more detailed features (may be computationally expensive)
                detailed_features = self._extract_detailed_features(shape_safe_objective)
                features.update(detailed_features)
            
            self.logger.info(f"Feature extraction completed: {len(features)} features extracted")
            return features
            
        except Exception as e:
            self.logger.error(f"Error during feature extraction: {str(e)}")
            return self._get_default_features()
    
    def _get_default_features(self):
        """Return default feature values when extraction fails."""
        return {
            'dimension': self.dim,
            'y_mean': 0.0,
            'y_std': 1.0,
            'y_min': -5.0,
            'y_max': 5.0,
            'y_range': 10.0,
            'y_skewness': 0.0,
            'y_kurtosis': 0.0,
            'y_normality': 0.5,
            'grad_mean_magnitude': 1.0,
            'grad_std_magnitude': 0.5,
            'hessian_mean': 0.0,
            'convexity_ratio': 0.5,
            'estimated_local_minima': 1,
            'multimodality_estimate': 0.5,
            'ruggedness': 0.5,
            'response_ratio': 1.0,
            'separability': 0.5,
            'plateau_ratio': 0.0
        }
    
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
        
        try:
            # Add dimension as a feature
            features['dimension'] = self.dim
            
            # Filter out non-finite values first
            valid_mask = np.isfinite(y)
            y_valid = y[valid_mask]
            
            if len(y_valid) < 10:
                self.logger.warning("Too few valid values for feature extraction. Using defaults.")
                return self._get_default_features()
            
            # Basic statistics with error handling
            try:
                features['y_mean'] = float(np.mean(y_valid))
                features['y_std'] = float(np.std(y_valid))
            except Exception as e:
                self.logger.warning(f"Error calculating basic statistics: {str(e)}")
                features['y_mean'] = 0.0
                features['y_std'] = 1.0
            
            # Handle min/max/range calculation
            try:
                y_min = float(np.min(y_valid))
                y_max = float(np.max(y_valid))
                
                if np.isfinite(y_min) and np.isfinite(y_max):
                    features['y_min'] = y_min
                    features['y_max'] = y_max
                    features['y_range'] = y_max - y_min
                    
                    # Additional validation for y_range
                    if not np.isfinite(features['y_range']):
                        self.logger.warning("Non-finite y_range computed. Using default.")
                        features['y_range'] = 10.0
                else:
                    self.logger.warning("Non-finite min/max values. Using defaults.")
                    features['y_min'] = -5.0
                    features['y_max'] = 5.0
                    features['y_range'] = 10.0
            except Exception as e:
                self.logger.warning(f"Error calculating min/max/range: {str(e)}")
                features['y_min'] = -5.0
                features['y_max'] = 5.0
                features['y_range'] = 10.0
            
            # Distribution characteristics
            try:
                features['y_skewness'] = float(skew(y_valid))
            except:
                features['y_skewness'] = 0.0
                
            try:
                features['y_kurtosis'] = float(kurtosis(y_valid))
            except:
                features['y_kurtosis'] = 0.0
            
            # Normality test (approximate)
            try:
                z_scores = (y_valid - features['y_mean']) / (features['y_std'] + 1e-10)
                features['y_normality'] = float(np.mean(np.abs(z_scores) < 2))
            except:
                features['y_normality'] = 0.5
                
        except Exception as e:
            self.logger.warning(f"Error extracting statistical features: {str(e)}")
            return self._get_default_features()
        
        return features
    
    def _extract_landscape_features(self, X, y, objective_func, y_range):
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
            
            # Ensure x is 1D and proper shape
            x = x.reshape(-1) if hasattr(x, 'ndim') and x.ndim > 1 else x
            
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
        features['response_ratio'] = y_range / self.dim
        
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
                    # Ensure xi is properly shaped (scalar to array)
                    if np.isscalar(xi):
                        xi = np.array([xi])
                    x = global_min_point.copy()
                    x[i] = xi[0]  # Use first element if xi is array
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
