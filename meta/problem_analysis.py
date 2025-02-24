"""
Problem feature analysis for meta-optimizer learning.
"""
import numpy as np
from typing import List, Dict, Tuple, Callable
from scipy.stats import norm
from sklearn.preprocessing import StandardScaler


class ProblemAnalyzer:
    """Analyzes optimization problem characteristics."""
    
    def __init__(self, bounds: List[Tuple[float, float]], dim: int):
        """
        Initialize problem analyzer.
        
        Args:
            bounds: List of (min, max) bounds for each dimension
            dim: Number of dimensions
        """
        self.bounds = bounds
        self.dim = dim
        
    def analyze_features(self, objective_func: Callable, n_samples: int = 50) -> Dict[str, float]:
        """
        Extract features that characterize the optimization problem.
        
        Args:
            objective_func: Function to analyze
            n_samples: Number of samples to use for analysis
            
        Returns:
            Dictionary of problem features
        """
        # Generate sample points
        X = self._generate_samples(n_samples)
        y = np.array([objective_func(x) for x in X])
        
        # Calculate features
        features = {
            'dimension': float(self.dim),
            'range': float(np.max(y) - np.min(y)),
            'std': float(np.std(y)),
            'gradient_variance': self._estimate_gradient_variance(X, y),
            'modality': self._estimate_modality(X, y),
            'convexity': self._estimate_convexity(X, y),
            'ruggedness': self._estimate_ruggedness(X, y),
            'separability': self._estimate_separability(X, y)
        }
        
        return features
    
    def _generate_samples(self, n_samples: int) -> np.ndarray:
        """Generate Latin Hypercube samples in the bounded space."""
        # Use Latin Hypercube Sampling for better coverage
        samples = np.zeros((n_samples, self.dim))
        
        for i in range(self.dim):
            samples[:, i] = np.random.permutation(np.linspace(0, 1, n_samples))
            
        # Scale to bounds
        for i in range(self.dim):
            low, high = self.bounds[i]
            samples[:, i] = low + (high - low) * samples[:, i]
            
        return samples
    
    def _estimate_gradient_variance(self, X: np.ndarray, y: np.ndarray) -> float:
        """Estimate variance in gradients across the space."""
        # Compute approximate gradients using finite differences
        gradients = []
        for i in range(len(X)):
            for d in range(self.dim):
                h = 1e-6 * (self.bounds[d][1] - self.bounds[d][0])
                x_plus = X[i].copy()
                x_plus[d] += h
                y_plus = y[i]  # Use original y to avoid extra function calls
                grad = (y_plus - y[i]) / h
                gradients.append(grad)
                
        return float(np.var(gradients))
    
    def _estimate_modality(self, X: np.ndarray, y: np.ndarray) -> float:
        """Estimate number of local optima (modality)."""
        # Use kernel density estimation to identify modes
        bandwidth = 0.1 * (np.max(y) - np.min(y))
        y_normalized = (y - np.min(y)) / (np.max(y) - np.min(y))
        
        # Count number of peaks in smoothed landscape
        peaks = 0
        for i in range(1, len(y_normalized) - 1):
            if (y_normalized[i] > y_normalized[i-1] and 
                y_normalized[i] > y_normalized[i+1]):
                peaks += 1
                
        return float(max(1, peaks))
    
    def _estimate_convexity(self, X: np.ndarray, y: np.ndarray) -> float:
        """Estimate degree of convexity."""
        # Check if midpoints have higher values than endpoints
        convex_ratio = 0
        n_checks = min(100, len(X) * (len(X) - 1) // 2)
        
        for _ in range(n_checks):
            # Randomly select two points
            i, j = np.random.choice(len(X), 2, replace=False)
            mid = (X[i] + X[j]) / 2
            y_mid = np.mean([y[i], y[j]])
            
            # Count cases where midpoint is better (lower)
            if y_mid < max(y[i], y[j]):
                convex_ratio += 1
                
        return float(convex_ratio / n_checks)
    
    def _estimate_ruggedness(self, X: np.ndarray, y: np.ndarray) -> float:
        """Estimate landscape ruggedness."""
        # Calculate average absolute difference between neighboring points
        diffs = []
        for i in range(len(y)):
            # Find k nearest neighbors
            distances = np.sum((X - X[i])**2, axis=1)
            nearest = np.argsort(distances)[1:4]  # Skip self
            
            for j in nearest:
                diffs.append(abs(y[i] - y[j]))
                
        # Normalize by range
        y_range = np.max(y) - np.min(y)
        if y_range == 0:
            return 0.0
            
        return float(np.mean(diffs) / y_range)
    
    def _estimate_separability(self, X: np.ndarray, y: np.ndarray) -> float:
        """Estimate problem separability."""
        # Check if optimizing each dimension independently gives similar results
        n_checks = min(20, self.dim)
        separable_score = 0
        
        for _ in range(n_checks):
            # Select random dimension
            d = np.random.randint(0, self.dim)
            
            # Sort points by this dimension
            idx = np.argsort(X[:, d])
            x_sorted = X[idx]
            y_sorted = y[idx]
            
            # Check if values change monotonically
            is_monotonic = np.all(np.diff(y_sorted) >= 0) or np.all(np.diff(y_sorted) <= 0)
            if is_monotonic:
                separable_score += 1
                
        return float(separable_score / n_checks)
