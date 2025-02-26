"""
Module for analyzing optimization problem characteristics.
"""
from typing import Dict, List, Tuple, Callable
import numpy as np

class ProblemAnalyzer:
    """Class for analyzing optimization problem characteristics."""
    
    def __init__(self, bounds: List[Tuple[float, float]], dim: int):
        """
        Initialize problem analyzer.
        
        Args:
            bounds: List of (min, max) bounds for each dimension
            dim: Problem dimensionality
        """
        self.bounds = bounds
        self.dim = dim
        self.n_samples = min(100, 10 * dim)  # Number of samples for feature extraction
        
    def analyze_features(self, objective_func: Callable) -> Dict[str, float]:
        """
        Extract features from the objective function.
        
        Args:
            objective_func: Function to analyze
            
        Returns:
            Dictionary of problem features
        """
        # Generate random samples
        samples = np.zeros((self.n_samples, self.dim))
        for i, (lower, upper) in enumerate(self.bounds):
            samples[:, i] = np.random.uniform(lower, upper, self.n_samples)
            
        # Evaluate samples
        scores = np.array([objective_func(x) for x in samples])
        
        # Extract features
        features = {
            'mean': float(np.mean(scores)),
            'std': float(np.std(scores)),
            'min': float(np.min(scores)),
            'max': float(np.max(scores)),
            'range': float(np.max(scores) - np.min(scores)),
            'dimensionality': float(self.dim),
            'search_space_volume': float(np.prod([upper - lower for lower, upper in self.bounds]))
        }
        
        return features
