"""
Problem feature analysis for meta-optimizer learning.
"""
import numpy as np
from typing import List, Dict, Tuple, Callable
from scipy.stats import norm, entropy
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors


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
        self.n_neighbors = min(10, dim)
        
    def analyze_features(self, objective_func: Callable, n_samples: int = 100) -> Dict[str, float]:
        """
        Extract features that characterize the optimization problem.
        
        Args:
            objective_func: Function to analyze
            n_samples: Number of samples to use for analysis
            
        Returns:
            Dictionary of problem features
        """
        # Generate sample points using improved sampling
        X = self._generate_samples(n_samples)
        y = np.array([objective_func(x) for x in X])
        
        # Calculate enhanced features
        features = {
            'dimension': float(self.dim),
            'range': float(np.max(y) - np.min(y)),
            'std': float(np.std(y)),
            'gradient_variance': self._estimate_gradient_variance(X, y),
            'modality': self._estimate_modality(X, y),
            'convexity': self._estimate_convexity(X, y),
            'ruggedness': self._estimate_ruggedness(X, y),
            'separability': self._estimate_separability(X, y),
            'local_structure': self._analyze_local_structure(X, y),
            'global_structure': self._analyze_global_structure(X, y),
            'fitness_distance_correlation': self._compute_fdc(X, y),
            'information_content': self._compute_information_content(y),
            'basin_ratio': self._estimate_basin_ratio(X, y),
            'gradient_homogeneity': self._estimate_gradient_homogeneity(X, y)
        }
        
        return features
    
    def _generate_samples(self, n_samples: int) -> np.ndarray:
        """Generate improved Latin Hypercube samples with corner points."""
        # Adjust number of samples to accommodate corner points
        n_corners = min(2**self.dim, n_samples // 2)  # Limit corners in high dimensions
        n_lhs = max(n_samples - n_corners, n_samples // 2)  # Ensure minimum LHS samples
        
        # Generate Latin Hypercube samples
        samples = np.zeros((n_lhs, self.dim))
        for i in range(self.dim):
            samples[:, i] = np.random.permutation(np.linspace(0, 1, n_lhs))
        
        # Add selected corner points
        if n_corners > 0:
            corners = np.array(list(np.ndindex(tuple([2] * self.dim))))[:n_corners]
            samples = np.vstack([samples, corners])
        
        # Scale to bounds
        for i in range(self.dim):
            low, high = self.bounds[i]
            samples[:, i] = low + (high - low) * samples[:, i]
            
        return samples
    
    def _estimate_gradient_variance(self, X: np.ndarray, y: np.ndarray) -> float:
        """Estimate variance in gradients using improved finite differences."""
        gradients = []
        h = np.array([1e-6 * (b[1] - b[0]) for b in self.bounds])
        
        for i in range(len(X)):
            grad = np.zeros(self.dim)
            for d in range(self.dim):
                x_plus = X[i].copy()
                x_plus[d] += h[d]
                x_minus = X[i].copy()
                x_minus[d] -= h[d]
                grad[d] = (y[i+1] - y[i-1]) / (2*h[d]) if 0 < i < len(X)-1 else 0
            gradients.append(np.linalg.norm(grad))
                
        return float(np.var(gradients))
    
    def _estimate_modality(self, X: np.ndarray, y: np.ndarray) -> float:
        """Estimate number of local optima using clustering and peak detection."""
        # Normalize values
        y_normalized = (y - np.min(y)) / (np.max(y) - np.min(y))
        
        # Use k-nearest neighbors to identify local structures
        nbrs = NearestNeighbors(n_neighbors=self.n_neighbors).fit(X)
        distances, indices = nbrs.kneighbors(X)
        
        # Count local optima
        local_min = 0
        local_max = 0
        for i in range(len(y)):
            neighbors = indices[i][1:]  # Exclude self
            if all(y[i] < y[j] for j in neighbors):
                local_min += 1
            if all(y[i] > y[j] for j in neighbors):
                local_max += 1
                
        return float(max(1, local_min + local_max))
    
    def _estimate_convexity(self, X: np.ndarray, y: np.ndarray) -> float:
        """Estimate degree of convexity using random triplets."""
        n_tests = min(1000, len(X) * (len(X) - 1) * (len(X) - 2) // 6)
        convex_count = 0
        
        for _ in range(n_tests):
            # Select random triplet
            i, j, k = np.random.choice(len(X), 3, replace=False)
            
            # Check convexity condition
            x_mid = 0.5 * (X[i] + X[j])
            y_mid = 0.5 * (y[i] + y[j])
            
            if y[k] > y_mid:
                convex_count += 1
                
        return float(convex_count / n_tests)
    
    def _estimate_ruggedness(self, X: np.ndarray, y: np.ndarray) -> float:
        """Estimate landscape ruggedness using improved neighbor analysis."""
        nbrs = NearestNeighbors(n_neighbors=self.n_neighbors).fit(X)
        distances, indices = nbrs.kneighbors(X)
        
        ruggedness_scores = []
        for i in range(len(y)):
            local_y = y[indices[i]]
            local_distances = distances[i]
            weights = np.exp(-local_distances)
            weighted_diffs = weights * np.abs(local_y - y[i])
            local_ruggedness = np.sum(weighted_diffs) / np.sum(weights)
            ruggedness_scores.append(local_ruggedness)
            
        y_range = np.max(y) - np.min(y)
        if y_range == 0:
            return 0.0
            
        return float(np.mean(ruggedness_scores) / y_range)
    
    def _analyze_local_structure(self, X: np.ndarray, y: np.ndarray) -> float:
        """Analyze local neighborhood structure."""
        nbrs = NearestNeighbors(n_neighbors=self.n_neighbors).fit(X)
        distances, indices = nbrs.kneighbors(X)
        
        local_variations = []
        for i in range(len(y)):
            local_y = y[indices[i]]
            local_variations.append(np.std(local_y) / (np.max(local_y) - np.min(local_y) + 1e-10))
            
        return float(np.mean(local_variations))
    
    def _analyze_global_structure(self, X: np.ndarray, y: np.ndarray) -> float:
        """Analyze global landscape structure."""
        # Compute pairwise distances and value differences
        n_samples = min(1000, len(X) * (len(X) - 1) // 2)
        dist_val_corr = []
        
        for _ in range(n_samples):
            i, j = np.random.choice(len(X), 2, replace=False)
            dist = np.linalg.norm(X[i] - X[j])
            val_diff = abs(y[i] - y[j])
            dist_val_corr.append((dist, val_diff))
            
        dist_val_corr = np.array(dist_val_corr)
        return float(np.corrcoef(dist_val_corr.T)[0, 1])
    
    def _compute_fdc(self, X: np.ndarray, y: np.ndarray) -> float:
        """Compute Fitness Distance Correlation."""
        best_idx = np.argmin(y)
        distances = np.array([np.linalg.norm(x - X[best_idx]) for x in X])
        
        return float(np.corrcoef(distances, y)[0, 1])
    
    def _compute_information_content(self, y: np.ndarray) -> float:
        """Compute information content of the landscape."""
        # Use histogram to estimate probability distribution
        hist, _ = np.histogram(y, bins='auto', density=True)
        return float(entropy(hist + 1e-10))
    
    def _estimate_basin_ratio(self, X: np.ndarray, y: np.ndarray) -> float:
        """Estimate ratio of attraction basins."""
        nbrs = NearestNeighbors(n_neighbors=self.n_neighbors).fit(X)
        distances, indices = nbrs.kneighbors(X)
        
        basin_count = 0
        visited = set()
        
        for i in range(len(y)):
            if i in visited:
                continue
                
            # Find local minimum
            current = i
            while True:
                neighbors = indices[current]
                best_neighbor = neighbors[np.argmin(y[neighbors])]
                if y[best_neighbor] >= y[current]:
                    break
                current = best_neighbor
                
            basin_count += 1
            visited.update(indices[current])
            
        return float(basin_count / len(X))
    
    def _estimate_gradient_homogeneity(self, X: np.ndarray, y: np.ndarray) -> float:
        """Estimate how homogeneous the gradient field is."""
        gradients = []
        h = np.array([1e-6 * (b[1] - b[0]) for b in self.bounds])
        
        for i in range(1, len(X)-1):
            grad = np.zeros(self.dim)
            for d in range(self.dim):
                grad[d] = (y[i+1] - y[i-1]) / (2*h[d])
            gradients.append(grad)
            
        # Compute average angle between gradients
        angles = []
        for i in range(len(gradients)):
            for j in range(i+1, len(gradients)):
                cos_angle = np.dot(gradients[i], gradients[j]) / (
                    np.linalg.norm(gradients[i]) * np.linalg.norm(gradients[j]) + 1e-10)
                angles.append(np.arccos(np.clip(cos_angle, -1.0, 1.0)))
                
        return float(np.std(angles) / np.pi)
    
    def _estimate_separability(self, X: np.ndarray, y: np.ndarray) -> float:
        """Estimate how separable the problem is."""
        # Compute pairwise distances and value differences
        n_samples = min(1000, len(X) * (len(X) - 1) // 2)
        dist_val_corr = []
        
        for _ in range(n_samples):
            i, j = np.random.choice(len(X), 2, replace=False)
            dist = np.linalg.norm(X[i] - X[j])
            val_diff = abs(y[i] - y[j])
            dist_val_corr.append((dist, val_diff))
            
        dist_val_corr = np.array(dist_val_corr)
        return float(np.corrcoef(dist_val_corr.T)[0, 1])
