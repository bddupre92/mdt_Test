"""
Dynamic benchmark functions for optimization with concept drift.

This module provides tools for creating dynamic benchmark functions
that change over time, simulating concept drift in optimization problems.
"""

import numpy as np
import math
from typing import Callable, List, Tuple, Optional, Union, Dict, Any

class DynamicFunction:
    """
    Creates a dynamic test function that changes over time.
    
    This wrapper transforms a static benchmark function into a dynamic one
    by applying various types of changes over time (iterations).
    """
    
    def __init__(self, 
                 base_function: Callable[[np.ndarray], float],
                 drift_type: str = 'linear',
                 drift_rate: float = 0.01,
                 drift_interval: int = 100,
                 noise_level: float = 0.0,
                 dim: int = 2,
                 bounds: Optional[List[Tuple[float, float]]] = None,
                 severity: float = 1.0):
        """
        Initialize a dynamic function.
        
        Parameters
        ----------
        base_function : Callable[[np.ndarray], float]
            The static benchmark function to make dynamic
        drift_type : str
            Type of drift to apply:
            - 'linear': Linear shift in the function
            - 'oscillatory': Periodic shift
            - 'sudden': Abrupt change at regular intervals
            - 'incremental': Gradual change that accelerates
            - 'random': Random shifts
            - 'noise': Increasing noise over time
        drift_rate : float
            Rate parameter controlling how quickly the function changes
        drift_interval : int
            Number of function evaluations between drift changes (for sudden drift)
        noise_level : float
            Level of noise to add to function evaluations
        dim : int
            Dimensionality of the function
        bounds : Optional[List[Tuple[float, float]]]
            Bounds for each dimension [(min1, max1), (min2, max2), ...]
        severity : float
            Scaling factor for the drift (higher = more severe drift)
        """
        self.base_function = base_function
        self.drift_type = drift_type
        self.drift_rate = drift_rate * severity
        self.drift_interval = drift_interval
        self.noise_level = noise_level
        self.dim = dim
        self.bounds = bounds or [(-5, 5)] * dim
        self.evaluations = 0
        self.time_step = 0
        self.severity = severity
        
        # For tracking drift characteristics
        self.drift_magnitude = 0.0
        self.drift_history = []
        
        # Initialize the current optimal value
        self.current_optimal = 0.0
        self._estimate_current_optimal()
        
    def _estimate_current_optimal(self):
        """Estimate the current optimal value by sampling points."""
        best_value = float('inf')
        n_samples = 100
        
        # Sample random points to find an approximate optimum
        for _ in range(n_samples):
            x = np.array([np.random.uniform(low, high) for low, high in self.bounds])
            drifted_x = self._apply_drift(x)
            value = self.base_function(drifted_x)
            if value < best_value:
                best_value = value
        
        self.current_optimal = best_value
        
    def reset(self) -> None:
        """Reset the function's state to initial conditions."""
        self.evaluations = 0
        self.time_step = 0
        self.drift_magnitude = 0.0
        self.drift_history = []
        self._estimate_current_optimal()
    
    def evaluate(self, x: np.ndarray) -> float:
        """
        Evaluate the dynamic function.
        
        Parameters
        ----------
        x : np.ndarray
            Input vector to evaluate
        
        Returns
        -------
        float
            Function value with drift applied
        """
        # Check input dimension
        if len(x) != self.dim:
            raise ValueError(f"Expected input of dimension {self.dim}, got {len(x)}")
        
        # Apply drift to input vector
        drifted_x = self._apply_drift(x)
        
        # Evaluate base function with drifted parameters
        result = self.base_function(drifted_x)
        
        # Add noise if specified
        if self.noise_level > 0:
            noise = np.random.normal(0, self.noise_level * (1 + self.time_step * 0.01))
            result += noise
        
        # Update evaluation counter
        self.evaluations += 1
        
        # Update time step for time-dependent dynamics
        if self.evaluations % self.drift_interval == 0:
            self.time_step += 1
            # Re-estimate the optimal value when the function changes
            self._estimate_current_optimal()
            
        return result
    
    def _apply_drift(self, x: np.ndarray) -> np.ndarray:
        """
        Apply drift transformation to input vector.
        
        Parameters
        ----------
        x : np.ndarray
            Original input vector
        
        Returns
        -------
        np.ndarray
            Transformed input vector with drift applied
        """
        x = np.asarray(x).copy()
        
        if self.drift_type == 'linear':
            # Linear shift in the function's landscape
            drift_factor = 1 + (self.time_step * self.drift_rate)
            drifted_x = x * drift_factor
            self.drift_magnitude = abs(drift_factor - 1)
            
        elif self.drift_type == 'oscillatory':
            # Periodic shift
            drift_factor = 1 + self.severity * math.sin(self.time_step * self.drift_rate)
            drifted_x = x * drift_factor
            self.drift_magnitude = abs(drift_factor - 1)
            
        elif self.drift_type == 'sudden':
            # Abrupt change at regular intervals
            if self.time_step % int(1 / self.drift_rate) == 0 and self.time_step > 0:
                drift_factor = 1 + self.severity
            else:
                drift_factor = 1
            drifted_x = x * drift_factor
            self.drift_magnitude = abs(drift_factor - 1)
            
        elif self.drift_type == 'incremental':
            # Gradual change that accelerates
            drift_factor = 1 + (self.time_step**2 * self.drift_rate)
            drifted_x = x * drift_factor
            self.drift_magnitude = abs(drift_factor - 1)
            
        elif self.drift_type == 'random':
            # Random shifts
            random_shift = np.random.uniform(-1, 1, size=self.dim) * self.drift_rate * self.time_step
            drifted_x = x + random_shift
            self.drift_magnitude = np.linalg.norm(random_shift)
            
        elif self.drift_type == 'noise':
            # Function remains the same, but noise increases over time
            drifted_x = x
            self.drift_magnitude = self.noise_level * self.time_step
            
        else:
            # Unknown drift type
            drifted_x = x
            self.drift_magnitude = 0
            
        # Track drift characteristics
        self.drift_history.append(self.drift_magnitude)
            
        # Enforce bounds
        for i in range(self.dim):
            min_val, max_val = self.bounds[i]
            drifted_x[i] = max(min_val, min(max_val, drifted_x[i]))
            
        return drifted_x
    
    def get_drift_characteristics(self) -> Dict[str, Any]:
        """
        Get information about the drift characteristics.
        
        Returns
        -------
        Dict[str, Any]
            Dictionary with drift information:
            - drift_type: Type of drift
            - drift_rate: Rate of drift
            - current_magnitude: Current drift magnitude
            - average_magnitude: Average drift magnitude over time
            - drift_history: List of drift magnitudes over time
        """
        return {
            'drift_type': self.drift_type,
            'drift_rate': self.drift_rate,
            'current_magnitude': self.drift_magnitude,
            'average_magnitude': np.mean(self.drift_history) if self.drift_history else 0,
            'drift_history': self.drift_history.copy()
        }


def create_dynamic_benchmark(base_function: Callable[[np.ndarray], float],
                           dim: int = 2,
                           bounds: Optional[List[Tuple[float, float]]] = None,
                           drift_type: str = 'linear',
                           drift_rate: float = 0.01,
                           drift_interval: int = 100,
                           noise_level: float = 0.0,
                           severity: float = 1.0) -> DynamicFunction:
    """
    Create a dynamic benchmark function.
    
    Parameters
    ----------
    base_function : Callable[[np.ndarray], float]
        The static benchmark function to make dynamic
    dim : int
        Dimensionality of the function
    bounds : Optional[List[Tuple[float, float]]]
        Bounds for each dimension [(min1, max1), (min2, max2), ...]
    drift_type : str
        Type of drift to apply ('linear', 'oscillatory', 'sudden', etc.)
    drift_rate : float
        Rate parameter controlling how quickly the function changes
    drift_interval : int
        Number of function evaluations between drift changes
    noise_level : float
        Level of noise to add to function evaluations
    severity : float
        Scaling factor for the drift (higher = more severe drift)
    
    Returns
    -------
    DynamicFunction
        A dynamic benchmark function
    """
    return DynamicFunction(
        base_function=base_function,
        drift_type=drift_type,
        drift_rate=drift_rate,
        drift_interval=drift_interval,
        noise_level=noise_level,
        dim=dim,
        bounds=bounds,
        severity=severity
    )


# Example usage:
if __name__ == "__main__":
    from benchmarking.test_functions import ClassicalTestFunctions
    import matplotlib.pyplot as plt
    
    # Create a dynamic version of the sphere function
    dynamic_sphere = create_dynamic_benchmark(
        ClassicalTestFunctions.sphere,
        dim=2,
        drift_type='oscillatory',
        drift_rate=0.02
    )
    
    # Evaluate the function multiple times to see how it changes
    evaluations = 1000
    results = []
    
    for i in range(evaluations):
        # Create a random point
        x = np.random.uniform(-5, 5, size=2)
        
        # Evaluate
        value = dynamic_sphere.evaluate(x)
        results.append(value)
        
    # Plot the results
    plt.figure(figsize=(10, 6))
    plt.plot(results)
    plt.title('Dynamic Sphere Function (Oscillatory Drift)')
    plt.xlabel('Evaluation Number')
    plt.ylabel('Function Value')
    plt.grid(True)
    plt.savefig('dynamic_sphere.png')
    print(f"Plot saved to dynamic_sphere.png")
    
    # Get drift characteristics
    drift_info = dynamic_sphere.get_drift_characteristics()
    print("\nDrift Characteristics:")
    for key, value in drift_info.items():
        if key != 'drift_history':
            print(f"  {key}: {value}") 