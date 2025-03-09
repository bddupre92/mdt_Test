"""
Test functions for benchmarking optimization algorithms.
"""
import numpy as np
from typing import Dict, List, Tuple, Callable, Optional, Any


class TestFunction:
    """Base class for test functions."""
    
    def __init__(self, dim: int, bounds: List[Tuple[float, float]]):
        """
        Initialize the test function.
        
        Parameters
        ----------
        dim : int
            Dimensionality of the function
        bounds : List[Tuple[float, float]]
            Bounds for each dimension as (lower, upper)
        """
        self.dim = dim
        self.bounds = bounds
        
    def evaluate(self, x: np.ndarray) -> float:
        """
        Evaluate the function at point x.
        
        Parameters
        ----------
        x : numpy.ndarray
            The point to evaluate
            
        Returns
        -------
        float
            The function value at the given point
        """
        raise NotImplementedError("Subclasses must implement this method")
    
    def get_global_minimum(self) -> Tuple[np.ndarray, float]:
        """
        Get the global minimum of the function.
        
        Returns
        -------
        Tuple[numpy.ndarray, float]
            The global minimum point and value
        """
        raise NotImplementedError("Subclasses must implement this method")


class Sphere(TestFunction):
    """Sphere function: f(x) = sum(x^2)"""
    
    def evaluate(self, x: np.ndarray) -> float:
        """Evaluate the Sphere function at point x."""
        return np.sum(x**2)
    
    def get_global_minimum(self) -> Tuple[np.ndarray, float]:
        """Global minimum at the origin with value 0."""
        return np.zeros(self.dim), 0.0


class Rosenbrock(TestFunction):
    """Rosenbrock function: f(x) = sum(100 * (x[i+1] - x[i]^2)^2 + (x[i] - 1)^2)"""
    
    def evaluate(self, x: np.ndarray) -> float:
        """Evaluate the Rosenbrock function at point x."""
        return np.sum(100 * (x[1:] - x[:-1]**2)**2 + (x[:-1] - 1)**2)
    
    def get_global_minimum(self) -> Tuple[np.ndarray, float]:
        """Global minimum at (1,...,1) with value 0."""
        return np.ones(self.dim), 0.0


class Rastrigin(TestFunction):
    """Rastrigin function: f(x) = 10*n + sum(x^2 - 10*cos(2*pi*x))"""
    
    def evaluate(self, x: np.ndarray) -> float:
        """Evaluate the Rastrigin function at point x."""
        return 10 * self.dim + np.sum(x**2 - 10 * np.cos(2 * np.pi * x))
    
    def get_global_minimum(self) -> Tuple[np.ndarray, float]:
        """Global minimum at the origin with value 0."""
        return np.zeros(self.dim), 0.0


class Ackley(TestFunction):
    """Ackley function"""
    
    def evaluate(self, x: np.ndarray) -> float:
        """Evaluate the Ackley function at point x."""
        a, b, c = 20, 0.2, 2 * np.pi
        term1 = -a * np.exp(-b * np.sqrt(np.sum(x**2) / self.dim))
        term2 = -np.exp(np.sum(np.cos(c * x)) / self.dim)
        return term1 + term2 + a + np.exp(1)
    
    def get_global_minimum(self) -> Tuple[np.ndarray, float]:
        """Global minimum at the origin with value 0."""
        return np.zeros(self.dim), 0.0


class Griewank(TestFunction):
    """Griewank function: f(x) = 1 + sum(x^2)/4000 - prod(cos(x_i/sqrt(i)))"""
    
    def evaluate(self, x: np.ndarray) -> float:
        """Evaluate the Griewank function at point x."""
        sum_term = np.sum(x**2) / 4000
        prod_term = np.prod(np.cos(x / np.sqrt(np.arange(1, len(x) + 1))))
        return 1 + sum_term - prod_term
    
    def get_global_minimum(self) -> Tuple[np.ndarray, float]:
        """Global minimum at the origin with value 0."""
        return np.zeros(self.dim), 0.0


class Schwefel(TestFunction):
    """Schwefel function: f(x) = 418.9829*n - sum(x_i * sin(sqrt(|x_i|)))"""
    
    def evaluate(self, x: np.ndarray) -> float:
        """Evaluate the Schwefel function at point x."""
        return 418.9829 * self.dim - np.sum(x * np.sin(np.sqrt(np.abs(x))))
    
    def get_global_minimum(self) -> Tuple[np.ndarray, float]:
        """Global minimum at (420.9687,...,420.9687) with value 0."""
        return np.ones(self.dim) * 420.9687, 0.0


class Levy(TestFunction):
    """Levy function"""
    
    def evaluate(self, x: np.ndarray) -> float:
        """Evaluate the Levy function at point x."""
        w = 1 + (x - 1) / 4
        term1 = np.sin(np.pi * w[0])**2
        term2 = np.sum((w[:-1] - 1)**2 * (1 + 10 * np.sin(np.pi * w[:-1] + 1)**2))
        term3 = (w[-1] - 1)**2 * (1 + np.sin(2 * np.pi * w[-1])**2)
        return term1 + term2 + term3
    
    def get_global_minimum(self) -> Tuple[np.ndarray, float]:
        """Global minimum at (1,...,1) with value 0."""
        return np.ones(self.dim), 0.0


# Dictionary mapping function names to their constructors
TEST_FUNCTIONS = {
    "sphere": Sphere,
    "rosenbrock": Rosenbrock,
    "rastrigin": Rastrigin,
    "ackley": Ackley,
    "griewank": Griewank,
    "schwefel": Schwefel,
    "levy": Levy
}


def create_test_suite() -> Dict[str, Callable[[int, List[Tuple[float, float]]], TestFunction]]:
    """
    Create a test suite of benchmark functions.
    
    Returns
    -------
    Dict[str, Callable]
        Dictionary mapping function names to their constructors
    """
    return TEST_FUNCTIONS
