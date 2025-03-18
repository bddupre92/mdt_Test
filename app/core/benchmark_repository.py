"""
Benchmark Repository Module

This module provides a collection of standard benchmark functions for optimization testing.
"""

import numpy as np
from typing import Tuple, List, Optional, Dict, Any
from dataclasses import dataclass

@dataclass
class BenchmarkFunction:
    """Base class for benchmark functions."""
    name: str
    dimension: int
    bounds: Tuple[float, float]  # (lower_bound, upper_bound) for all dimensions
    optimal_value: float = 0.0
    
    def evaluate(self, x: np.ndarray) -> float:
        """Evaluate the function at point x."""
        raise NotImplementedError
    
    def get_info(self) -> Dict[str, Any]:
        """Get function metadata."""
        return {
            "name": self.name,
            "dimension": self.dimension,
            "bounds": self.bounds,
            "optimal_value": self.optimal_value
        }

class Sphere(BenchmarkFunction):
    """Sphere function: sum(x_i^2)."""
    def __init__(self, dimension: int = 2, bounds: Tuple[float, float] = (-5.0, 5.0)):
        super().__init__("Sphere", dimension, bounds)
    
    def evaluate(self, x: np.ndarray) -> float:
        return np.sum(x**2)

class Rosenbrock(BenchmarkFunction):
    """Rosenbrock function: sum(100*(x_{i+1} - x_i^2)^2 + (1 - x_i)^2)."""
    def __init__(self, dimension: int = 2, bounds: Tuple[float, float] = (-5.0, 5.0)):
        super().__init__("Rosenbrock", dimension, bounds)
    
    def evaluate(self, x: np.ndarray) -> float:
        return np.sum(100.0 * (x[1:] - x[:-1]**2)**2 + (1 - x[:-1])**2)

class Rastrigin(BenchmarkFunction):
    """Rastrigin function: 10*n + sum(x_i^2 - 10*cos(2*pi*x_i))."""
    def __init__(self, dimension: int = 2, bounds: Tuple[float, float] = (-5.12, 5.12)):
        super().__init__("Rastrigin", dimension, bounds)
    
    def evaluate(self, x: np.ndarray) -> float:
        return 10 * len(x) + np.sum(x**2 - 10 * np.cos(2 * np.pi * x))

class Ackley(BenchmarkFunction):
    """Ackley function."""
    def __init__(self, dimension: int = 2, bounds: Tuple[float, float] = (-32.768, 32.768)):
        super().__init__("Ackley", dimension, bounds)
    
    def evaluate(self, x: np.ndarray) -> float:
        n = len(x)
        sum_sq = np.sum(x**2)
        sum_cos = np.sum(np.cos(2 * np.pi * x))
        return -20 * np.exp(-0.2 * np.sqrt(sum_sq/n)) - np.exp(sum_cos/n) + 20 + np.e

class Griewank(BenchmarkFunction):
    """Griewank function."""
    def __init__(self, dimension: int = 2, bounds: Tuple[float, float] = (-600.0, 600.0)):
        super().__init__("Griewank", dimension, bounds)
    
    def evaluate(self, x: np.ndarray) -> float:
        sum_sq = np.sum(x**2) / 4000
        prod_cos = np.prod(np.cos(x / np.sqrt(np.arange(1, len(x) + 1))))
        return 1 + sum_sq - prod_cos

class Schwefel(BenchmarkFunction):
    """Schwefel function."""
    def __init__(self, dimension: int = 2, bounds: Tuple[float, float] = (-500.0, 500.0)):
        super().__init__("Schwefel", dimension, bounds)
        self.optimal_value = -418.9829 * dimension
    
    def evaluate(self, x: np.ndarray) -> float:
        return -np.sum(x * np.sin(np.sqrt(np.abs(x))))

class BenchmarkRepository:
    """Repository of benchmark functions."""
    
    def __init__(self):
        """Initialize the repository with default functions."""
        self.functions: Dict[str, BenchmarkFunction] = {}
        self._initialize_defaults()
    
    def _initialize_defaults(self):
        """Initialize default benchmark functions."""
        default_functions = [
            Sphere(),
            Rosenbrock(),
            Rastrigin(),
            Ackley(),
            Griewank(),
            Schwefel()
        ]
        for func in default_functions:
            self.functions[func.name.lower()] = func
    
    def get_function(self, name: str, dimension: int = 2, bounds: Optional[Tuple[float, float]] = None) -> BenchmarkFunction:
        """Get a benchmark function by name with specified dimension and bounds."""
        if name.lower() not in self.functions:
            raise ValueError(f"Unknown function: {name}")
        
        func_class = type(self.functions[name.lower()])
        if bounds:
            return func_class(dimension=dimension, bounds=bounds)
        return func_class(dimension=dimension)
    
    def get_all_functions(self, dimension: int = 2) -> List[BenchmarkFunction]:
        """Get all available benchmark functions with specified dimension."""
        return [self.get_function(name, dimension) for name in self.functions.keys()]
    
    def list_functions(self) -> List[str]:
        """List names of all available functions."""
        return list(self.functions.keys()) 