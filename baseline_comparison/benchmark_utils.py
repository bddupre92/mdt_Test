"""
Benchmark functions for optimization problems

This module provides standard benchmark functions for testing
optimization algorithms and algorithm selectors.
"""

import numpy as np
from typing import Tuple, Optional, Union, List, Callable

class BenchmarkFunction:
    """Base class for benchmark functions"""
    
    def __init__(self, name: str, dims: int):
        """
        Initialize the benchmark function
        
        Args:
            name: Name of the function
            dims: Number of dimensions
        """
        self.name = name
        self.dims = dims
        
    def evaluate(self, x: np.ndarray) -> float:
        """
        Evaluate the function
        
        Args:
            x: Input point
            
        Returns:
            Function value
        """
        raise NotImplementedError("Subclasses must implement evaluate")
    
    def __str__(self) -> str:
        return f"{self.name} ({self.dims}D)"


class Sphere(BenchmarkFunction):
    """Sphere function: sum(x_i^2)"""
    
    def __init__(self, dims: int = 10):
        """
        Initialize the Sphere function
        
        Args:
            dims: Number of dimensions
        """
        super().__init__("sphere", dims)
        self.bounds = (-5.12, 5.12)
        self.optimum = np.zeros(dims)
        self.optimum_value = 0.0
        
    def evaluate(self, x: np.ndarray) -> float:
        """
        Evaluate the Sphere function
        
        Args:
            x: Input point
            
        Returns:
            Function value
        """
        return np.sum(x**2)


class Rosenbrock(BenchmarkFunction):
    """Rosenbrock function: sum(100*(x_{i+1} - x_i^2)^2 + (1 - x_i)^2)"""
    
    def __init__(self, dims: int = 10):
        """
        Initialize the Rosenbrock function
        
        Args:
            dims: Number of dimensions
        """
        super().__init__("rosenbrock", dims)
        self.bounds = (-5.0, 10.0)
        self.optimum = np.ones(dims)
        self.optimum_value = 0.0
        
    def evaluate(self, x: np.ndarray) -> float:
        """
        Evaluate the Rosenbrock function
        
        Args:
            x: Input point
            
        Returns:
            Function value
        """
        return np.sum(100.0 * (x[1:] - x[:-1]**2)**2 + (1 - x[:-1])**2)


class Ackley(BenchmarkFunction):
    """Ackley function"""
    
    def __init__(self, dims: int = 10):
        """
        Initialize the Ackley function
        
        Args:
            dims: Number of dimensions
        """
        super().__init__("ackley", dims)
        self.bounds = (-32.768, 32.768)
        self.optimum = np.zeros(dims)
        self.optimum_value = 0.0
        
    def evaluate(self, x: np.ndarray) -> float:
        """
        Evaluate the Ackley function
        
        Args:
            x: Input point
            
        Returns:
            Function value
        """
        a = 20
        b = 0.2
        c = 2 * np.pi
        
        term1 = -a * np.exp(-b * np.sqrt(np.mean(x**2)))
        term2 = -np.exp(np.mean(np.cos(c * x)))
        
        return term1 + term2 + a + np.e


class Rastrigin(BenchmarkFunction):
    """Rastrigin function: 10*n + sum(x_i^2 - 10*cos(2*pi*x_i))"""
    
    def __init__(self, dims: int = 10):
        """
        Initialize the Rastrigin function
        
        Args:
            dims: Number of dimensions
        """
        super().__init__("rastrigin", dims)
        self.bounds = (-5.12, 5.12)
        self.optimum = np.zeros(dims)
        self.optimum_value = 0.0
        
    def evaluate(self, x: np.ndarray) -> float:
        """
        Evaluate the Rastrigin function
        
        Args:
            x: Input point
            
        Returns:
            Function value
        """
        return 10 * self.dims + np.sum(x**2 - 10 * np.cos(2 * np.pi * x))


class Griewank(BenchmarkFunction):
    """Griewank function"""
    
    def __init__(self, dims: int = 10):
        """
        Initialize the Griewank function
        
        Args:
            dims: Number of dimensions
        """
        super().__init__("griewank", dims)
        self.bounds = (-600.0, 600.0)
        self.optimum = np.zeros(dims)
        self.optimum_value = 0.0
        
    def evaluate(self, x: np.ndarray) -> float:
        """
        Evaluate the Griewank function
        
        Args:
            x: Input point
            
        Returns:
            Function value
        """
        term1 = np.sum(x**2) / 4000.0
        term2 = np.prod(np.cos(x / np.sqrt(np.arange(1, self.dims + 1))))
        
        return term1 - term2 + 1.0


class Schwefel(BenchmarkFunction):
    """Schwefel function"""
    
    def __init__(self, dims: int = 10):
        """
        Initialize the Schwefel function
        
        Args:
            dims: Number of dimensions
        """
        super().__init__("schwefel", dims)
        self.bounds = (-500.0, 500.0)
        self.optimum = np.ones(dims) * 420.9687
        self.optimum_value = 0.0
        
    def evaluate(self, x: np.ndarray) -> float:
        """
        Evaluate the Schwefel function
        
        Args:
            x: Input point
            
        Returns:
            Function value
        """
        return 418.9829 * self.dims - np.sum(x * np.sin(np.sqrt(np.abs(x))))


class DynamicFunction(BenchmarkFunction):
    """
    Wrapper for creating dynamic benchmark functions with time-varying components
    
    This class wraps a base benchmark function and adds time-varying
    components to create a dynamic optimization problem.
    """
    
    def __init__(
        self,
        base_function: BenchmarkFunction,
        drift_type: str = "linear",
        drift_speed: float = 0.01,
        severity: float = 1.0
    ):
        """
        Initialize the dynamic function
        
        Args:
            base_function: The base benchmark function
            drift_type: Type of drift ("linear", "oscillatory", "random", "abrupt")
            drift_speed: Speed of the drift
            severity: Severity of the drift
        """
        super().__init__(f"dynamic_{base_function.name}_{drift_type}", base_function.dims)
        self.base_function = base_function
        self.drift_type = drift_type
        self.drift_speed = drift_speed
        self.severity = severity
        self.bounds = base_function.bounds
        
        # Initialize time step
        self.t = 0
        
        # Initialize dynamic optimum
        self.optimum = base_function.optimum.copy()
        self.optimum_value = base_function.optimum_value
        
    def evaluate(self, x: np.ndarray) -> float:
        """
        Evaluate the dynamic function
        
        Args:
            x: Input point
            
        Returns:
            Function value
        """
        # Apply drift to the input
        x_t = self._apply_drift(x)
        
        # Evaluate the base function
        return self.base_function.evaluate(x_t)
    
    def _apply_drift(self, x: np.ndarray) -> np.ndarray:
        """
        Apply drift to the input
        
        Args:
            x: Input point
            
        Returns:
            Transformed input point
        """
        # Calculate drift vector
        drift = self._calculate_drift()
        
        # Apply drift to the input
        x_t = x - drift
        
        # Increment time step
        self.t += 1
        
        return x_t
    
    def _calculate_drift(self) -> np.ndarray:
        """
        Calculate the drift vector
        
        Returns:
            Drift vector
        """
        if self.drift_type == "linear":
            # Linear drift
            drift = self.severity * self.drift_speed * self.t * np.ones(self.dims)
        
        elif self.drift_type == "oscillatory":
            # Oscillatory drift
            freq = 2 * np.pi * self.drift_speed
            drift = self.severity * np.sin(freq * self.t / 100) * np.ones(self.dims)
        
        elif self.drift_type == "random":
            # Random drift
            drift = self.severity * self.drift_speed * np.random.randn(self.dims)
        
        elif self.drift_type == "abrupt":
            # Abrupt drift
            if self.t % 100 == 0:
                drift = self.severity * np.random.randn(self.dims)
            else:
                drift = np.zeros(self.dims)
        
        else:
            # No drift
            drift = np.zeros(self.dims)
        
        return drift


def get_benchmark_function(name: str, dims: int = 10, **kwargs) -> BenchmarkFunction:
    """
    Get a benchmark function by name
    
    Args:
        name: Name of the benchmark function
        dims: Number of dimensions
        **kwargs: Additional parameters for dynamic functions
        
    Returns:
        Benchmark function instance
    """
    # Create a dictionary of available functions
    function_map = {
        "sphere": Sphere,
        "rosenbrock": Rosenbrock,
        "ackley": Ackley,
        "rastrigin": Rastrigin,
        "griewank": Griewank,
        "schwefel": Schwefel
    }
    
    # Check if it's a dynamic function
    if name.startswith("dynamic_"):
        parts = name.split("_")
        if len(parts) >= 3:
            base_name = parts[1]
            drift_type = parts[2]
            
            # Get the base function
            if base_name in function_map:
                base_function = function_map[base_name](dims)
                
                # Create the dynamic function
                return DynamicFunction(
                    base_function,
                    drift_type=drift_type,
                    **kwargs
                )
    
    # Regular function
    if name in function_map:
        return function_map[name](dims)
    
    # Default to Sphere if name not found
    return Sphere(dims)


def get_all_benchmark_functions(dims: int = 10) -> List[BenchmarkFunction]:
    """
    Get a list of all available benchmark functions
    
    Args:
        dims: Number of dimensions
        
    Returns:
        List of benchmark function instances
    """
    functions = [
        Sphere(dims),
        Rosenbrock(dims),
        Ackley(dims),
        Rastrigin(dims),
        Griewank(dims),
        Schwefel(dims)
    ]
    
    return functions


def get_dynamic_benchmark_functions(dims: int = 10) -> List[BenchmarkFunction]:
    """
    Get a list of dynamic benchmark functions
    
    Args:
        dims: Number of dimensions
        
    Returns:
        List of dynamic benchmark function instances
    """
    base_functions = get_all_benchmark_functions(dims)
    drift_types = ["linear", "oscillatory", "random", "abrupt"]
    
    dynamic_functions = []
    
    for base_func in base_functions:
        for drift_type in drift_types:
            dynamic_functions.append(
                DynamicFunction(base_func, drift_type=drift_type)
            )
    
    return dynamic_functions 