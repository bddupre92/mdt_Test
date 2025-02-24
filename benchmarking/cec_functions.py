"""
cec_functions.py
---------------
Implementation of CEC benchmark functions for optimization.
Includes shifted, rotated, and hybrid functions from CEC competitions.
"""

import numpy as np
from typing import List, Tuple, Callable
from dataclasses import dataclass
from .test_functions import TestFunction

class CECTestFunctions:
    @staticmethod
    def shifted_sphere(x: np.ndarray, shift: np.ndarray = None) -> float:
        """Shifted Sphere Function f1"""
        x = np.asarray(x)
        if shift is None:
            shift = np.zeros_like(x)
        z = x - shift
        return np.sum(z**2)
    
    @staticmethod
    def shifted_schwefel(x: np.ndarray, shift: np.ndarray = None) -> float:
        """Shifted Schwefel's Problem 1.2 f2"""
        x = np.asarray(x)
        if shift is None:
            shift = np.zeros_like(x)
        z = x - shift
        n = len(z)
        return np.sum([np.sum(z[:i+1])**2 for i in range(n)])
    
    @staticmethod
    def shifted_rotated_high_conditioned_elliptic(
            x: np.ndarray, 
            shift: np.ndarray = None,
            rotation: np.ndarray = None
        ) -> float:
        """Shifted Rotated High Conditioned Elliptic Function f3"""
        x = np.asarray(x)
        if shift is None:
            shift = np.zeros_like(x)
        if rotation is None:
            rotation = np.eye(len(x))
        z = rotation @ (x - shift)
        d = len(z)
        return np.sum([(1e6)**(i/(d-1)) * z[i]**2 for i in range(d)])
    
    @staticmethod
    def shifted_rotated_rosenbrock(
            x: np.ndarray,
            shift: np.ndarray = None,
            rotation: np.ndarray = None
        ) -> float:
        """Shifted Rotated Rosenbrock's Function f4"""
        x = np.asarray(x)
        if shift is None:
            shift = np.zeros_like(x)
        if rotation is None:
            rotation = np.eye(len(x))
        z = rotation @ (x - shift)
        return np.sum(100.0*(z[1:] - z[:-1]**2)**2 + (1 - z[:-1])**2)
    
    @staticmethod
    def shifted_rotated_ackley_with_global_optimum_on_bounds(
            x: np.ndarray,
            shift: np.ndarray = None,
            rotation: np.ndarray = None
        ) -> float:
        """Shifted Rotated Ackley's Function with Global Optimum on Bounds f5"""
        x = np.asarray(x)
        if shift is None:
            shift = np.zeros_like(x)
        if rotation is None:
            rotation = np.eye(len(x))
        z = rotation @ (x - shift)
        
        a, b, c = 20, 0.2, 2*np.pi
        d = len(z)
        sum1 = np.sum(z**2)
        sum2 = np.sum(np.cos(c*z))
        
        return -a * np.exp(-b*np.sqrt(sum1/d)) - np.exp(sum2/d) + a + np.exp(1)
    
    @staticmethod
    def shifted_rotated_weierstrass(
            x: np.ndarray,
            shift: np.ndarray = None,
            rotation: np.ndarray = None,
            a: float = 0.5,
            b: float = 3,
            k_max: int = 20
        ) -> float:
        """Shifted Rotated Weierstrass Function f6"""
        x = np.asarray(x)
        if shift is None:
            shift = np.zeros_like(x)
        if rotation is None:
            rotation = np.eye(len(x))
        z = rotation @ (x - shift)
        
        def weierstrass_sum(x):
            return sum(a**k * np.cos(2*np.pi*b**k * (x + 0.5)) for k in range(k_max))
        
        return sum(weierstrass_sum(z[i]) for i in range(len(z))) - \
               len(z) * sum(a**k * np.cos(2*np.pi*b**k * 0.5) for k in range(k_max))

def create_cec_suite(dim: int, bounds: List[Tuple[float, float]]) -> dict:
    """Create a suite of CEC benchmark functions with specified dimension and bounds"""
    # Generate random shifts and rotations for each function
    shift = np.random.uniform(-80, 80, dim)
    rotation = np.random.normal(0, 1, (dim, dim))
    rotation = rotation @ rotation.T  # Make rotation matrix orthogonal
    
    return {
        'shifted_sphere': TestFunction(
            name='Shifted Sphere',
            func=lambda x: CECTestFunctions.shifted_sphere(x, shift),
            dim=dim,
            bounds=bounds,
            global_minimum=0.0,
            characteristics={'multimodal': False, 'separable': True}
        ),
        'shifted_schwefel': TestFunction(
            name='Shifted Schwefel',
            func=lambda x: CECTestFunctions.shifted_schwefel(x, shift),
            dim=dim,
            bounds=bounds,
            global_minimum=0.0,
            characteristics={'multimodal': False, 'separable': False}
        ),
        'shifted_rotated_elliptic': TestFunction(
            name='Shifted Rotated Elliptic',
            func=lambda x: CECTestFunctions.shifted_rotated_high_conditioned_elliptic(x, shift, rotation),
            dim=dim,
            bounds=bounds,
            global_minimum=0.0,
            characteristics={'multimodal': False, 'separable': False}
        ),
        'shifted_rotated_rosenbrock': TestFunction(
            name='Shifted Rotated Rosenbrock',
            func=lambda x: CECTestFunctions.shifted_rotated_rosenbrock(x, shift, rotation),
            dim=dim,
            bounds=bounds,
            global_minimum=0.0,
            characteristics={'multimodal': False, 'separable': False}
        ),
        'shifted_rotated_ackley': TestFunction(
            name='Shifted Rotated Ackley',
            func=lambda x: CECTestFunctions.shifted_rotated_ackley_with_global_optimum_on_bounds(x, shift, rotation),
            dim=dim,
            bounds=bounds,
            global_minimum=0.0,
            characteristics={'multimodal': True, 'separable': False}
        ),
        'shifted_rotated_weierstrass': TestFunction(
            name='Shifted Rotated Weierstrass',
            func=lambda x: CECTestFunctions.shifted_rotated_weierstrass(x, shift, rotation),
            dim=dim,
            bounds=bounds,
            global_minimum=0.0,
            characteristics={'multimodal': True, 'separable': False}
        )
    }
