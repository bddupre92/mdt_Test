"""
optimizer_factory.py
------------------
Factory for creating optimizer instances.
"""

from typing import Dict, Any, List, Tuple, Callable
import numpy as np
from scipy.optimize import differential_evolution

class DifferentialEvolutionWrapper:
    """Wrapper for scipy's differential evolution optimizer"""
    def __init__(self, dim: int, bounds: List[Tuple[float, float]], adaptive: bool = True):
        self.dim = dim
        self.bounds = bounds
        self.adaptive = adaptive
        self.name = 'DE (Adaptive)' if adaptive else 'DE'
    
    def optimize(self, objective_fn: Callable[[np.ndarray], float], max_evals: int = 100) -> Tuple[np.ndarray, float]:
        """Run optimization"""
        result = differential_evolution(
            objective_fn,
            bounds=self.bounds,
            maxiter=max_evals,
            popsize=10,
            mutation=(0.5, 1.0),
            recombination=0.7,
            updating='deferred' if not self.adaptive else 'immediate',
            workers=1
        )
        return result.x, result.fun

def create_optimizers(dim: int, bounds: List[Tuple[float, float]], include_meta: bool = False) -> Dict[str, Any]:
    """
    Create optimizer instances
    
    Args:
        dim: Problem dimensionality
        bounds: Parameter bounds [(min1, max1), (min2, max2), ...]
        include_meta: Whether to include meta-optimizer
    
    Returns:
        Dictionary mapping optimizer names to instances
    """
    optimizers = {}
    
    # Create differential evolution optimizers
    optimizers['DE (Standard)'] = DifferentialEvolutionWrapper(
        dim=dim, bounds=bounds, adaptive=False
    )
    optimizers['DE (Adaptive)'] = DifferentialEvolutionWrapper(
        dim=dim, bounds=bounds, adaptive=True
    )
    
    return optimizers
