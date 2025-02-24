"""
optimizer_factory.py
------------------
Factory for creating optimizer instances.
"""

from typing import Dict, Any, List, Tuple
from .de import DifferentialEvolutionOptimizer
from .gwo import GreyWolfOptimizer
from .es import EvolutionStrategyOptimizer
from .aco import AntColonyOptimizer

def create_optimizers(dim: int, bounds: List[Tuple[float, float]]) -> Dict[str, Any]:
    """
    Create optimizer instances.
    
    Args:
        dim: Problem dimensionality
        bounds: Parameter bounds
        
    Returns:
        Dictionary mapping optimizer names to optimizer instances
    """
    optimizers = {}
    
    # Create DE optimizers
    optimizers['DE (Standard)'] = DifferentialEvolutionOptimizer(
        dim=dim, bounds=bounds, adaptive=False
    )
    optimizers['DE (Adaptive)'] = DifferentialEvolutionOptimizer(
        dim=dim, bounds=bounds, adaptive=True
    )
    
    # Create GWO optimizers
    optimizers['GWO (Standard)'] = GreyWolfOptimizer(
        dim=dim, bounds=bounds, adaptive=False
    )
    optimizers['GWO (Adaptive)'] = GreyWolfOptimizer(
        dim=dim, bounds=bounds, adaptive=True
    )
    
    # Create ES optimizers
    optimizers['ES (Standard)'] = EvolutionStrategyOptimizer(
        dim=dim, bounds=bounds, adaptive=False
    )
    optimizers['ES (Adaptive)'] = EvolutionStrategyOptimizer(
        dim=dim, bounds=bounds, adaptive=True
    )
    
    # Create ACO optimizers
    optimizers['ACO (Standard)'] = AntColonyOptimizer(
        dim=dim, bounds=bounds, adaptive=False
    )
    optimizers['ACO (Adaptive)'] = AntColonyOptimizer(
        dim=dim, bounds=bounds, adaptive=True
    )
    
    return optimizers
