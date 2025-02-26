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
from meta.meta_optimizer import MetaOptimizer as BaseMetaOptimizer
from app.core.optimization.meta_optimizer import OptimizationConfig

def create_optimizers(dim: int, bounds: List[Tuple[float, float]], include_meta: bool = True) -> Dict[str, Any]:
    """
    Create optimizer instances.
    
    Args:
        dim: Problem dimensionality
        bounds: Parameter bounds
        include_meta: Whether to include meta-optimizer
        
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
    
    # Add meta-optimizer if requested
    if include_meta:
        config = OptimizationConfig(
            population_size=50,
            max_iterations=100,
            feature_subset_size=dim,
            crossover_rate=0.8,
            mutation_rate=0.1,
            performance_threshold=0.7,
            drift_adaptation_rate=0.3
        )
        
        # Create meta-optimizer with all base optimizers
        meta_opt = BaseMetaOptimizer(
            dim=dim,
            bounds=bounds,
            optimizers=optimizers,
            history_file='meta/history.json',
            selection_file='meta/selection.json',
            n_parallel=2
        )
        optimizers['Meta-Optimizer'] = meta_opt
    
    return optimizers
