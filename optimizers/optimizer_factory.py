"""
optimizer_factory.py
-----------------
Factory for creating optimization algorithms
"""

from typing import Dict, Any, Optional, List, Tuple, Callable
import numpy as np
from scipy.optimize import differential_evolution
import time
import logging

from .base_optimizer import BaseOptimizer

class DifferentialEvolutionWrapper(BaseOptimizer):
    def __init__(self, dim: int = 5, bounds: Optional[List[Tuple[float, float]]] = None, population_size: int = 15, adaptive: bool = True):
        if bounds is None:
            bounds = [(0, 1)] * dim
        super().__init__(dim=dim, bounds=bounds, population_size=population_size, adaptive=adaptive)
        self.logger = logging.getLogger(__name__)
        
    def _optimize(self, objective: Callable) -> Tuple[np.ndarray, float]:
        """Run differential evolution optimization"""
        self.logger.info(f"Starting optimization with DE (adaptive={self.adaptive})")
        self.start_time = time.time()
        
        try:
            result = differential_evolution(
                objective,
                bounds=self.bounds,
                maxiter=self.max_evals // self.population_size,
                popsize=self.population_size,
                updating='deferred',
                workers=1,  # For progress tracking
                strategy='best1bin' if not self.adaptive else 'best1exp'
            )
            
            self.end_time = time.time()
            self.evaluations = result.nfev
            self.best_solution = result.x
            self.best_score = result.fun
            
            # Update convergence curve
            self.convergence_curve = [self.best_score]
            
            self.logger.info(f"DE optimization completed: score={self.best_score:.6f}, evals={self.evaluations}")
            return self.best_solution, self.best_score
            
        except Exception as e:
            self.logger.error(f"DE optimization failed: {str(e)}")
            # Set a default solution in case of failure
            self.best_solution = np.zeros(self.dim)
            self.best_score = float('inf')
            return self.best_solution, self.best_score
        
    def reset(self):
        """Reset optimizer state."""
        super().reset()
        self.evaluations = 0
        self.best_solution = None
        self.best_score = float('inf')
        self.convergence_curve = []
        self.start_time = None
        self.end_time = None
        
    def suggest(self) -> Dict[str, Any]:
        """Suggest a new configuration to evaluate.
        
        Returns:
            Dictionary with configuration parameters
        """
        # Generate a random solution within bounds
        solution = np.array([
            np.random.uniform(low, high)
            for low, high in self.bounds
        ])
        
        # Convert to dictionary format expected by MetaLearner
        config = {
            'n_estimators': int(100 + solution[0] * 900),  # 100-1000
            'max_depth': int(5 + solution[1] * 25),  # 5-30
            'min_samples_split': 2,
            'min_samples_leaf': 1,
            'max_features': 'sqrt',
            'bootstrap': True,
            'class_weight': None
        }
        
        return config
        
    def evaluate(self, config: Dict[str, Any]) -> float:
        """Placeholder for evaluation function.
        
        In the MetaLearner context, evaluation is handled externally.
        This is just a placeholder to satisfy the interface.
        
        Args:
            config: Configuration to evaluate
            
        Returns:
            Placeholder score
        """
        return 0.0
        
    def update(self, config: Dict[str, Any], score: float) -> None:
        """Update optimizer state with evaluation results.
        
        Args:
            config: Configuration that was evaluated
            score: Score from evaluation
        """
        # Track best solution
        if score > self.best_score:
            self.best_score = score
            
            # Convert config back to solution format
            solution = np.zeros(self.dim)
            if 'n_estimators' in config:
                solution[0] = (config['n_estimators'] - 100) / 900
            if 'max_depth' in config:
                solution[1] = (config['max_depth'] - 5) / 25
                
            self.best_solution = solution
            
        # Update convergence tracking
        self.convergence_curve.append(score)
        self.evaluations += 1

def create_optimizers(dim: int = 5, bounds: Optional[List[Tuple[float, float]]] = None) -> Dict[str, BaseOptimizer]:
    """Create dictionary of optimization algorithms
    
    Args:
        dim: Problem dimension
        bounds: Optional bounds for each dimension
        
    Returns:
        Dictionary mapping algorithm names to optimizer instances
    """
    if bounds is None:
        bounds = [(0, 1)] * dim
        
    return {
        'DE (Standard)': DifferentialEvolutionWrapper(dim=dim, bounds=bounds, adaptive=False),
        'DE (Adaptive)': DifferentialEvolutionWrapper(dim=dim, bounds=bounds, population_size=20, adaptive=True)
    }
