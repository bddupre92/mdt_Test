"""
base_optimizer.py
-----------------
Base class for optimization algorithms with common functionality
and adaptive parameter management.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Optional, Any, Callable
import numpy as np
import time
import logging
import pandas as pd
from dataclasses import dataclass, field


@dataclass
class OptimizerState:
    """Container for optimizer state."""
    best_solution: Optional[np.ndarray] = None
    best_score: float = float('inf')
    population: Optional[np.ndarray] = None
    evaluations: int = 0
    iteration: int = 0
    start_time: float = 0.0
    end_time: float = 0.0
    success_history: List[bool] = field(default_factory=list)
    diversity_history: List[float] = field(default_factory=list)
    convergence_curve: List[float] = field(default_factory=list)
    history: List[Dict[str, Any]] = field(default_factory=list)


class BaseOptimizer(ABC):
    """Base class for optimization algorithms."""
    
    def __init__(self, dim: int, bounds: List[Tuple[float, float]], 
                 population_size: Optional[int] = None,
                 adaptive: bool = True):
        """
        Initialize optimizer.
        
        Args:
            dim: Number of dimensions
            bounds: List of (min, max) bounds for each dimension
            population_size: Optional population size
            adaptive: Whether to use adaptive parameters
        """
        self.dim = dim
        self.bounds = bounds
        self.population_size = population_size or min(100, 10 * dim)
        self.adaptive = adaptive
        
        # Initialize state
        self.objective_func = None
        self.max_evals = None
        self.best_solution = None
        self.best_score = float('inf')
        self.population = None
        self.evaluations = 0
        self._current_iteration = 0
        self.start_time = 0
        self.end_time = 0
        
        # Performance tracking
        self.success_history = []
        self.diversity_history = []
        self.convergence_curve = []
        self.history = []
        
        # Progress callback for live visualization
        self.progress_callback = None
        
        # Configure logging
        self.logger = logging.getLogger(f"{self.__class__.__name__}")
        self.logger.setLevel(logging.DEBUG)
        
        # Log initialization
        self.logger.info(f"Initializing {self.__class__.__name__} with dim={dim}")
        self.logger.debug(f"Bounds: {bounds}")
        self.logger.debug(f"Population size: {self.population_size}")
        
    def set_objective(self, func: Callable) -> None:
        """Set objective function."""
        self.objective_func = func
        
    def _init_population(self) -> np.ndarray:
        """Initialize population using Latin Hypercube Sampling."""
        population = np.zeros((self.population_size, self.dim))
        
        # Generate Latin Hypercube samples
        for i in range(self.dim):
            population[:, i] = np.random.permutation(
                np.linspace(0, 1, self.population_size)
            )
            
        # Scale to bounds
        for i in range(self.dim):
            low, high = self.bounds[i]
            population[:, i] = low + (high - low) * population[:, i]
            
        return population
        
    def _update_diversity(self) -> None:
        """Update population diversity metrics."""
        if self.population is None:
            return
            
        # Calculate mean pairwise distance
        distances = []
        for i in range(min(len(self.population), 100)):  # Limit computation
            idx = np.random.choice(len(self.population), 2, replace=False)
            dist = np.linalg.norm(self.population[idx[0]] - self.population[idx[1]])
            distances.append(dist)
            
        diversity = np.mean(distances) if distances else 0.0
        self.diversity_history.append(diversity)
        
    def _check_convergence(self) -> bool:
        """Check if optimization should stop."""
        if self.max_evals and self.evaluations >= self.max_evals:
            return True
            
        if self.best_score < 1e-8:  # Optimal solution found
            return True
            
        # Check for stagnation
        if len(self.convergence_curve) > 50:
            recent_improvement = (self.convergence_curve[-50] - 
                                self.convergence_curve[-1])
            if recent_improvement < 1e-8:
                return True
                
        return False
        
    def _update_parameters(self) -> None:
        """Update adaptive parameters based on progress."""
        pass  # Implemented by concrete optimizers
        
    def get_convergence_curve(self) -> List[float]:
        """Get convergence curve"""
        return self.convergence_curve
        
    def run(self, objective_func: Optional[Callable] = None, max_evals: Optional[int] = None, record_history: bool = True) -> Dict[str, Any]:
        """Run the optimization process.
        
        Args:
            objective_func: Optional objective function to use
            max_evals: Maximum number of function evaluations
            record_history: Whether to record convergence history
            
        Returns:
            Dictionary containing optimization results
        """
        if objective_func is not None:
            self.set_objective(objective_func)
            
        if max_evals is not None:
            self.max_evals = max_evals
            
        self.start_time = time.time()
        
        try:
            # Initialize population if not already done
            if not hasattr(self, 'population') or self.population is None:
                self.population = self._init_population()
            
            # Main optimization loop
            while not self._check_convergence():
                # Perform one iteration
                self._iterate()
                
                # Update diversity and parameters
                self._update_diversity()
                if self.adaptive:
                    self._update_parameters()
                    
                # Record state
                if record_history:
                    self.history.append({
                        'iteration': self._current_iteration,
                        'best_score': float(self.best_score),
                        'evaluations': self.evaluations,
                        'diversity': self.diversity_history[-1] if self.diversity_history else 0.0
                    })
                    
                # Call progress callback if set
                if self.progress_callback:
                    self.progress_callback(
                        optimizer_name=getattr(self, 'name', self.__class__.__name__),
                        iteration=self._current_iteration,
                        score=float(self.best_score),
                        evaluations=self.evaluations
                    )
                
            self.end_time = time.time()
            runtime = self.end_time - self.start_time
            
            # Prepare results
            results = {
                'solution': self.best_solution.tolist() if self.best_solution is not None else None,
                'score': float(self.best_score),
                'evaluations': self.evaluations,
                'runtime': runtime,
                'convergence': self.convergence_curve,
                'history': self.history,
                'success_rate': float(np.mean(self.success_history)),
                'final_diversity': self.diversity_history[-1] if self.diversity_history else 0.0
            }
            
            return results
            
        except Exception as e:
            self.end_time = time.time()
            runtime = self.end_time - self.start_time
            
            # Log the error and return partial results
            self.logger.error(f"Error in optimization: {str(e)}")
            
            return {
                'solution': None,
                'score': float('inf'),
                'evaluations': self.evaluations,
                'runtime': runtime,
                'error': str(e)
            }
    
    @abstractmethod
    def _iterate(self):
        """Perform one iteration of the optimization algorithm.
        This method must be implemented by concrete optimizer classes."""
        pass
    
    def get_state(self) -> 'OptimizerState':
        """Get current optimizer state."""
        return OptimizerState(
            best_solution=self.best_solution,
            best_score=self.best_score,
            population=self.population,
            evaluations=self.evaluations,
            iteration=self._current_iteration,
            start_time=self.start_time,
            end_time=self.end_time,
            success_history=self.success_history,
            diversity_history=self.diversity_history,
            convergence_curve=self.convergence_curve,
            history=self.history
        )
        
    def set_state(self, state: 'OptimizerState') -> None:
        """Set optimizer state."""
        self.best_solution = state.best_solution
        self.best_score = state.best_score
        self.population = state.population
        self.evaluations = state.evaluations
        self._current_iteration = state.iteration
        self.start_time = state.start_time
        self.end_time = state.end_time
        self.success_history = state.success_history
        self.diversity_history = state.diversity_history
        self.convergence_curve = state.convergence_curve
        self.history = state.history
        
    def reset(self) -> None:
        """Reset optimizer state."""
        self.best_solution = None
        self.best_score = float('inf')
        self.population = None
        self.evaluations = 0
        self._current_iteration = 0
        self.start_time = 0
        self.end_time = 0
        self.success_history = []
        self.diversity_history = []
        self.convergence_curve = []
        self.history = []
