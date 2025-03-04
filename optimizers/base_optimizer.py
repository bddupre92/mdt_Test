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
    # Basic metrics
    best_solution: Optional[np.ndarray] = None
    best_score: float = float('inf')
    population: Optional[np.ndarray] = None
    evaluations: int = 0
    iteration: int = 0
    start_time: float = 0.0
    end_time: float = 0.0
    
    # Performance tracking
    success_history: List[bool] = field(default_factory=list)
    diversity_history: List[float] = field(default_factory=list)
    convergence_curve: List[float] = field(default_factory=list)
    history: List[Dict[str, Any]] = field(default_factory=list)
    
    # Advanced metrics
    time_per_iteration: List[float] = field(default_factory=list)
    last_improvement_iter: int = 0
    convergence_rate: Optional[float] = None
    stagnation_count: int = 0
    
    # Problem characteristics
    gradient_estimates: List[float] = field(default_factory=list)
    local_optima_count: int = 0
    landscape_ruggedness: Optional[float] = None
    
    # Exploration/exploitation balance
    exploration_phase: bool = True
    selection_pressure: List[float] = field(default_factory=list)
    
    # Parameter adaptation
    parameter_history: Dict[str, List[float]] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert state to dictionary for serialization."""
        return {
            # Basic metrics
            'best_score': float(self.best_score),
            'evaluations': self.evaluations,
            'iteration': self.iteration,
            'runtime': self.end_time - self.start_time if self.end_time > 0 else 0,
            
            # Performance tracking
            'success_rate': float(np.mean(self.success_history)) if self.success_history else 0,
            'current_diversity': float(self.diversity_history[-1]) if self.diversity_history else 0,
            'convergence_history': self.convergence_curve[-10:] if self.convergence_curve else [],
            
            # Advanced metrics
            'avg_time_per_iter': float(np.mean(self.time_per_iteration)) if self.time_per_iteration else 0,
            'iterations_since_improvement': self.iteration - self.last_improvement_iter,
            'convergence_rate': float(self.convergence_rate) if self.convergence_rate is not None else 0,
            'stagnation_count': self.stagnation_count,
            
            # Problem characteristics
            'estimated_local_optima': self.local_optima_count,
            'landscape_ruggedness': float(self.landscape_ruggedness) if self.landscape_ruggedness is not None else 0,
            
            # Exploration/exploitation
            'exploration_phase': self.exploration_phase,
            'selection_pressure': float(np.mean(self.selection_pressure[-10:])) if self.selection_pressure else 0,
            
            # Parameter adaptation summary
            'parameter_history': {k: v[-5:] for k, v in self.parameter_history.items()} if self.parameter_history else {}
        }


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
        
        # Advanced metrics
        self.time_per_iteration = []
        self.last_improvement_iter = 0
        self.convergence_rate = None
        self.stagnation_count = 0
        
        # Problem characteristics
        self.gradient_estimates = []
        self.local_optima_count = 0
        self.landscape_ruggedness = None
        
        # Exploration/exploitation balance
        self.exploration_phase = True
        self.selection_pressure = []
        
        # Parameter adaptation
        self.parameter_history = {}
        
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
                # Record iteration start time
                iter_start_time = time.time()
                
                # Perform one iteration
                self._iterate(self.objective_func)
                
                # Calculate iteration time
                iter_time = time.time() - iter_start_time
                self.time_per_iteration.append(iter_time)
                
                # Update diversity and parameters
                self._update_diversity()
                if self.adaptive:
                    self._update_parameters()
                
                # Check for improvement
                if len(self.convergence_curve) > 1:
                    if self.convergence_curve[-1] < self.convergence_curve[-2]:
                        self.last_improvement_iter = self._current_iteration
                        
                        # Calculate convergence rate
                        improvement = self.convergence_curve[-2] - self.convergence_curve[-1]
                        self.convergence_rate = improvement / iter_time
                    else:
                        # Increment stagnation counter
                        self.stagnation_count += 1
                
                # Estimate problem characteristics
                self._estimate_problem_characteristics()
                
                # Update exploration/exploitation phase
                if self._current_iteration > 10:
                    recent_diversity = self.diversity_history[-10:]
                    if np.mean(recent_diversity) < 0.1 * self.diversity_history[0]:
                        self.exploration_phase = False
                
                # Calculate selection pressure
                if len(self.success_history) >= 10:
                    self.selection_pressure.append(np.mean(self.success_history[-10:]))
                    
                # Record state
                if record_history:
                    self.history.append({
                        'iteration': self._current_iteration,
                        'best_score': float(self.best_score),
                        'evaluations': self.evaluations,
                        'diversity': self.diversity_history[-1] if self.diversity_history else 0.0,
                        'time_per_iter': self.time_per_iteration[-1] if self.time_per_iteration else 0.0,
                        'convergence_rate': float(self.convergence_rate) if self.convergence_rate is not None else 0.0,
                        'stagnation_count': self.stagnation_count,
                        'exploration_phase': self.exploration_phase
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
                'best_score': float(self.best_score),
                'evaluations': self.evaluations,
                'runtime': runtime,
                'convergence': self.convergence_curve,
                'history': self.history,
                'success_rate': float(np.mean(self.success_history)) if self.success_history else 0.0,
                'final_diversity': self.diversity_history[-1] if self.diversity_history else 0.0,
                'state': self.get_state().to_dict()
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
                'best_score': float('inf'),
                'evaluations': self.evaluations,
                'runtime': runtime,
                'error': str(e)
            }
    
    @abstractmethod
    def _iterate(self, objective_func: Callable):
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
            history=self.history,
            time_per_iteration=self.time_per_iteration,
            last_improvement_iter=self.last_improvement_iter,
            convergence_rate=self.convergence_rate,
            stagnation_count=self.stagnation_count,
            gradient_estimates=self.gradient_estimates,
            local_optima_count=self.local_optima_count,
            landscape_ruggedness=self.landscape_ruggedness,
            exploration_phase=self.exploration_phase,
            selection_pressure=self.selection_pressure,
            parameter_history=self.parameter_history
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
        self.time_per_iteration = state.time_per_iteration
        self.last_improvement_iter = state.last_improvement_iter
        self.convergence_rate = state.convergence_rate
        self.stagnation_count = state.stagnation_count
        self.gradient_estimates = state.gradient_estimates
        self.local_optima_count = state.local_optima_count
        self.landscape_ruggedness = state.landscape_ruggedness
        self.exploration_phase = state.exploration_phase
        self.selection_pressure = state.selection_pressure
        self.parameter_history = state.parameter_history
        
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
        self.time_per_iteration = []
        self.last_improvement_iter = 0
        self.convergence_rate = None
        self.stagnation_count = 0
        self.gradient_estimates = []
        self.local_optima_count = 0
        self.landscape_ruggedness = None
        self.exploration_phase = True
        self.selection_pressure = []
        self.parameter_history = {}
    
    def optimize(self, objective_func: Callable, max_evals: Optional[int] = None) -> np.ndarray:
        """
        Optimize the objective function and return the best solution.
        This method is designed to be compatible with the OptimizerAnalyzer.
        
        Args:
            objective_func: Objective function to minimize
            max_evals: Maximum number of function evaluations
            
        Returns:
            Best solution found (numpy array)
        """
        self.logger.info(f"Starting {self.__class__.__name__}.optimize with max_evals={max_evals}")
        
        # Run the optimization process
        result = self.run(objective_func, max_evals)
        
        # Extract the best solution to return
        best_solution = result.get('solution', None)
        
        # Convert to numpy array if needed
        if best_solution is not None and not isinstance(best_solution, np.ndarray):
            if isinstance(best_solution, list):
                best_solution = np.array(best_solution)
            else:
                # Try to convert or fallback to zeros
                try:
                    best_solution = np.array(best_solution)
                except:
                    self.logger.error(f"Could not convert solution to numpy array: {best_solution}")
                    best_solution = np.zeros(self.dim)
        
        # Return the best solution
        self.logger.info(f"Completed optimize with best score: {self.best_score}")
        return best_solution
    
    def _bound_solution(self, solution: np.ndarray) -> np.ndarray:
        """Ensure solution stays within bounds.
        
        Args:
            solution: Solution vector to bound
            
        Returns:
            Bounded solution vector
        """
        bounded_solution = solution.copy()
        for i in range(self.dim):
            low, high = self.bounds[i]
            bounded_solution[i] = np.clip(bounded_solution[i], low, high)
        return bounded_solution

    def _estimate_problem_characteristics(self) -> None:
        """Estimate problem characteristics."""
        pass  # Implemented by concrete optimizers

    def _update_state(self, solution: np.ndarray, score: float) -> None:
        """Update optimizer state with a new solution and its score.
        
        Args:
            solution: Solution vector
            score: Objective function value
        """
        # Increment evaluation counter
        self.evaluations += 1
        
        # Store in solutions and evaluations lists if we have a state object
        if hasattr(self, 'state'):
            if not hasattr(self.state, 'solutions'):
                self.state.solutions = []
            if not hasattr(self.state, 'evaluations'):
                self.state.evaluations = []
                
            self.state.solutions.append(solution.copy())
            self.state.evaluations.append(score)
        
        # Update best solution if needed
        if score < self.best_score:
            self.best_score = score
            self.best_solution = solution.copy()
            self.last_improvement_iter = self._current_iteration
            self.success_history.append(True)
            self.convergence_curve.append(self.best_score)
        else:
            self.success_history.append(False)
            if self.convergence_curve:
                self.convergence_curve.append(self.convergence_curve[-1])
            else:
                self.convergence_curve.append(self.best_score)
