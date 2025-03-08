"""
optimizer_factory.py
-----------------
Factory for creating optimization algorithms
"""

from typing import Dict, Any, Optional, List, Tuple, Callable, Union, Type
import numpy as np
import logging
import scipy.optimize as optimize
import time
from optimizers.base_optimizer import BaseOptimizer
from optimizers.ant_colony import AntColonyOptimizer
from optimizers.grey_wolf import GreyWolfOptimizer
from optimizers.differential_evolution import DifferentialEvolutionOptimizer
from optimizers.evolution_strategy import EvolutionStrategyOptimizer
from meta.meta_optimizer import MetaOptimizer
import importlib
import sys
import os

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import and reload optimizer modules
from optimizers.ant_colony import AntColonyOptimizer
from optimizers.grey_wolf import GreyWolfOptimizer
from optimizers.differential_evolution import DifferentialEvolutionOptimizer
from optimizers.evolution_strategy import EvolutionStrategyOptimizer
from meta.meta_optimizer import MetaOptimizer

# Force reload of modules
importlib.reload(sys.modules['optimizers.ant_colony'])
importlib.reload(sys.modules['optimizers.grey_wolf'])
importlib.reload(sys.modules['optimizers.differential_evolution'])
importlib.reload(sys.modules['optimizers.evolution_strategy'])
importlib.reload(sys.modules['meta.meta_optimizer'])

class DifferentialEvolutionWrapper(DifferentialEvolutionOptimizer):
    """Wrapper for Differential Evolution optimizer that implements BaseOptimizer interface."""
    
    def __init__(self, dim: int = 5, bounds: Optional[List[Tuple[float, float]]] = None, 
                 population_size: int = 15, adaptive: bool = True, name: str = "DE",
                 F: float = 0.8, CR: float = 0.7):
        """
        Initialize DE wrapper.
        
        Args:
            dim: Number of dimensions
            bounds: List of (min, max) bounds for each dimension
            population_size: Number of individuals in population
            adaptive: Whether to use adaptive parameters
            name: Name of the optimizer
            F: Mutation factor
            CR: Crossover probability
        """
        if bounds is None:
            bounds = [(0, 1)] * dim
        super().__init__(dim=dim, bounds=bounds, population_size=population_size, 
                         adaptive=adaptive, F=F, CR=CR)
        self.logger = logging.getLogger(__name__)
        self.name = name
        self.max_evals = 10000
        
    def optimize(self, objective_func: Callable, max_evals: Optional[int] = None, record_history: bool = False) -> Tuple[np.ndarray, float]:
        """Run optimization
        
        Args:
            objective_func: Objective function to minimize
            max_evals: Maximum number of function evaluations
            record_history: Whether to record the convergence history
            
        Returns:
            Tuple of (best solution, best score)
        """
        if max_evals is not None:
            self.max_evals = max_evals
            
        self.logger.info(f"Starting DE optimization with max_evals={self.max_evals}")
        
        # Initialize
        self.reset()  # Use the reset method from the parent class
        self.best_solution = None
        self.best_score = float('inf')
        self.evaluations = 0
        self.convergence_curve = []
        
        # Run optimization
        self._iterate(objective_func)
        
        if record_history:
            return self.best_solution, self.best_score, self.convergence_curve
        return self.best_solution, self.best_score
    
    def get_parameters(self) -> Dict[str, Any]:
        """Get current parameter settings
        
        Returns:
            Dictionary of parameter settings
        """
        return {
            "population_size": self.population_size,
            "F": self.F,
            "CR": self.CR,
            "adaptive": self.adaptive,
            "max_evals": self.max_evals,
            "dim": self.dim
        }
        
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


    def _optimize_scipy(self, objective: Callable) -> Tuple[np.ndarray, float]:
        """Run scipy's differential evolution optimization"""
        self.logger.info(f"Starting optimization with scipy DE (adaptive={self.adaptive})")
        self.start_time = time.time()
        
        # Create a wrapper function to count evaluations and enforce the limit
        self.evaluations = 0
        max_evals = self.max_evals
        
        def objective_with_limit(x):
            if self.evaluations >= max_evals:
                return float('inf')  # Return a bad score if we exceed evaluations
            self.evaluations += 1
            return objective(x)
        
        try:
            # Calculate maxiter to keep total evaluations close to max_evals
            # In scipy DE, total evals ≈ maxiter * popsize
            maxiter = max(1, max_evals // (self.population_size * 2))
            
            result = optimize.differential_evolution(
                objective_with_limit,
                bounds=self.bounds,
                maxiter=maxiter,
                popsize=self.population_size,
                updating='deferred',
                workers=1,  # For progress tracking
                strategy='best1bin' if not self.adaptive else 'best1exp'
            )
            
            self.end_time = time.time()
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
        
    def _iterate(self, objective_func: Callable) -> Tuple[np.ndarray, float]:
        """Perform one iteration of the optimization algorithm
        
        Args:
            objective_func: Objective function to minimize
            
        Returns:
            Tuple of (best solution, best score)
        """
        return self._optimize_scipy(objective_func)

    def run(self, objective_func: Optional[Callable] = None, max_evals: Optional[int] = None) -> Dict[str, Any]:
        """Run optimization
        
        Args:
            objective_func: Objective function to minimize
            max_evals: Maximum number of function evaluations
            
        Returns:
            Dictionary with optimization results
        """
        try:
            # Use the objective_func passed as parameter or the one set via set_objective
            if objective_func is not None:
                self.set_objective(objective_func)
                
            if self.objective_func is None:
                raise ValueError("No objective function provided. Call set_objective() or provide objective_func parameter.")
                
            # Run optimization
            solution, score = self.optimize(self.objective_func, max_evals)
            
            # Return results
            return {
                'solution': solution,
                'score': score,
                'evaluations': self.evaluations,
                'runtime': 0.0,  # Not tracked
                'convergence_curve': self.convergence_curve if hasattr(self, 'convergence_curve') else []
            }
        except Exception as e:
            self.logger.error(f"Error in optimization: {str(e)}")
            return {
                'solution': None,
                'score': float('inf'),
                'evaluations': 0,
                'runtime': 0.0,
                'error': str(e)
            }


class EvolutionStrategyWrapper(EvolutionStrategyOptimizer):
    """Wrapper for Evolution Strategy optimizer that implements BaseOptimizer interface."""
    
    def __init__(self, dim: int = 5, bounds: Optional[List[Tuple[float, float]]] = None, 
                 population_size: int = 100, adaptive: bool = True, name: str = "ES"):
        """
        Initialize ES wrapper.
        
        Args:
            dim: Number of dimensions
            bounds: List of (min, max) bounds for each dimension
            population_size: Population size (lambda)
            adaptive: Whether to use adaptive step size
            name: Name of the optimizer
        """
        if bounds is None:
            bounds = [(0, 1)] * dim
        super().__init__(dim=dim, bounds=bounds, population_size=population_size, 
                         adaptive=adaptive)
        self.logger = logging.getLogger(__name__)
        self.name = name
        self.max_evals = 10000
        
    def get_parameters(self) -> Dict[str, Any]:
        """Get optimizer parameters
        
        Returns:
            Dictionary of parameter settings
        """
        return {
            "population_size": self.population_size,
            "adaptive": self.adaptive,
            "max_evals": self.max_evals,
            "dim": self.dim
        }
        
    def run(self, objective_func: Optional[Callable] = None, max_evals: Optional[int] = None) -> Dict[str, Any]:
        """Run optimization
        
        Args:
            objective_func: Objective function to minimize
            max_evals: Maximum number of function evaluations
            
        Returns:
            Dictionary with optimization results
        """
        try:
            # Use the objective_func passed as parameter or the one set via set_objective
            if objective_func is not None:
                self.set_objective(objective_func)
                
            if self.objective_func is None:
                raise ValueError("No objective function provided. Call set_objective() or provide objective_func parameter.")
                
            # Run optimization
            solution, score = self.optimize(self.objective_func, max_evals)
            
            # Return results
            return {
                'solution': solution,
                'score': score,
                'evaluations': self.evaluations,
                'runtime': 0.0,  # Not tracked
                'convergence_curve': self.convergence_curve if hasattr(self, 'convergence_curve') else []
            }
        except Exception as e:
            self.logger.error(f"Error in optimization: {str(e)}")
            return {
                'solution': None,
                'score': float('inf'),
                'evaluations': 0,
                'runtime': 0.0,
                'error': str(e)
            }
        
    def optimize(self, objective_func: Callable, max_evals: Optional[int] = None, record_history: bool = False) -> Tuple[np.ndarray, float]:
        """Run optimization
        
        Args:
            objective_func: Objective function to minimize
            max_evals: Maximum number of function evaluations
            record_history: Whether to record the convergence history
            
        Returns:
            Tuple of (best solution, best score)
        """
        if max_evals is not None:
            self.max_evals = max_evals
            
        self.logger.info(f"Starting ES optimization with max_evals={self.max_evals}")
        
        # Initialize
        self.reset()  # Use the reset method from the parent class
        self.best_solution = None
        self.best_score = float('inf')
        self.evaluations = 0
        self.convergence_curve = []
        
        # Run optimization
        self._iterate(objective_func)
        
        if record_history:
            return self.best_solution, self.best_score, self.convergence_curve
        return self.best_solution, self.best_score
        
    def _iterate(self, objective_func: Callable) -> Tuple[np.ndarray, float]:
        """Perform one iteration of the optimization algorithm
        
        Args:
            objective_func: Objective function to minimize
            
        Returns:
            Tuple of (best solution, best score)
        """
        # Initialize population if not already done
        if not hasattr(self, 'population') or self.population is None:
            self.population = np.array([
                np.array([np.random.uniform(low, high) for low, high in self.bounds])
                for _ in range(self.population_size)
            ])
            self.population_scores = np.array([objective_func(ind) for ind in self.population])
            self.evaluations += self.population_size
        
        # Track best solution
        best_idx = np.argmin(self.population_scores)
        if self.population_scores[best_idx] < self.best_score:
            self.best_score = self.population_scores[best_idx]
            self.best_solution = self.population[best_idx].copy()
        
        # Simple ES iteration
        for _ in range(self.max_evals // self.population_size):
            # Check if max evaluations reached
            if self.evaluations >= self.max_evals:
                break
                
            # Generate offspring
            offspring = []
            for i in range(self.population_size):
                # Select parents
                parent_indices = np.random.choice(self.population_size, 2, replace=False)
                parent1 = self.population[parent_indices[0]]
                parent2 = self.population[parent_indices[1]]
                
                # Recombine
                child = (parent1 + parent2) / 2.0
                
                # Mutate
                sigma = 0.1
                child = child + np.random.normal(0, sigma, size=self.dim)
                
                # Bound
                child = np.clip(child, 
                               [low for low, _ in self.bounds],
                               [high for _, high in self.bounds])
                
                offspring.append(child)
            
            # Evaluate offspring
            offspring_scores = np.array([objective_func(ind) for ind in offspring])
            self.evaluations += len(offspring)
            
            # Select survivors (mu + lambda)
            combined = np.vstack([self.population, offspring])
            combined_scores = np.concatenate([self.population_scores, offspring_scores])
            
            # Sort by score
            sorted_indices = np.argsort(combined_scores)
            self.population = combined[sorted_indices[:self.population_size]]
            self.population_scores = combined_scores[sorted_indices[:self.population_size]]
            
            # Update best solution
            if self.population_scores[0] < self.best_score:
                self.best_score = self.population_scores[0]
                self.best_solution = self.population[0].copy()
                
            # Update convergence curve
            if hasattr(self, 'convergence_curve'):
                self.convergence_curve.append(self.best_score)
        
        return self.best_solution, self.best_score
        
    def suggest(self) -> Dict[str, Any]:
        """Suggest a new configuration to evaluate."""
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
        """Placeholder for evaluation function."""
        return 0.0
        
    def update(self, config: Dict[str, Any], score: float) -> None:
        """Update optimizer state with evaluation results."""
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


class AntColonyWrapper(AntColonyOptimizer):
    """Wrapper for Ant Colony Optimizer that implements BaseOptimizer interface."""
    
    def __init__(self, dim: int = 5, bounds: Optional[List[Tuple[float, float]]] = None, 
                 population_size: int = 50, adaptive: bool = True, name: str = "ACO",
                 alpha: float = 1.0, beta: float = 2.0, 
                 evaporation_rate: float = 0.1, q: float = 1.0):
        """
        Initialize ACO wrapper.
        
        Args:
            dim: Number of dimensions
            bounds: List of (min, max) bounds for each dimension
            population_size: Number of ants
            adaptive: Whether to use adaptive parameters
            name: Name of the optimizer
            alpha: Pheromone importance
            beta: Heuristic importance
            evaporation_rate: Pheromone evaporation rate
            q: Pheromone deposit factor
        """
        if bounds is None:
            bounds = [(0, 1)] * dim
        super().__init__(dim=dim, bounds=bounds, population_size=population_size, 
                         adaptive=adaptive, alpha=alpha, beta=beta, 
                         evaporation_rate=evaporation_rate, q=q)
        self.logger = logging.getLogger(__name__)
        self.name = name
        self.max_evals = 10000
        
    def get_parameters(self) -> Dict[str, Any]:
        """Get optimizer parameters
        
        Returns:
            Dictionary of parameter settings
        """
        return {
            "population_size": self.population_size,
            "alpha": self.alpha,
            "beta": self.beta,
            "evaporation_rate": self.evaporation_rate,
            "q": self.q if hasattr(self, 'q') else None,
            "adaptive": self.adaptive,
            "max_evals": self.max_evals,
            "dim": self.dim
        }
        
    def run(self, objective_func: Optional[Callable] = None, max_evals: Optional[int] = None) -> Dict[str, Any]:
        """Run optimization
        
        Args:
            objective_func: Objective function to minimize
            max_evals: Maximum number of function evaluations
            
        Returns:
            Dictionary with optimization results
        """
        try:
            # Use the objective_func passed as parameter or the one set via set_objective
            if objective_func is not None:
                self.set_objective(objective_func)
                
            if self.objective_func is None:
                raise ValueError("No objective function provided. Call set_objective() or provide objective_func parameter.")
                
            # Run optimization
            solution, score = self.optimize(self.objective_func, max_evals)
            
            # Return results
            return {
                'solution': solution,
                'score': score,
                'evaluations': self.evaluations,
                'runtime': 0.0,  # Not tracked
                'convergence_curve': self.convergence_curve if hasattr(self, 'convergence_curve') else []
            }
        except Exception as e:
            self.logger.error(f"Error in optimization: {str(e)}")
            return {
                'solution': None,
                'score': float('inf'),
                'evaluations': 0,
                'runtime': 0.0,
                'error': str(e)
            }
        
    def _evaluate(self, solution: np.ndarray, objective_func: Callable) -> float:
        """Evaluate a solution
        
        Args:
            solution: Solution to evaluate
            objective_func: Objective function to evaluate
            
        Returns:
            Objective function value
        """
        self.evaluations += 1
        return objective_func(solution)
        
    def _generate_solutions(self) -> np.ndarray:
        """Generate solutions for each ant
        
        Returns:
            Array of solutions
        """
        solutions = []
        for ant in range(self.population_size):
            # Construct solution
            solution = np.zeros(self.dim)
            for d in range(self.dim):
                # Calculate probabilities
                pheromone = self.pheromone[d]
                heuristic = 1.0 / (1.0 + np.abs(np.linspace(self.bounds[d][0], self.bounds[d][1], self.num_points)))
                
                # Combine pheromone and heuristic information
                probs = (pheromone ** self.alpha) * (heuristic ** self.beta)
                probs = probs / np.sum(probs)
                
                # Select value
                point_idx = np.random.choice(self.num_points, p=probs)
                solution[d] = self._value_from_index(d, point_idx)
            
            solutions.append(solution)
            
        return np.array(solutions)
        
    def optimize(self, objective_func: Callable, max_evals: Optional[int] = None, record_history: bool = False) -> Tuple[np.ndarray, float]:
        """Run optimization
        
        Args:
            objective_func: Objective function to minimize
            max_evals: Maximum number of function evaluations
            record_history: Whether to record the convergence history
            
        Returns:
            Tuple of (best solution, best score)
        """
        if max_evals is not None:
            self.max_evals = max_evals
            
        self.logger.info(f"Starting ACO optimization with max_evals={self.max_evals}")
        
        # Initialize
        self.reset()
        self.best_solution = None
        self.best_score = float('inf')
        self.evaluations = 0
        self.convergence_curve = []
        
        # Run optimization
        self._iterate(objective_func)
        
        if record_history:
            return self.best_solution, self.best_score, self.convergence_curve
        return self.best_solution, self.best_score
        
    def _iterate(self, objective_func: Callable) -> Tuple[np.ndarray, float]:
        """Run one iteration of the algorithm
        
        Args:
            objective_func: Objective function to minimize
            
        Returns:
            Tuple of (best solution, best score)
        """
        # Initialize
        if self.best_solution is None:
            self.best_solution = np.zeros(self.dim)
            self.best_score = float('inf')
            
        # Run optimization for max_evals iterations
        for _ in range(self.max_evals):
            # Check if max evaluations reached
            if self.evaluations >= self.max_evals:
                break
                
            # Generate solutions
            solutions = self._generate_solutions()
            
            # Evaluate solutions
            scores = np.array([objective_func(solution) for solution in solutions])
            self.evaluations += len(scores)
            
            # Update best solution
            best_idx = np.argmin(scores)
            if scores[best_idx] < self.best_score:
                self.best_solution = solutions[best_idx].copy()
                self.best_score = scores[best_idx]
                
            # Update pheromone levels
            self._update_pheromones(solutions, scores)
            
            # Update parameters if adaptive
            if self.adaptive:
                self._update_parameters()
                
            # Track convergence
            self.convergence_curve.append(self.best_score)
            
        return self.best_solution, self.best_score
        
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
            'min_samples_split': int(2 + solution[2] * 18),  # 2-20
            'min_samples_leaf': int(1 + solution[3] * 9),  # 1-10
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
            if 'min_samples_split' in config:
                solution[2] = (config['min_samples_split'] - 2) / 18
            if 'min_samples_leaf' in config:
                solution[3] = (config['min_samples_leaf'] - 1) / 9
                
            self.best_solution = solution
            
        # Update convergence tracking
        self.convergence_curve.append(score)
        self.evaluations += 1


class GreyWolfWrapper(GreyWolfOptimizer):
    """Wrapper for Grey Wolf Optimizer that implements BaseOptimizer interface."""
    
    def __init__(self, dim: int = 5, bounds: Optional[List[Tuple[float, float]]] = None, 
                 population_size: int = 50, name: str = "GWO"):
        """
        Initialize GWO wrapper.
        
        Args:
            dim: Number of dimensions
            bounds: List of (min, max) bounds for each dimension
            population_size: Population size
            name: Name of the optimizer
        """
        if bounds is None:
            bounds = [(0, 1)] * dim
        super().__init__(dim=dim, bounds=bounds, population_size=population_size)
        self.logger = logging.getLogger(__name__)
        self.name = name
        self.max_evals = 10000
        
    def run(self, objective_func: Optional[Callable] = None, max_evals: Optional[int] = None) -> Dict[str, Any]:
        """Run optimization
        
        Args:
            objective_func: Objective function to minimize
            max_evals: Maximum number of function evaluations
            
        Returns:
            Dictionary with optimization results
        """
        try:
            # Use the objective_func passed as parameter or the one set via set_objective
            if objective_func is not None:
                self.set_objective(objective_func)
                
            if self.objective_func is None:
                raise ValueError("No objective function provided. Call set_objective() or provide objective_func parameter.")
                
            # Run optimization
            solution, score = self.optimize(self.objective_func, max_evals)
            
            # Return results
            return {
                'solution': solution,
                'score': score,
                'evaluations': self.evaluations,
                'runtime': 0.0,  # Not tracked
                'convergence_curve': self.convergence_curve if hasattr(self, 'convergence_curve') else []
            }
        except Exception as e:
            self.logger.error(f"Error in optimization: {str(e)}")
            return {
                'solution': None,
                'score': float('inf'),
                'evaluations': 0,
                'runtime': 0.0,
                'error': str(e)
            }
        
    def optimize(self, objective_func: Callable, max_evals: Optional[int] = None, record_history: bool = False) -> Tuple[np.ndarray, float]:
        """Run optimization
        
        Args:
            objective_func: Objective function to minimize
            max_evals: Maximum number of function evaluations
            record_history: Whether to record the convergence history
            
        Returns:
            Tuple of (best solution, best score)
        """
        if max_evals is not None:
            self.max_evals = max_evals
            
        self.logger.info(f"Starting GWO optimization with max_evals={self.max_evals}")
        
        # Initialize
        self.reset()
        self.best_solution = None
        self.best_score = float('inf')
        self.evaluations = 0
        self.convergence_curve = []
        self.success_history = []  # Initialize success history
        
        # Run optimization
        self._iterate(objective_func)
        
        if record_history:
            return self.best_solution, self.best_score, self.convergence_curve
        return self.best_solution, self.best_score
        
    def get_parameters(self) -> Dict[str, Any]:
        """Get optimizer parameters
        
        Returns:
            Dictionary of parameter settings
        """
        return {
            "population_size": self.population_size,
            "max_evals": self.max_evals,
            "dim": self.dim
        }
        
    def _iterate(self, objective_func: Callable) -> Tuple[np.ndarray, float]:
        """Perform one iteration of the optimization algorithm
        
        Args:
            objective_func: Objective function to minimize
            
        Returns:
            Tuple of (best solution, best score)
        """
        # Initialize population
        self.population = self._init_population()
        self.population_scores = np.array([objective_func(ind) for ind in self.population])
        self.evaluations += self.population_size
        
        # Track best solution
        best_idx = np.argmin(self.population_scores)
        self.best_solution = self.population[best_idx].copy()
        self.best_score = self.population_scores[best_idx]
        
        # Initialize alpha, beta, delta wolves
        sorted_indices = np.argsort(self.population_scores)
        alpha_idx, beta_idx, delta_idx = sorted_indices[:3]
        
        alpha = self.population[alpha_idx].copy()
        beta = self.population[beta_idx].copy()
        delta = self.population[delta_idx].copy()
        
        # Main loop
        iteration = 0
        max_iterations = self.max_evals // self.population_size
        
        while iteration < max_iterations and self.evaluations < self.max_evals:
            # Update a parameter
            a = 2 - iteration * (2 / max_iterations)
            
            # Update each wolf position
            for i in range(self.population_size):
                # Update position based on alpha, beta, delta
                A1 = 2 * a * np.random.random() - a
                A2 = 2 * a * np.random.random() - a
                A3 = 2 * a * np.random.random() - a
                C1 = 2 * np.random.random()
                C2 = 2 * np.random.random()
                C3 = 2 * np.random.random()
                
                D_alpha = np.abs(C1 * alpha - self.population[i])
                D_beta = np.abs(C2 * beta - self.population[i])
                D_delta = np.abs(C3 * delta - self.population[i])
                
                X1 = alpha - A1 * D_alpha
                X2 = beta - A2 * D_beta
                X3 = delta - A3 * D_delta
                
                # New position
                new_position = (X1 + X2 + X3) / 3
                
                # Bound position
                new_position = np.clip(new_position, 
                                     [low for low, _ in self.bounds],
                                     [high for _, high in self.bounds])
                
                # Evaluate new position
                new_score = objective_func(new_position)
                self.evaluations += 1
                
                # Update if better
                if new_score < self.population_scores[i]:
                    self.population[i] = new_position
                    self.population_scores[i] = new_score
                    self.success_history.append(1)
                else:
                    self.success_history.append(0)
                
                # Update best solution
                if new_score < self.best_score:
                    self.best_solution = new_position.copy()
                    self.best_score = new_score
                    
                # Update alpha, beta, delta
                if new_score < self.population_scores[alpha_idx]:
                    delta = beta.copy()
                    beta = alpha.copy()
                    alpha = new_position.copy()
                    alpha_idx, beta_idx, delta_idx = i, alpha_idx, beta_idx
                elif new_score < self.population_scores[beta_idx]:
                    delta = beta.copy()
                    beta = new_position.copy()
                    beta_idx, delta_idx = i, beta_idx
                elif new_score < self.population_scores[delta_idx]:
                    delta = new_position.copy()
                    delta_idx = i
                    
                # Update convergence curve
                self.convergence_curve.append(self.best_score)
                
                # Check if max evaluations reached
                if self.evaluations >= self.max_evals:
                    break
            
            iteration += 1
            
        return self.best_solution, self.best_score
        
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
            'min_samples_split': int(2 + solution[2] * 18) if self.dim > 2 else 2,  # 2-20
            'min_samples_leaf': int(1 + solution[3] * 9) if self.dim > 3 else 1,  # 1-10
            'max_features': 'sqrt',
            'bootstrap': True,
            'model_type': 'random_forest'
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
            if self.dim > 2 and 'min_samples_split' in config:
                solution[2] = (config['min_samples_split'] - 2) / 18
            if self.dim > 3 and 'min_samples_leaf' in config:
                solution[3] = (config['min_samples_leaf'] - 1) / 9
                
            self.best_solution = solution
            
        # Update convergence tracking
        self.convergence_curve.append(score)
        self.evaluations += 1


class OptimizerFactory:
    """Factory class for creating optimization algorithms"""
    
    def __init__(self):
        """Initialize the optimizer factory"""
        self.logger = logging.getLogger(__name__)
        self.optimizers = {
            'differential_evolution': DifferentialEvolutionOptimizer,
            'evolution_strategy': EvolutionStrategyOptimizer,
            'ant_colony': AntColonyOptimizer,
            'grey_wolf': GreyWolfOptimizer,
        }
    
    def create_optimizer(self, optimizer_type: str, **kwargs) -> BaseOptimizer:
        """
        Create an optimizer instance
        
        Args:
            optimizer_type: Type of optimizer to create
            **kwargs: Additional parameters for the optimizer
            
        Returns:
            Optimizer instance
        """
        if optimizer_type not in self.optimizers:
            raise ValueError(f"Unknown optimizer type: {optimizer_type}. Available types: {list(self.optimizers.keys())}")
        
        optimizer_class = self.optimizers[optimizer_type]
        return optimizer_class(**kwargs)
    
    def get_available_optimizers(self) -> List[str]:
        """
        Get list of available optimizer types
        
        Returns:
            List of optimizer type names
        """
        return list(self.optimizers.keys())


def create_optimizers(dim, population_size=None, timeout=60, iteration_timeout=10, **kwargs):
    """Create a dictionary of optimizers with proper timeout handling."""
    optimizers = {}
    
    # Common parameters for all optimizers
    optimizer_params = {
        'dim': dim,
        'population_size': population_size,
        'timeout': timeout,
        'iteration_timeout': iteration_timeout
    }
    
    # Base optimizers
    optimizers['ACO'] = AntColonyOptimizer(**optimizer_params)
    optimizers['GWO'] = GreyWolfOptimizer(**optimizer_params)
    optimizers['DE'] = DifferentialEvolutionOptimizer(**optimizer_params)
    optimizers['ES'] = EvolutionStrategyOptimizer(**optimizer_params)
    
    # Adaptive variants
    adaptive_params = optimizer_params.copy()
    adaptive_params['adaptive'] = True
    optimizers['DE (Adaptive)'] = DifferentialEvolutionOptimizer(**adaptive_params)
    optimizers['ES (Adaptive)'] = EvolutionStrategyOptimizer(**adaptive_params)
    
    # Meta-optimizer
    base_optimizers = {
        'DE': optimizers['DE'],
        'ES': optimizers['ES']
    }
    meta_params = optimizer_params.copy()
    meta_params.update({
        'optimizers': base_optimizers,
        'n_parallel': 2
    })
    optimizers['Meta-Optimizer'] = MetaOptimizer(**meta_params)
    
    return optimizers
