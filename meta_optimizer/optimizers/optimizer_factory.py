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
from .base_optimizer import BaseOptimizer
from .de import DifferentialEvolutionOptimizer
from .es import EvolutionStrategyOptimizer
from .aco import AntColonyOptimizer
from .gwo import GreyWolfOptimizer
from .hybrid import HybridEvolutionaryOptimizer

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
            'hybrid_evolutionary': HybridEvolutionaryOptimizer,
        }
        
        # Expert-specific optimizer configurations
        self.expert_configs = {
            'physiological': {
                'optimizer_type': 'differential_evolution',
                'hyperparameter_space': {
                    'population_size': (20, 50),
                    'mutation': (0.5, 1.0),
                    'recombination': (0.7, 0.9),
                    'strategy': ['best1bin', 'best1exp', 'rand1exp'],
                    'tol': (1e-5, 1e-3)
                },
                'problem_characteristics': {
                    'high_dimensional': True,
                    'noisy': True,
                    'multimodal': False,
                    'time_series_dependent': True
                },
                'evaluation_function': 'rmse_with_smoothness_penalty',
                'early_stopping': {
                    'patience': 5,
                    'min_delta': 0.001,
                    'monitor': 'val_loss'
                }
            },
            'environmental': {
                'optimizer_type': 'evolution_strategy',
                'hyperparameter_space': {
                    'population_size': (30, 100),
                    'num_offspring': (60, 200),
                    'mutation_rate': (0.05, 0.3),
                    'elite_ratio': (0.1, 0.3)
                },
                'problem_characteristics': {
                    'high_dimensional': True,
                    'noisy': True,
                    'multimodal': True,
                    'seasonal_patterns': True
                },
                'evaluation_function': 'mae_with_lag_penalty',
                'early_stopping': {
                    'patience': 7,
                    'min_delta': 0.005,
                    'monitor': 'val_mae'
                }
            },
            'behavioral': {
                'optimizer_type': 'ant_colony',
                'hyperparameter_space': {
                    'n_ants': (10, 30),
                    'evaporation_rate': (0.1, 0.5),
                    'alpha': (1.0, 3.0),
                    'beta': (1.0, 5.0),
                    'min_pheromone': (0.1, 0.3)
                },
                'problem_characteristics': {
                    'high_dimensional': False,
                    'noisy': True,
                    'sparse_features': True,
                    'temporal_patterns': True
                },
                'evaluation_function': 'weighted_rmse_mae',
                'early_stopping': {
                    'patience': 10,
                    'min_delta': 0.01,
                    'monitor': 'val_score'
                }
            },
            'medication_history': {
                'optimizer_type': 'hybrid_evolutionary',
                'hyperparameter_space': {
                    'population_size': (20, 50),
                    'crossover_rate': (0.6, 0.9),
                    'mutation_rate': (0.1, 0.4),
                    'local_search_iterations': (3, 10),
                    'local_search_radius': (0.05, 0.2)
                },
                'problem_characteristics': {
                    'mixed_variable_types': True,
                    'high_dimensional': False,
                    'temporal_dependencies': True,
                    'sparse_interactions': True
                },
                'evaluation_function': 'treatment_response_score',
                'early_stopping': {
                    'patience': 8,
                    'min_delta': 0.008,
                    'monitor': 'val_response_score'
                }
            }
        }
    
    def _get_optimizer_parameters(self, optimizer_class) -> List[str]:
        """
        Get the accepted parameter names for an optimizer class by inspecting its __init__ method
        
        Args:
            optimizer_class: The optimizer class to inspect
            
        Returns:
            List of parameter names accepted by the optimizer
        """
        import inspect
        
        try:
            # Get the signature of the __init__ method
            sig = inspect.signature(optimizer_class.__init__)
            
            # Extract parameter names, excluding 'self'
            params = [param for param in sig.parameters if param != 'self']
            
            self.logger.debug(f"Detected parameters for {optimizer_class.__name__}: {params}")
            return params
            
        except (TypeError, ValueError) as e:
            self.logger.warning(f"Could not inspect parameters for {optimizer_class.__name__}: {e}")
            
            # Fallback to a predefined set of common parameters
            common_params = ['dim', 'bounds', 'population_size', 'max_evals', 'max_iterations', 
                            'fitness_function', 'random_seed', 'verbose']
            self.logger.warning(f"Using fallback parameters: {common_params}")
            return common_params

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
        
        # Get the accepted parameters for this optimizer class
        accepted_params = self._get_optimizer_parameters(optimizer_class)
        
        # Filter kwargs to only include accepted parameters
        filtered_kwargs = {}
        early_stopping_params = {}
        for param, value in kwargs.items():
            # Check if it's an early stopping parameter
            if param.startswith('early_stopping_'):
                # Store in separate dict for the EarlyStoppingCallback, don't pass to optimizer
                early_stopping_params[param] = value
            # Otherwise include if it's in the accepted params list
            elif param in accepted_params:
                filtered_kwargs[param] = value
            else:
                self.logger.debug(f"Ignoring unsupported parameter for {optimizer_type}: {param}")
                
        # Early stopping params can be accessed later when setting up early stopping
        # We don't pass them to the optimizer constructor
            
        # Ensure required parameters are included - but skip for hybrid_evolutionary optimizer
        # since it calculates dimensions from bounds internally
        if optimizer_type != 'hybrid_evolutionary':
            if 'bounds' in filtered_kwargs and 'dim' not in filtered_kwargs:
                # If bounds are provided but dim is not, derive dim from bounds
                filtered_kwargs['dim'] = len(filtered_kwargs['bounds'])
            elif 'dim' in filtered_kwargs and 'bounds' not in filtered_kwargs:
                # If dim is provided but bounds are not, create default bounds
                filtered_kwargs['bounds'] = [(0, 1)] * filtered_kwargs['dim']
        elif optimizer_type == 'hybrid_evolutionary' and 'dim' in filtered_kwargs:
            # For hybrid_evolutionary, remove dim parameter as it's not needed
            filtered_kwargs.pop('dim')
            self.logger.debug("Removed 'dim' parameter for hybrid_evolutionary optimizer")
            
        # Convert specific numeric parameters to appropriate types
        int_parameters = ['population_size', 'max_iterations', 'max_evals']
        for param in int_parameters:
            if param in filtered_kwargs and not isinstance(filtered_kwargs[param], int):
                try:
                    filtered_kwargs[param] = int(filtered_kwargs[param])
                except (ValueError, TypeError):
                    self.logger.warning(f"Could not convert {param}={filtered_kwargs[param]} to integer. Using as-is.")
        
        self.logger.info(f"Creating {optimizer_type} optimizer with parameters: {filtered_kwargs}")
        
        # For specific optimizer types, use adapter classes instead of raw optimizers
        # This ensures the adapter provides the required interface
        if optimizer_type == 'ant_colony':
            # Import here to avoid circular imports
            from moe_framework.experts.optimizer_adapters import AntColonyAdapter
            
            # Use the adapter class with the required methods
            return AntColonyAdapter(
                bounds=filtered_kwargs.get('bounds'),
                population_size=filtered_kwargs.get('population_size', 50),
                max_iterations=filtered_kwargs.get('max_iterations', 30),
                alpha=filtered_kwargs.get('alpha', 1.0),
                beta=filtered_kwargs.get('beta', 2.0),
                evaporation_rate=filtered_kwargs.get('evaporation_rate', 0.1),
                random_seed=filtered_kwargs.get('random_seed', 42)
            )
        
        return optimizer_class(**filtered_kwargs)
    
    def create_expert_optimizer(self, expert_type: str, **kwargs) -> BaseOptimizer:
        """
        Create an optimizer instance tailored for a specific expert type
        
        Args:
            expert_type: Type of expert ('physiological', 'environmental', 'behavioral', 'medication_history')
            **kwargs: Additional parameters to override default configuration
            
        Returns:
            Optimizer instance configured for the specific expert domain
        """
        # Use the dedicated method if available
        expert_methods = {
            'physiological': self.create_physiological_optimizer,
            'environmental': self.create_environmental_optimizer,
            'behavioral': self.create_behavioral_optimizer,
            'medication_history': self.create_medication_history_optimizer
        }
        
        if expert_type in expert_methods:
            self.logger.info(f"Using dedicated optimizer creation method for {expert_type} expert")
            return expert_methods[expert_type](**kwargs)
        
        if expert_type not in self.expert_configs:
            raise ValueError(f"Unknown expert type: {expert_type}. Available types: {list(self.expert_configs.keys())}")
        
        # Get the default configuration for this expert type
        config = self.expert_configs[expert_type]
        optimizer_type = config['optimizer_type']
        
        # Merge default configuration with provided parameters
        combined_kwargs = {}
        
        # First add any directly passed parameters that are critical (like dim and bounds)
        for key, value in kwargs.items():
            if key in ['dim', 'bounds', 'fitness_function']:
                combined_kwargs[key] = value
        
        # Then add default parameters from expert config if not already set
        for key, value in self._get_default_parameters(expert_type).items():
            if key not in combined_kwargs:  # Don't override already set parameters
                combined_kwargs[key] = kwargs.get(key, value)
        
        # Finally add any remaining kwargs that weren't handled above
        for key, value in kwargs.items():
            if key not in combined_kwargs:
                combined_kwargs[key] = value
                
        # Parameter name mappings for different optimizer types
        param_mappings = {
            'evolution_strategy': {
                'num_offspring': 'offspring_ratio',  # The ES optimizer uses offspring_ratio, not num_offspring
                'mutation_rate': 'adaptation_rate',  # ES uses adaptation_rate instead of mutation_rate
                'elite_ratio': None,  # This parameter is not used by ES optimizer, should be removed
            },
            'ant_colony': {
                'n_ants': 'population_size',  # ACO uses population_size instead of n_ants
            },
            'differential_evolution': {
                # No special mappings needed
            },
            'hybrid_evolutionary': {
                'dim': None,  # HybridEvolutionaryOptimizer calculates dim from bounds internally
            }
        }
        
        # Apply parameter name mappings
        if optimizer_type in param_mappings:
            for original_param, mapped_param in param_mappings[optimizer_type].items():
                if original_param in combined_kwargs:
                    if mapped_param is None:
                        # If mapped_param is None, remove the parameter as it's not used
                        combined_kwargs.pop(original_param)
                        self.logger.debug(f"Removed unused parameter {original_param} for {optimizer_type}")
                    else:
                        # Transfer the value to the correctly named parameter
                        combined_kwargs[mapped_param] = combined_kwargs.pop(original_param)
                        self.logger.debug(f"Mapped parameter {original_param} to {mapped_param}")
        
        # Convert specific numeric parameters to appropriate types
        int_parameters = ['population_size', 'n_ants', 'max_iterations', 'max_evals']
        for param in int_parameters:
            if param in combined_kwargs and not isinstance(combined_kwargs[param], int):
                try:
                    combined_kwargs[param] = int(combined_kwargs[param])
                except (ValueError, TypeError):
                    self.logger.warning(f"Could not convert {param}={combined_kwargs[param]} to integer. Using as-is.")
                    
        # For evolution_strategy, convert num_offspring to offspring_ratio if needed
        # and ensure it's properly processed regardless of mapping
        if optimizer_type == 'evolution_strategy' and 'num_offspring' in combined_kwargs and 'population_size' in combined_kwargs:
            # Calculate offspring_ratio from num_offspring and population_size
            num_offspring = combined_kwargs.pop('num_offspring')
            try:
                offspring_ratio = float(num_offspring) / float(combined_kwargs['population_size'])
                combined_kwargs['offspring_ratio'] = offspring_ratio
                self.logger.debug(f"Calculated offspring_ratio={offspring_ratio} from num_offspring={num_offspring}")
            except (ValueError, TypeError, ZeroDivisionError):
                self.logger.warning(f"Could not calculate offspring_ratio from num_offspring={num_offspring}. Using default.")
        
        # Ensure required parameters are present
        if optimizer_type in ['differential_evolution', 'evolution_strategy', 'ant_colony'] and 'dim' not in combined_kwargs:
            # Critical error - can't proceed without dimensions
            self.logger.error(f"Missing required 'dim' parameter for {optimizer_type}")
            raise ValueError(f"Missing required 'dim' parameter for {optimizer_type}")
        
        # Get the optimizer class to check for accepted parameters
        optimizer_class = self.optimizers[optimizer_type]
        accepted_params = self._get_optimizer_parameters(optimizer_class)
        
        # Filter out parameters that are not accepted by the optimizer class
        # to avoid TypeError for unexpected keyword arguments
        filtered_kwargs = {}
        early_stopping_kwargs = {}
        for param, value in combined_kwargs.items():
            if param.startswith('early_stopping_'):
                # Store early stopping parameters separately - don't pass to constructor
                early_stopping_kwargs[param] = value
            elif param in accepted_params:
                filtered_kwargs[param] = value
            else:
                self.logger.debug(f"Filtering out unsupported parameter for {optimizer_type}: {param}")
                
        # Store the early stopping parameters for later use in the ExpertOptimizerIntegration class
        # We'll handle them separately and not pass them to the optimizer constructor
        
        # Create optimizer with the filtered parameters
        self.logger.info(f"Creating {optimizer_type} optimizer for {expert_type} expert with params: {filtered_kwargs}")
        return self.create_optimizer(optimizer_type, **filtered_kwargs)
    
    def _get_default_parameters(self, expert_type: str) -> Dict[str, Any]:
        """
        Get default parameters for a specific expert type
        
        Args:
            expert_type: Type of expert
            
        Returns:
            Dictionary of default parameters
        """
        config = self.expert_configs[expert_type]
        
        # Convert hyperparameter space to concrete values for defaults
        defaults = {}
        for param, value in config['hyperparameter_space'].items():
            if isinstance(value, tuple) and len(value) == 2:
                # For ranges, use the middle value
                defaults[param] = (value[0] + value[1]) / 2
            elif isinstance(value, list):
                # For categorical values, use the first one
                defaults[param] = value[0]
            else:
                defaults[param] = value
                
        # Add early stopping parameters
        if 'early_stopping' in config:
            for param, value in config['early_stopping'].items():
                defaults[f"early_stopping_{param}"] = value
        
        # Add required parameters for specific optimizer types
        optimizer_type = config['optimizer_type']
        if optimizer_type == 'ant_colony':
            # For ant colony optimizer, we need to provide dim and bounds
            # Create default bounds if not already included
            if 'bounds' not in defaults:
                # Create 4 dimensional bounds for typical hyperparameters
                defaults['bounds'] = [
                    (50, 200),   # n_estimators
                    (1, 10),    # max_depth
                    (2, 20),    # min_samples_split
                    (1, 10)     # min_samples_leaf
                ]
                
            # Set dimension based on bounds length if not already included
            if 'dim' not in defaults and 'bounds' in defaults:
                defaults['dim'] = len(defaults['bounds'])
                
        return defaults
    
    def get_available_optimizers(self) -> List[str]:
        """
        Get list of available optimizer types
        
        Returns:
            List of optimizer type names
        """
        return list(self.optimizers.keys())
    
    def get_available_expert_types(self) -> List[str]:
        """
        Get list of available expert types with specialized optimizer configurations
        
        Returns:
            List of expert type names
        """
        return list(self.expert_configs.keys())
    
    def get_problem_characterization(self, expert_type: str) -> Dict[str, Any]:
        """
        Get problem characterization for a specific expert type
        
        Args:
            expert_type: Type of expert
            
        Returns:
            Dictionary of problem characteristics
        """
        if expert_type not in self.expert_configs:
            raise ValueError(f"Unknown expert type: {expert_type}")
        return self.expert_configs[expert_type]['problem_characteristics']
    
    def get_evaluation_function_type(self, expert_type: str) -> str:
        """
        Get the type of evaluation function recommended for a specific expert type
        
        Args:
            expert_type: Type of expert
            
        Returns:
            Name of the recommended evaluation function
        """
        if expert_type not in self.expert_configs:
            raise ValueError(f"Unknown expert type: {expert_type}")
        return self.expert_configs[expert_type]['evaluation_function']
        
    def create_physiological_optimizer(self, **kwargs) -> BaseOptimizer:
        """
        Create an optimizer specifically tailored for physiological data experts
        
        Physiological data often has continuous temporal patterns with
        temporal dependencies and requires smoothness constraints.
        
        Args:
            **kwargs: Additional parameters to override default configuration
            
        Returns:
            Differential Evolution optimizer configured for physiological data
        """
        self.logger.info("Creating optimizer for physiological expert")
        # Get base parameters from expert configs
        base_params = self.expert_configs.get('physiological', {})
        
        # Specify specialized configurations for physiological domain
        domain_params = {
            'adaptive': True,  # Enable adaptive mutation for handling smooth transitions
            'strategy': 'best1bin',  # Strategy that works well for smooth continuous spaces
            'mutation': 0.7,  # Higher mutation factor for escaping local optima
            'recombination': 0.9,  # Higher crossover rate for effective space exploration
            'init': 'latin'  # Latin hypercube sampling for better initial coverage
        }
        
        # Combine kwargs with domain parameters (kwargs takes precedence)
        combined_params = {**domain_params, **kwargs}
        
        # Add the expert type for logging and tracking
        combined_params['expert_type'] = 'physiological'
        
        # Set the optimizer type
        return self.create_optimizer('differential_evolution', **combined_params)
        
    def create_environmental_optimizer(self, **kwargs) -> BaseOptimizer:
        """
        Create an optimizer specifically tailored for environmental data experts
        
        Environmental data often has seasonal patterns, external influences,
        and requires handling of lag effects and multi-modal patterns.
        
        Args:
            **kwargs: Additional parameters to override default configuration
            
        Returns:
            Evolution Strategy optimizer configured for environmental data
        """
        self.logger.info("Creating optimizer for environmental expert")
        # Get base parameters from expert configs
        base_params = self.expert_configs.get('environmental', {})
        
        # Specify specialized configurations for environmental domain
        domain_params = {
            'population_size': 50,  # Larger population helps with multimodal spaces
            'adaptation_rate': 0.2,  # Moderate adaptation rate for seasonal pattern detection
            'offspring_ratio': 2.0,  # Generate more offspring to increase exploration
            'selective_pressure': 0.3,  # Higher selective pressure for multimodal spaces
            'recombination': 'intermediate'  # Intermediate recombination good for smooth landscapes
        }
        
        # Combine kwargs with domain parameters (kwargs takes precedence)
        combined_params = {**domain_params, **kwargs}
        
        # Add the expert type for logging and tracking
        combined_params['expert_type'] = 'environmental'
        
        # Set the optimizer type
        return self.create_optimizer('evolution_strategy', **combined_params)
        
    def create_behavioral_optimizer(self, **kwargs) -> BaseOptimizer:
        """
        Create an optimizer specifically tailored for behavioral data experts
        
        Behavioral data often has sparse feature interactions, categorical variables,
        and complex temporal patterns with irregularities.
        
        Args:
            **kwargs: Additional parameters to override default configuration
            
        Returns:
            Ant Colony optimizer configured for behavioral data
        """
        self.logger.info("Creating optimizer for behavioral expert")
        # Get base parameters from expert configs
        base_params = self.expert_configs.get('behavioral', {})
        
        # Specify specialized configurations for behavioral domain
        domain_params = {
            'population_size': 30,  # Good balance for behavioral feature selection
            'alpha': 2.0,  # Higher pheromone importance for variable interactions
            'beta': 4.0,  # Higher heuristic importance for behavioral patterns
            'evaporation_rate': 0.15,  # Moderate evaporation for temporal patterns
            'q': 2.0,  # Higher deposit factor for strong features
            'elitist': True  # Enable elitism to preserve good feature combinations
        }
        
        # Combine kwargs with domain parameters (kwargs takes precedence)
        combined_params = {**domain_params, **kwargs}
        
        # Add the expert type for logging and tracking
        combined_params['expert_type'] = 'behavioral'
        
        # Set the optimizer type
        return self.create_optimizer('ant_colony', **combined_params)
        
    def create_medication_history_optimizer(self, **kwargs) -> BaseOptimizer:
        """
        Create an optimizer specifically tailored for medication history data experts
        
        Medication history data often has discrete events, treatment responses,
        mixed variable types, and critical temporal dependencies.
        
        Args:
            **kwargs: Additional parameters to override default configuration
            
        Returns:
            Hybrid Evolutionary optimizer configured for medication history data
        """
        self.logger.info("Creating optimizer for medication history expert")
        # Get base parameters from expert configs
        base_params = self.expert_configs.get('medication_history', {})
        
        # Specify specialized configurations for medication history domain
        domain_params = {
            'population_size': 30,  # Moderate population for medication patterns
            'crossover_rate': 0.8,  # Higher crossover for effective treatment combinations
            'mutation_rate': 0.2,  # Moderate mutation for exploring treatment alternatives
            'local_search': 'pattern',  # Pattern search for local optimization of treatment patterns
            'local_search_iterations': 5,  # Moderate local search iterations
            'mixed_integer': True  # Support for mixed integer parameters common in medication
        }
        
        # Combine kwargs with domain parameters (kwargs takes precedence)
        combined_params = {**domain_params, **kwargs}
        
        # Add the expert type for logging and tracking
        combined_params['expert_type'] = 'medication_history'
        
        # Set the optimizer type
        return self.create_optimizer('hybrid_evolutionary', **combined_params)
    
    def create_all(self, dim: int = 5, bounds: Optional[List[Tuple[float, float]]] = None) -> Dict[str, BaseOptimizer]:
        """
        Create instances of all available optimizers
        
        Args:
            dim: Problem dimension
            bounds: Optional bounds for each dimension
            
        Returns:
            Dictionary mapping algorithm names to optimizer instances
        """
        return create_optimizers(dim, bounds)


def create_optimizers(dim: int = 5, bounds: Optional[List[Tuple[float, float]]] = None) -> Dict[str, Any]:
    """
    Create dictionary of optimization algorithms
    
    Args:
        dim: Problem dimension
        bounds: Optional bounds for each dimension
        
    Returns:
        Dictionary mapping algorithm names to optimizer instances
    """
    if bounds is None:
        bounds = [(0, 1)] * dim
        
    return {
        'DE (Standard)': DifferentialEvolutionWrapper(dim=dim, bounds=bounds, adaptive=False, name="DE (Standard)"),
        'DE (Adaptive)': DifferentialEvolutionWrapper(dim=dim, bounds=bounds, population_size=20, adaptive=True, name="DE (Adaptive)"),
        'ES (Standard)': EvolutionStrategyWrapper(dim=dim, bounds=bounds, adaptive=False, name="ES (Standard)"),
        'ES (Adaptive)': EvolutionStrategyWrapper(dim=dim, bounds=bounds, adaptive=True, name="ES (Adaptive)"),
        'ACO': AntColonyWrapper(dim=dim, bounds=bounds, adaptive=True, name="ACO"),
        'GWO': GreyWolfWrapper(dim=dim, bounds=bounds, name="GWO")
    }
