"""
Optimizer Adapters Module

This module provides adapter classes to ensure compatibility between expert models
and their optimizers with different parameter requirements.
"""

import logging
import numpy as np
from typing import List, Tuple, Callable, Dict, Any, Optional

from meta_optimizer.optimizers.de import DifferentialEvolutionOptimizer
from meta_optimizer.optimizers.es import EvolutionStrategyOptimizer
from meta_optimizer.optimizers.aco import AntColonyOptimizer
from meta_optimizer.optimizers.hybrid import HybridEvolutionaryOptimizer

# Configure logging
logger = logging.getLogger(__name__)


class DifferentialEvolutionAdapter:
    """Adapter class for DifferentialEvolutionOptimizer that handles initialization and provides a simplified interface."""
    
    def __init__(self, 
                 fitness_function: Callable,
                 bounds: List[Tuple[float, float]],
                 population_size: int = 10,
                 max_iterations: int = 30,
                 crossover_probability: float = 0.7,
                 differential_weight: float = 0.8,
                 random_seed: int = 42):
        """
        Initialize the DifferentialEvolutionAdapter.
        
        Args:
            fitness_function: Function to optimize (minimize)
            bounds: List of (min, max) bounds for each parameter
            population_size: Size of the population
            max_iterations: Maximum number of iterations
            crossover_probability: Probability of crossover (CR)
            differential_weight: Differential weight (F)
            random_seed: Random seed for reproducibility
        """
        self.fitness_function = fitness_function
        self.bounds = bounds
        self.population_size = population_size
        self.max_iterations = max_iterations
        self.crossover_probability = crossover_probability
        self.differential_weight = differential_weight
        self.random_seed = random_seed
        self.dim = len(bounds)
        
        # Initialize the optimizer
        self.optimizer = DifferentialEvolutionOptimizer(
            dim=self.dim,
            bounds=bounds,
            population_size=population_size,
            F=differential_weight,
            CR=crossover_probability
        )
        
        # Store the fitness function for later use
        self.fitness_function = fitness_function
        
        logger.info(f"Initialized DifferentialEvolutionAdapter with {self.dim} dimensions")
    
    def optimize(self) -> Tuple[List[float], float]:
        """
        Run the optimization process.
        
        Returns:
            Tuple of (best parameters, best fitness)
        """
        logger.info(f"Starting DE optimization with {self.max_iterations} iterations")
        best_params, best_fitness = self.optimizer.optimize(self.fitness_function, max_evals=self.max_iterations*self.population_size)
        
        # Convert numpy array to list for easier handling
        if isinstance(best_params, np.ndarray):
            best_params = best_params.tolist()
            
        return best_params, best_fitness


class EvolutionStrategyAdapter:
    """Adapter class for EvolutionStrategyOptimizer that handles initialization and provides a simplified interface."""
    
    def __init__(self, 
                 fitness_function: Callable,
                 bounds: List[Tuple[float, float]],
                 population_size: int = 20,
                 max_iterations: int = 30,
                 initial_step_size: float = 0.3,
                 adaptation_rate: float = 0.2,
                 random_seed: int = 42):
        """
        Initialize the EvolutionStrategyAdapter.
        
        Args:
            fitness_function: Function to optimize (minimize)
            bounds: List of (min, max) bounds for each parameter
            population_size: Size of the population
            max_iterations: Maximum number of iterations
            initial_step_size: Initial step size for mutations
            adaptation_rate: Rate of parameter adaptation
            random_seed: Random seed for reproducibility
        """
        self.fitness_function = fitness_function
        self.bounds = bounds
        self.population_size = population_size
        self.max_iterations = max_iterations
        self.initial_step_size = initial_step_size
        self.adaptation_rate = adaptation_rate
        self.random_seed = random_seed
        self.dim = len(bounds)
        
        # Initialize the optimizer
        self.optimizer = EvolutionStrategyOptimizer(
            dim=self.dim,
            bounds=bounds,
            population_size=population_size,
            initial_step_size=initial_step_size,
            adaptation_rate=adaptation_rate,
            verbose=False
        )
        
        # Store the fitness function for later use
        self.fitness_function = fitness_function
        
        logger.info(f"Initialized EvolutionStrategyAdapter with {self.dim} dimensions")
    
    def optimize(self) -> Tuple[List[float], float]:
        """
        Run the optimization process.
        
        Returns:
            Tuple of (best parameters, best fitness)
        """
        logger.info(f"Starting ES optimization with {self.max_iterations} iterations")
        best_params, best_fitness = self.optimizer.optimize(self.fitness_function, max_evals=self.max_iterations*self.population_size)
        
        # Convert numpy array to list for easier handling
        if isinstance(best_params, np.ndarray):
            best_params = best_params.tolist()
            
        return best_params, best_fitness


class AntColonyAdapter:
    """Adapter class for AntColonyOptimizer that handles initialization and provides a simplified interface."""
    
    def __init__(self, 
                 fitness_function: Callable = None,
                 bounds: List[Tuple[float, float]] = None,
                 population_size: int = 10,
                 max_iterations: int = 30,
                 alpha: float = 1.0,
                 beta: float = 2.0,
                 evaporation_rate: float = 0.1,
                 random_seed: int = 42):
        """
        Initialize the AntColonyAdapter.
        
        Args:
            fitness_function: Function to optimize (minimize)
            bounds: List of (min, max) bounds for each parameter
            population_size: Number of ants
            max_iterations: Maximum number of iterations
            alpha: Pheromone importance
            beta: Heuristic importance
            evaporation_rate: Pheromone evaporation rate
            random_seed: Random seed for reproducibility
        """
        self.fitness_function = fitness_function
        self.bounds = bounds
        self.population_size = population_size
        self.max_iterations = max_iterations
        self.alpha = alpha
        self.beta = beta
        self.evaporation_rate = evaporation_rate
        self.random_seed = random_seed
        self.dim = len(bounds) if bounds else None
        
        # Initialize the optimizer only if bounds and dimensions are provided
        if bounds and self.dim:
            self.optimizer = AntColonyOptimizer(
                dim=self.dim,
                bounds=bounds,
                population_size=population_size,
                alpha=alpha,
                beta=beta,
                evaporation_rate=evaporation_rate
            )
            logger.info(f"Initialized AntColonyAdapter with {self.dim} dimensions")
        else:
            self.optimizer = None
    
    def set_fitness_function(self, fitness_function: Callable):
        """
        Set the fitness function to be optimized.
        
        Args:
            fitness_function: Function to optimize (minimize)
        """
        self.fitness_function = fitness_function
    
    def set_bounds(self, bounds: List[Tuple[float, float]]):
        """
        Set the parameter bounds.
        
        Args:
            bounds: List of (min, max) bounds for each parameter
        """
        self.bounds = bounds
        self.dim = len(bounds)
        
        # Reinitialize the optimizer with the new bounds if it wasn't initialized before
        if self.optimizer is None:
            self.optimizer = AntColonyOptimizer(
                dim=self.dim,
                bounds=bounds,
                population_size=self.population_size,
                alpha=self.alpha,
                beta=self.beta,
                evaporation_rate=self.evaporation_rate
            )
            logger.info(f"Initialized AntColonyAdapter with {self.dim} dimensions")
    
    def optimize(self, max_iterations: Optional[int] = None) -> Tuple[List[float], float]:
        """
        Run the optimization process.
        
        Args:
            max_iterations: Optional override for the maximum number of iterations
            
        Returns:
            Tuple of (best parameters, best fitness)
        """
        if self.optimizer is None:
            raise ValueError("Optimizer not initialized. Please set bounds first.")
        
        if self.fitness_function is None:
            raise ValueError("Fitness function not set. Please set fitness function first.")
        
        # Use provided max_iterations if specified, otherwise use the default
        iterations = max_iterations or self.max_iterations
        
        logger.info(f"Starting ACO optimization with {iterations} iterations")
        best_params, best_fitness = self.optimizer.optimize(self.fitness_function, max_evals=iterations*self.population_size)
        
        # Convert numpy array to list for easier handling
        if isinstance(best_params, np.ndarray):
            best_params = best_params.tolist()
            
        return best_params, best_fitness


class HybridEvolutionaryAdapter:
    """Adapter class for HybridEvolutionaryOptimizer that handles initialization and provides a simplified interface."""
    
    def __init__(self, 
                 fitness_function: Callable,
                 bounds: List[Tuple[float, float]],
                 population_size: int = 15,
                 max_iterations: int = 30,
                 crossover_rate: float = 0.7,
                 mutation_rate: float = 0.1,
                 local_search_iterations: int = 5,
                 random_seed: int = 42):
        """
        Initialize the HybridEvolutionaryAdapter.
        
        Args:
            fitness_function: Function to optimize (minimize)
            bounds: List of (min, max) bounds for each parameter
            population_size: Size of the population
            max_iterations: Maximum number of iterations
            crossover_rate: Probability of crossover
            mutation_rate: Mutation rate
            local_search_iterations: Number of local search iterations
            random_seed: Random seed for reproducibility
        """
        self.fitness_function = fitness_function
        self.bounds = bounds
        self.population_size = population_size
        self.max_iterations = max_iterations
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.local_search_iterations = local_search_iterations
        self.random_seed = random_seed
        self.dim = len(bounds)
        
        # Initialize the optimizer
        self.optimizer = HybridEvolutionaryOptimizer(
            fitness_function=fitness_function,
            bounds=bounds,
            population_size=population_size,
            max_iterations=max_iterations,
            crossover_rate=crossover_rate,
            mutation_rate=mutation_rate,
            local_search_iterations=local_search_iterations,
            random_seed=random_seed,
            verbose=True
        )
        
        # Store the fitness function for later use
        self.fitness_function = fitness_function
        
        logger.info(f"Initialized HybridEvolutionaryAdapter with {self.dim} dimensions")
    
    def optimize(self) -> Tuple[List[float], float]:
        """
        Run the optimization process.
        
        Returns:
            Tuple of (best parameters, best fitness)
        """
        logger.info(f"Starting Hybrid optimization with {self.max_iterations} iterations")
        # The HybridEvolutionaryOptimizer already has its fitness function and max_iterations set during initialization
        best_params, best_fitness = self.optimizer.optimize()
        
        # Convert numpy array to list for easier handling
        if isinstance(best_params, np.ndarray):
            best_params = best_params.tolist()
            
        return best_params, best_fitness


def create_optimizer_adapter(optimizer_type: str, 
                            fitness_function: Callable,
                            bounds: List[Tuple[float, float]],
                            **kwargs) -> Any:
    """
    Create an appropriate optimizer adapter based on the optimizer type.
    
    Args:
        optimizer_type: Type of optimizer to create
        fitness_function: Function to optimize (minimize)
        bounds: List of (min, max) bounds for each parameter
        **kwargs: Additional parameters for the optimizer
        
    Returns:
        Optimizer adapter instance
    """
    optimizer_type = optimizer_type.lower()
    
    if optimizer_type in ['de', 'differential_evolution']:
        return DifferentialEvolutionAdapter(
            fitness_function=fitness_function,
            bounds=bounds,
            **kwargs
        )
    elif optimizer_type in ['es', 'evolution_strategy']:
        return EvolutionStrategyAdapter(
            fitness_function=fitness_function,
            bounds=bounds,
            **kwargs
        )
    elif optimizer_type in ['aco', 'ant_colony']:
        return AntColonyAdapter(
            fitness_function=fitness_function,
            bounds=bounds,
            **kwargs
        )
    elif optimizer_type in ['hybrid', 'hybrid_evolutionary']:
        return HybridEvolutionaryAdapter(
            fitness_function=fitness_function,
            bounds=bounds,
            **kwargs
        )
    else:
        raise ValueError(f"Unknown optimizer type: {optimizer_type}")
