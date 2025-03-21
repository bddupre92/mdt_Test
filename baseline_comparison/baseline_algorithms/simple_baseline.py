import random
import logging
from typing import List, Optional, Tuple
import numpy as np

logger = logging.getLogger(__name__)

class SimpleBaselineSelector:
    """
    A simple baseline selector that randomly selects algorithms from a pool.
    This serves as the most basic baseline for comparison.
    """
    
    def __init__(self, algorithms: Optional[List[str]] = None, random_seed: Optional[int] = None):
        """
        Initialize the simple baseline selector.
        
        Args:
            algorithms: List of available algorithms to select from
            random_seed: Optional random seed for reproducibility
        """
        self.available_algorithms = algorithms or [
            "differential_evolution",
            "particle_swarm",
            "genetic_algorithm", 
            "simulated_annealing",
            "cma_es"
        ]
        
        if random_seed is not None:
            random.seed(random_seed)
            
        self.last_selected = None
        logger.info(f"Initialized SimpleBaselineSelector with {len(self.available_algorithms)} algorithms")
        
    def get_available_algorithms(self) -> List[str]:
        """Get list of available algorithms."""
        return self.available_algorithms.copy()
        
    def set_available_algorithms(self, algorithms: List[str]):
        """Set the list of available algorithms."""
        self.available_algorithms = algorithms.copy()
        logger.info(f"Updated available algorithms: {algorithms}")
        
    def select_algorithm(self) -> str:
        """Randomly select an algorithm from the available pool."""
        if not self.available_algorithms:
            raise ValueError("No algorithms available for selection")
            
        self.last_selected = random.choice(self.available_algorithms)
        return self.last_selected
        
    def get_selected_algorithm(self) -> Optional[str]:
        """Get the last selected algorithm."""
        return self.last_selected
        
    def optimize(self, problem, max_evaluations: int) -> Tuple[np.ndarray, float]:
        """
        Optimize the given problem using a randomly selected algorithm.
        
        Args:
            problem: The optimization problem to solve
            max_evaluations: Maximum number of function evaluations
            
        Returns:
            Tuple of (best_solution, best_fitness)
        """
        # Select a random algorithm
        algorithm = self.select_algorithm()
        logger.info(f"Selected algorithm: {algorithm}")
        
        # Get dimensions from the problem, checking for different attribute names
        if hasattr(problem, 'dims'):
            dimensions = problem.dims
        elif hasattr(problem, 'dimension'):
            dimensions = problem.dimension
        elif hasattr(problem, 'dimensions'):
            dimensions = problem.dimensions
        else:
            # Default to 2D if no dimension attribute is found
            logger.warning("Problem has no dimension attribute, defaulting to 2D")
            dimensions = 2
        
        # Create a zero solution as placeholder (actual optimization would be implemented here)
        solution = np.zeros(dimensions)
        fitness = problem.evaluate(solution)
        
        return solution, fitness 