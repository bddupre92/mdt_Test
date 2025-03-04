"""Evolution Strategy optimizer implementation."""
from typing import Dict, List, Tuple, Optional, Any, Callable
import numpy as np
import logging
from .base_optimizer import BaseOptimizer


class EvolutionStrategyOptimizer(BaseOptimizer):
    """Evolution Strategy (μ + λ) optimizer."""
    
    def __init__(self, dim: int, bounds: List[Tuple[float, float]], 
                 population_size: Optional[int] = None,
                 mu: Optional[int] = None,
                 sigma: float = 0.1,
                 adaptive: bool = True):
        """
        Initialize ES optimizer.
        
        Args:
            dim: Number of dimensions
            bounds: List of (min, max) bounds for each dimension
            population_size: Optional population size (λ)
            mu: Parent population size (μ)
            sigma: Initial step size
            adaptive: Whether to use adaptive parameters
        """
        super().__init__(dim, bounds, population_size, adaptive)
        
        # ES-specific parameters
        self.mu = mu or self.population_size // 4
        self.sigma = sigma
        self.sigma_history = []
        
        # Success-based step size adaptation
        self.success_threshold = 0.2
        self.adaptation_speed = 0.2
        
        # Initialize tracking variables
        self.success_history = []
        self.parent_fitness = None
        
        # Log ES-specific parameters
        self.logger.debug(f"mu: {self.mu}, sigma: {self.sigma}")
        
    def _iterate(self, objective_func: Callable):
        """Perform one iteration of ES."""
        # Generate offspring
        offspring = np.zeros((self.population_size, self.dim))
        offspring_fitness = np.zeros(self.population_size)
        
        for i in range(self.population_size):
            # Select random parent
            parent_idx = np.random.randint(self.mu)
            parent = self.population[parent_idx]
            
            # Generate offspring with mutation
            offspring[i] = parent + self.sigma * np.random.normal(0, 1, self.dim)
            
            # Apply bounds
            offspring[i] = self._bound_solution(offspring[i])
            
            # Evaluate fitness
            offspring_fitness[i] = objective_func(offspring[i])
            self.evaluations += 1
        
        # Combine parent and offspring population
        combined_pop = np.vstack((self.population, offspring))
        combined_fitness = np.zeros(len(combined_pop))
        
        # Evaluate parents if not already evaluated
        if not hasattr(self, 'parent_fitness') or self.parent_fitness is None:
            self.parent_fitness = np.zeros(len(self.population))
            for i in range(len(self.population)):
                self.parent_fitness[i] = objective_func(self.population[i])
                self.evaluations += 1
        
        # Combine fitness values
        combined_fitness[:len(self.population)] = self.parent_fitness
        combined_fitness[len(self.population):] = offspring_fitness
        
        # Select the mu best individuals
        best_indices = np.argsort(combined_fitness)[:self.mu]
        
        # Ensure best_indices is a valid shape and not empty
        if len(best_indices) > 0:
            # Make a copy of the selected individuals to avoid broadcasting issues
            try:
                new_population = np.array([combined_pop[i].copy() for i in best_indices])
                
                # Ensure the new population has the right shape
                if new_population.shape != (self.mu, self.dim):
                    self.logger.warning(f"Population shape mismatch: {new_population.shape} vs expected {(self.mu, self.dim)}")
                    # Fix shape if possible
                    if len(new_population.shape) == 2 and new_population.shape[1] == self.dim:
                        # Just wrong number of rows
                        if new_population.shape[0] < self.mu:
                            # Need to add more rows
                            extra_rows = np.zeros((self.mu - new_population.shape[0], self.dim))
                            new_population = np.vstack((new_population, extra_rows))
                        elif new_population.shape[0] > self.mu:
                            # Need to truncate
                            new_population = new_population[:self.mu]
                
                self.population = new_population
                self.parent_fitness = combined_fitness[best_indices]
                
                # Sanity check on population and parent_fitness dimensions
                if self.population.shape[0] != len(self.parent_fitness):
                    self.logger.warning("Population size doesn't match fitness size, correcting")
                    # Resize parent_fitness to match population size
                    if len(self.parent_fitness) < self.population.shape[0]:
                        self.parent_fitness = np.pad(
                            self.parent_fitness, 
                            (0, self.population.shape[0] - len(self.parent_fitness)),
                            'constant', 
                            constant_values=float('inf')
                        )
                    else:
                        self.parent_fitness = self.parent_fitness[:self.population.shape[0]]
                    
            except Exception as e:
                self.logger.error(f"Error updating population: {str(e)}")
                # Recover by reinitializing
                self.population = self._init_population()
                self.parent_fitness = None
        else:
            self.logger.warning("No valid solutions found during selection")
            
        # Update best solution
        best_idx = np.argmin(combined_fitness)
        current_best = combined_pop[best_idx]
        current_best_score = combined_fitness[best_idx]
        
        if current_best_score < self.best_score:
            self.best_solution = current_best.copy()
            self.best_score = current_best_score
            self.success_history.append(True)
        else:
            self.success_history.append(False)
            
        # Update convergence tracking
        self.convergence_curve.append(self.best_score)
        
        # Adaptive sigma
        if self.adaptive and len(self.success_history) >= 10:
            self._update_sigma()
            
        self._current_iteration += 1
        
    def _update_sigma(self):
        """Update step size sigma."""
        # Calculate success rate
        success_rate = sum(self.success_history[-10:]) / 10
        
        # Safely update parameter history
        try:
            # Initialize parameter_history if it doesn't exist
            if not hasattr(self, 'parameter_history'):
                self.parameter_history = {}
            
            # Ensure keys exist in parameter_history
            if 'success_rate' not in self.parameter_history:
                self.parameter_history['success_rate'] = []
                
            self.parameter_history['success_rate'].append(success_rate)
        except Exception as e:
            self.logger.warning(f"Could not update parameter history: {str(e)}")
        
        # Update sigma based on 1/5 success rule
        if success_rate < self.success_threshold:
            self.sigma *= (1 - self.adaptation_speed)  # Decrease step size
        else:
            self.sigma *= (1 + self.adaptation_speed)  # Increase step size
            
        # Keep sigma in reasonable bounds
        min_sigma = 1e-5
        max_sigma = np.mean([b[1] - b[0] for b in self.bounds]) / 2
        self.sigma = max(min_sigma, min(max_sigma, self.sigma))
        
        # Track sigma history
        self.sigma_history.append(self.sigma)
        
        # Safely update sigma in parameter history
        try:
            if 'sigma' not in self.parameter_history:
                self.parameter_history['sigma'] = []
            self.parameter_history['sigma'].append(self.sigma)
        except Exception as e:
            self.logger.warning(f"Could not update sigma history: {str(e)}")
    
    def get_parameters(self) -> Dict[str, Any]:
        """
        Get optimizer parameters.
        
        Returns:
            Dictionary of parameters
        """
        return {
            'dim': self.dim,
            'population_size': self.population_size,
            'mu': self.mu,
            'sigma': self.sigma,
            'adaptive': self.adaptive,
            'evaluations': self.evaluations,
            'iterations': self._current_iteration
        }
        
    def _estimate_problem_characteristics(self):
        pass

    def reset(self) -> None:
        """Reset optimizer state including ES-specific attributes."""
        # First call the parent class reset method
        super().reset()
        
        # Reset ES-specific parameters
        self.sigma = 0.1  # Reset to initial value
        self.sigma_history = []
        self.success_history = []
        self.parent_fitness = None
        
        # Ensure mu is properly set based on reset population size
        self.mu = self.population_size // 4
        
        # Clear any other state that might cause issues
        if hasattr(self, 'parameter_history'):
            self.parameter_history = {}
            
    def optimize(self, objective_func: Callable, max_evals: Optional[int] = None) -> np.ndarray:
        """
        Optimize the objective function and return the best solution.
        This method overrides the base optimize method to add better error handling.
        
        Args:
            objective_func: Objective function to minimize
            max_evals: Maximum number of function evaluations
            
        Returns:
            Best solution found (numpy array)
        """
        self.logger.info(f"Starting {self.__class__.__name__}.optimize with max_evals={max_evals}")
        
        try:
            # Reset key attributes to avoid issues from previous runs
            if self.population is not None and hasattr(self, 'mu') and self.population.shape[0] != self.mu:
                self.logger.warning(f"Population size mismatch: {self.population.shape[0]} vs mu={self.mu}, reinitializing")
                self.population = None  # Force reinitialization
            
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
            
            # Ensure the solution has the right shape
            if best_solution is None or best_solution.shape != (self.dim,):
                self.logger.warning(f"Invalid solution shape, resetting to zeros")
                best_solution = np.zeros(self.dim)
            
            # Return the best solution
            self.logger.info(f"Completed optimize with best score: {self.best_score}")
            return best_solution
            
        except Exception as e:
            self.logger.error(f"Error in optimization: {str(e)}")
            # Return a fallback solution
            self.best_score = float('inf')
            return np.zeros(self.dim)
