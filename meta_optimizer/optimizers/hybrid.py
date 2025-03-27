"""
hybrid.py
---------
Hybrid Evolutionary Optimizer that combines multiple evolutionary approaches.

This optimizer is specifically designed for the MedicationHistoryExpert to handle
the complex patterns in medication response data, which may have both discrete
(medication types) and continuous (dosage, timing) parameters.
"""

import numpy as np
from typing import Dict, Any, Optional, List, Tuple, Callable
from .base_optimizer import BaseOptimizer
import time
import logging
from tqdm.auto import tqdm

# Configure logging
logger = logging.getLogger(__name__)

class HybridEvolutionaryOptimizer(BaseOptimizer):
    def __init__(self,
                 fitness_function: Callable,
                 bounds: List[Tuple[float, float]],
                 population_size: int = 20,
                 max_iterations: int = 30,
                 local_search_iterations: int = 5,
                 crossover_rate: float = 0.7,
                 mutation_rate: float = 0.3,
                 mutation_scale: float = 0.5,
                 local_search_radius: float = 0.1,
                 random_seed: Optional[int] = None,
                 verbose: bool = False):
        """
        Initialize Hybrid Evolutionary Optimizer.
        
        Args:
            fitness_function: Function to optimize (minimize)
            bounds: List of (min, max) bounds for each dimension
            population_size: Size of the population
            max_iterations: Maximum number of iterations
            local_search_iterations: Number of local search iterations
            crossover_rate: Probability of crossover (GA phase)
            mutation_rate: Probability of mutation (GA phase)
            mutation_scale: Scale factor for mutation (DE phase)
            local_search_radius: Radius for local search
            random_seed: Random seed for reproducibility
            verbose: Whether to show progress bars
        """
        # Initialize base optimizer
        dim = len(bounds)
        super().__init__(dim=dim, bounds=bounds, population_size=population_size, adaptive=True)
        
        # Store parameters
        self.fitness_function = fitness_function
        self.max_iterations = max_iterations
        self.local_search_iterations = local_search_iterations
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.mutation_scale = mutation_scale
        self.local_search_radius = local_search_radius
        self.verbose = verbose
        
        # Set random seed if provided
        if random_seed is not None:
            np.random.seed(random_seed)
        
        # Initialize population
        self.initialize_population()
        
        # Initialize fitness values
        self.fitness = np.zeros(self.population_size)
        
        # Track best solution
        self.best_solution = None
        self.best_fitness = float('inf')
        
        # Track evaluations
        self.evaluations = 0
        
        # Track execution time
        self.execution_time = 0
    
    def initialize_population(self):
        """Initialize the population with random values within bounds."""
        self.population = np.zeros((self.population_size, self.dim))
        for i in range(self.dim):
            lower, upper = self.bounds[i]
            self.population[:, i] = np.random.uniform(lower, upper, self.population_size)
    
    def evaluate_population(self):
        """Evaluate the fitness of the entire population."""
        for i in range(self.population_size):
            self.fitness[i] = self.fitness_function(self.population[i])
            self.evaluations += 1
            
            # Update best solution if needed
            if self.fitness[i] < self.best_fitness:
                self.best_fitness = self.fitness[i]
                self.best_solution = self.population[i].copy()
    
    def ga_phase(self):
        """Genetic Algorithm phase: selection, crossover, and mutation."""
        new_population = np.zeros_like(self.population)
        
        # Tournament selection and crossover
        for i in range(0, self.population_size, 2):
            # Select parents using tournament selection
            parent1_idx = self._tournament_selection(2)
            parent2_idx = self._tournament_selection(2)
            
            # Crossover
            if np.random.rand() < self.crossover_rate:
                child1, child2 = self._crossover(
                    self.population[parent1_idx], 
                    self.population[parent2_idx]
                )
            else:
                child1 = self.population[parent1_idx].copy()
                child2 = self.population[parent2_idx].copy()
            
            # Mutation
            child1 = self._mutate(child1)
            child2 = self._mutate(child2)
            
            # Add to new population
            if i < self.population_size - 1:
                new_population[i] = child1
                new_population[i+1] = child2
            else:
                new_population[i] = child1
        
        # Ensure all solutions are within bounds
        for i in range(self.population_size):
            new_population[i] = self._ensure_bounds(new_population[i])
        
        return new_population
    
    def de_phase(self):
        """Differential Evolution phase."""
        new_population = np.zeros_like(self.population)
        
        for i in range(self.population_size):
            # Select three random individuals different from i
            candidates = list(range(self.population_size))
            candidates.remove(i)
            r1, r2, r3 = np.random.choice(candidates, 3, replace=False)
            
            # Create trial vector
            mutant = self.population[r1] + self.mutation_scale * (
                self.population[r2] - self.population[r3]
            )
            
            # Ensure bounds
            mutant = self._ensure_bounds(mutant)
            
            # Crossover
            trial = np.zeros(self.dim)
            crossover_points = np.random.rand(self.dim) < self.crossover_rate
            
            # Ensure at least one dimension is crossed over
            if not np.any(crossover_points):
                crossover_points[np.random.randint(0, self.dim)] = True
            
            # Create trial vector
            trial = np.where(crossover_points, mutant, self.population[i])
            
            # Add to new population
            new_population[i] = trial
        
        return new_population
    
    def local_search(self):
        """Local search phase around the best solution."""
        # Start with the best solution
        best_copy = self.best_solution.copy()
        best_fitness = self.best_fitness
        
        # Perform local search iterations
        for _ in range(self.local_search_iterations):
            # Create a perturbed solution
            perturbed = best_copy.copy()
            
            # Perturb each dimension with small random changes
            for i in range(self.dim):
                lower, upper = self.bounds[i]
                range_i = upper - lower
                perturbed[i] += np.random.uniform(
                    -self.local_search_radius * range_i,
                    self.local_search_radius * range_i
                )
            
            # Ensure bounds
            perturbed = self._ensure_bounds(perturbed)
            
            # Evaluate perturbed solution
            perturbed_fitness = self.fitness_function(perturbed)
            self.evaluations += 1
            
            # Update if better
            if perturbed_fitness < best_fitness:
                best_copy = perturbed.copy()
                best_fitness = perturbed_fitness
        
        # Update global best if local search found a better solution
        if best_fitness < self.best_fitness:
            self.best_solution = best_copy.copy()
            self.best_fitness = best_fitness
            
            # Replace worst solution in population with new best
            worst_idx = np.argmax(self.fitness)
            self.population[worst_idx] = best_copy.copy()
            self.fitness[worst_idx] = best_fitness
    
    def _tournament_selection(self, tournament_size):
        """Tournament selection for GA phase."""
        candidates = np.random.choice(
            range(self.population_size), 
            tournament_size, 
            replace=False
        )
        winner = candidates[0]
        for candidate in candidates[1:]:
            if self.fitness[candidate] < self.fitness[winner]:
                winner = candidate
        return winner
    
    def _crossover(self, parent1, parent2):
        """Perform crossover between two parents."""
        # Single-point crossover
        crossover_point = np.random.randint(1, self.dim)
        child1 = np.concatenate([parent1[:crossover_point], parent2[crossover_point:]])
        child2 = np.concatenate([parent2[:crossover_point], parent1[crossover_point:]])
        return child1, child2
    
    def _mutate(self, individual):
        """Perform mutation on an individual."""
        mutated = individual.copy()
        for i in range(self.dim):
            if np.random.rand() < self.mutation_rate:
                lower, upper = self.bounds[i]
                mutated[i] = np.random.uniform(lower, upper)
        return mutated
    
    def _ensure_bounds(self, individual):
        """Ensure individual is within bounds."""
        bounded = individual.copy()
        for i in range(self.dim):
            lower, upper = self.bounds[i]
            bounded[i] = np.clip(bounded[i], lower, upper)
        return bounded
    
    def optimize(self) -> Tuple[np.ndarray, float]:
        """
        Run the hybrid optimization process.
        
        Returns:
            Tuple containing:
                - Best solution found (numpy array)
                - Best fitness value (float)
        """
        start_time = time.time()
        
        # Initial evaluation
        self.evaluate_population()
        
        # Main optimization loop
        iterator = range(self.max_iterations)
        if self.verbose:
            iterator = tqdm(iterator, desc="Hybrid Optimization")
            
        for iteration in iterator:
            # Phase 1: Genetic Algorithm
            if iteration < self.max_iterations // 3:
                new_population = self.ga_phase()
            # Phase 2: Differential Evolution
            else:
                new_population = self.de_phase()
            
            # Evaluate new population
            new_fitness = np.zeros(self.population_size)
            for i in range(self.population_size):
                new_fitness[i] = self.fitness_function(new_population[i])
                self.evaluations += 1
                
                # Update best solution if needed
                if new_fitness[i] < self.best_fitness:
                    self.best_fitness = new_fitness[i]
                    self.best_solution = new_population[i].copy()
            
            # Selection (keep the best solutions)
            combined_population = np.vstack([self.population, new_population])
            combined_fitness = np.concatenate([self.fitness, new_fitness])
            
            # Sort by fitness
            sorted_indices = np.argsort(combined_fitness)
            
            # Keep the best solutions
            self.population = combined_population[sorted_indices[:self.population_size]]
            self.fitness = combined_fitness[sorted_indices[:self.population_size]]
            
            # Phase 3: Local Search (every 5 iterations)
            if iteration % 5 == 0:
                self.local_search()
            
            # Log progress
            if self.verbose and iteration % 5 == 0:
                logger.info(f"Iteration {iteration}: Best fitness = {self.best_fitness:.6f}")
        
        # Final local search
        self.local_search()
        
        # Record execution time
        self.execution_time = time.time() - start_time
        
        if self.verbose:
            logger.info(f"Optimization completed in {self.execution_time:.2f} seconds")
            logger.info(f"Function evaluations: {self.evaluations}")
            logger.info(f"Best fitness: {self.best_fitness:.6f}")
        
        return self.best_solution, self.best_fitness
    
    def _iterate(self, objective_func: Callable):
        """
        Perform one iteration of the hybrid evolutionary algorithm.
        
        Args:
            objective_func: The objective function to optimize
        """
        # Increment iteration counter
        self._current_iteration += 1
        
        # Store current best for comparison
        previous_best = self.best_fitness
        
        # Phase 1: Genetic Algorithm
        self.ga_phase()
        
        # Phase 2: Differential Evolution
        self.de_phase()
        
        # Phase 3: Local Search (every 5 iterations)
        if self._current_iteration % 5 == 0:
            self.local_search()
        
        # Evaluate population
        self.evaluate_population()
        
        # Update convergence curve
        self.convergence_curve.append(float(self.best_fitness))
        
        # Update success history
        success = self.best_fitness < previous_best
        self.success_history.append(success)
        
        if success:
            self.last_improvement_iter = self._current_iteration
            
        # Log progress
        if self.verbose and self._current_iteration % 5 == 0:
            self.logger.info(f"Iteration {self._current_iteration}: Best fitness = {self.best_fitness:.6f}")
    
    def get_optimization_results(self) -> Dict[str, Any]:
        """
        Get detailed optimization results.
        
        Returns:
            Dictionary with optimization results
        """
        return {
            'best_solution': self.best_solution.tolist() if self.best_solution is not None else None,
            'best_fitness': float(self.best_fitness),
            'evaluations': self.evaluations,
            'execution_time': self.execution_time,
            'population_size': self.population_size,
            'dimensions': self.dim,
            'algorithm': 'HybridEvolutionaryOptimizer'
        }
