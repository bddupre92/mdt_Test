"""
es.py
------
Evolution Strategy optimizer with adaptive parameters.
"""

import numpy as np
from typing import Dict, Any, Optional, List, Tuple, Callable
from .base_optimizer import BaseOptimizer
import time
import logging
from collections import defaultdict
from tqdm.auto import tqdm

class EvolutionStrategyOptimizer(BaseOptimizer):
    def __init__(self,
                 dim: int,
                 bounds: List[Tuple[float, float]],
                 population_size: int = 20,  # Reduced default population size
                 offspring_ratio: float = 1.0,  # Increased offspring ratio
                 initial_step_size: float = 0.3,  # Reduced initial step size
                 adaptation_rate: float = 0.2,  # Increased adaptation rate
                 max_evals: int = 10000,
                 timeout: float = 30.0,
                 adaptive: bool = True,
                 name: str = None,
                 verbose: bool = False):
        """
        Initialize Evolution Strategy optimizer.
        
        Args:
            dim: Number of dimensions
            bounds: List of (min, max) bounds for each dimension
            population_size: Population size
            offspring_ratio: Ratio of offspring to population size
            initial_step_size: Initial step size for mutation
            adaptation_rate: Rate of parameter adaptation
            max_evals: Maximum number of function evaluations
            timeout: Maximum runtime in seconds
            adaptive: Whether to use adaptive parameters
            name: Optional name for the optimizer
            verbose: Whether to show progress bars and additional output
        """
        super().__init__(dim, bounds, population_size, adaptive)
        
        # Store name
        self.name = name or "ES"
        
        # ES-specific parameters
        self.initial_step_size = initial_step_size
        self.offspring_ratio = offspring_ratio
        self.offspring_size = max(1, int(offspring_ratio * population_size))
        self.adaptation_rate = adaptation_rate
        self.max_evals = max_evals
        self.timeout = timeout  # Store timeout as an instance variable
        self.verbose = verbose
        
        # Initialize strategy parameters
        self.strategy_params = np.ones(population_size) * initial_step_size
        
        # Parameter history for tracking adaptation
        self.param_history = {
            'step_size': [],
            'success_rate': [],
            'diversity': []
        }
        
        # Initialize population with size based on max_evals
        self.population_size = min(population_size, max(5, max_evals // 10))  # Ensure reasonable population size
        self.offspring_size = max(1, int(offspring_ratio * self.population_size))
        
        # Initialize population
        self.population = self._init_population()
        self.population_scores = np.full(self.population_size, np.inf)
        
        # Initialize tracking variables
        self.evaluations = 0
        self.start_time = 0
        self.end_time = 0
        
        # Initialize history tracking
        self.convergence_curve = []
        self.time_curve = []
        self.eval_curve = []
        self.diversity_history = []
        
        # Success tracking - use numpy array with fixed size
        self.success_history = np.zeros(20)  # Track last 20 iterations
        self.success_idx = 0
        self.last_improvement_iter = 0
        self._current_iteration = 0
        self.selection_pressure = []  # Initialize as a list
        
        self.start_time = None
        
        # Initialize sigmas for mutation control
        self.initial_sigma = initial_step_size
        self.sigmas = np.ones((population_size, dim)) * initial_step_size
    
    def _calculate_diversity(self) -> float:
        """
        Calculate population diversity using mean distance from centroid.
        
        Returns:
            Normalized diversity measure
        """
        if self.population is None or len(self.population) <= 1:
            return 0.0
            
        # Calculate centroid
        centroid = np.mean(self.population, axis=0)
        
        # Calculate distances from centroid
        distances = np.sqrt(np.sum((self.population - centroid)**2, axis=1))
        mean_dist = np.mean(distances)
        
        # Normalize by bounds range
        bounds_range = np.mean([ub - lb for lb, ub in self.bounds])
        if bounds_range == 0:
            return 0.0
            
        # Return normalized diversity
        return mean_dist / (bounds_range * np.sqrt(self.dim))
    
    def _generate_offspring(self) -> np.ndarray:
        """
        Generate offspring using recombination and mutation.
        
        Returns:
            New candidate solution
        """
        # Select parents using tournament selection
        tournament_size = min(3, self.population_size // 2)
        parent_indices = []
        
        # Select two parents
        for _ in range(2):
            candidates = np.random.choice(self.population_size, tournament_size, replace=False)
            parent_indices.append(candidates[np.argmin(self.population_scores[candidates])])
        
        # Recombine parents (intermediate recombination)
        parent1 = self.population[parent_indices[0]]
        parent2 = self.population[parent_indices[1]]
        
        # Dynamic weighting based on fitness values
        if self.population_scores[parent_indices[0]] < self.population_scores[parent_indices[1]]:
            # First parent is better
            alpha = 0.75  # Weight more towards the better parent
        else:
            # Second parent is better
            alpha = 0.25  # Weight more towards the better parent
            
        # Weighted recombination
        child = alpha * parent1 + (1 - alpha) * parent2
        
        # Apply mutation with adaptive step size
        # Calculate progress through the optimization
        progress = min(1.0, self.evaluations / (self.max_evals * 0.7))
        
        # Higher variance early, lower variance later (simulated annealing effect)
        global_step_factor = (1.0 - progress * 0.9)
        
        # Get individual step size from strategy parameters
        individual_step_size = np.mean([self.strategy_params[i] for i in parent_indices])
        
        # Calculate final step size with global cooling factor
        step_size = global_step_factor * individual_step_size
        
        # Apply non-isotropic mutation (different step size per dimension)
        # This helps navigate fitness landscapes with different scaling per dimension
        dimension_factors = 0.5 + np.random.random(self.dim)
        mutation = np.random.normal(0, step_size, size=self.dim) * dimension_factors
        
        # Apply mutation
        child = child + mutation
        
        # Ensure solution stays within bounds
        return self._bound_solution(child)
    
    def _recombine(self, parent1: np.ndarray, parent2: np.ndarray) -> np.ndarray:
        """Recombine two parents to create a child"""
        # Intermediate recombination (average)
        return (parent1 + parent2) / 2.0
    
    def _update_parameters(self):
        """
        Update optimizer parameters based on performance using the 1/5 success rule.
        
        The 1/5 success rule is a well-established heuristic in Evolution Strategies
        that adjusts step sizes to maintain a success rate of approximately 1/5.
        """
        if not self.adaptive:
            return
            
        # Calculate success rate from success history
        # Add safety check for empty array
        if np.any(self.success_history):
            success_rate = np.mean(self.success_history)
        else:
            success_rate = 0.0
        
        # Calculate population diversity
        diversity = self._calculate_diversity()
        self.diversity_history.append(diversity)
        
        # Apply 1/5 success rule with damping factor
        target_success_rate = 0.2  # The target 1/5 success rate
        c = 0.9  # Damping factor to prevent oscillations
        
        if success_rate > target_success_rate:
            # Too many successes, increase step size
            adaptation_factor = 1 + c * (success_rate - target_success_rate)
        else:
            # Too few successes, decrease step size
            adaptation_factor = 1 - c * (target_success_rate - success_rate)
            
        # Apply adaptation factor with bounds
        self.strategy_params *= adaptation_factor
        
        # Ensure step sizes stay within reasonable bounds relative to the problem scale
        bounds_range = np.mean([ub - lb for lb, ub in self.bounds])
        min_step = 0.0001 * bounds_range  # Minimum step size
        max_step = 0.2 * bounds_range     # Maximum step size
        
        # Clip step sizes
        self.strategy_params = np.clip(self.strategy_params, min_step, max_step)
        
        # Record parameter values for analysis
        self.param_history['step_size'].append(float(np.mean(self.strategy_params)))
        self.param_history['success_rate'].append(float(success_rate))
        self.param_history['diversity'].append(float(diversity))
    
    def _iterate(self, objective_func, *args, **kwargs):
        """Perform one iteration of the optimization algorithm."""
        # Generate offspring
        offspring = self._generate_offspring()
        offspring_score = objective_func(offspring)
        self.evaluations += 1
        
        # Record history if requested (only occasionally to save memory)
        if hasattr(self, 'record_history') and self.record_history and self._current_iteration % 10 == 0:
            self._record_history()
        
        # Update parameters if adaptive optimization is enabled
        if self._current_iteration % 5 == 0 and self.adaptive:
            self._update_parameters()
        
        # Update population if offspring is better than worst
        worst_idx = np.argmax(self.population_scores)
        if offspring_score < self.population_scores[worst_idx]:
            self.population[worst_idx] = offspring
            self.population_scores[worst_idx] = offspring_score
            # Update success history
            self.success_history[self.success_idx] = 1
            self.last_improvement_iter = self._current_iteration  # Update improvement tracker
        else:
            # Update success history
            self.success_history[self.success_idx] = 0
        
        # Update success index
        self.success_idx = (self.success_idx + 1) % len(self.success_history)
        
        # Update best solution if offspring is better
        if offspring_score < self.best_score:
            self.best_solution = offspring.copy()
            self.best_score = offspring_score
            self.last_improvement_iter = self._current_iteration  # Update improvement tracker
            
        # Check for timeout
        if time.time() - self.start_time > self.timeout:
            return self.best_solution, self.best_score
            
        return self.best_solution, self.best_score
    
    def reset(self):
        """Reset optimizer state"""
        super().reset()
        
        # Reset parameters
        self.strategy_params = np.ones(self.population_size) * 0.01
        
        # Initialize population
        self.population = self._init_population()
        self.population_scores = np.full(self.population_size, np.inf)
        
        # Initialize parameter history
        if self.adaptive:
            self.param_history = {
                'step_size': [0.01],
                'success_rate': [],
                'diversity': []
            }
        else:
            self.param_history = {
                'diversity': []
            }
    
    def _optimize(self, objective_func: Callable, context: Optional[Dict[str, Any]] = None) -> Tuple[np.ndarray, float]:
        """Run ES optimization"""
        # Initialize population
        self.population = self._init_population()
        self.population_scores = np.full(self.population_size, np.inf)
        
        # Initialize strategy parameters
        self.sigmas = np.ones((self.population_size, self.dim)) * self.initial_sigma
        
        # Initialize success rates
        success_rates = np.zeros(self.population_size)
        
        # Evaluate initial population
        for i in range(self.population_size):
            self.population_scores[i] = self._evaluate(self.population[i], objective_func)
        
        # Track initial diversity
        self._update_diversity()
        
        # Number of generations without improvement
        stagnation_count = 0
        best_score_history = []
        
        while not self._check_convergence():
            # Generate offspring
            offspring = []
            offspring_scores = []
            offspring_sigmas = []
            
            for i in range(self.offspring_size):
                # Select parents using tournament selection
                parent_indices = np.random.choice(self.population_size, 4, replace=False)
                tournament_scores = self.population_scores[parent_indices]
                parent1_idx = parent_indices[np.argmin(tournament_scores[:2])]
                parent2_idx = parent_indices[np.argmin(tournament_scores[2:])]
                
                parent1 = self.population[parent1_idx]
                parent2 = self.population[parent2_idx]
                parent1_sigma = self.sigmas[parent1_idx]
                parent2_sigma = self.sigmas[parent2_idx]
                
                # Recombine
                child = self._recombine(parent1, parent2)
                child_sigma = (parent1_sigma + parent2_sigma) / 2
                
                # Mutate strategy parameters
                tau = 1 / np.sqrt(2 * self.dim)
                tau_prime = 1 / np.sqrt(2 * np.sqrt(self.dim))
                
                # Update sigmas
                child_sigma *= np.exp(tau_prime * np.random.normal() + 
                                   tau * np.random.normal(size=self.dim))
                child_sigma = np.clip(child_sigma, 1e-10, 1.0)
                
                # Mutate solution
                child += child_sigma * np.random.normal(size=self.dim)
                child = self._bound_solution(child)
                
                # Local search with probability
                if np.random.random() < 0.1:  # 10% chance
                    # Try small perturbations
                    for _ in range(5):  # Try 5 local moves
                        perturb = np.random.normal(0, 0.1, size=self.dim)
                        new_child = self._bound_solution(child + perturb)
                        new_score = objective_func(new_child)
                        if new_score < objective_func(child):
                            child = new_child
                
                # Evaluate
                score = self._evaluate(child, objective_func)
                
                offspring.append(child)
                offspring_scores.append(score)
                offspring_sigmas.append(child_sigma)
            
            # Selection
            combined_pop = np.vstack([self.population, offspring])
            combined_scores = np.concatenate([self.population_scores, offspring_scores])
            combined_sigmas = np.vstack([self.sigmas, offspring_sigmas])
            
            # Select best individuals
            indices = np.argsort(combined_scores)[:self.population_size]
            self.population = combined_pop[indices]
            self.population_scores = combined_scores[indices]
            self.sigmas = combined_sigmas[indices]
            
            # Check for improvement
            current_best = np.min(self.population_scores)
            if len(best_score_history) > 0 and current_best >= best_score_history[-1]:
                stagnation_count += 1
            else:
                stagnation_count = 0
            best_score_history.append(current_best)
            
            # If stagnating, increase mutation strength
            if stagnation_count > 10:  # After 10 generations of no improvement
                self.sigmas *= 1.5  # Increase exploration
                stagnation_count = 0
            
            # Update parameters
            if self.adaptive:
                self._update_parameters()
            
            # Track diversity
            self._update_diversity()
        
        return self.best_solution, self.best_score

    def optimize(self, objective_func: Callable,
                max_evals: Optional[int] = None,
                record_history: bool = True,
                context: Optional[Dict[str, Any]] = None) -> Tuple[np.ndarray, float]:
        """
        Run Evolution Strategy optimization.
        
        Args:
            objective_func: Function to minimize
            max_evals: Maximum number of function evaluations
            record_history: Whether to record convergence history
            context: Optional context dictionary for problem-specific information
            
        Returns:
            Best solution found and its score
        """
        # Update max_evals if provided
        if max_evals is not None:
            self.max_evals = max_evals
            
        self.start_time = time.time()
        
        # Re-initialize success_history to ensure it's a numpy array
        self.success_history = np.zeros(20)
        self.success_idx = 0
        
        # Initialize population using random sampling within bounds
        self.population = self._init_population()
        self.population_scores = np.zeros(self.population_size)
            
        # Evaluate initial population
        for i in range(self.population_size):
            self.population_scores[i] = objective_func(self.population[i])
            self.evaluations += 1
        
        # Sort initial population
        sort_idx = np.argsort(self.population_scores)
        self.population = self.population[sort_idx]
        self.population_scores = self.population_scores[sort_idx]
        
        # Initialize best solution
        self.best_solution = self.population[0].copy()
        self.best_score = self.population_scores[0]
        self.last_improvement_iter = 0  # Initialize improvement tracker
        
        # Main optimization loop
        remaining_evals = max(0, self.max_evals - self.evaluations)
        max_iters = remaining_evals // (self.offspring_size + 1)  # Account for offspring and potential local search
        iter_count = 0
        
        logging.debug(f"ES: Starting optimization loop with max_evals={remaining_evals}, max_iters={max_iters}")
        
        # Create progress bar tracking evaluations instead of iterations
        with tqdm(total=remaining_evals, desc="ES Optimization", unit="evals", position=1, leave=True) as pbar:
            while not self._check_convergence() and iter_count < max_iters:
                # Record current best solution
                self.convergence_curve.append(float(self.best_score))
                
                # Log progress periodically
                if iter_count % 50 == 0:
                    logging.debug(f"ES: Iteration {iter_count}, best score: {self.best_score:.6f}, evals: {self.evaluations}")
                    runtime = time.time() - self.start_time
                    evals_per_sec = self.evaluations / max(runtime, 0.001)
                    success_rate = np.mean(self.success_history) * 100
                    pbar.set_postfix({
                        "best": f"{self.best_score:.2e}",
                        "e/s": f"{evals_per_sec:.1f}",
                        "success": f"{success_rate:.0f}%",
                        "div": f"{self._calculate_diversity():.2f}"
                    })
                    
                # Main iteration
                new_best, score = self._iterate(objective_func)
                self._current_iteration = iter_count
                
                # Update progress based on actual evaluations
                pbar.n = self.evaluations - (self.max_evals - remaining_evals)
                pbar.refresh()
                
                # Increment iteration counter
                iter_count += 1
                
                # Check if we're close to evaluation budget
                if self.evaluations >= self.max_evals - self.offspring_size:
                    break
                
        # Add final local search phase for the best solution
        if self.best_score > 0.01:  # Only if we haven't already converged well
            logging.debug(f"ES: Starting final local search phase from score={self.best_score:.6f}")
            
            # Use smaller step sizes for fine-tuning
            local_step_size = 0.01
            
            # Calculate remaining evaluations for local search
            remaining_local_evals = max(0, self.max_evals - self.evaluations)
            local_search_iters = min(100, remaining_local_evals)
            with tqdm(total=local_search_iters, desc="ES Local Search", unit="iter", position=1, leave=True) as ls_pbar:
                for ls_iter in range(local_search_iters):  # Local search iterations
                    if self.evaluations >= self.max_evals or time.time() - self.start_time > self.timeout:
                        break
                        
                    # Create variations of the best solution
                    variations = []
                    scores = []
                    
                    # Create multiple variations with decreasing step sizes
                    decay_factor = 0.95
                    curr_step = local_step_size
                    
                    for _ in range(10):  # Try 10 variations per iteration
                        # Create a variation with decreasing step size
                        variation = self.best_solution + np.random.normal(0, curr_step, size=self.dim)
                        variation = self._bound_solution(variation)
                        score = objective_func(variation)
                        self.evaluations += 1
                        
                        variations.append(variation)
                        scores.append(score)
                        
                        # Decrease step size for next variation
                        curr_step *= decay_factor
                    
                    # Check if any variations are better
                    best_var_idx = np.argmin(scores)
                    if scores[best_var_idx] < self.best_score:
                        self.best_solution = variations[best_var_idx].copy()
                        self.best_score = scores[best_var_idx]
                        logging.debug(f"ES: Local search improved score to {self.best_score:.6f}")
                        runtime = time.time() - self.start_time
                        evals_per_sec = self.evaluations / max(runtime, 0.001)
                        ls_pbar.set_postfix({
                            "best": f"{self.best_score:.2e}",
                            "e/s": f"{evals_per_sec:.1f}",
                            "step": f"{local_step_size:.2e}"
                        })
                    else:
                        # No improvement, reduce step size
                        local_step_size *= 0.5
                        if local_step_size < 1e-6:
                            logging.debug("ES: Local search converged (step size too small)")
                            break
                    
                    # Update progress bar
                    ls_pbar.update(1)
        
        # Record final best
        self.convergence_curve.append(float(self.best_score))
        
        self.end_time = time.time()
        return self.best_solution.copy(), self.best_score

    def _update_diversity(self):
        """Track population diversity"""
        diversity = np.mean(np.std(self.population, axis=0))
        self.diversity_history.append(diversity)
        self.param_history['diversity'].append(diversity)
    
    def _evaluate(self, solution: np.ndarray, objective_func: Callable) -> float:
        """
        Evaluate a solution using the objective function.
        
        Args:
            solution: Solution to evaluate
            objective_func: Objective function
            
        Returns:
            Fitness score
        """
        self.evaluations += 1
        return objective_func(solution)

    def _record_history(self):
        """Record optimization history"""
        # Record convergence history
        self.convergence_curve.append(self.best_score)
        
        # Record evaluation count
        self.eval_curve.append(self.evaluations)
        
        # Record time
        self.time_curve.append(time.time() - self.start_time)

    def get_parameters(self) -> Dict[str, Any]:
        """
        Get optimizer parameters.
        
        Returns:
            Dictionary of optimizer parameters
        """
        return {
            'population_size': self.population_size,
            'offspring_size': self.offspring_size,
            'step_size': np.mean(self.strategy_params) if hasattr(self, 'strategy_params') else self.initial_step_size,
            'max_evals': self.max_evals,
            'adaptive': self.adaptive,
            'evaluations': self.evaluations,
            'runtime': self.end_time - self.start_time if hasattr(self, 'end_time') and self.end_time > 0 else 0,
            'diversity': self.diversity_history[-1] if hasattr(self, 'diversity_history') and self.diversity_history else 0
        }

    def _check_convergence(self) -> bool:
        """
        Check if optimization has converged based on multiple criteria.
        
        Returns:
            True if converged, False otherwise
        """
        # Check if max evaluations reached
        if self.evaluations >= self.max_evals:
            logging.debug("ES: Convergence due to max evaluations")
            return True
            
        # Check if timeout reached
        if hasattr(self, 'timeout') and time.time() - self.start_time > self.timeout:
            logging.debug("ES: Convergence due to timeout")
            return True
            
        # Check if population has converged (std of fitness below threshold)
        if np.std(self.population_scores) < 1e-8:
            logging.debug("ES: Convergence due to low population diversity")
            return True
            
        # Check if best score is very close to zero (for minimization)
        if self.best_score < 1e-6:
            logging.debug(f"ES: Convergence due to optimal score: {self.best_score}")
            return True
            
        # Check if solution has not improved for many iterations
        if hasattr(self, 'last_improvement_iter') and self._current_iteration - self.last_improvement_iter > 50:
            logging.debug("ES: Convergence due to no improvement")
            return True
            
        return False

    def _bound_solution(self, solution: np.ndarray) -> np.ndarray:
        """
        Ensure solution stays within the specified bounds.
        
        Args:
            solution: Solution to bound
            
        Returns:
            Bounded solution
        """
        lower_bounds = np.array([b[0] for b in self.bounds])
        upper_bounds = np.array([b[1] for b in self.bounds])
        return np.clip(solution, lower_bounds, upper_bounds)

    def _init_population(self) -> np.ndarray:
        """
        Initialize population using uniform random sampling within bounds.
        
        Returns:
            Initial population as numpy array of shape (population_size, dim)
        """
        population = np.zeros((self.population_size, self.dim))
        
        # Get lower and upper bounds
        lower_bounds = np.array([b[0] for b in self.bounds])
        upper_bounds = np.array([b[1] for b in self.bounds])
        
        # Generate initial population within bounds
        for i in range(self.population_size):
            population[i] = lower_bounds + np.random.random(self.dim) * (upper_bounds - lower_bounds)
            
        return population
