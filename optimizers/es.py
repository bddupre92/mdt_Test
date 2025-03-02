"""
es.py
------
Evolution Strategy optimizer with adaptive parameters.
"""

import numpy as np
from typing import Dict, Any, Optional, List, Tuple, Callable
from .base_optimizer import BaseOptimizer
import time

class EvolutionStrategyOptimizer(BaseOptimizer):
    def __init__(self,
                 dim: int,
                 bounds: List[Tuple[float, float]],
                 population_size: int = 50,
                 max_evals: int = 5000,
                 adaptive: bool = True,
                 offspring_size: int = None,
                 initial_step_size: float = 0.01,
                 timeout: float = 30.0,
                 **kwargs):
        """
        Initialize ES optimizer.
        
        Args:
            dim: Problem dimensionality
            bounds: Parameter bounds
            population_size: Population size
            max_evals: Maximum function evaluations
            adaptive: Whether to use parameter adaptation
            offspring_size: Number of offspring (default: population_size)
            initial_step_size: Initial step size (default: 0.01)
            timeout: Timeout in seconds (default: 30.0)
        """
        super().__init__(dim=dim, bounds=bounds, population_size=population_size,
                        adaptive=adaptive)
        
        # ES parameters
        self.offspring_size = offspring_size or population_size
        self.strategy_params = np.ones(population_size) * initial_step_size
        
        # Store max_evals as instance variable
        self.max_evals = max_evals
        
        # Initialize population
        self.population = self._init_population()
        self.population_scores = np.full(population_size, np.inf)
        
        # Initialize parameter history
        if adaptive:
            self.param_history = {
                'step_size': [initial_step_size],  # Initial mean step size
                'success_rate': [],
                'diversity': []
            }
        else:
            self.param_history = {
                'diversity': []
            }
        
        self.timeout = timeout
        self.start_time = None
    
    def _calculate_diversity(self) -> float:
        """Calculate population diversity"""
        centroid = np.mean(self.population, axis=0)
        distances = np.sqrt(np.sum((self.population - centroid)**2, axis=1))
        return np.mean(distances)
    
    def _generate_offspring(self) -> np.ndarray:
        """Generate offspring using recombination and mutation"""
        # Select parents using tournament selection
        tournament_size = 3
        parent_indices = []
        for _ in range(2):  # Select 2 parents
            candidates = np.random.choice(self.population_size, tournament_size, replace=False)
            parent_indices.append(candidates[np.argmin(self.population_scores[candidates])])
        
        # Recombine parents
        parent1 = self.population[parent_indices[0]]
        parent2 = self.population[parent_indices[1]]
        child = (parent1 + parent2) / 2.0
        
        # Apply mutation with adaptive step size
        step_size = np.mean([self.strategy_params[i] for i in parent_indices])
        mutation = np.random.normal(0, step_size, size=self.dim)
        child = child + mutation
        
        return self._bound_solution(child)
    
    def _update_parameters(self):
        """Update optimizer parameters based on performance"""
        if not self.adaptive:
            return
            
        # Calculate success rate
        success_rate = np.mean(self.success_history)
        
        # Calculate population diversity
        diversity = self._calculate_diversity()
        
        # Update step sizes based on both success rate and diversity
        target_success_rate = 0.2
        target_diversity = 0.1  # Relative to bounds range
        
        # Compute adaptation factors
        success_factor = 1.0 + 0.2 * (success_rate - target_success_rate)
        diversity_factor = 1.0 + 0.2 * (target_diversity - diversity)
        
        # Apply combined adaptation
        adaptation_factor = np.clip(success_factor * diversity_factor, 0.8, 1.2)
        self.strategy_params *= adaptation_factor
        
        # Keep step sizes within even tighter bounds
        bounds_range = np.mean([ub - lb for lb, ub in self.bounds])
        min_step = 0.0001 * bounds_range
        max_step = 0.1 * bounds_range
        self.strategy_params = np.clip(self.strategy_params, min_step, max_step)
        
        # Record parameter values
        self.param_history['step_size'].append(np.mean(self.strategy_params))
        self.param_history['success_rate'].append(success_rate)
        self.param_history['diversity'].append(diversity)
    
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
        Run ES optimization.
        
        Args:
            objective_func: Function to minimize
            max_evals: Maximum number of function evaluations (overrides init value)
            record_history: Whether to record convergence and parameter history
            context: Optional problem context
            
        Returns:
            Best solution found and its score
        """
        # Update max_evals if provided
        if max_evals is not None:
            self.max_evals = max_evals
            
        self.start_time = time.time()
        
        # Initialize population scores
        for i in range(self.population_size):
            self.population_scores[i] = self._evaluate(self.population[i], objective_func)
        
        # Sort initial population
        sort_idx = np.argsort(self.population_scores)
        self.population = self.population[sort_idx]
        self.population_scores = self.population_scores[sort_idx]
        
        # Track initial diversity
        self._update_diversity()
        
        # Track best solution
        self.best_solution = self.population[0].copy()
        self.best_score = self.population_scores[0]
        
        num_evals = len(self.population)
        no_improvement = 0
        
        while num_evals < self.max_evals:
            # Check timeout
            if time.time() - self.start_time > self.timeout:
                print(f"ES optimization stopped due to timeout after {num_evals} evaluations")
                break
                
            # Generate and evaluate offspring
            offspring = self._generate_offspring()
            offspring_score = self._evaluate(offspring, objective_func)
            num_evals += 1
            
            # Update population if offspring is better than worst
            worst_idx = np.argmax(self.population_scores)
            if offspring_score < self.population_scores[worst_idx]:
                self.population[worst_idx] = offspring
                self.population_scores[worst_idx] = offspring_score
                self.success_history[self.success_idx] = 1
                no_improvement = 0
            else:
                self.success_history[self.success_idx] = 0
                no_improvement += 1
            
            self.success_idx = (self.success_idx + 1) % len(self.success_history)
            
            # Update best solution
            if offspring_score < self.best_score:
                self.best_solution = offspring
                self.best_score = offspring_score
                
            # Update parameters
            if self.adaptive and num_evals % self.population_size == 0:
                self._update_parameters()
                
            # Early stopping if no improvement for a while
            if no_improvement > 100:
                print(f"ES optimization stopped due to no improvement after {num_evals} evaluations")
                break
                
        self.end_time = time.time()
        return self.best_solution, self.best_score
