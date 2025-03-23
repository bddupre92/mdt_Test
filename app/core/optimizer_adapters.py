"""
Optimizer Adapters Module

This module provides adapters for different optimization algorithms to ensure a consistent interface.
"""

import numpy as np
from typing import Tuple, Callable, Optional, List, Dict, Any
from abc import ABC, abstractmethod

class OptimizerAdapter(ABC):
    """Base class for optimizer adapters."""
    
    @abstractmethod
    def optimize(self, 
                objective: Callable[[np.ndarray], float],
                dimension: int,
                bounds: Tuple[float, float],
                max_evaluations: int = 1000) -> Tuple[np.ndarray, float, int]:
        """Run optimization.
        
        Args:
            objective: Objective function to minimize
            dimension: Problem dimension
            bounds: Function bounds (min, max)
            max_evaluations: Maximum function evaluations
            
        Returns:
            Tuple of (best_position, best_fitness, evaluations)
        """
        pass

class DifferentialEvolutionAdapter(OptimizerAdapter):
    """Adapter for Differential Evolution."""
    
    def __init__(self, population_size: int = 50, F: float = 0.8, CR: float = 0.7):
        """Initialize DE optimizer.
        
        Args:
            population_size: Population size
            F: Mutation factor
            CR: Crossover probability
        """
        self.population_size = population_size
        self.F = F
        self.CR = CR
    
    def optimize(self, 
                objective: Callable[[np.ndarray], float],
                dimension: int,
                bounds: Tuple[float, float],
                max_evaluations: int = 1000) -> Tuple[np.ndarray, float, int]:
        """Run Differential Evolution optimization."""
        # Initialize population
        population = np.random.uniform(
            bounds[0], bounds[1], 
            size=(self.population_size, dimension)
        )
        fitness = np.array([objective(ind) for ind in population])
        evaluations = self.population_size
        
        best_idx = np.argmin(fitness)
        best_position = population[best_idx].copy()
        best_fitness = fitness[best_idx]
        
        while evaluations < max_evaluations:
            for i in range(self.population_size):
                # Select three random individuals
                idxs = [idx for idx in range(self.population_size) if idx != i]
                a, b, c = population[np.random.choice(idxs, 3, replace=False)]
                
                # Create mutant
                mutant = a + self.F * (b - c)
                
                # Crossover
                cross_points = np.random.rand(dimension) < self.CR
                if not np.any(cross_points):
                    cross_points[np.random.randint(0, dimension)] = True
                
                trial = np.where(cross_points, mutant, population[i])
                
                # Bound constraint
                trial = np.clip(trial, bounds[0], bounds[1])
                
                # Selection
                trial_fitness = objective(trial)
                evaluations += 1
                
                if trial_fitness < fitness[i]:
                    population[i] = trial
                    fitness[i] = trial_fitness
                    
                    if trial_fitness < best_fitness:
                        best_position = trial.copy()
                        best_fitness = trial_fitness
                
                if evaluations >= max_evaluations:
                    break
        
        return best_position, best_fitness, evaluations

class EvolutionStrategyAdapter(OptimizerAdapter):
    """Adapter for Evolution Strategy (μ + λ)."""
    
    def __init__(self, mu: int = 10, lambda_: int = 20, sigma: float = 0.1):
        """Initialize ES optimizer.
        
        Args:
            mu: Parent population size
            lambda_: Offspring population size
            sigma: Initial step size
        """
        self.mu = mu
        self.lambda_ = lambda_
        self.sigma = sigma
    
    def optimize(self, 
                objective: Callable[[np.ndarray], float],
                dimension: int,
                bounds: Tuple[float, float],
                max_evaluations: int = 1000) -> Tuple[np.ndarray, float, int]:
        """Run Evolution Strategy optimization."""
        # Initialize parent population
        parents = np.random.uniform(
            bounds[0], bounds[1], 
            size=(self.mu, dimension)
        )
        fitness = np.array([objective(ind) for ind in parents])
        evaluations = self.mu
        
        best_idx = np.argmin(fitness)
        best_position = parents[best_idx].copy()
        best_fitness = fitness[best_idx]
        
        while evaluations < max_evaluations:
            # Generate offspring
            offspring = np.zeros((self.lambda_, dimension))
            for i in range(self.lambda_):
                # Select random parent
                parent = parents[np.random.randint(self.mu)]
                
                # Add Gaussian noise
                offspring[i] = parent + self.sigma * np.random.randn(dimension)
                
                # Bound constraint
                offspring[i] = np.clip(offspring[i], bounds[0], bounds[1])
            
            # Evaluate offspring
            offspring_fitness = np.array([objective(ind) for ind in offspring])
            evaluations += self.lambda_
            
            # Select best individuals for next generation
            population = np.vstack((parents, offspring))
            population_fitness = np.hstack((fitness, offspring_fitness))
            
            best_indices = np.argsort(population_fitness)[:self.mu]
            parents = population[best_indices]
            fitness = population_fitness[best_indices]
            
            # Update best solution
            if fitness[0] < best_fitness:
                best_position = parents[0].copy()
                best_fitness = fitness[0]
            
            # Update step size
            self.sigma *= 0.99
            
            if evaluations >= max_evaluations:
                break
        
        return best_position, best_fitness, evaluations

class AntColonyAdapter(OptimizerAdapter):
    """Adapter for Ant Colony Optimization."""
    
    def __init__(self, n_ants: int = 30, evaporation: float = 0.1, q: float = 1.0):
        """Initialize ACO optimizer.
        
        Args:
            n_ants: Number of ants
            evaporation: Pheromone evaporation rate
            q: Pheromone deposit factor
        """
        self.n_ants = n_ants
        self.evaporation = evaporation
        self.q = q
    
    def optimize(self, 
                objective: Callable[[np.ndarray], float],
                dimension: int,
                bounds: Tuple[float, float],
                max_evaluations: int = 1000) -> Tuple[np.ndarray, float, int]:
        """Run Ant Colony Optimization."""
        # Initialize pheromone matrix
        n_points = 20  # Discretization points per dimension
        pheromone = np.ones((dimension, n_points)) / n_points
        
        # Initialize best solution
        best_position = np.random.uniform(bounds[0], bounds[1], dimension)
        best_fitness = objective(best_position)
        evaluations = 1
        
        while evaluations < max_evaluations:
            # Generate solutions for all ants
            solutions = np.zeros((self.n_ants, dimension))
            fitness = np.zeros(self.n_ants)
            
            for i in range(self.n_ants):
                # Construct solution
                solution = np.zeros(dimension)
                for j in range(dimension):
                    # Select value based on pheromone levels
                    probs = pheromone[j] / np.sum(pheromone[j])
                    idx = np.random.choice(n_points, p=probs)
                    solution[j] = bounds[0] + (bounds[1] - bounds[0]) * idx / (n_points - 1)
                
                solutions[i] = solution
                fitness[i] = objective(solution)
                evaluations += 1
                
                # Update best solution
                if fitness[i] < best_fitness:
                    best_position = solution.copy()
                    best_fitness = fitness[i]
                
                if evaluations >= max_evaluations:
                    break
            
            # Update pheromone levels
            pheromone *= (1 - self.evaporation)
            
            # Add new pheromone
            for i in range(self.n_ants):
                solution = solutions[i]
                delta = self.q / (fitness[i] + 1e-10)
                
                for j in range(dimension):
                    # Find closest discretization point
                    idx = int(round((solution[j] - bounds[0]) / (bounds[1] - bounds[0]) * (n_points - 1)))
                    pheromone[j, idx] += delta
            
            if evaluations >= max_evaluations:
                break
        
        return best_position, best_fitness, evaluations

class GreyWolfAdapter(OptimizerAdapter):
    """Adapter for Grey Wolf Optimization."""
    
    def __init__(self, n_wolves: int = 30):
        """Initialize GWO optimizer.
        
        Args:
            n_wolves: Number of wolves in the pack
        """
        self.n_wolves = n_wolves
    
    def optimize(self, 
                objective: Callable[[np.ndarray], float],
                dimension: int,
                bounds: Tuple[float, float],
                max_evaluations: int = 1000) -> Tuple[np.ndarray, float, int]:
        """Run Grey Wolf Optimization."""
        # Initialize population
        population = np.random.uniform(
            bounds[0], bounds[1], 
            size=(self.n_wolves, dimension)
        )
        fitness = np.array([objective(ind) for ind in population])
        evaluations = self.n_wolves
        
        # Initialize alpha, beta, and delta wolves
        sorted_indices = np.argsort(fitness)
        alpha = population[sorted_indices[0]].copy()
        alpha_score = fitness[sorted_indices[0]]
        beta = population[sorted_indices[1]].copy()
        beta_score = fitness[sorted_indices[1]]
        delta = population[sorted_indices[2]].copy()
        delta_score = fitness[sorted_indices[2]]
        
        best_position = alpha.copy()
        best_fitness = alpha_score
        
        while evaluations < max_evaluations:
            # Update a (linearly decreased from 2 to 0)
            a = 2 * (1 - evaluations / max_evaluations)
            
            for i in range(self.n_wolves):
                # Update position
                A1 = 2 * a * np.random.rand(dimension) - a
                C1 = 2 * np.random.rand(dimension)
                D_alpha = abs(C1 * alpha - population[i])
                X1 = alpha - A1 * D_alpha
                
                A2 = 2 * a * np.random.rand(dimension) - a
                C2 = 2 * np.random.rand(dimension)
                D_beta = abs(C2 * beta - population[i])
                X2 = beta - A2 * D_beta
                
                A3 = 2 * a * np.random.rand(dimension) - a
                C3 = 2 * np.random.rand(dimension)
                D_delta = abs(C3 * delta - population[i])
                X3 = delta - A3 * D_delta
                
                # New position
                population[i] = (X1 + X2 + X3) / 3
                
                # Bound constraint
                population[i] = np.clip(population[i], bounds[0], bounds[1])
                
                # Evaluate new position
                fitness[i] = objective(population[i])
                evaluations += 1
                
                # Update alpha, beta, and delta
                if fitness[i] < alpha_score:
                    alpha = population[i].copy()
                    alpha_score = fitness[i]
                    best_position = alpha.copy()
                    best_fitness = alpha_score
                elif fitness[i] < beta_score:
                    beta = population[i].copy()
                    beta_score = fitness[i]
                elif fitness[i] < delta_score:
                    delta = population[i].copy()
                    delta_score = fitness[i]
                
                if evaluations >= max_evaluations:
                    break
            
            if evaluations >= max_evaluations:
                break
        
        return best_position, best_fitness, evaluations

class MetaOptimizerAdapter(OptimizerAdapter):
    """Adapter for Meta-Optimizer."""
    
    def __init__(self, base_optimizers: Optional[List[OptimizerAdapter]] = None):
        """Initialize meta-optimizer.
        
        Args:
            base_optimizers: List of base optimizers to choose from
        """
        if base_optimizers is None:
            base_optimizers = [
                DifferentialEvolutionAdapter(),
                EvolutionStrategyAdapter(),
                AntColonyAdapter(),
                GreyWolfAdapter()
            ]
        self.base_optimizers = base_optimizers
        self.selected_optimizers = []
    
    def optimize(self, 
                objective: Callable[[np.ndarray], float],
                dimension: int,
                bounds: Tuple[float, float],
                max_evaluations: int = 1000) -> Tuple[np.ndarray, float, int]:
        """Run meta-optimization."""
        # Allocate evaluations to optimizers
        evaluations_per_optimizer = max_evaluations // len(self.base_optimizers)
        remaining_evaluations = max_evaluations % len(self.base_optimizers)
        
        best_position = None
        best_fitness = float('inf')
        total_evaluations = 0
        
        # Run each optimizer
        for i, optimizer in enumerate(self.base_optimizers):
            # Allocate evaluations
            if i == len(self.base_optimizers) - 1:
                opt_evaluations = evaluations_per_optimizer + remaining_evaluations
            else:
                opt_evaluations = evaluations_per_optimizer
            
            # Run optimization
            position, fitness, evaluations = optimizer.optimize(
                objective, dimension, bounds, opt_evaluations
            )
            total_evaluations += evaluations
            
            # Track selected optimizer
            self.selected_optimizers.append(optimizer.__class__.__name__)
            
            # Update best solution
            if fitness < best_fitness:
                best_position = position.copy()
                best_fitness = fitness
        
        return best_position, best_fitness, total_evaluations
    
    def get_selected_optimizers(self) -> List[str]:
        """Get list of selected optimizers in order of use."""
        return self.selected_optimizers.copy() 