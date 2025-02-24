"""
de.py
-----
Enhanced Differential Evolution with adaptive parameters and
improved exploration/exploitation balance.
"""

from typing import Tuple, List, Callable, Dict, Any, Optional
import numpy as np
from .base_optimizer import BaseOptimizer
import time
from collections import deque

class DifferentialEvolutionOptimizer(BaseOptimizer):
    def __init__(self,
                 dim: int,
                 bounds: List[Tuple[float, float]],
                 population_size: int = 50,
                 F: float = 0.8,
                 CR: float = 0.5,
                 strategy: str = 'best/1/bin',
                 adaptive: bool = True,
                 **kwargs):
        """
        Initialize DE optimizer with adaptive parameters.
        
        Args:
            dim: Problem dimensionality
            bounds: Parameter bounds
            population_size: Population size
            F: Initial mutation factor
            CR: Initial crossover rate
            strategy: DE variant strategy
            adaptive: Whether to use parameter adaptation
        """
        super().__init__(dim, bounds, population_size, adaptive=adaptive, **kwargs)
        
        self.F = F
        self.CR = CR
        self.strategy = strategy
        
        # Adaptive parameters
        if adaptive:
            self.F_range = (0.1, 1.0)
            self.CR_range = (0.1, 0.9)
            self.success_threshold = 0.6
            
            # Track parameter performance
            self.param_history = {'F': [F], 'CR': [CR]}
            self.param_memory = {
                'F': deque(maxlen=50),
                'CR': deque(maxlen=50)
            }
    
    def _mutate(self, population: np.ndarray, target_idx: int) -> np.ndarray:
        """
        Generate mutant vector using specified strategy.
        """
        if self.strategy == 'best/1/bin':
            best_idx = np.argmin(self.current_scores)
            r1, r2 = self._select_parents(target_idx, 2)
            
            mutant = (population[best_idx] + 
                     self.F * (population[r1] - population[r2]))
            
        elif self.strategy == 'rand/1/bin':
            r1, r2, r3 = self._select_parents(target_idx, 3)
            
            mutant = (population[r1] + 
                     self.F * (population[r2] - population[r3]))
            
        elif self.strategy == 'current-to-best/1/bin':
            best_idx = np.argmin(self.current_scores)
            r1, r2 = self._select_parents(target_idx, 2)
            
            mutant = (population[target_idx] +
                     self.F * (population[best_idx] - population[target_idx]) +
                     self.F * (population[r1] - population[r2]))
        
        return self._clip_to_bounds(mutant)
    
    def _crossover(self, target: np.ndarray, mutant: np.ndarray) -> np.ndarray:
        """
        Perform binomial crossover between target and mutant vectors.
        """
        mask = np.random.rand(self.dim) <= self.CR
        # Ensure at least one component is taken from mutant
        if not np.any(mask):
            mask[np.random.randint(0, self.dim)] = True
        
        return np.where(mask, mutant, target)
    
    def _select_parents(self, target_idx: int, count: int) -> List[int]:
        """Select random distinct parents excluding target_idx"""
        available = list(range(self.population_size))
        available.remove(target_idx)
        return np.random.choice(available, size=count, replace=False)
    
    def _adapt_parameters(self) -> None:
        """
        Adapt F and CR based on recent success history.
        Uses the JADE-inspired parameter adaptation.
        """
        if not self.adaptive:
            return
            
        # Calculate success rate over recent history
        if len(self.success_history) >= 10:
            success_rate = sum(self.success_history) / len(self.success_history)
            
            # Adapt F
            if success_rate > self.success_threshold:
                # Increase exploration
                self.F = min(self.F * 1.1, self.F_range[1])
            else:
                # Increase exploitation
                self.F = max(self.F * 0.9, self.F_range[0])
            
            # Adapt CR similarly
            if success_rate > self.success_threshold:
                self.CR = min(self.CR * 1.1, self.CR_range[1])
            else:
                self.CR = max(self.CR * 0.9, self.CR_range[0])
            
            # Store parameter history
            self.param_history['F'].append(self.F)
            self.param_history['CR'].append(self.CR)
    
    def optimize(self,
                objective_func: Callable,
                context: Optional[Dict[str, Any]] = None) -> Tuple[np.ndarray, float]:
        """
        Run the DE optimization process.
        
        Args:
            objective_func: Function to minimize
            context: Optional problem context
            
        Returns:
            Tuple of (best_solution, best_score)
        """
        start_time = time.time()
        
        # Initialize population
        population = np.array([
            self._random_solution()
            for _ in range(self.population_size)
        ])
        
        # Evaluate initial population
        self.current_scores = np.array([
            objective_func(ind) for ind in population
        ])
        self.state.evaluations += self.population_size
        
        best_idx = np.argmin(self.current_scores)
        self._update_state(
            population[best_idx],
            self.current_scores[best_idx]
        )
        
        # Main optimization loop
        while not self._check_convergence():
            self.state.generation += 1
            
            # Adapt parameters if needed
            if self.adaptive and self.state.generation % 10 == 0:
                self._adapt_parameters()
            
            # Evolution step
            for i in range(self.population_size):
                # Generate trial vector
                mutant = self._mutate(population, i)
                trial = self._crossover(population[i], mutant)
                
                # Evaluate trial vector
                trial_score = objective_func(trial)
                self.state.evaluations += 1
                
                # Selection
                if trial_score < self.current_scores[i]:
                    population[i] = trial
                    self.current_scores[i] = trial_score
                    self._update_state(trial, trial_score)
        
        self.state.runtime = time.time() - start_time
        return self.state.best_solution, self.state.best_score
