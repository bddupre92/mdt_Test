"""Differential Evolution optimizer implementation."""
from typing import Dict, List, Tuple, Optional, Any, Callable
import numpy as np
import logging
import time
from .base_optimizer import BaseOptimizer


class DifferentialEvolutionOptimizer(BaseOptimizer):
    """Differential Evolution optimizer."""
    
    def __init__(self, dim: int, bounds: List[Tuple[float, float]], 
                 population_size: Optional[int] = None,
                 F: float = 0.8,
                 CR: float = 0.5,
                 adaptive: bool = True):
        """
        Initialize DE optimizer.
        
        Args:
            dim: Number of dimensions
            bounds: List of (min, max) bounds for each dimension
            population_size: Optional population size
            F: Mutation factor
            CR: Crossover rate
            adaptive: Whether to use adaptive parameters
        """
        super().__init__(dim, bounds, population_size, adaptive)
        
        # DE-specific parameters
        self.F = F
        self.CR = CR
        
        # Initialize parameter history
        self.parameter_history = {
            'F': [self.F],
            'CR': [self.CR],
            'success_rate': [],
            'population_diversity': []
        }
        
        # Initialize tracking variables
        self.success_history = []
        self.selection_pressure = []
        self.diversity_history = []
        self.last_improvement_iter = 0
        
        # Log DE-specific parameters
        self.logger = logging.getLogger(__name__)
        self.logger.debug(f"F: {self.F}, CR: {self.CR}")
        
    def _iterate(self, objective_func: Callable):
        """Perform one iteration of DE."""
        # Track success in this iteration
        iteration_success = 0
        iteration_trials = 0
        
        for i in range(self.population_size):
            # Select random indices for mutation
            idxs = [idx for idx in range(self.population_size) if idx != i]
            a, b, c = np.random.choice(idxs, 3, replace=False)
            
            # Create mutant vector
            mutant = self.population[a] + self.F * (self.population[b] - self.population[c])
            
            # Ensure mutant is within bounds
            for j in range(self.dim):
                if mutant[j] < self.bounds[j][0]:
                    mutant[j] = self.bounds[j][0]
                elif mutant[j] > self.bounds[j][1]:
                    mutant[j] = self.bounds[j][1]
            
            # Crossover
            trial = np.copy(self.population[i])
            j_rand = np.random.randint(self.dim)
            for j in range(self.dim):
                if np.random.random() < self.CR or j == j_rand:
                    trial[j] = mutant[j]
            
            # Selection
            f_trial = objective_func(trial)
            f_target = objective_func(self.population[i])
            
            self.evaluations += 2
            iteration_trials += 1
            
            if f_trial < f_target:
                self.population[i] = trial
                iteration_success += 1
                if f_trial < self.best_score:
                    self.best_score = f_trial
                    self.best_solution = trial.copy()
                    self.success_history.append(True)
                    
                    # Record improvement
                    self.last_improvement_iter = self._current_iteration
                else:
                    self.success_history.append(False)
            else:
                self.success_history.append(False)
                
        # Calculate selection pressure for this iteration
        if iteration_trials > 0:
            selection_pressure = iteration_success / iteration_trials
            
            # Safely store selection pressure
            if not hasattr(self, 'selection_pressure'):
                self.selection_pressure = []
            self.selection_pressure.append(selection_pressure)
            
            # Safely update parameter history
            try:
                self.parameter_history['success_rate'].append(selection_pressure)
            except KeyError:
                # Create the key if it doesn't exist
                self.parameter_history['success_rate'] = [selection_pressure]
            except Exception as e:
                self.logger.warning(f"Could not update success_rate: {str(e)}")
                
        # Update convergence curve
        self.convergence_curve.append(self.best_score)
        
        # Adaptive parameter update
        if self.adaptive:
            self._update_parameters()
            
        self._current_iteration += 1
        
    def _update_parameters(self):
        """Update DE control parameters."""
        # Ensure success_history exists
        if not hasattr(self, 'success_history'):
            self.success_history = []
            
        if len(self.success_history) < 10:
            return
            
        # Calculate success rate over last 10 iterations
        recent_success = np.mean(self.success_history[-10:])
        
        # Safely update parameter history
        try:
            # Initialize parameter_history if it doesn't exist
            if not hasattr(self, 'parameter_history'):
                self.parameter_history = {}
                
            # Ensure keys exist in parameter_history
            for key in ['F', 'CR', 'success_rate', 'population_diversity']:
                if key not in self.parameter_history:
                    self.parameter_history[key] = []
                
            self.parameter_history['success_rate'].append(recent_success)
        except Exception as e:
            self.logger.warning(f"Could not update parameter history: {str(e)}")
        
        # Adjust F based on success rate
        if recent_success < 0.2:
            self.F *= 0.9  # Reduce step size
        elif recent_success > 0.8:
            self.F *= 1.1  # Increase step size
            
        # Keep F in reasonable bounds
        self.F = np.clip(self.F, 0.1, 2.0)
        
        # Safely update F in parameter history
        try:
            self.parameter_history['F'].append(self.F)
        except Exception as e:
            self.logger.warning(f"Could not update F history: {str(e)}")
        
        # Ensure diversity_history exists
        if not hasattr(self, 'diversity_history'):
            self.diversity_history = []
        
        # Adjust CR based on diversity
        if len(self.diversity_history) > 1:
            current_diversity = self.diversity_history[-1]
            self.parameter_history['population_diversity'].append(current_diversity)
            
            if current_diversity < 0.1:
                self.CR = min(0.9, self.CR * 1.1)  # Increase mixing
            elif current_diversity > 0.5:
                self.CR = max(0.1, self.CR * 0.9)  # Reduce mixing
                
        # Safely update CR in parameter history
        try:
            self.parameter_history['CR'].append(self.CR)
        except Exception as e:
            self.logger.warning(f"Could not update CR history: {str(e)}")
        
    def get_parameters(self) -> Dict[str, Any]:
        """
        Get optimizer parameters.
        
        Returns:
            Dictionary of parameters
        """
        return {
            'dim': self.dim,
            'population_size': self.population_size,
            'F': self.F,
            'CR': self.CR,
            'adaptive': self.adaptive,
            'evaluations': self.evaluations,
            'iterations': self._current_iteration
        }
        
    def _estimate_problem_characteristics(self):
        """Estimate problem characteristics."""
        if self.population is None or len(self.population) < 10:
            return
            
        # Estimate gradient by sampling random points
        if self._current_iteration % 5 == 0:  # Only check periodically
            gradients = []
            for _ in range(min(10, self.population_size // 2)):
                # Select two random points
                idx1, idx2 = np.random.choice(len(self.population), 2, replace=False)
                p1, p2 = self.population[idx1], self.population[idx2]
                
                # Calculate function values
                f1 = self.objective_func(p1)
                f2 = self.objective_func(p2)
                self.evaluations += 2
                
                # Calculate approximate gradient magnitude
                dist = np.linalg.norm(p1 - p2)
                if dist > 1e-10:  # Avoid division by zero
                    grad_mag = abs(f1 - f2) / dist
                    gradients.append(grad_mag)
            
            if gradients:
                avg_gradient = np.mean(gradients)
                self.gradient_estimates.append(avg_gradient)
                
                # Estimate landscape ruggedness from gradient variation
                if len(self.gradient_estimates) > 1:
                    self.landscape_ruggedness = np.std(self.gradient_estimates) / np.mean(self.gradient_estimates)
                
        # Estimate number of local optima
        if self._current_iteration % 10 == 0 and len(self.population) >= 10:
            # Cluster solutions to estimate number of local optima
            from sklearn.cluster import DBSCAN
            
            # Only use top 50% of solutions
            scores = np.array([self.objective_func(x) for x in self.population])
            self.evaluations += len(self.population)
            
            # Sort by score
            sorted_indices = np.argsort(scores)
            top_indices = sorted_indices[:len(sorted_indices) // 2]
            top_solutions = self.population[top_indices]
            
            # Cluster solutions
            try:
                # Normalize solutions for clustering
                normalized_solutions = (top_solutions - np.min(top_solutions, axis=0)) / (np.max(top_solutions, axis=0) - np.min(top_solutions, axis=0) + 1e-10)
                
                # Use DBSCAN to find clusters
                clustering = DBSCAN(eps=0.1, min_samples=2).fit(normalized_solutions)
                n_clusters = len(set(clustering.labels_)) - (1 if -1 in clustering.labels_ else 0)
                
                # Update local optima count
                self.local_optima_count = max(1, n_clusters)
            except Exception as e:
                self.logger.warning(f"Error estimating local optima: {str(e)}")
