"""
Comprehensive analysis of MetaOptimizer performance.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
import logging
from tqdm.auto import tqdm
import pandas as pd
from datetime import datetime
from typing import Dict, List, Any, Tuple
import warnings
import sys
import os
import time

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.plot_utils import save_plot

# Filter warnings
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

# Import optimizers
from optimizers.de import DifferentialEvolutionOptimizer
from optimizers.es import EvolutionStrategyOptimizer
from optimizers.gwo import GreyWolfOptimizer
from optimizers.aco import AntColonyOptimizer
from optimizers.ml_optimizers.surrogate_optimizer import SurrogateOptimizer
from optimizers.base_optimizer import OptimizerState

# Custom optimizer implementations with required abstract methods
class CustomDEOptimizer(DifferentialEvolutionOptimizer):
    def _iterate(self):
        """Implement the abstract _iterate method from BaseOptimizer"""
        # Increment iteration counter
        self._current_iteration += 1
        
        # Track success in this iteration
        success_in_iteration = False
        
        # Process each individual in the population
        for i in range(self.population_size):
            # Create mutant vector
            mutant = self._mutate(i)
            
            # Create trial vector through crossover
            trial = self._crossover(self.population[i], mutant)
            
            # Evaluate trial vector
            if self.objective_func is None:
                raise ValueError("Objective function not set")
                
            trial_score = self.objective_func(trial)
            self.evaluations += 1
            
            # Selection
            if trial_score <= self.population_scores[i]:
                # Trial is better or equal, replace target
                self.population[i] = trial
                self.population_scores[i] = trial_score
                success_in_iteration = True
                
                # Update best solution if needed
                if trial_score < self.best_score:
                    self.best_score = trial_score
                    self.best_solution = trial.copy()
                    self.convergence_curve.append(self.best_score)
        
        # Update success history
        self.success_history.append(success_in_iteration)
        if len(self.success_history) > 20:  # Keep only last 20
            self.success_history.pop(0)
    
    def optimize(self, objective_func, max_evals=None):
        """Wrapper for the run method to match the interface expected by benchmarking code"""
        self.set_objective(objective_func)
        result = self.run(max_evals=max_evals)
        return self.best_solution

class CustomESOptimizer(EvolutionStrategyOptimizer):
    def _iterate(self):
        """Implement the abstract _iterate method from BaseOptimizer"""
        # Increment iteration counter
        self._current_iteration += 1
        
        # Generate offspring through mutation and recombination
        offspring = self._generate_offspring()
        
        # Evaluate offspring
        offspring_score = self.objective_func(offspring)
        self.evaluations += 1
        
        # Update best solution if better
        if offspring_score < self.best_score:
            self.best_solution = offspring.copy()
            self.best_score = offspring_score
            success_in_iteration = True
        else:
            success_in_iteration = False
            
        # Selection: replace worst individual in population if offspring is better
        self._selection(offspring, offspring_score)
        
        # Update convergence curve
        self.convergence_curve.append(self.best_score)
        
        # Update success history
        self.success_history.append(success_in_iteration)
        if len(self.success_history) > 20:  # Keep only last 20
            self.success_history.pop(0)
    
    def _selection(self, offspring, offspring_score):
        """Replace worst individual in population if offspring is better"""
        worst_idx = np.argmax(self.population_scores)
        if offspring_score < self.population_scores[worst_idx]:
            self.population[worst_idx] = offspring.copy()
            self.population_scores[worst_idx] = offspring_score
    
    def optimize(self, objective_func, max_evals=None):
        """Wrapper for the run method to match the interface expected by benchmarking code"""
        self.set_objective(objective_func)
        result = self.run(max_evals=max_evals)
        return self.best_solution

class CustomGWOOptimizer(GreyWolfOptimizer):
    def _iterate(self):
        """Implement the abstract _iterate method from BaseOptimizer"""
        # Increment iteration counter
        self._current_iteration += 1
        
        # Set max_iterations if not already set
        if not hasattr(self, 'max_iterations'):
            self.max_iterations = max(1, self.max_evals // max(1, self.population_size))
        
        # Update alpha, beta, and delta positions
        self._update_wolf_positions()
        
        # Update all wolf positions
        for i in range(self.population_size):
            # Update position
            new_position = self._update_position(i)
            
            # Evaluate new position
            new_score = self.objective_func(new_position)
            self.evaluations += 1
            
            # Update wolf if better
            if new_score < self.population_scores[i]:
                self.population[i] = new_position
                self.population_scores[i] = new_score
                
                # Update best solution if needed
                if new_score < self.best_score:
                    self.best_score = new_score
                    self.best_solution = new_position.copy()
                    self.convergence_curve.append(self.best_score)
    
    def _update_wolf_positions(self):
        """Update alpha, beta, and delta wolf positions based on fitness"""
        # Sort population by fitness
        sorted_indices = np.argsort(self.population_scores)
        
        # Initialize wolf hierarchy if not already done
        if not hasattr(self, 'alpha') or self.alpha is None:
            self.alpha = self.population[sorted_indices[0]].copy()
            self.alpha_score = self.population_scores[sorted_indices[0]]
            self.beta = self.population[sorted_indices[1]].copy()
            self.beta_score = self.population_scores[sorted_indices[1]]
            self.delta = self.population[sorted_indices[2]].copy()
            self.delta_score = self.population_scores[sorted_indices[2]]
        
        # Update alpha (best solution)
        if self.population_scores[sorted_indices[0]] < self.alpha_score:
            self.alpha = self.population[sorted_indices[0]].copy()
            self.alpha_score = self.population_scores[sorted_indices[0]]
            
        # Update beta (second best solution)
        if self.population_scores[sorted_indices[1]] < self.beta_score:
            self.beta = self.population[sorted_indices[1]].copy()
            self.beta_score = self.population_scores[sorted_indices[1]]
            
        # Update delta (third best solution)
        if self.population_scores[sorted_indices[2]] < self.delta_score:
            self.delta = self.population[sorted_indices[2]].copy()
            self.delta_score = self.population_scores[sorted_indices[2]]
    
    def _update_position(self, wolf_idx):
        """Update position of a wolf based on alpha, beta, and delta positions"""
        # Calculate a parameter (decreases linearly from 2 to 0)
        a = 2 - self._current_iteration * (2 / self.max_iterations)
        
        # Generate random coefficients
        r1 = np.random.rand(self.dim)
        r2 = np.random.rand(self.dim)
        A1 = 2 * a * r1 - a
        C1 = 2 * r2
        
        r1 = np.random.rand(self.dim)
        r2 = np.random.rand(self.dim)
        A2 = 2 * a * r1 - a
        C2 = 2 * r2
        
        r1 = np.random.rand(self.dim)
        r2 = np.random.rand(self.dim)
        A3 = 2 * a * r1 - a
        C3 = 2 * r2
        
        # Calculate distance to leaders
        D_alpha = np.abs(C1 * self.alpha - self.population[wolf_idx])
        D_beta = np.abs(C2 * self.beta - self.population[wolf_idx])
        D_delta = np.abs(C3 * self.delta - self.population[wolf_idx])
        
        # Calculate position updates
        X1 = self.alpha - A1 * D_alpha
        X2 = self.beta - A2 * D_beta
        X3 = self.delta - A3 * D_delta
        
        # Average position updates
        new_position = (X1 + X2 + X3) / 3
        
        # Bound solution
        return self._bound_solution(new_position)
    
    def optimize(self, objective_func, max_evals=None):
        """Wrapper for the run method to match the interface expected by benchmarking code"""
        self.set_objective(objective_func)
        result = self.run(max_evals=max_evals)
        return self.best_solution

class CustomACOOptimizer(AntColonyOptimizer):
    def __init__(self, dim, bounds, population_size=None, **kwargs):
        # Convert population_size to pop_size
        pop_size = population_size if population_size is not None else None
        super().__init__(dim, bounds, pop_size=pop_size, **kwargs)
        self.state = OptimizerState()
        self.state.solutions = []
        self.state.evaluations = []
        
        # Initialize pheromones
        n_steps = 100  # Number of discrete steps for each dimension
        self.pheromones = []
        for d in range(self.dim):
            # Initialize with equal pheromone levels
            self.pheromones.append(np.ones(n_steps) / n_steps)
            
        # Track evaluated solutions to avoid duplicates
        self.evaluated_solutions = set()
        self.max_attempts = 20  # Increased maximum attempts to generate a unique solution
        self.precision = 1e-6  # Precision for rounding to avoid floating point comparison issues
        
        # Adaptive exploration parameters
        self.exploration_rate = 0.2  # Initial exploration rate
        self.min_exploration_rate = 0.05  # Minimum exploration rate
        self.exploration_decay = 0.95  # Decay rate for exploration
    
    def _iterate(self):
        """Implement the abstract _iterate method from BaseOptimizer"""
        # Increment iteration counter
        self._current_iteration += 1
        
        # Generate solutions for all ants
        new_solutions = []
        new_scores = []
        
        # Update exploration rate
        self.exploration_rate = max(self.min_exploration_rate, 
                                   self.exploration_rate * self.exploration_decay)
        
        # Track unique solutions in this iteration
        unique_solutions_found = 0
        
        for i in range(self.population_size):
            # Generate new solution (with duplicate prevention)
            new_solution = self._generate_unique_solution()
            
            if new_solution is not None:
                unique_solutions_found += 1
                
                # Evaluate solution
                new_score = self.objective_func(new_solution)
                self.evaluations += 1
                
                # Add to evaluated solutions with proper rounding
                solution_tuple = tuple(round(float(v), int(-np.log10(self.precision))) for v in new_solution)
                self.evaluated_solutions.add(solution_tuple)
                
                new_solutions.append(new_solution)
                new_scores.append(new_score)
                
                # Update best solution if needed
                if new_score < self.best_score:
                    self.best_score = new_score
                    self.best_solution = new_solution.copy()
                    self.convergence_curve.append(self.best_score)
        
        # If we found at least one unique solution, update pheromones
        if len(new_solutions) > 0:
            # Update pheromone trails
            self._update_pheromones(new_solutions, new_scores)
            
            # Update population
            self.population = np.array(new_solutions)
            self.population_scores = np.array(new_scores)
        else:
            # If no unique solutions were found, increase exploration
            self.exploration_rate = min(0.8, self.exploration_rate * 2)
            logging.warning(f"ACO: No unique solutions found in iteration {self._current_iteration}, increasing exploration to {self.exploration_rate:.2f}")
    
    def _generate_unique_solution(self):
        """Generate a unique solution that hasn't been evaluated before"""
        for attempt in range(self.max_attempts):
            # Increase exploration as attempts increase
            current_exploration = self.exploration_rate * (1 + attempt / self.max_attempts)
            solution = self._generate_solution(exploration_rate=current_exploration)
            
            # Round to specified precision for comparison
            solution_tuple = tuple(round(float(v), int(-np.log10(self.precision))) for v in solution)
            
            if solution_tuple not in self.evaluated_solutions:
                return solution
                
            # If we're having trouble finding unique solutions, add more randomness
            if attempt > self.max_attempts // 2:
                # Add more noise for exploration
                for d in range(self.dim):
                    lower, upper = self.bounds[d]
                    noise_scale = (upper - lower) * 0.2 * (attempt / self.max_attempts)
                    solution[d] += np.random.normal(0, noise_scale)
                    solution[d] = max(lower, min(upper, solution[d]))
                
                # Check again with the modified solution
                solution_tuple = tuple(round(float(v), int(-np.log10(self.precision))) for v in solution)
                if solution_tuple not in self.evaluated_solutions:
                    return solution
        
        # If we still can't find a unique solution after max attempts
        logging.warning("ACO: Could not generate a unique solution after max attempts")
        
        # Try a completely random solution as a last resort
        for _ in range(5):  # Try a few random solutions
            random_solution = np.array([np.random.uniform(low, high) for low, high in self.bounds])
            random_tuple = tuple(round(float(v), int(-np.log10(self.precision))) for v in random_solution)
            
            if random_tuple not in self.evaluated_solutions:
                return random_solution
        
        # If even random solutions are duplicates, return None to signal failure
        return None
    
    def _generate_solution(self, exploration_rate=None):
        """Generate a solution based on pheromone trails with exploration rate"""
        solution = np.zeros(self.dim)
        
        # Use instance exploration rate if none provided
        if exploration_rate is None:
            exploration_rate = self.exploration_rate
            
        # Discretize the search space
        n_steps = 100  # Number of discrete steps for each dimension
        
        for d in range(self.dim):
            lower, upper = self.bounds[d]
            step_size = (upper - lower) / n_steps
            
            # Decide whether to explore or exploit
            if np.random.random() < exploration_rate:
                # Exploration: choose a random step
                selected_step = np.random.randint(0, n_steps)
            else:
                # Exploitation: use pheromone levels
                step_probs = self.pheromones[d] / np.sum(self.pheromones[d])
                selected_step = np.random.choice(n_steps, p=step_probs)
            
            # Convert step to continuous value
            solution[d] = lower + selected_step * step_size
            
            # Add some noise for exploration (scaled by exploration rate)
            noise = np.random.normal(0, step_size * exploration_rate)
            solution[d] += noise
            
            # Ensure bounds are respected
            solution[d] = max(lower, min(upper, solution[d]))
        
        return solution
    
    def _update_pheromones(self, solutions, scores):
        """Update pheromone trails based on solution quality"""
        n_steps = len(self.pheromones[0])
        
        # Apply evaporation
        evaporation_rate = 0.1
        for d in range(self.dim):
            self.pheromones[d] *= (1 - evaporation_rate)
        
        # Add new pheromones based on solution quality
        for solution, score in zip(solutions, scores):
            # Invert score for maximization (higher pheromone for better solutions)
            quality = 1.0 / (score + 1e-10)
            
            for d in range(self.dim):
                lower, upper = self.bounds[d]
                step_size = (upper - lower) / n_steps
                
                # Determine which step this solution falls into
                step_idx = min(n_steps - 1, int((solution[d] - lower) / step_size))
                
                # Update pheromone level
                self.pheromones[d][step_idx] += quality
    
    def reset(self):
        """Reset the optimizer state"""
        super().reset()
        
        # Reset pheromones
        n_steps = 100  # Number of discrete steps for each dimension
        self.pheromones = []
        for d in range(self.dim):
            # Initialize with equal pheromone levels
            self.pheromones.append(np.ones(n_steps) / n_steps)
        
        # Reset exploration rate
        self.exploration_rate = 0.2
        
        # Clear evaluated solutions
        self.evaluated_solutions = set()
    
    def optimize(self, objective_func, max_evals=None):
        """Wrapper for the run method to match the interface expected by benchmarking code"""
        self.set_objective(objective_func)
        result = self.run(max_evals=max_evals)
        return self.best_solution

class CustomSurrogateOptimizer(SurrogateOptimizer):
    def __init__(self, dim, bounds, population_size=None, **kwargs):
        # Convert population_size to pop_size
        pop_size = population_size if population_size is not None else None
        super().__init__(dim, bounds, pop_size=pop_size, **kwargs)
        self.state = OptimizerState()
        self.state.solutions = []
        self.state.evaluations = []
        self.n_direct_eval = 5  # Number of points to evaluate directly
        
        # Track evaluated solutions to avoid duplicates
        self.evaluated_solutions = set()
        self.max_attempts = 20  # Increased maximum attempts to generate a unique solution
        self.precision = 1e-6  # Precision for rounding to avoid floating point comparison issues
        
        # Adaptive exploration parameters
        self.exploration_rate = 0.2  # Initial exploration rate
        self.min_exploration_rate = 0.05  # Minimum exploration rate
        self.exploration_decay = 0.95  # Decay rate for exploration
        
        # Initialize surrogate model if not already done
        if not hasattr(self, 'model') or self.model is None:
            from sklearn.ensemble import RandomForestRegressor
            self.model = RandomForestRegressor(n_estimators=50, max_depth=10)
    
    def _iterate(self):
        """Implement the abstract _iterate method from BaseOptimizer"""
        # Increment iteration counter
        self._current_iteration += 1
        
        # Update exploration rate
        self.exploration_rate = max(self.min_exploration_rate, 
                                   self.exploration_rate * self.exploration_decay)
        
        # Update surrogate model
        self._update_model()
        
        # Generate and evaluate candidates
        candidates = self._generate_unique_candidates()
        
        # Evaluate best candidate if we have any
        if len(candidates) > 0:
            best_candidate = candidates[0]
            score = self.objective_func(best_candidate)
            self.evaluations += 1
            
            # Add to evaluated solutions with proper rounding
            solution_tuple = tuple(round(float(v), int(-np.log10(self.precision))) for v in best_candidate)
            self.evaluated_solutions.add(solution_tuple)
            
            # Store solution and evaluation for surrogate model training
            self.state.solutions.append(best_candidate)
            self.state.evaluations.append(score)
            
            # Update best solution if needed
            if score < self.best_score:
                self.best_score = score
                self.best_solution = best_candidate.copy()
                self.convergence_curve.append(self.best_score)
        else:
            # If no unique candidates were found, increase exploration
            self.exploration_rate = min(0.8, self.exploration_rate * 2)
            logging.warning(f"Surrogate: No unique candidates found in iteration {self._current_iteration}, increasing exploration to {self.exploration_rate:.2f}")
            
            # Try a completely random solution as a last resort
            random_solution = self._generate_unique_random_solution(force_random=True)
            if random_solution is not None:
                score = self.objective_func(random_solution)
                self.evaluations += 1
                
                # Add to evaluated solutions
                solution_tuple = tuple(round(float(v), int(-np.log10(self.precision))) for v in random_solution)
                self.evaluated_solutions.add(solution_tuple)
                
                # Store solution and evaluation for surrogate model training
                self.state.solutions.append(random_solution)
                self.state.evaluations.append(score)
                
                # Update best solution if needed
                if score < self.best_score:
                    self.best_score = score
                    self.best_solution = random_solution.copy()
                    self.convergence_curve.append(self.best_score)
    
    def _update_model(self):
        """Update the surrogate model with current data"""
        if hasattr(self, 'model') and self.model is not None:
            X_train = np.array(self.state.solutions)
            y_train = np.array(self.state.evaluations)
            
            if len(X_train) > 0 and len(y_train) > 0:
                try:
                    self.model.fit(X_train, y_train)
                except Exception as e:
                    # Handle potential errors during model fitting
                    logging.warning(f"Error updating surrogate model: {e}")
    
    def _generate_unique_candidates(self, n_candidates=10):
        """Generate unique candidate solutions using the surrogate model"""
        candidates = []
        
        # If we don't have enough data or model isn't ready, use random sampling
        if not hasattr(self, 'model') or self.model is None or len(self.state.solutions) < self.n_direct_eval:
            for _ in range(n_candidates):
                candidate = self._generate_unique_random_solution()
                if candidate is not None:
                    candidates.append(candidate)
            return candidates
        
        # Generate random candidates
        random_candidates = []
        max_candidates = n_candidates * 30  # Generate more candidates than needed to ensure uniqueness
        
        # First, try to generate unique random solutions
        for _ in range(max_candidates):
            candidate = self._generate_unique_random_solution()
            if candidate is not None:
                random_candidates.append(candidate)
                
                # Stop if we have enough unique candidates
                if len(random_candidates) >= n_candidates * 10:
                    break
        
        # If we couldn't generate enough random candidates, return what we have
        if len(random_candidates) == 0:
            return []
        
        # Predict scores using the surrogate model
        try:
            predicted_scores = self.model.predict(np.array(random_candidates))
            
            # Sort candidates by predicted score
            sorted_indices = np.argsort(predicted_scores)
            
            # Select top candidates, ensuring uniqueness
            for idx in sorted_indices:
                candidate = random_candidates[idx]
                candidate_tuple = tuple(round(float(v), int(-np.log10(self.precision))) for v in candidate)
                
                if candidate_tuple not in self.evaluated_solutions:
                    candidates.append(candidate)
                
                if len(candidates) >= n_candidates:
                    break
                    
            # If we couldn't find enough candidates, add some exploration candidates
            if len(candidates) < n_candidates // 2:
                # Add some random candidates for exploration
                exploration_indices = np.random.choice(
                    len(random_candidates), 
                    min(n_candidates - len(candidates), len(random_candidates)),
                    replace=False
                )
                for idx in exploration_indices:
                    candidate = random_candidates[idx]
                    candidate_tuple = tuple(round(float(v), int(-np.log10(self.precision))) for v in candidate)
                    
                    if candidate_tuple not in self.evaluated_solutions and candidate not in candidates:
                        candidates.append(candidate)
                        
        except Exception as e:
            # Fallback to random candidates if prediction fails
            logging.warning(f"Error predicting with surrogate model: {e}")
            candidates = random_candidates[:min(n_candidates, len(random_candidates))]
        
        return candidates
    
    def _generate_unique_random_solution(self, force_random=False):
        """Generate a unique random solution"""
        for attempt in range(self.max_attempts):
            # Decide whether to use Latin Hypercube Sampling or pure random
            if force_random or np.random.random() < self.exploration_rate:
                # Pure random sampling
                candidate = np.array([np.random.uniform(low, high) for low, high in self.bounds])
            else:
                # Latin Hypercube Sampling for better space coverage
                # Divide each dimension into segments
                segments = 10
                segment_indices = np.random.permutation(segments)
                
                candidate = np.zeros(self.dim)
                for d in range(self.dim):
                    lower, upper = self.bounds[d]
                    segment_width = (upper - lower) / segments
                    
                    # Select a random point within the chosen segment
                    segment_idx = segment_indices[d % segments]
                    segment_start = lower + segment_idx * segment_width
                    candidate[d] = segment_start + np.random.uniform(0, segment_width)
            
            # Round to specified precision for comparison
            candidate_tuple = tuple(round(float(v), int(-np.log10(self.precision))) for v in candidate)
            
            if candidate_tuple not in self.evaluated_solutions:
                return candidate
                
            # If we're having trouble finding unique solutions, add more randomness
            if attempt > self.max_attempts // 2:
                # Increase the range slightly to find more unique solutions
                candidate = np.array([np.random.uniform(low - (high-low)*0.1, high + (high-low)*0.1) 
                                     for low, high in self.bounds])
                # Ensure bounds are respected
                for d in range(self.dim):
                    lower, upper = self.bounds[d]
                    candidate[d] = max(lower, min(upper, candidate[d]))
                    
                # Check again with the modified solution
                candidate_tuple = tuple(round(float(v), int(-np.log10(self.precision))) for v in candidate)
                if candidate_tuple not in self.evaluated_solutions:
                    return candidate
        
        # If we still can't find a unique solution after max attempts
        if not force_random:
            logging.warning("Surrogate: Could not generate a unique solution after max attempts")
        
        # As a last resort, try a completely random solution with high variance
        for _ in range(5):  # Try a few random solutions
            # Generate a solution with high variance
            candidate = np.array([np.random.uniform(low - (high-low)*0.2, high + (high-low)*0.2) 
                                 for low, high in self.bounds])
            
            # Ensure bounds are respected
            for d in range(self.dim):
                lower, upper = self.bounds[d]
                candidate[d] = max(lower, min(upper, candidate[d]))
                
            # Check if it's unique
            candidate_tuple = tuple(round(float(v), int(-np.log10(self.precision))) for v in candidate)
            if candidate_tuple not in self.evaluated_solutions:
                return candidate
        
        # If even random solutions are duplicates, return None
        return None
    
    def reset(self):
        """Reset the optimizer state"""
        super().reset()
        
        # Reset state
        self.state = OptimizerState()
        self.state.solutions = []
        self.state.evaluations = []
        
        # Reset exploration rate
        self.exploration_rate = 0.2
        
        # Clear evaluated solutions
        self.evaluated_solutions = set()
        
        # Reinitialize model
        from sklearn.ensemble import RandomForestRegressor
        self.model = RandomForestRegressor(n_estimators=50, max_depth=10)
    
    def optimize(self, objective_func, max_evals=None):
        """Wrapper for the run method to match the interface expected by benchmarking code"""
        self.set_objective(objective_func)
        result = self.run(max_evals=max_evals)
        return self.best_solution

# Meta-learning
from meta.meta_optimizer import MetaOptimizer

# Analysis
from analysis.theoretical_analysis import ConvergenceAnalyzer, StabilityAnalyzer
from benchmarking.test_functions import create_test_suite
from benchmarking.statistical_analysis import run_statistical_tests
from visualization.optimizer_analysis import OptimizerAnalyzer

def setup_output_dirs() -> Dict[str, Path]:
    """Create output directories."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_dir = Path('results') / f'comprehensive_analysis_{timestamp}'
    
    dirs = {
        'main': base_dir,
        'plots': base_dir / 'plots',
        'data': base_dir / 'data',
        'analysis': base_dir / 'analysis'
    }
    
    for dir_path in dirs.values():
        dir_path.mkdir(parents=True, exist_ok=True)
        
    return dirs

def setup_logging(output_dir: Path):
    """Configure logging."""
    log_file = output_dir / 'analysis.log'
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )

def create_quick_test_suite():
    """Create a simplified test suite for quick testing."""
    return {
        'unimodal': {
            'sphere': {
                'func': lambda x: np.sum(x**2),  # Simple sphere function
                'dim': 2,
                'bounds': [(-5.12, 5.12)],
                'optimal': 0.0,
                'multimodal': 0,
                'discrete_vars': 0
            }
        },
        'multimodal': {
            'rastrigin': {
                'func': lambda x: 10*len(x) + np.sum(x**2 - 10*np.cos(2*np.pi*x)),
                'dim': 2,
                'bounds': [(-5.12, 5.12)],
                'optimal': 0.0,
                'multimodal': 1,
                'discrete_vars': 0
            }
        }
    }

class EvaluationTracker:
    """
    Wrapper for objective functions to track evaluations and detect duplicates.
    This class helps to identify when optimizers are evaluating the same points multiple times.
    """
    def __init__(self, func, name="Unknown", precision=1e-6):
        self.func = func
        self.name = name
        self.precision = precision  # Precision for rounding to avoid floating point comparison issues
        self.evaluated_points = {}  # Dictionary to store evaluated points and their results
        self.total_count = 0  # Total number of evaluations
        self.duplicate_count = 0  # Number of duplicate evaluations
        self.evaluation_history = []  # History of all evaluations for analysis
        self.duplicate_history = []  # History of duplicate evaluations
        self.time_spent = 0  # Total time spent on evaluations
        
    def __call__(self, x):
        """
        Call the wrapped function and track evaluations.
        
        Args:
            x: Input to the objective function
            
        Returns:
            Result of the objective function
        """
        start_time = time.time()
        
        # Convert to tuple for hashability, with proper rounding
        x_tuple = tuple(round(float(v), int(-np.log10(self.precision))) for v in x)
        
        self.total_count += 1
        self.evaluation_history.append(x_tuple)
        
        if x_tuple in self.evaluated_points:
            self.duplicate_count += 1
            self.duplicate_history.append(x_tuple)
            logging.warning(f"Duplicate evaluation detected in {self.name}! Point: {x_tuple}")
            result = self.evaluated_points[x_tuple]
        else:
            result = self.func(x)
            self.evaluated_points[x_tuple] = result
        
        end_time = time.time()
        self.time_spent += (end_time - start_time)
        
        return result
    
    def get_stats(self):
        """
        Get statistics about evaluations.
        
        Returns:
            Dictionary with evaluation statistics
        """
        unique_count = len(self.evaluated_points)
        duplicate_percentage = (self.duplicate_count / self.total_count * 100) if self.total_count > 0 else 0
        
        return {
            "name": self.name,
            "total_evaluations": self.total_count,
            "unique_evaluations": unique_count,
            "duplicate_evaluations": self.duplicate_count,
            "duplicate_percentage": duplicate_percentage,
            "time_spent": self.time_spent,
            "avg_time_per_eval": self.time_spent / self.total_count if self.total_count > 0 else 0
        }
    
    def get_detailed_stats(self):
        """
        Get detailed statistics about evaluations, including patterns of duplicates.
        
        Returns:
            Dictionary with detailed evaluation statistics
        """
        stats = self.get_stats()
        
        # Analyze patterns in duplicates
        if len(self.duplicate_history) > 0:
            # Count frequency of each duplicate point
            duplicate_frequencies = {}
            for point in self.duplicate_history:
                if point in duplicate_frequencies:
                    duplicate_frequencies[point] += 1
                else:
                    duplicate_frequencies[point] = 1
            
            # Find the most frequently duplicated points
            most_frequent = sorted(duplicate_frequencies.items(), key=lambda x: x[1], reverse=True)[:5]
            
            stats["most_frequent_duplicates"] = most_frequent
            stats["unique_duplicates"] = len(duplicate_frequencies)
            
            # Calculate average number of times a point is duplicated
            if len(duplicate_frequencies) > 0:
                stats["avg_duplications_per_point"] = sum(duplicate_frequencies.values()) / len(duplicate_frequencies)
            else:
                stats["avg_duplications_per_point"] = 0
        
        return stats
    
    def reset(self):
        """Reset the tracker"""
        self.evaluated_points = {}
        self.total_count = 0
        self.duplicate_count = 0
        self.evaluation_history = []
        self.duplicate_history = []
        self.time_spent = 0

def run_quick_benchmarks(test_func: callable, dim: int, bounds: List[Tuple[float, float]], 
                   n_trials: int = 2, max_evals: int = 100) -> Dict[str, Any]:
    """
    Run benchmarks for all optimizers with duplicate evaluation tracking.
    
    Args:
        test_func: Test function to optimize
        dim: Dimensionality of the problem
        bounds: Bounds for each dimension
        n_trials: Number of trials to run
        max_evals: Maximum number of evaluations per trial
        
    Returns:
        Dictionary with benchmark results
    """
    # Create full bounds list
    full_bounds = [(bounds[0][0], bounds[0][1]) for _ in range(dim)]
    
    # Create base optimizers
    base_optimizers = {
        "de": CustomDEOptimizer(dim=dim, bounds=full_bounds, population_size=10),
        "es": CustomESOptimizer(dim=dim, bounds=full_bounds, population_size=10),
        "gwo": CustomGWOOptimizer(dim=dim, bounds=full_bounds, population_size=10),
        "aco": CustomACOOptimizer(dim=dim, bounds=full_bounds, population_size=10),
        "surrogate": CustomSurrogateOptimizer(dim=dim, bounds=full_bounds, population_size=10)
    }
    
    # Results dictionary
    results = {
        "optimizer_results": {},
        "best_scores": {},
        "convergence_curves": {},
        "evaluation_stats": {},
        "detailed_stats": {}
    }
    
    # Run benchmarks for each optimizer
    for name, optimizer in tqdm(base_optimizers.items(), desc="Optimizers"):
        best_scores = []
        convergence_curves = []
        histories = []
        all_evaluation_stats = []
        
        # Progress bar for trials
        trial_pbar = tqdm(range(n_trials), desc=f"Trial", leave=False)
        
        for trial in trial_pbar:
            # Reset optimizer
            optimizer.reset()
            
            # Create a new tracker for each trial
            tracker = EvaluationTracker(test_func, name=f"{name.upper()}-Trial{trial+1}")
            
            # Set objective function
            optimizer.set_objective(tracker)
            
            # Run optimization
            try:
                optimizer.run(max_evals=max_evals)
                
                # Store results
                best_scores.append(optimizer.best_score)
                convergence_curves.append(optimizer.convergence_curve)
                
                # Get evaluation statistics
                eval_stats = tracker.get_stats()
                all_evaluation_stats.append(eval_stats)
                
                # Get detailed statistics for the last trial
                if trial == n_trials - 1:
                    detailed_stats = tracker.get_detailed_stats()
                    results["detailed_stats"][name] = detailed_stats
                
                # Update progress bar
                trial_pbar.set_postfix({
                    "best": f"{optimizer.best_score:.4f}", 
                    "duplicates": f"{eval_stats['duplicate_percentage']:.1f}%"
                })
                
            except Exception as e:
                logging.error(f"Error running {name}: {e}")
                best_scores.append(float('inf'))
                convergence_curves.append([float('inf')] * max_evals)
        
        # Store results
        results["optimizer_results"][name] = {
            "best_solution": optimizer.best_solution,
            "best_score": np.min(best_scores),
            "mean_score": np.mean(best_scores),
            "std_score": np.std(best_scores)
        }
        results["best_scores"][name] = best_scores
        results["convergence_curves"][name] = convergence_curves
        
        # Aggregate evaluation statistics
        avg_stats = {
            "total_evaluations": np.mean([stats["total_evaluations"] for stats in all_evaluation_stats]),
            "unique_evaluations": np.mean([stats["unique_evaluations"] for stats in all_evaluation_stats]),
            "duplicate_evaluations": np.mean([stats["duplicate_evaluations"] for stats in all_evaluation_stats]),
            "duplicate_percentage": np.mean([stats["duplicate_percentage"] for stats in all_evaluation_stats]),
            "time_spent": np.mean([stats["time_spent"] for stats in all_evaluation_stats]),
            "avg_time_per_eval": np.mean([stats["avg_time_per_eval"] for stats in all_evaluation_stats])
        }
        results["evaluation_stats"][name] = avg_stats
        
        # Print summary for this optimizer
        print(f"\n{name.upper()} Summary:")
        print(f"  Best Score: {np.min(best_scores):.6f}")
        print(f"  Mean Score: {np.mean(best_scores):.6f} ± {np.std(best_scores):.6f}")
        print(f"  Duplicate Evaluations: {avg_stats['duplicate_percentage']:.2f}%")
        print(f"  Unique Evaluations: {avg_stats['unique_evaluations']:.1f} / {avg_stats['total_evaluations']:.1f}")
    
    # Print overall summary
    print("\nOverall Summary:")
    for name in base_optimizers.keys():
        stats = results["evaluation_stats"][name]
        print(f"{name.upper()}: {results['optimizer_results'][name]['best_score']:.6f} (Duplicates: {stats['duplicate_percentage']:.2f}%)")
    
    # Rank optimizers by performance
    ranked_optimizers = sorted(
        base_optimizers.keys(),
        key=lambda x: results["optimizer_results"][x]["best_score"]
    )
    
    print("\nRanked by Performance:")
    for i, name in enumerate(ranked_optimizers):
        print(f"{i+1}. {name.upper()}: {results['optimizer_results'][name]['best_score']:.6f}")
    
    # Rank optimizers by duplicate percentage (lower is better)
    ranked_by_duplicates = sorted(
        base_optimizers.keys(),
        key=lambda x: results["evaluation_stats"][x]["duplicate_percentage"]
    )
    
    print("\nRanked by Duplicate Percentage (lower is better):")
    for i, name in enumerate(ranked_by_duplicates):
        stats = results["evaluation_stats"][name]
        print(f"{i+1}. {name.upper()}: {stats['duplicate_percentage']:.2f}%")
    
    return results

def plot_convergence(results: Dict[str, Any], output_dir: Path, title: str):
    """Plot convergence curves with confidence intervals."""
    plt.figure(figsize=(12, 8))
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(results)))
    for (name, data), color in zip(results.items(), colors):
        if 'histories' in data and data['histories']:
            # Extract evaluation counts and scores
            eval_counts = []
            scores = []
            for history in data['histories']:
                counts, hist_scores = zip(*history)
                eval_counts.append(counts)
                scores.append(hist_scores)
            
            # Convert to numpy arrays for calculations
            eval_counts = np.array(eval_counts)
            scores = np.array(scores)
            
            # Calculate mean and std across trials
            mean_scores = np.mean(scores, axis=0)
            std_scores = np.std(scores, axis=0)
            
            # Plot mean line with confidence interval
            plt.plot(eval_counts[0], mean_scores, label=name.upper(), color=color, linewidth=2)
            plt.fill_between(eval_counts[0], 
                           mean_scores - std_scores,
                           mean_scores + std_scores,
                           alpha=0.2, color=color)
    
    plt.yscale('log')
    plt.xlabel('Function Evaluations')
    plt.ylabel('Best Score (log scale)')
    plt.title(f'Convergence Analysis - {title}')
    plt.legend()
    plt.grid(True, which="both", ls="-", alpha=0.2)
    plt.tight_layout()
    
    # Save plot using save_plot
    save_plot(plt.gcf(), f'convergence_{title.lower()}.png', plot_type='benchmarks')
    plt.close()

def plot_performance_distribution(results: Dict[str, Any], output_dir: Path, title: str):
    """Plot performance distribution across trials."""
    plt.figure(figsize=(12, 6))
    
    data = []
    labels = []
    for name, res in results.items():
        data.append(res['scores'])
        labels.extend([name.upper()] * len(res['scores']))
    
    plt.boxplot(data, tick_labels=[name.upper() for name in results.keys()])
    plt.yscale('log')
    plt.ylabel('Best Score (log scale)')
    plt.title(f'Performance Distribution - {title}')
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.tight_layout()
    
    # Save plot using save_plot
    save_plot(plt.gcf(), f'distribution_{title.lower()}.png', plot_type='benchmarks')
    plt.close()

def plot_feature_importance(meta_opt: MetaOptimizer, output_dir: Path):
    """Plot feature importance for optimizer selection."""
    if not hasattr(meta_opt, 'selection_tracker'):
        return
        
    # Get feature correlations from selection tracker
    correlations = meta_opt.selection_tracker.get_feature_correlations()
    if not correlations:
        return
        
    # Prepare data for plotting
    features = []
    importances = []
    
    for opt_name, feat_corrs in correlations.items():
        for feat, corr in feat_corrs.items():
            if feat not in features:
                features.append(feat)
                importances.append(abs(corr))
            else:
                idx = features.index(feat)
                importances[idx] = max(importances[idx], abs(corr))
    
    if not features:
        return
        
    plt.figure(figsize=(10, 6))
    y_pos = np.arange(len(features))
    plt.barh(y_pos, importances)
    plt.yticks(y_pos, features)
    plt.xlabel('Maximum Absolute Correlation')
    plt.title('Feature Importance in Optimizer Selection')
    plt.tight_layout()
    
    # Save plot using save_plot
    save_plot(plt.gcf(), 'feature_importance.png', plot_type='explainability')
    plt.close()

def plot_optimizer_selection(meta_opt: MetaOptimizer, output_dir: Path):
    """Plot optimizer selection patterns."""
    if not hasattr(meta_opt, 'selection_tracker'):
        return
        
    # Get selection statistics
    stats = meta_opt.selection_tracker.get_selection_stats()
    if stats is None:
        return
        
    plt.figure(figsize=(10, 6))
    if isinstance(stats, pd.Series):
        stats.plot(kind='bar')
    else:
        # Handle numpy array or dict
        if isinstance(stats, dict):
            names = list(stats.keys())
            values = list(stats.values())
        else:
            names = [f'Optimizer {i}' for i in range(len(stats))]
            values = stats
            
        plt.bar(range(len(values)), values)
        plt.xticks(range(len(names)), names, rotation=45)
    
    plt.title('Optimizer Selection Frequency')
    plt.xlabel('Optimizer')
    plt.ylabel('Selection Count')
    plt.tight_layout()
    
    # Save plot using save_plot
    save_plot(plt.gcf(), 'optimizer_selection.png', plot_type='performance')
    plt.close()

def analyze_meta_optimizer_behavior(meta_opt: MetaOptimizer, output_dir: Path):
    """Analyze and visualize meta-optimizer behavior."""
    if not hasattr(meta_opt, 'history'):
        return
        
    # Plot exploration vs exploitation
    if hasattr(meta_opt, 'exploration_rates'):
        plt.figure(figsize=(10, 6))
        plt.plot(meta_opt.exploration_rates)
        plt.title('Exploration Rate Over Time')
        plt.xlabel('Iteration')
        plt.ylabel('Exploration Rate')
        plt.grid(True)
        
        # Save plot using save_plot
        save_plot(plt.gcf(), 'exploration_rate.png', plot_type='performance')
        plt.close()
    
    # Plot confidence scores
    if hasattr(meta_opt, 'confidence_scores'):
        plt.figure(figsize=(10, 6))
        for opt, scores in meta_opt.confidence_scores.items():
            plt.plot(scores, label=opt)
        plt.title('Optimizer Confidence Scores')
        plt.xlabel('Iteration')
        plt.ylabel('Confidence Score')
        plt.legend()
        plt.grid(True)
        
        # Save plot using save_plot
        save_plot(plt.gcf(), 'confidence_scores.png', plot_type='performance')
        plt.close()

def quick_test():
    """Run a quick test with reduced parameters."""
    # Setup
    dirs = setup_output_dirs()
    setup_logging(dirs['main'])
    logging.info("Starting quick test analysis...")
    
    # Get test suite
    suite = create_quick_test_suite()
    
    # Run analysis for each function type
    for suite_name, functions in suite.items():
        logging.info(f"\nAnalyzing {suite_name} functions")
        
        for func_name, func_info in functions.items():
            logging.info(f"Testing {func_name}")
            
            # Run benchmarks with reduced parameters
            results = run_quick_benchmarks(
                test_func=func_info['func'],
                dim=func_info['dim'],
                bounds=func_info['bounds'],
                n_trials=2,  # Reduced trials
                max_evals=100  # Reduced evaluations
            )
            
            # Generate visualizations
            plot_convergence(results, dirs['plots'], func_name)
            plot_performance_distribution(results, dirs['plots'], func_name)
            
            # Save results
            with open(dirs['data'] / f'{func_name}_results.json', 'w') as f:
                json.dump({k: {
                    'mean': float(v['mean']),
                    'std': float(v['std']),
                    'best': float(v['best']),
                    'worst': float(v['worst']),
                    'evaluation_stats': v['evaluation_stats'] if 'evaluation_stats' in v else {}
                } for k, v in results.items()}, f, indent=2)
            
            # Log summary
            logging.info("\nResults Summary:")
            for name, res in results.items():
                logging.info(f"{name:15} Mean: {res['mean']:.2e} ± {res['std']:.2e}")
                if 'evaluation_stats' in res:
                    stats = res['evaluation_stats']
                    logging.info(f"{name:15} Duplicates: {stats['duplicate_evaluations']} ({stats['duplicate_percentage']:.2f}%)")
    
    logging.info(f"\nQuick test complete! Results saved to {dirs['main']}")

if __name__ == '__main__':
    quick_test()  # Run the quick test instead of the full analysis
