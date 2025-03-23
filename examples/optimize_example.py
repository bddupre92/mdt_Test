"""Example optimization using the meta-optimizer framework."""
import numpy as np
import logging
from typing import Dict, Any
import matplotlib.pyplot as plt
from pathlib import Path
import os
import sys
from datetime import datetime

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from meta.meta_optimizer import MetaOptimizer
from optimizers.de import DifferentialEvolutionOptimizer
from optimizers.es import EvolutionStrategyOptimizer
from optimizers.gwo import GreyWolfOptimizer
from utils.plot_utils import save_plot

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
        # Return the solution vector directly, not the dictionary
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
        # Return the solution vector directly, not the dictionary
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


def setup_logging():
    """Configure logging."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )


def sphere(x: np.ndarray) -> float:
    """Sphere test function."""
    return np.sum(x**2)


def rastrigin(x: np.ndarray) -> float:
    """Rastrigin test function."""
    return 10 * len(x) + np.sum(x**2 - 10 * np.cos(2 * np.pi * x))


def plot_convergence(optimizer_results: Dict[str, Any], save_path: str):
    """Plot convergence curves."""
    plt.figure(figsize=(10, 6))
    
    # Plot best solution's convergence curve
    if 'convergence' in optimizer_results['best_solution']:
        plt.plot(
            optimizer_results['best_solution']['convergence'],
            label='Best Solution',
            linewidth=2
        )
    
    # Plot all optimizers' convergence curves
    for result in optimizer_results['history']:
        if 'convergence' in result:
            plt.plot(result['convergence'], alpha=0.7)
            
    plt.yscale('log')
    plt.xlabel('Iteration')
    plt.ylabel('Best Score')
    plt.title('Convergence Curves')
    plt.legend()
    plt.grid(True)
    
    # Save plot using save_plot
    filename = os.path.basename(save_path)
    save_plot(plt.gcf(), filename, plot_type='performance')
    plt.close()


def main():
    """Run optimization example."""
    # Setup logging
    setup_logging()
    logger = logging.getLogger(__name__)
    
    # Create results directory
    results_dir = Path('results')
    results_dir.mkdir(exist_ok=True)
    
    # Problem parameters
    dim = 30
    bounds = [(-5.12, 5.12)] * dim
    max_evals = 10000
    
    # Initialize optimizers
    optimizers = {
        'DifferentialEvolution': CustomDEOptimizer(
            dim=dim,
            bounds=bounds,
            population_size=50,
            F=0.8,
            CR=0.5
        ),
        'EvolutionStrategy': CustomESOptimizer(
            dim=dim,
            bounds=bounds,
            population_size=100,
            mu=20,
            sigma=0.1
        ),
        'GreyWolfOptimizer': CustomGWOOptimizer(
            dim=dim,
            bounds=bounds,
            population_size=50
        )
    }
    
    # Initialize meta-optimizer
    meta_opt = MetaOptimizer(
        dim=dim,
        bounds=bounds,
        optimizers=optimizers,
        history_file=str(results_dir / 'history.json'),
        selection_file=str(results_dir / 'selection.json')
    )
    
    # Test functions
    test_functions = {
        'sphere': sphere,
        'rastrigin': rastrigin
    }
    
    # Run optimization for each test function
    for func_name, func in test_functions.items():
        logger.info(f"Optimizing {func_name} function")
        
        # Run meta-optimizer
        results = meta_opt.optimize(
            objective_func=func,
            max_evals=max_evals,
            context={'problem_type': func_name}
        )
        
        # Log results
        logger.info(f"Best score: {results['best_score']:.3e}")
        logger.info(f"Total evaluations: {sum(r['evaluations'] for r in results['history'])}")
        
        # Plot convergence
        plot_convergence(
            results,
            str(results_dir / f'{func_name}_convergence.png')
        )
        
        # Reset for next function
        meta_opt.reset()


if __name__ == '__main__':
    main()
