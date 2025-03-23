"""
theoretical_comparison.py
------------------------
Comprehensive comparison of meta-learning framework against traditional optimizers.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
import logging
import pandas as pd
import warnings
from typing import Dict, List, Any
from datetime import datetime
import time
import sys
import os
from meta.selection_tracker import SelectionTracker
from scipy.stats import norm
from optimizers.base_optimizer import OptimizerState

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.plot_utils import save_plot

# Filter sklearn warnings
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

# Optimizers
from optimizers.de import DifferentialEvolutionOptimizer
from optimizers.es import EvolutionStrategyOptimizer
from optimizers.gwo import GreyWolfOptimizer
from optimizers.aco import AntColonyOptimizer
from optimizers.ml_optimizers.surrogate_optimizer import SurrogateOptimizer

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
        super().__init__(dim, bounds, population_size, **kwargs)
        self.state = OptimizerState()
        self.state.solutions = []
        self.state.evaluations = []
        
    def _iterate(self):
        """Implement the abstract _iterate method from BaseOptimizer"""
        # Increment iteration counter
        self._current_iteration += 1
        
        # Generate solutions for all ants
        new_solutions = []
        new_scores = []
        
        for i in range(self.population_size):
            # Generate new solution
            new_solution = self._generate_solution()
            
            # Evaluate solution
            score = self.objective_func(new_solution)
            
            # Update state
            self._update_state(new_solution, score)
            
            # Store solution and score
            new_solutions.append(new_solution)
            new_scores.append(score)
            
            # Update best solution if needed
            if score < self.best_score:
                self.best_score = score
                self.best_solution = new_solution.copy()
        
        # Update pheromone matrix
        self._update_pheromones(new_solutions, new_scores)
        
        return self.best_score
    
    def _generate_solution(self):
        """Generate a new solution based on pheromone matrix"""
        # If pheromone matrix is not initialized or in early iterations, use random sampling
        if self._current_iteration < 2 or not hasattr(self, 'pheromone_matrix'):
            return np.random.uniform(
                low=[b[0] for b in self.bounds],
                high=[b[1] for b in self.bounds],
                size=self.dim
            )
        
        # Initialize solution
        solution = np.zeros(self.dim)
        
        # For each dimension
        for d in range(self.dim):
            # Calculate selection probabilities based on pheromone levels
            probabilities = self.pheromone_matrix[d] / np.sum(self.pheromone_matrix[d])
            
            # Select bin based on pheromone levels
            bin_idx = np.random.choice(len(probabilities), p=probabilities)
            
            # Convert bin index to continuous value within bounds
            bin_width = (self.bounds[d][1] - self.bounds[d][0]) / self.n_bins
            # Add some noise within the bin
            solution[d] = self.bounds[d][0] + bin_idx * bin_width + np.random.uniform(0, bin_width)
        
        return solution
    
    def _update_pheromones(self, solutions, scores):
        """Update pheromone matrix based on solution quality"""
        # Initialize pheromone matrix if not already done
        if not hasattr(self, 'pheromone_matrix'):
            self.n_bins = 10  # Number of bins per dimension
            self.pheromone_matrix = np.ones((self.dim, self.n_bins))
            self.evaporation_rate = 0.1
        
        # Evaporate pheromones
        self.pheromone_matrix *= (1 - self.evaporation_rate)
        
        # Convert scores to pheromone updates (higher pheromone for better solutions)
        max_score = max(scores)
        min_score = min(scores)
        score_range = max_score - min_score if max_score > min_score else 1.0
        
        # Update pheromones for each solution
        for solution, score in zip(solutions, scores):
            # Normalize score (invert so better solutions have higher values)
            normalized_score = (max_score - score) / score_range if score_range > 0 else 1.0
            
            # Update pheromone for each dimension
            for d in range(self.dim):
                # Calculate bin index
                bin_width = (self.bounds[d][1] - self.bounds[d][0]) / self.n_bins
                bin_idx = min(int((solution[d] - self.bounds[d][0]) / bin_width), self.n_bins - 1)
                
                # Update pheromone
                self.pheromone_matrix[d, bin_idx] += normalized_score
    
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
    
    def _iterate(self):
        """Implement the abstract _iterate method from BaseOptimizer"""
        # Increment iteration counter
        self._current_iteration += 1
        
        # Update surrogate model
        self._update_model()
        
        # Generate candidate solutions
        candidates = self._generate_candidates()
        
        # Evaluate promising candidates
        for candidate in candidates[:self.n_direct_eval]:
            score = self.objective_func(candidate)
            self._update_state(candidate, score)
            
            if score < self.best_score:
                self.best_score = score
                self.best_solution = candidate.copy()
                
        return self.best_score
    
    def _update_model(self):
        """Update the surrogate model with observed data"""
        if len(self.state.evaluations) < 2:  # Need at least 2 points for GP
            return
            
        X = np.array(self.state.solutions)
        y = np.array(self.state.evaluations)
        
        # Subsample if too many points
        if len(X) > self.max_gp_size:
            indices = np.random.choice(len(X), self.max_gp_size, replace=False)
            X_train = X[indices]
            y_train = y[indices]
        else:
            X_train = X
            y_train = y
            
        # Normalize y values
        y_mean = np.mean(y_train)
        y_std = np.std(y_train) + 1e-8
        y_norm = (y_train - y_mean) / y_std
        
        # Update model
        try:
            self.model.fit(self.scale_points(X_train), y_norm)
        except Exception as e:
            self.logger.warning(f"GP model update failed: {str(e)}")
    
    def _generate_candidates(self):
        """Generate candidate solutions using the surrogate model"""
        # If we don't have enough data yet, use random sampling
        if len(self.state.evaluations) < 5:
            return np.random.uniform(
                low=[b[0] for b in self.bounds],
                high=[b[1] for b in self.bounds],
                size=(self.pop_size, self.dim)
            )
            
        # Generate random candidates
        candidates = np.random.uniform(
            low=[b[0] for b in self.bounds],
            high=[b[1] for b in self.bounds],
            size=(self.pop_size * 2, self.dim)  # Generate more candidates
        )
        
        # Predict values and uncertainties
        try:
            mu, sigma = self.model.predict(self.scale_points(candidates), return_std=True)
            
            # Calculate acquisition function (Expected Improvement)
            best_y = self.best_score
            gamma = (best_y - mu) / (sigma + 1e-9)
            ei = sigma * (gamma * norm.cdf(gamma) + norm.pdf(gamma))
            
            # Select candidates with best EI
            best_indices = np.argsort(ei)[-self.pop_size:]
            return candidates[best_indices]
            
        except Exception as e:
            self.logger.warning(f"Candidate generation failed: {str(e)}")
            # Fallback to random sampling
            return candidates[:self.pop_size]
    
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

def setup_logging(output_dir: str):
    """Configure logging"""
    log_file = Path(output_dir) / 'comparison.log'
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )

def create_output_dirs():
    """Create timestamped output directories"""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    base_dir = Path(f'results/comparison_{timestamp}')
    
    dirs = {
        'main': base_dir,
        'plots': base_dir / 'plots',
        'data': base_dir / 'data',
        'analysis': base_dir / 'analysis',
        'results': base_dir / 'results'
    }
    
    for dir_path in dirs.values():
        dir_path.mkdir(parents=True, exist_ok=True)
    
    return dirs

def compare_optimizers(
    test_func,
    dim,
    bounds,
    n_trials=5,
    max_evals=1000,
    population_size=50,
    dirs=None,
    problem_type=None
):
    """Compare meta-optimizer against individual optimizers"""
    
    if dirs is None:
        dirs = {}
    
    # Create individual optimizers
    de_optimizer = CustomDEOptimizer(dim, bounds, population_size=population_size)
    es_optimizer = CustomESOptimizer(dim, bounds, population_size=population_size)
    gwo_optimizer = CustomGWOOptimizer(dim, bounds, population_size=population_size)
    aco_optimizer = CustomACOOptimizer(dim, bounds, population_size=population_size)
    surrogate_optimizer = CustomSurrogateOptimizer(dim, bounds, population_size=population_size)
    
    # Create meta-optimizer with improved settings
    history_file = None
    selection_file = None
    if 'results' in dirs:
        history_file = str(dirs['results'] / 'meta_optimizer_history.json')
        selection_file = str(dirs['results'] / 'optimizer_selections.json')
        
    meta_optimizer = MetaOptimizer(
        dim=dim,
        bounds=bounds,
        optimizers={
            'de': de_optimizer,
            'es': es_optimizer,
            'gwo': gwo_optimizer,
            'aco': aco_optimizer,
            'surrogate': surrogate_optimizer
        },
        history_file=history_file,
        selection_file=selection_file
    )
    
    # Run trials
    results = {
        'individual': {},
        'meta': {'bayesian': []}
    }
    
    # Test individual optimizers
    for name, optimizer in {
        'de': de_optimizer,
        'es': es_optimizer,
        'gwo': gwo_optimizer,
        'aco': aco_optimizer,
        'surrogate': surrogate_optimizer
    }.items():
        logging.info(f"Testing {name}")
        results['individual'][name] = []
        
        for trial in range(n_trials):
            optimizer.reset()
            start_time = time.time()
            solution = optimizer.optimize(test_func, max_evals=max_evals)
            runtime = time.time() - start_time
            
            if solution is not None:
                value = test_func(solution)
                logging.info(f"  Trial {trial + 1}: {value:.2e}")
                
                history = optimizer.history
                if isinstance(history, pd.DataFrame):
                    history = history.to_dict('records')
                
                results['individual'][name].append({
                    'solution': solution.tolist(),  # Convert to list
                    'value': value,
                    'runtime': runtime,
                    'history': history,
                    'n_evals': getattr(optimizer, 'n_evals', max_evals)
                })
    
    # Test meta-optimizer
    logging.info("Testing meta-optimizer")
    
    for trial in range(n_trials):
        meta_optimizer.reset()
        start_time = time.time()
        solution = meta_optimizer.optimize(test_func, max_evals=max_evals)
        runtime = time.time() - start_time
        
        if solution is not None:
            value = test_func(solution)
            logging.info(f"  Trial {trial + 1}: {value:.2e}")
            
            # Store trial results
            results['meta']['bayesian'].append({
                'solution': solution.tolist(),
                'value': value,
                'runtime': runtime,
                'history': meta_optimizer.optimization_history,
                'total_evaluations': meta_optimizer.total_evaluations
            })
    
    return results

def plot_convergence(results, plot_dir, func_name):
    """Plot convergence curves for all optimizers."""
    plt.figure(figsize=(10, 6))
    plt.title(f"Convergence on {func_name}")
    plt.xlabel("Function Evaluations")
    plt.ylabel("Objective Value (log scale)")
    plt.yscale('log')
    
    # Plot standard optimizers
    for name, trials in results.items():
        if name == 'meta':
            continue
            
        all_scores = {}
        for trial in trials:
            if 'convergence' in trial:
                for idx, score in trial['convergence']:
                    if idx not in all_scores:
                        all_scores[idx] = []
                    all_scores[idx].append(score)
        
        if all_scores:
            iterations = sorted(all_scores.keys())
            mean_scores = [np.mean(all_scores[i]) for i in iterations]
            std_scores = [np.std(all_scores[i]) for i in iterations]
            
            plt.plot(iterations, mean_scores, label=name.upper(), alpha=0.8)
            plt.fill_between(
                iterations,
                [m - s for m, s in zip(mean_scores, std_scores)],
                [m + s for m, s in zip(mean_scores, std_scores)],
                alpha=0.2
            )
    
    # Plot meta-optimizer
    if 'meta' in results:
        for mode, trials in results['meta'].items():
            if not trials:  # Skip if no successful trials
                continue
                
            # Process meta-optimizer history
            all_meta_scores = {}
            for trial in trials:
                if 'history' in trial:
                    history = trial['history']
                    for entry in history:
                        evals = entry.get('evaluations', 0)
                        if evals not in all_meta_scores:
                            all_meta_scores[evals] = []
                        all_meta_scores[evals].append(entry.get('score', float('inf')))
            
            if all_meta_scores:
                meta_iterations = sorted(all_meta_scores.keys())
                meta_mean_scores = [np.mean(all_meta_scores[i]) for i in meta_iterations]
                meta_std_scores = [np.std(all_meta_scores[i]) for i in meta_iterations]
                
                plt.plot(meta_iterations, meta_mean_scores, label=f"META-{mode.upper()}", 
                        linestyle='--', linewidth=2, alpha=0.8)
                plt.fill_between(
                    meta_iterations,
                    [m - s for m, s in zip(meta_mean_scores, meta_std_scores)],
                    [m + s for m, s in zip(meta_mean_scores, meta_std_scores)],
                    alpha=0.2
                )
    
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Save plot
    os.makedirs(plot_dir, exist_ok=True)
    plot_path = os.path.join(plot_dir, f"convergence_{func_name}.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return plot_path

def plot_optimizer_selection(results: dict, output_dir: Path, title: str):
    """Plot meta-optimizer selection patterns"""
    if 'meta' not in results or not results['meta']['bayesian']:
        logging.warning(f"No meta-optimizer results for {title}")
        return
        
    selections = []
    for trial in results['meta']['bayesian']:
        if isinstance(trial['history'], list):
            for record in trial['history']:
                if isinstance(record, dict) and 'optimizer' in record:
                    selections.append(record['optimizer'])
    
    if not selections:
        logging.warning(f"No optimizer selections recorded for {title}")
        return
    
    # Count selections
    selection_counts = pd.Series(selections).value_counts()
    
    plt.figure(figsize=(10, 6))
    selection_counts.plot(kind='bar')
    plt.title(f'Optimizer Selection Pattern - {title}', fontsize=14)
    plt.xlabel('Optimizer', fontsize=12)
    plt.ylabel('Times Selected', fontsize=12)
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    save_plot(plt.gcf(), f'selection_pattern_{title.lower().replace(" ", "_")}.png', plot_type='benchmarks')
    plt.close()

def create_summary_table(results: dict, output_dir: Path, title: str, success_threshold: float = 1e-4):
    """Create summary statistics table"""
    summary = {
        'Optimizer': [],
        'Mean': [],
        'Std': [],
        'Best': [],
        'Worst': [],
        'Success Rate': [],
        'Avg Iterations': []
    }
    
    # Individual optimizers
    for name, trials in results['individual'].items():
        if not trials:  # Skip if no successful trials
            continue
            
        values = [trial['value'] for trial in trials]
        iterations = [trial['n_evals'] for trial in trials]
        success_rate = sum(1 for v in values if v < success_threshold) / len(values)
        
        summary['Optimizer'].append(name.upper())
        summary['Mean'].append(np.mean(values))
        summary['Std'].append(np.std(values))
        summary['Best'].append(np.min(values))
        summary['Worst'].append(np.max(values))
        summary['Success Rate'].append(success_rate)
        summary['Avg Iterations'].append(np.mean(iterations))
    
    # Meta-optimizer
    if 'meta' in results:
        for mode, trials in results['meta'].items():
            if not trials:  # Skip if no successful trials
                continue
                
            values = [trial['value'] for trial in trials]
            iterations = [trial['total_evaluations'] for trial in trials]  # Use stored total_evaluations
            success_rate = sum(1 for v in values if v < success_threshold) / len(values)
            
            summary['Optimizer'].append('META-OPTIMIZER')
            summary['Mean'].append(np.mean(values))
            summary['Std'].append(np.std(values))
            summary['Best'].append(np.min(values))
            summary['Worst'].append(np.max(values))
            summary['Success Rate'].append(success_rate)
            summary['Avg Iterations'].append(np.mean(iterations))
    
    # Create DataFrame and format
    df = pd.DataFrame(summary)
    df = df.round({
        'Mean': 6,
        'Std': 6,
        'Best': 6,
        'Worst': 6,
        'Success Rate': 1,
        'Avg Iterations': 1
    })
    
    # Save to file
    df.to_csv(output_dir / f'summary_{title.lower().replace(" ", "_")}.csv', index=False)
    
    return df

def save_results(results: dict, dirs: dict):
    """Save results to file"""
    for problem, result in results.items():
        with open(dirs['data'] / f'{problem}_results.json', 'w') as f:
            json.dump(result, f, indent=2)

def main():
    """Main entry point"""
    # Setup directories
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_dir = Path('results') / f'comparison_{timestamp}'
    
    dirs = {
        'main': base_dir,
        'plots': base_dir / 'plots',
        'data': base_dir / 'data',
        'analysis': base_dir / 'analysis',
        'results': base_dir
    }
    
    for dir_path in dirs.values():
        dir_path.mkdir(parents=True, exist_ok=True)
        
    setup_logging(dirs['main'])
    
    logging.info("Starting comprehensive comparison...")
    
    # Get test suite
    suite = create_test_suite()
    
    # Run comparison on each function type
    all_results = {}
    summary_tables = {}
    
    for suite_name, functions in suite.items():
        logging.info(f"\nTesting {suite_name} functions")
        suite_results = {}
        
        for func_name, func_info in functions.items():
            logging.info(f"Testing {func_name}")
            
            # Run comparison
            results = compare_optimizers(
                test_func=func_info['func'],
                dim=func_info['dim'],
                bounds=func_info['bounds'] * func_info['dim'],
                n_trials=5,
                max_evals=1000,
                dirs=dirs,
                problem_type=func_name
            )
            
            # Save results
            save_results({func_name: results}, dirs)
            
            # Create visualizations
            plot_convergence(results, dirs['plots'], f"{func_name}")
            plot_optimizer_selection(results, dirs['plots'], f"{func_name}")
            
            # Create summary table
            summary_tables[func_name] = create_summary_table(
                results, 
                dirs['analysis'],
                func_name
            )
            
            suite_results[func_name] = results
            
            # Log summary
            logging.info(f"\nResults Summary for {func_name}:")
            print(summary_tables[func_name].to_string())
            
        all_results[suite_name] = suite_results
    
    # Save overall results
    with open(dirs['data'] / 'all_results.json', 'w') as f:
        json.dump(all_results, f, indent=2)
    
    # Print selection analysis
    if Path(dirs['results'] / 'optimizer_selections.json').exists():
        selection_tracker = SelectionTracker(str(dirs['results'] / 'optimizer_selections.json'))
        stats = selection_tracker.get_selection_stats()
        if not stats.empty:
            logging.info("\nOptimizer Selection Analysis:")
            logging.info("\nSelection Statistics:")
            logging.info(stats.to_string())
            
            for problem in all_results.keys():
                correlations = selection_tracker.get_feature_correlations(problem)
                if correlations:
                    logging.info(f"\nFeature Correlations for {problem}:")
                    for opt, feat_corrs in correlations.items():
                        logging.info(f"\n{opt}:")
                        for feat, corr in feat_corrs.items():
                            logging.info(f"  {feat}: {corr:.3f}")
    
    logging.info(f"\nAll results saved to {dirs['main']}")
    logging.info("Comparison complete!")

if __name__ == '__main__':
    main()
