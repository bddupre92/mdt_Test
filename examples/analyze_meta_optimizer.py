"""
analyze_meta_optimizer.py
-----------------------
Comprehensive analysis of meta-learning optimizer performance with incremental testing
and detailed analysis of meta-learner effectiveness.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any, List, Tuple
import json
import os
from datetime import datetime
from pathlib import Path
import sys

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.plot_utils import save_plot

from meta.meta_optimizer import MetaOptimizer
from optimizers.ml_optimizers.surrogate_optimizer import SurrogateOptimizer
from optimizers.de import DifferentialEvolutionOptimizer
from optimizers.es import EvolutionStrategyOptimizer
from optimizers.gwo import GreyWolfOptimizer

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
    def __init__(self, dim, bounds, population_size, max_evals):
        super().__init__(dim, bounds, population_size)
        self.max_evals = max_evals
        self.max_iterations = max(1, max_evals // max(1, population_size))

    def _iterate(self):
        """Implement the abstract _iterate method from BaseOptimizer"""
        # Increment iteration counter
        self._current_iteration += 1
        
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

class CustomSurrogateOptimizer(SurrogateOptimizer):
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
            self.evaluations += 1
            
            # Add to training data
            self._add_training_point(candidate, score)
            
            # Update best solution if needed
            if score < self.best_score:
                self.best_score = score
                self.best_solution = candidate.copy()
                self.convergence_curve.append(self.best_score)
    
    def optimize(self, objective_func, max_evals=None):
        """Wrapper for the run method to match the interface expected by benchmarking code"""
        self.set_objective(objective_func)
        result = self.run(max_evals=max_evals)
        return self.best_solution

def create_test_suite():
    """Create test suite with functions grouped by characteristics"""
    def sphere(x):
        x = np.asarray(x)
        return float(np.sum(x**2))
        
    def rastrigin(x):
        x = np.asarray(x)
        return float(10 * len(x) + np.sum(x**2 - 10 * np.cos(2 * np.pi * x)))
        
    def rosenbrock(x):
        x = np.asarray(x)
        return float(np.sum(100.0 * (x[1:] - x[:-1]**2)**2 + (1 - x[:-1])**2))
    
    suites = {
        'unimodal': {
            'sphere': {
                'func': sphere,
                'bounds': [(-5.12, 5.12)],
                'dim': 2,
                'multimodal': 0,
                'discrete_vars': 0,
                'optimal': 0.0
            }
        },
        'multimodal': {
            'rastrigin': {
                'func': rastrigin,
                'bounds': [(-5.12, 5.12)],
                'dim': 2,
                'multimodal': 1,
                'discrete_vars': 0,
                'optimal': 0.0
            }
        },
        'high_dimensional': {
            'rosenbrock': {
                'func': rosenbrock,
                'bounds': [(-2.048, 2.048)],
                'dim': 10,
                'multimodal': 0,
                'discrete_vars': 0,
                'optimal': 0.0
            }
        }
    }
    return suites

def create_optimizers(dim: int, bounds: List[Tuple[float, float]]):
    """Create optimizer instances"""
    return {
        'surrogate': CustomSurrogateOptimizer(
            dim=dim,
            bounds=bounds,
            pop_size=30,
            n_initial=10
        ),
        'de': CustomDEOptimizer(
            dim=dim,
            bounds=bounds,
            population_size=30
        ),
        'es': CustomESOptimizer(
            dim=dim,
            bounds=bounds,
            population_size=30
        ),
        'gwo': CustomGWOOptimizer(
            dim=dim,
            bounds=bounds,
            population_size=30,
            max_evals=1000
        )
    }

class OptimizationAnalyzer:
    def __init__(self, base_dir: str = 'results/meta_analysis'):
        """Initialize analyzer with results directory"""
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Create run directory
        self.run_dir = self.base_dir / self.timestamp
        self.run_dir.mkdir(parents=True, exist_ok=True)
        
    def run_single_test(self, func_name: str, func_info: Dict, n_trials: int = 5):
        """Run test for a single function and save results"""
        print(f"\nTesting {func_name}")
        
        # Results storage
        results = {
            'scores': [],
            'optimizer_choices': [],
            'convergence': [],
            'meta_learner_accuracy': [],
            'errors': []
        }
        
        # Create bounds list
        bounds = func_info['bounds'] * func_info['dim']
        
        for trial in range(n_trials):
            print(f"  Trial {trial + 1}/{n_trials}")
            
            try:
                # Create optimizers
                optimizers = create_optimizers(func_info['dim'], bounds)
                
                # Create meta-optimizer
                meta_opt = MetaOptimizer(optimizers, mode='bayesian')
                
                # Create context
                context = {
                    'dim': func_info['dim'],
                    'multimodal': func_info['multimodal'],
                    'discrete_vars': func_info['discrete_vars']
                }
                
                # Run optimization
                best_solution = meta_opt.optimize(func_info['func'], context)
                best_score = float(func_info['func'](np.asarray(best_solution)))
                
                # Store results
                results['scores'].append(best_score)
                results['optimizer_choices'].append(
                    meta_opt.performance_history['optimizer'].tolist()
                )
                results['convergence'].append([
                    float(score) for score in meta_opt.performance_history['score']
                ])
                
                # Analyze meta-learner accuracy
                if len(meta_opt.performance_history) > 1:
                    # Compare predicted vs actual performance
                    predicted_best = meta_opt.select_optimizer(context)
                    actual_scores = meta_opt.performance_history.groupby('optimizer')['score'].mean()
                    actual_best = actual_scores.idxmin()
                    results['meta_learner_accuracy'].append(predicted_best == actual_best)
                
                print(f"    Best score: {best_score:.2e}")
                print("    Optimizer usage:")
                usage = meta_opt.performance_history['optimizer'].value_counts()
                for opt, count in usage.items():
                    print(f"      {opt}: {count}")
                    
            except Exception as e:
                print(f"    Error in trial {trial + 1}: {str(e)}")
                results['errors'].append({
                    'trial': trial + 1,
                    'error': str(e)
                })
                continue
        
        # Save results only if we have some successful trials
        if results['scores']:
            results_file = self.run_dir / f"{func_name}_results.json"
            with open(results_file, 'w') as f:
                json.dump({
                    'function_info': {
                        'dim': func_info['dim'],
                        'multimodal': func_info['multimodal'],
                        'discrete_vars': func_info['discrete_vars'],
                        'optimal': func_info['optimal']
                    },
                    'results': {
                        'scores': results['scores'],
                        'optimizer_choices': results['optimizer_choices'],
                        'convergence': results['convergence'],
                        'meta_learner_accuracy': results['meta_learner_accuracy'],
                        'errors': results['errors']
                    }
                }, f, indent=2)
            
            # Analyze results if we have any successful trials
            self.analyze_results(func_name, results)
        else:
            print(f"  No successful trials for {func_name}")
        
        return results
    
    def analyze_results(self, func_name: str, results: Dict):
        """Analyze and visualize results for a single function"""
        if not results['scores']:
            print(f"\nNo results to analyze for {func_name}")
            if results['errors']:
                print("\nErrors encountered:")
                for error in results['errors']:
                    print(f"  Trial {error['trial']}: {error['error']}")
            return
            
        print(f"\nAnalyzing {func_name}")
        
        # Create function directory
        func_dir = self.run_dir / func_name
        func_dir.mkdir(parents=True, exist_ok=True)
        
        # 1. Performance Statistics
        scores = np.array(results['scores'])
        stats = {
            'mean': float(np.mean(scores)),
            'std': float(np.std(scores)),
            'min': float(np.min(scores)),
            'max': float(np.max(scores)),
            'median': float(np.median(scores)),
            'successful_trials': len(scores),
            'failed_trials': len(results['errors'])
        }
        
        print("\nPerformance Statistics:")
        for key, value in stats.items():
            print(f"  {key}: {value}")
        
        # Save statistics
        with open(func_dir / 'statistics.json', 'w') as f:
            json.dump(stats, f, indent=2)
        
        # 2. Convergence Plot
        plt.figure(figsize=(10, 6))
        for i, conv in enumerate(results['convergence']):
            plt.plot(conv, alpha=0.5, label=f'Trial {i+1}')
        plt.yscale('log')
        plt.xlabel('Iteration')
        plt.ylabel('Objective Value')
        plt.title(f'Convergence Plot - {func_name}')
        plt.legend()
        plt.grid(True)
        save_plot(plt.gcf(), 'convergence.png', plot_type='benchmarks')
        plt.close()
        
        # 3. Optimizer Usage Analysis
        optimizer_counts = []
        for choices in results['optimizer_choices']:
            counts = pd.Series(choices).value_counts()
            optimizer_counts.append(counts)
        
        usage_df = pd.DataFrame(optimizer_counts).fillna(0)
        
        plt.figure(figsize=(10, 6))
        usage_df.mean().plot(kind='bar')
        plt.title(f'Average Optimizer Usage - {func_name}')
        plt.xlabel('Optimizer')
        plt.ylabel('Average Number of Calls')
        plt.tight_layout()
        save_plot(plt.gcf(), 'optimizer_usage.png', plot_type='benchmarks')
        plt.close()
        
        # 4. Meta-learner Analysis
        if results['meta_learner_accuracy']:
            accuracy = np.mean(results['meta_learner_accuracy'])
            print(f"\nMeta-learner accuracy: {accuracy:.2%}")
            
            with open(func_dir / 'meta_learner_accuracy.json', 'w') as f:
                json.dump({
                    'accuracy': float(accuracy),
                    'predictions': len(results['meta_learner_accuracy'])
                }, f, indent=2)

def main():
    """Run complete analysis"""
    # Create analyzer
    analyzer = OptimizationAnalyzer()
    
    # Get test suites
    suites = create_test_suite()
    
    # Test each suite
    all_results = {}
    for suite_name, suite in suites.items():
        print(f"\nTesting suite: {suite_name}")
        suite_results = {}
        
        for func_name, func_info in suite.items():
            try:
                # Run tests
                results = analyzer.run_single_test(func_name, func_info)
                
                # Analyze results
                analyzer.analyze_results(func_name, results)
                
                suite_results[func_name] = results
                
            except Exception as e:
                print(f"Error testing {func_name}: {str(e)}")
                continue
        
        all_results[suite_name] = suite_results
    
    # Save overall summary
    summary = {
        'timestamp': analyzer.timestamp,
        'suites_tested': list(suites.keys()),
        'functions_tested': sum(len(suite) for suite in suites.values()),
        'successful_tests': sum(
            len(suite_results) for suite_results in all_results.values()
        )
    }
    
    with open(analyzer.run_dir / 'summary.json', 'w') as f:
        json.dump(summary, f, indent=2)

if __name__ == '__main__':
    main()
