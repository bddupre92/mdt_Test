"""
test_enhanced_optimizers.py
--------------------------
Test script for evaluating enhanced optimizers with meta-learning.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Any, List, Tuple
import time
from pathlib import Path
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed

from optimizers.de import DifferentialEvolutionOptimizer
from optimizers.gwo import GreyWolfOptimizer
from optimizers.es import EvolutionStrategyOptimizer
from optimizers.aco import AntColonyOptimizer
from meta.meta_optimizer import MetaOptimizer
from utils.plot_utils import save_plot

# Test functions
def rosenbrock(x: np.ndarray) -> float:
    """Rosenbrock function (banana function)"""
    return np.sum(100.0 * (x[1:] - x[:-1]**2)**2 + (1 - x[:-1])**2)

def rastrigin(x: np.ndarray) -> float:
    """Rastrigin function (highly multimodal)"""
    return 10 * len(x) + np.sum(x**2 - 10 * np.cos(2 * np.pi * x))

def sphere(x: np.ndarray) -> float:
    """Sphere function (unimodal, convex)"""
    return np.sum(x**2)

TEST_FUNCTIONS = {
    'rosenbrock': rosenbrock,
    'rastrigin': rastrigin,
    'sphere': sphere
}

class OptimizationExperiment:
    def __init__(self, dim: int = 30, max_evals: int = 10000):
        self.dim = dim
        self.max_evals = max_evals
        self.bounds = [(-5.12, 5.12)] * dim
        self.results_dir = Path("results")
        self.results_dir.mkdir(exist_ok=True)
        
        # Initialize optimizers
        self.optimizers = {
            'de_standard': DifferentialEvolutionOptimizer(
                dim=dim, bounds=self.bounds, strategy='rand/1/bin',
                adaptive=False, max_evaluations=max_evals
            ),
            'de_adaptive': DifferentialEvolutionOptimizer(
                dim=dim, bounds=self.bounds, strategy='best/1/bin',
                adaptive=True, max_evaluations=max_evals
            ),
            'de_current_to_best': DifferentialEvolutionOptimizer(
                dim=dim, bounds=self.bounds, strategy='current-to-best/1/bin',
                adaptive=True, max_evaluations=max_evals
            ),
            'gwo_standard': GreyWolfOptimizer(
                dim=dim, bounds=self.bounds,
                adaptive=False, max_evaluations=max_evals
            ),
            'gwo_adaptive': GreyWolfOptimizer(
                dim=dim, bounds=self.bounds,
                adaptive=True, max_evaluations=max_evals
            ),
            'es_standard': EvolutionStrategyOptimizer(
                dim=dim, bounds=self.bounds,
                adaptive=False, max_evaluations=max_evals
            ),
            'es_adaptive': EvolutionStrategyOptimizer(
                dim=dim, bounds=self.bounds,
                adaptive=True, max_evaluations=max_evals
            ),
            'aco_standard': AntColonyOptimizer(
                dim=dim, bounds=self.bounds,
                adaptive=False, max_evaluations=max_evals
            ),
            'aco_adaptive': AntColonyOptimizer(
                dim=dim, bounds=self.bounds,
                adaptive=True, max_evaluations=max_evals
            )
        }
        
        # Initialize meta-optimizer
        self.meta_opt = MetaOptimizer(self.optimizers)
        
    def run_single_optimization(self, func_name: str, func: callable, run: int,
                              optimizer_name: str = None) -> Dict[str, Any]:
        """Run a single optimization trial"""
        context = {
            'dim': self.dim,
            'multimodal': func_name in ['rastrigin'],
            'discrete_vars': 0
        }
        
        # Select optimizer
        if optimizer_name is None:
            optimizer_name = self.meta_opt.select_optimizer(context)
        optimizer = self.optimizers[optimizer_name]
        
        # Run optimization
        start_time = time.time()
        solution, score = optimizer.optimize(func, context=context)
        runtime = time.time() - start_time
        
        # Get optimizer state and statistics
        state = optimizer.get_state()
        param_history = optimizer.get_parameter_history()
        
        # Plot convergence
        plt.figure(figsize=(10, 6))
        plt.plot(state.history, label=f'Run {run+1}')
        plt.yscale('log')
        plt.xlabel('Iteration')
        plt.ylabel('Objective Value')
        plt.title(f'{func_name} - {optimizer_name} Convergence')
        plt.legend()
        plt.grid(True)
        save_plot(plt.gcf(), f'{func_name}_{optimizer_name}_run{run+1}.png', plot_type='benchmarks')
        plt.close()
        
        # Update meta-optimizer
        self.meta_opt.update_performance(
            optimizer_name,
            context,
            runtime,
            score
        )
        
        return {
            'run': run,
            'optimizer': optimizer_name,
            'best_score': score,
            'runtime': runtime,
            'evaluations': state.evaluations,
            'convergence': state.history,
            'parameters': param_history
        }
    
    def run_experiment(self, num_runs: int = 5, parallel: bool = True) -> Dict[str, List[Dict[str, Any]]]:
        """Run full optimization experiment"""
        results = {}
        
        if parallel:
            with ProcessPoolExecutor(max_workers=mp.cpu_count()) as executor:
                futures = []
                
                for func_name, func in TEST_FUNCTIONS.items():
                    print(f"\nOptimizing {func_name}...")
                    results[func_name] = []
                    
                    for run in range(num_runs):
                        for opt_name in self.optimizers.keys():
                            future = executor.submit(
                                self.run_single_optimization,
                                func_name, func, run, opt_name
                            )
                            futures.append((func_name, future))
                
                for func_name, future in futures:
                    result = future.result()
                    results[func_name].append(result)
        else:
            for func_name, func in TEST_FUNCTIONS.items():
                print(f"\nOptimizing {func_name}...")
                results[func_name] = []
                
                for run in range(num_runs):
                    print(f"Run {run + 1}/{num_runs}")
                    for opt_name in self.optimizers.keys():
                        result = self.run_single_optimization(
                            func_name, func, run, opt_name
                        )
                        results[func_name].append(result)
        
        return results
    
    def print_summary(self, results: Dict[str, List[Dict[str, Any]]]):
        """Print summary statistics for all runs"""
        print("\nSummary of Results:")
        print("=" * 80)
        
        for func_name, func_results in results.items():
            print(f"\n{func_name.upper()}")
            print("-" * 40)
            
            # Group by optimizer
            by_optimizer = {}
            for result in func_results:
                opt = result['optimizer']
                if opt not in by_optimizer:
                    by_optimizer[opt] = []
                by_optimizer[opt].append(result)
            
            for opt, opt_results in by_optimizer.items():
                scores = [r['best_score'] for r in opt_results]
                times = [r['runtime'] for r in opt_results]
                evals = [r['evaluations'] for r in opt_results]
                
                print(f"\n{opt}:")
                print(f"  Best score:   {min(scores):.2e} ± {np.std(scores):.2e}")
                print(f"  Avg runtime:  {np.mean(times):.2f} ± {np.std(times):.2f} s")
                print(f"  Avg evals:    {np.mean(evals):.0f} ± {np.std(evals):.0f}")
        
        # Print meta-optimizer statistics
        print("\nMeta-Optimizer Statistics:")
        print("=" * 80)
        stats = self.meta_opt.get_optimizer_stats()
        for opt, opt_stats in stats.items():
            print(f"\n{opt}:")
            print(f"  Success rate: {opt_stats['success_rate']:.2%}")
            print(f"  Avg runtime:  {opt_stats['avg_runtime']:.2f} s")
            print(f"  Total runs:   {opt_stats['runs']}")

if __name__ == "__main__":
    # Create and run experiment
    experiment = OptimizationExperiment(dim=30, max_evals=10000)
    results = experiment.run_experiment(num_runs=5, parallel=True)
    
    # Print summary
    experiment.print_summary(results)
