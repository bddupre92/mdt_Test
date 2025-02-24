"""
tune_optimizers.py
----------------
Fine-tune optimization algorithms using grid search and advanced parameter ranges.
"""

import numpy as np
from optimizers.aco import AntColonyOptimizer
from optimizers.gwo import GreyWolfOptimizer
from optimizers.es import EvolutionStrategy
from optimizers.de import DifferentialEvolutionOptimizer
from benchmarking.test_functions import TestFunction, ClassicalTestFunctions
from visualization.optimizer_analysis import OptimizerAnalyzer
import os
import pandas as pd
from concurrent.futures import ProcessPoolExecutor
import itertools
import time

def evaluate_params(args):
    optimizer_class, func, params = args
    optimizer = optimizer_class(
        dim=func.dim,
        bounds=func.bounds,
        **params
    )
    try:
        start_time = time.time()
        best_solution, best_score = optimizer.optimize(func.func)
        execution_time = time.time() - start_time
        
        return {
            'optimizer': optimizer_class.__name__,
            'function': func.name,
            **params,
            'score': float(best_score),
            'time': execution_time
        }
    except Exception as e:
        print(f"Error with params {params}: {str(e)}")
        return None

def grid_search_optimizer(optimizer_class, func, param_grid, n_runs=5):
    param_combinations = [dict(zip(param_grid.keys(), v)) 
                        for v in itertools.product(*param_grid.values())]
    
    all_args = [(optimizer_class, func, params) 
                for params in param_combinations 
                for _ in range(n_runs)]
    
    results = []
    with ProcessPoolExecutor() as executor:
        for result in executor.map(evaluate_params, all_args):
            if result is not None:
                results.append(result)
    
    return pd.DataFrame(results)

def main():
    # Create output directory
    os.makedirs('results/tuning', exist_ok=True)
    
    # Test functions
    dim = 2
    test_functions = [
        TestFunction(
            name='Rosenbrock',
            func=ClassicalTestFunctions.rosenbrock,
            dim=dim,
            bounds=[(-2.048, 2.048)] * dim,
            global_minimum=0.0,
            characteristics={'continuous': True, 'non-convex': True, 'unimodal': True}
        )
    ]
    
    # Parameter grids for each optimizer
    param_grids = {
        GreyWolfOptimizer: {
            'population_size': [30, 50, 100, 200],
            'num_generations': [100, 200, 300, 500]
        },
        DifferentialEvolutionOptimizer: {
            'population_size': [30, 50, 100, 200],
            'num_generations': [100, 200, 300, 500],
            'F': [0.5, 0.7, 0.9],
            'CR': [0.3, 0.5, 0.7, 0.9]
        },
        EvolutionStrategy: {
            'population_size': [30, 50, 100, 200],
            'num_generations': [100, 200, 300, 500],
            'mutation_rate': [0.05, 0.1, 0.2],
            'crossover_rate': [0.5, 0.7, 0.9]
        },
        AntColonyOptimizer: {
            'population_size': [30, 50, 100, 200],
            'num_generations': [100, 200, 300, 500],
            'rho': [0.1, 0.3, 0.5],
            'alpha': [1.0, 2.0, 3.0],
            'beta': [1.0, 2.0, 3.0],
            'Q': [0.5, 1.0, 2.0]
        }
    }
    
    # Run grid search for each optimizer on each function
    for func in test_functions:
        print(f"\nTuning optimizers for {func.name} function...")
        
        all_results = []
        for optimizer_class, param_grid in param_grids.items():
            print(f"\nTuning {optimizer_class.__name__}...")
            results = grid_search_optimizer(optimizer_class, func, param_grid)
            all_results.append(results)
            
            # Save individual optimizer results
            results.to_csv(f'results/tuning/{func.name.lower()}_{optimizer_class.__name__.lower()}_tuning.csv', index=False)
            
            # Print best parameters
            best_idx = results.groupby(['optimizer', 'function'])['score'].idxmin()
            best_params = results.loc[best_idx]
            print(f"\nBest parameters for {optimizer_class.__name__}:")
            print(best_params.to_string())
        
        # Combine all results
        combined_results = pd.concat(all_results)
        combined_results.to_csv(f'results/tuning/{func.name.lower()}_all_tuning.csv', index=False)

if __name__ == '__main__':
    main()
