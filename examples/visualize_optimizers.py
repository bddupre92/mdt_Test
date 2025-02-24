"""
visualize_optimizers.py
----------------------
Example script showing how to use the optimizer visualization framework.
"""

import numpy as np
from optimizers.aco import AntColonyOptimizer
from optimizers.gwo import GreyWolfOptimizer
from optimizers.es import EvolutionStrategy
from optimizers.de import DifferentialEvolutionOptimizer
from benchmarking.test_functions import TestFunction, ClassicalTestFunctions
from visualization.optimizer_analysis import OptimizerAnalyzer
import os

def main():
    # Create output directory
    os.makedirs('results/plots', exist_ok=True)
    
    # Define dimensions and bounds for all problems
    dim = 2  # Using 2D for visualization purposes
    bounds = [(-5.12, 5.12)] * dim
    
    # Create test functions
    test_functions = [
        TestFunction(
            name='Sphere',
            func=ClassicalTestFunctions.sphere,
            dim=dim,
            bounds=bounds,
            global_minimum=0.0,
            characteristics={'continuous': True, 'convex': True, 'unimodal': True}
        ),
        TestFunction(
            name='Rastrigin',
            func=ClassicalTestFunctions.rastrigin,
            dim=dim,
            bounds=bounds,
            global_minimum=0.0,
            characteristics={'continuous': True, 'non-convex': True, 'multimodal': True}
        ),
        TestFunction(
            name='Rosenbrock',
            func=ClassicalTestFunctions.rosenbrock,
            dim=dim,
            bounds=[(-2.048, 2.048)] * dim,
            global_minimum=0.0,
            characteristics={'continuous': True, 'non-convex': True, 'unimodal': True}
        )
    ]
    
    # Create optimizers with consistent parameters
    population_size = 50
    num_generations = 100
    
    optimizers = [
        AntColonyOptimizer(
            dim=dim,
            bounds=bounds,
            population_size=population_size,
            num_generations=num_generations
        ),
        GreyWolfOptimizer(
            dim=dim,
            bounds=bounds,
            population_size=population_size,
            num_generations=num_generations
        ),
        EvolutionStrategy(
            dim=dim,
            bounds=bounds,
            population_size=population_size,
            num_generations=num_generations
        ),
        DifferentialEvolutionOptimizer(
            dim=dim,
            bounds=bounds,
            population_size=population_size,
            num_generations=num_generations
        )
    ]
    
    # Create analyzer
    analyzer = OptimizerAnalyzer(optimizers, test_functions)
    
    # Run comparison
    print("Running optimization comparison...")
    results = analyzer.run_comparison(n_runs=10, record_convergence=True)
    
    # Generate visualizations
    print("\nGenerating visualizations...")
    
    # 1. Convergence curves for each function
    print("- Plotting convergence curves...")
    for func in test_functions:
        analyzer.plot_convergence_curves(
            func.name,
            save_path=f'results/plots/convergence_{func.name.lower()}.png'
        )
    
    # 2. Performance heatmap
    print("- Creating performance heatmap...")
    analyzer.plot_performance_heatmap(
        metric='best_score',
        save_path='results/plots/performance_heatmap.png'
    )
    
    # 3. Create performance table
    print("- Generating performance table...")
    performance_table = analyzer.create_performance_table()
    performance_table.to_csv('results/performance_metrics.csv', index=False)
    
    # 4. Parameter sensitivity analysis
    print("- Analyzing parameter sensitivity...")
    param_ranges = {
        'population_size': [10, 20, 50, 100, 200],
        'num_generations': [50, 100, 200, 500]
    }
    
    for param, range_values in param_ranges.items():
        analyzer.plot_parameter_sensitivity(
            param,
            range_values,
            test_functions[0],  # Using Sphere function
            save_path=f'results/plots/sensitivity_{param}.png'
        )
    
    # 5. 3D fitness landscapes
    print("- Plotting fitness landscapes...")
    for func in test_functions:
        analyzer.plot_3d_fitness_landscape(
            func,
            n_points=50,
            save_path=f'results/plots/landscape_{func.name.lower()}.html'
        )
    
    # 6. Hyperparameter correlation
    print("- Analyzing hyperparameter correlations...")
    analyzer.plot_hyperparameter_correlation(
        save_path='results/plots/hyperparameter_correlation.png'
    )
    
    print("\nVisualization complete! Results saved in 'results' directory.")
    print("\nSummary of best results:")
    for func_name, func_results in results.items():
        best_result = min(func_results, key=lambda x: x.best_score)
        print(f"\n{func_name}:")
        print(f"  Best optimizer: {best_result.optimizer_name}")
        print(f"  Best score: {best_result.best_score:.2e}")
        print(f"  Execution time: {best_result.execution_time:.2f}s")

if __name__ == '__main__':
    main()
