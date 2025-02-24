"""
visualize_optimizers.py
----------------------
Example script showing how to use the optimizer visualization framework.
"""

import numpy as np
from optimizers.aco import AntColonyOptimizer
from optimizers.gwo import GreyWolfOptimizer
from optimizers.es import EvolutionStrategyOptimizer
from optimizers.de import DifferentialEvolutionOptimizer
from benchmarking.test_functions import TestFunction, ClassicalTestFunctions
from visualization.optimizer_analysis import OptimizerAnalyzer
from meta.meta_optimizer import MetaOptimizer
import os
import matplotlib.pyplot as plt

def main():
    # Create output directory
    os.makedirs('results/plots', exist_ok=True)
    
    # Define dimensions and bounds for all problems
    dim = 30  # Using higher dimension for real-world scenario
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
    
    # Create optimizers with both standard and adaptive variants
    optimizers = {
        'DE (Standard)': DifferentialEvolutionOptimizer(
            dim=dim, bounds=bounds, adaptive=False
        ),
        'DE (Adaptive)': DifferentialEvolutionOptimizer(
            dim=dim, bounds=bounds, adaptive=True
        ),
        'GWO (Standard)': GreyWolfOptimizer(
            dim=dim, bounds=bounds, adaptive=False
        ),
        'GWO (Adaptive)': GreyWolfOptimizer(
            dim=dim, bounds=bounds, adaptive=True
        ),
        'ES (Standard)': EvolutionStrategyOptimizer(
            dim=dim, bounds=bounds, adaptive=False
        ),
        'ES (Adaptive)': EvolutionStrategyOptimizer(
            dim=dim, bounds=bounds, adaptive=True
        ),
        'ACO (Standard)': AntColonyOptimizer(
            dim=dim, bounds=bounds, adaptive=False
        ),
        'ACO (Adaptive)': AntColonyOptimizer(
            dim=dim, bounds=bounds, adaptive=True
        )
    }
    
    # Create meta-optimizer
    meta_opt = MetaOptimizer(
        optimizers=optimizers,  # Pass the dictionary directly
        mode='bayesian'  # Using Bayesian optimization for selection
    )
    
    # Create analyzer
    analyzer = OptimizerAnalyzer(
        optimizers=list(optimizers.values()),  # Pass the dictionary values as a list
        test_functions=test_functions,
        meta_optimizer=meta_opt
    )
    
    # Run optimization comparison
    print("Running optimization comparison...")
    results = analyzer.run_comparison(n_runs=5, record_convergence=True)
    
    # Generate visualizations
    print("\nGenerating visualizations...")
    
    # 1. Convergence Comparison
    print("\n1. Plotting convergence comparisons...")
    for func in test_functions:
        analyzer.plot_convergence_comparison(
            function_name=func.name,
            log_scale=True
        )
        plt.savefig(f'results/plots/convergence_{func.name.lower()}.png')
        plt.clf()
    
    # 2. Performance Heatmap
    print("\n2. Creating performance heatmaps...")
    for metric in ['best_score', 'time', 'success_rate']:
        analyzer.plot_performance_heatmap(metric=metric)
        plt.savefig(f'results/plots/heatmap_{metric}.png')
        plt.clf()
    
    # 3. Parameter Adaptation Analysis
    print("\n3. Analyzing parameter adaptation...")
    for opt_name in optimizers:
        if 'Adaptive' in opt_name:
            for func in test_functions:
                analyzer.plot_parameter_adaptation(
                    optimizer_name=opt_name,
                    function_name=func.name
                )
                plt.savefig(f'results/plots/params_{opt_name.lower()}_{func.name.lower()}.png')
                plt.clf()
    
    # 4. Diversity Analysis
    print("\n4. Analyzing population diversity...")
    for opt_name in optimizers:
        for func in test_functions:
            analyzer.plot_diversity_analysis(
                optimizer_name=opt_name,
                function_name=func.name
            )
            plt.savefig(f'results/plots/diversity_{opt_name.lower()}_{func.name.lower()}.png')
            plt.clf()
    
    # 5. Meta-Optimizer Analysis
    print("\n5. Analyzing meta-optimizer performance...")
    analyzer.plot_meta_optimizer_analysis()
    plt.savefig('results/plots/meta_optimizer_analysis.png')
    plt.clf()
    
    # 6. Generate HTML Report
    print("\n6. Creating interactive HTML report...")
    analyzer.create_summary_report(output_file='results/optimization_report.html')
    
    print("\nVisualization complete! Results saved in 'results/plots' directory")
    print("Interactive report saved as 'results/optimization_report.html'")

if __name__ == '__main__':
    main()
