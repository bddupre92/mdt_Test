"""
visualize_optimizers.py
---------------------
Script to run optimization comparison and generate visualizations.
"""

from typing import Dict, List, Any
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import sys
from os.path import dirname, abspath
sys.path.append(dirname(dirname(abspath(__file__))))

from optimizers.de import DifferentialEvolutionOptimizer
from optimizers.gwo import GreyWolfOptimizer
from optimizers.es import EvolutionStrategyOptimizer
from optimizers.aco import AntColonyOptimizer
from benchmarking.test_functions import TEST_FUNCTIONS
from benchmarking.cec_functions import create_cec_suite
from benchmarking.statistical_analysis import StatisticalAnalyzer
from benchmarking.sota_comparison import SOTAComparison
from visualization.optimizer_analysis import OptimizerAnalyzer

def create_optimizers(dim: int, bounds: List[tuple]) -> Dict[str, Any]:
    """Create dictionary of optimizers to compare"""
    return {
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

def main():
    # Create output directories
    Path('results/plots').mkdir(parents=True, exist_ok=True)
    
    # Problem parameters
    dim = 30
    bounds = [(-100, 100)] * dim
    
    # Create optimizers
    optimizers = create_optimizers(dim, bounds)
    
    # Create analyzer
    analyzer = OptimizerAnalyzer(optimizers)
    
    # Get test functions (both classical and CEC)
    test_functions = {}
    
    # Add classical test functions
    for name, func_factory in TEST_FUNCTIONS.items():
        test_functions[name] = func_factory(dim, bounds)
    
    # Add CEC test functions
    cec_funcs = create_cec_suite(dim, bounds)
    test_functions.update(cec_funcs)
    
    print("Running optimization comparison...\n")
    
    # Run comparison
    results = analyzer.run_comparison(
        test_functions=test_functions,
        n_runs=30,
        record_convergence=True
    )
    
    print("\nGenerating visualizations...\n")
    
    # 1. Basic visualization
    print("1. Plotting convergence comparisons...")
    analyzer.plot_convergence_comparison()
    
    # 2. Performance heatmap
    print("\n2. Creating performance heatmaps...")
    analyzer.plot_performance_heatmap()
    
    # 3. Parameter adaptation analysis
    print("\n3. Analyzing parameter adaptation...")
    for optimizer_name in optimizers:
        if 'Adaptive' in optimizer_name:
            for func_name in test_functions:
                analyzer.plot_parameter_adaptation(optimizer_name, func_name)
    
    # 4. Diversity analysis
    print("\n4. Analyzing population diversity...")
    for optimizer_name in optimizers:
        for func_name in test_functions:
            analyzer.plot_diversity_analysis(optimizer_name, func_name)
    
    # 5. Statistical analysis
    print("\n5. Performing statistical analysis...")
    stat_analyzer = StatisticalAnalyzer()
    stat_results = stat_analyzer.compare_algorithms(results)
    stat_results.to_csv('results/statistical_analysis.csv', index=False)
    
    # 6. SOTA comparison
    print("\n6. Comparing with state-of-the-art variants...")
    sota_comparison = SOTAComparison()
    
    # Create reference results (you would normally load these from published results)
    reference_results = {
        func_name: [opt.best_score for opt in results[func_name].values()]
        for func_name in results
    }
    
    # Compare with reference
    comparison_results = sota_comparison.compare_with_reference(
        {name: results[name] for name in test_functions},
        reference_results
    )
    
    # Generate comparison report
    comparison_report = sota_comparison.generate_comparison_report(comparison_results)
    comparison_report.to_csv('results/sota_comparison.csv', index=False)
    
    # 7. Create interactive HTML report
    print("\n7. Creating interactive HTML report...")
    analyzer.create_html_report(
        statistical_results=stat_results,
        sota_results=comparison_report
    )
    
    print("\nVisualization complete! Results saved in 'results/plots' directory")
    print("Statistical analysis saved as 'results/statistical_analysis.csv'")
    print("SOTA comparison saved as 'results/sota_comparison.csv'")
    print("Interactive report saved as 'results/optimization_report.html'")

if __name__ == '__main__':
    main()
