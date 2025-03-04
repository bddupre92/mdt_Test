"""
optimizer_explainability_example.py
---------------------------------
Example script demonstrating optimizer explainability features
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

# Import necessary modules
from optimizers.optimizer_factory import OptimizerFactory
from explainability.optimizer_explainer import OptimizerExplainer
from utils.plot_utils import save_plot
from benchmarking.test_functions import ClassicalTestFunctions

def run_optimizer_explainability_example():
    """Run optimizer explainability example"""
    print("Running optimizer explainability example...")
    
    # Create optimizers
    factory = OptimizerFactory()
    optimizers = {
        'differential_evolution': factory.create_optimizer('differential_evolution', dim=10, bounds=[(-5, 5)] * 10),
        'evolution_strategy': factory.create_optimizer('evolution_strategy', dim=10, bounds=[(-5, 5)] * 10),
        'ant_colony': factory.create_optimizer('ant_colony', dim=10, bounds=[(-5, 5)] * 10)
    }
    
    # Test functions
    test_functions = {
        'sphere': ClassicalTestFunctions.sphere,
        'rosenbrock': ClassicalTestFunctions.rosenbrock,
        'rastrigin': ClassicalTestFunctions.rastrigin,
        'ackley': ClassicalTestFunctions.ackley
    }
    
    # Run optimizers on different functions
    results = {}
    for optimizer_name, optimizer in optimizers.items():
        optimizer_results = {}
        for func_name, func in test_functions.items():
            print(f"Running {optimizer_name} on {func_name}...")
            optimizer.reset()  # Reset optimizer state
            result = optimizer.run(func, max_evals=500)
            optimizer_results[func_name] = {
                'best_score': result['best_score'],
                'evaluations': result['evaluations'],
                'runtime': result['runtime'],
                'optimizer_state': optimizer.get_state()
            }
        results[optimizer_name] = optimizer_results
    
    # Create explainers for each optimizer
    explainers = {}
    for optimizer_name, optimizer in optimizers.items():
        explainer = OptimizerExplainer(optimizer)
        explainers[optimizer_name] = explainer
    
    # Generate explanations
    explanations = {}
    for optimizer_name, explainer in explainers.items():
        print(f"Generating explanation for {optimizer_name}...")
        explanation = explainer.explain()
        explanations[optimizer_name] = explanation
    
    # Generate plots for each optimizer
    for optimizer_name, explainer in explainers.items():
        print(f"Generating plots for {optimizer_name}...")
        
        # Get available plot types
        plot_types = explainer.supported_plot_types
        
        # Generate each plot type
        for plot_type in plot_types:
            try:
                print(f"  Generating {plot_type} plot...")
                fig = explainer.plot(plot_type)
                
                # Save plot
                filename = f"{optimizer_name}_{plot_type}.png"
                save_plot(fig, filename, plot_type='explainability')
                plt.close(fig)
            except Exception as e:
                print(f"  Error generating {plot_type} plot: {str(e)}")
    
    # Compare parameter sensitivity across optimizers
    print("\nParameter Sensitivity Analysis:")
    for optimizer_name, explanation in explanations.items():
        print(f"\n{optimizer_name}:")
        feature_importance = explanation['feature_importance']
        if feature_importance:
            for param, sensitivity in sorted(feature_importance.items(), key=lambda x: x[1], reverse=True):
                print(f"  {param}: {sensitivity:.4f}")
        else:
            print("  No parameter sensitivity data available")
    
    # Generate comparative plots
    print("\nGenerating comparative plots...")
    
    # Compare convergence curves
    fig, ax = plt.subplots(figsize=(10, 6))
    for optimizer_name, explanation in explanations.items():
        convergence_curve = explanation['convergence_curve']
        iterations = range(1, len(convergence_curve) + 1)
        ax.plot(iterations, convergence_curve, label=optimizer_name)
    
    ax.set_title("Convergence Comparison")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Objective Value")
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.legend()
    
    # Save comparative plot
    save_plot(fig, "optimizer_comparison_convergence.png", plot_type='explainability')
    plt.close(fig)
    
    # Compare landscape ruggedness detection
    fig, ax = plt.subplots(figsize=(10, 6))
    
    optimizer_names = []
    ruggedness_values = []
    local_optima_counts = []
    
    for optimizer_name, explanation in explanations.items():
        if explanation['landscape_ruggedness'] is not None:
            optimizer_names.append(optimizer_name)
            ruggedness_values.append(explanation['landscape_ruggedness'])
            local_optima_counts.append(explanation['local_optima_count'])
    
    if optimizer_names:
        x = np.arange(len(optimizer_names))
        width = 0.35
        
        ax.bar(x - width/2, ruggedness_values, width, label='Landscape Ruggedness')
        ax.bar(x + width/2, local_optima_counts, width, label='Local Optima Count')
        
        ax.set_title("Landscape Analysis Comparison")
        ax.set_xlabel("Optimizer")
        ax.set_xticks(x)
        ax.set_xticklabels(optimizer_names)
        ax.legend()
        
        # Save comparative plot
        save_plot(fig, "optimizer_comparison_landscape.png", plot_type='explainability')
    
    plt.close(fig)
    
    print("\nOptimizer explainability example completed!")
    print("Plots saved in results/explainability directory")

if __name__ == "__main__":
    run_optimizer_explainability_example()
