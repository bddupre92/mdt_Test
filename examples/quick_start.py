#!/usr/bin/env python3
"""
Quick Start Example for Meta-Optimizer Framework

This script demonstrates a simple use case of the meta-optimizer framework,
showing how to run optimization with different optimizers on benchmark functions.
"""

import numpy as np
import matplotlib.pyplot as plt
from meta_optimizer.meta.meta_optimizer import MetaOptimizer
from meta_optimizer.benchmark.test_functions import ClassicalTestFunctions
from meta_optimizer.visualization.optimizer_analysis import OptimizationResult, OptimizerAnalyzer
from meta_optimizer.optimizers.optimizer_factory import OptimizerFactory

def main():
    """Run a simple optimization example."""
    print("Meta-Optimizer Framework Quick Start Example")
    print("-------------------------------------------")
    
    # Set up parameters
    dim = 10
    max_evals = 1000
    n_runs = 3
    
    # Create benchmark function (Sphere)
    bounds = [(-5.12, 5.12)] * dim
    sphere_func = ClassicalTestFunctions.sphere
    
    # Create optimizers
    print("\nInitializing optimizers...")
    optimizers = OptimizerFactory.create_optimizers(dim=dim, bounds=bounds)
    
    # Run optimization with each optimizer
    print("\nRunning optimization...")
    all_results = {}
    optimization_results = {}
    
    for name, optimizer in optimizers.items():
        print(f"  Running {name}...")
        results = []
        convergence_curves = []
        
        for i in range(n_runs):
            start_time = np.datetime64('now')
            try:
                # Try with record_history parameter
                x_best, f_best = optimizer.optimize(sphere_func, max_evals=max_evals, record_history=True)
                has_history = True
            except (TypeError, ValueError):
                # If that fails, try without it
                x_best, f_best = optimizer.optimize(sphere_func, max_evals=max_evals)
                has_history = False
                
            end_time = np.datetime64('now')
            execution_time = (end_time - start_time) / np.timedelta64(1, 's')
            
            results.append((x_best, f_best))
            if has_history and hasattr(optimizer, 'history'):
                convergence_curves.append(optimizer.history)
        
        best_result = min(results, key=lambda x: x[1])
        all_results[name] = {
            'best_solution': best_result[0],
            'best_value': best_result[1],
            'all_runs': results
        }
        
        # Store formatted results for visualization
        if convergence_curves:
            for i, curve in enumerate(convergence_curves):
                # Create result with correct attributes
                result = OptimizationResult(
                    optimizer_name=name,
                    function_name="Sphere",
                    best_solution=results[i][0],
                    best_score=results[i][1],
                    convergence_curve=curve,
                    execution_time=execution_time,
                    hyperparameters=optimizer.get_parameters() if hasattr(optimizer, 'get_parameters') else {}
                )
                # Add the fitness_history attribute manually
                result.fitness_history = curve
                
                if "Sphere" not in optimization_results:
                    optimization_results["Sphere"] = {}
                if name not in optimization_results["Sphere"]:
                    optimization_results["Sphere"][name] = []
                optimization_results["Sphere"][name].append(result)
    
    # Run meta-optimizer
    print("\nRunning meta-optimizer...")
    # The MetaOptimizer handles selection of optimizers internally
    meta_opt = MetaOptimizer(optimizers=optimizers, dim=dim, bounds=bounds)
    
    meta_results = []
    meta_convergence_curves = []
    
    for i in range(n_runs):
        start_time = np.datetime64('now')
        
        # Run meta-optimizer - it returns only the solution, not a score
        x_best = meta_opt.optimize(sphere_func, max_evals=max_evals)
        # Evaluate the solution to get the score
        f_best = sphere_func(x_best)
            
        end_time = np.datetime64('now')
        execution_time = (end_time - start_time) / np.timedelta64(1, 's')
        
        meta_results.append((x_best, f_best))
        if hasattr(meta_opt, 'convergence_curve'):
            meta_convergence_curves.append(meta_opt.convergence_curve)
    
    best_meta_result = min(meta_results, key=lambda x: x[1])
    all_results['meta_optimizer'] = {
        'best_solution': best_meta_result[0],
        'best_value': best_meta_result[1],
        'all_runs': meta_results
    }
    
    # Store formatted results for meta-optimizer
    if meta_convergence_curves:
        for i, curve in enumerate(meta_convergence_curves):
            # Create result with correct attributes
            result = OptimizationResult(
                optimizer_name="Meta-Optimizer",
                function_name="Sphere",
                best_solution=meta_results[i][0],
                best_score=meta_results[i][1],
                convergence_curve=curve,
                execution_time=execution_time,
                hyperparameters=meta_opt.get_parameters() if hasattr(meta_opt, 'get_parameters') else {}
            )
            # Add the fitness_history attribute manually
            result.fitness_history = curve
            
            if "Sphere" not in optimization_results:
                optimization_results["Sphere"] = {}
            if "Meta-Optimizer" not in optimization_results["Sphere"]:
                optimization_results["Sphere"]["Meta-Optimizer"] = []
            optimization_results["Sphere"]["Meta-Optimizer"].append(result)
    
    # Display results
    print("\nResults Summary:")
    print("---------------")
    for name, result in all_results.items():
        print(f"{name}:")
        print(f"  Best value: {result['best_value']:.6f}")
    
    # Use OptimizerAnalyzer for visualization if we have results
    if optimization_results:
        print("\nGenerating visualization with OptimizerAnalyzer...")
        try:
            analyzer = OptimizerAnalyzer(optimizers)
            
            # Add our results to the analyzer
            analyzer.results = optimization_results
            
            # Generate and save plots
            analyzer.plot_convergence_comparison()
            plt.savefig('convergence_comparison.png')
            print("Visualization saved as 'convergence_comparison.png'")
        except Exception as e:
            print(f"Unable to use OptimizerAnalyzer: {e}")
            
            # Fallback to our custom plot
            print("Using fallback visualization...")
            plt.figure(figsize=(10, 6))
            
            for optimizer_name in all_results.keys():
                if optimizer_name in optimization_results.get("Sphere", {}):
                    results = optimization_results["Sphere"][optimizer_name]
                    # Get the convergence curves from all runs
                    curves = [r.fitness_history for r in results if hasattr(r, 'fitness_history') and r.fitness_history]
                    
                    if curves:
                        # Find the minimum length to align curves
                        min_length = min(len(curve) for curve in curves)
                        # Align and compute mean
                        aligned_curves = [curve[:min_length] for curve in curves]
                        mean_curve = np.mean(aligned_curves, axis=0)
                        plt.plot(range(len(mean_curve)), mean_curve, label=optimizer_name)
            
            plt.title('Convergence Comparison on Sphere Function')
            plt.xlabel('Iteration')
            plt.ylabel('Objective Value')
            plt.legend()
            plt.yscale('log')  # Log scale for better visualization
            plt.grid(True)
            plt.tight_layout()
            plt.savefig('convergence_comparison_fallback.png')
            print("Fallback visualization saved as 'convergence_comparison_fallback.png'")
    else:
        # Simple plotting if we don't have proper convergence curves
        print("\nGenerating simple visualization...")
        plt.figure(figsize=(10, 6))
        
        for name, result in all_results.items():
            values = [run[1] for run in result['all_runs']]
            plt.bar(name, np.mean(values), yerr=np.std(values))
        
        plt.title('Performance Comparison on Sphere Function')
        plt.ylabel('Mean Objective Value (with Std Dev)')
        plt.grid(True, axis='y')
        plt.tight_layout()
        plt.savefig('performance_comparison.png')
        print("Simple visualization saved as 'performance_comparison.png'")
    
    print("\nExample completed successfully!")

if __name__ == "__main__":
    main()
