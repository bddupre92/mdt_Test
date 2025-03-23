"""
Script to run benchmarks and generate reports.
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import seaborn as sns

from benchmark_runner import BenchmarkRunner
from optimizers.aco import AntColonyOptimizer
from optimizers.gwo import GreyWolfOptimizer
from optimizers.es import EvolutionStrategyOptimizer
from optimizers.de import DifferentialEvolutionOptimizer
from data.generate_synthetic import generate_synthetic_data
from data.preprocessing import preprocess_data
from data.domain_knowledge import add_migraine_features
from utils.plot_utils import save_plot

def plot_convergence(results_df, save_path='benchmark_results'):
    """Plot convergence curves for each optimizer"""
    # Create figure
    g = sns.FacetGrid(results_df, col="function", row="dimension", hue="optimizer")
    g.map(plt.plot, "evaluations", "best_score")
    g.add_legend()
    
    # Save figure using save_plot
    fig = plt.gcf()
    filename = f"{os.path.basename(save_path)}_convergence.png"
    save_plot(fig, filename, plot_type='benchmarks')
    plt.close()

def plot_boxplots(results_df, save_path='benchmark_results'):
    """Plot boxplots of final scores"""
    # Create figure
    plt.figure(figsize=(12, 6))
    sns.boxplot(data=results_df, x="optimizer", y="best_score", hue="function")
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    # Save figure using save_plot
    fig = plt.gcf()
    filename = f"{os.path.basename(save_path)}_boxplot.png"
    save_plot(fig, filename, plot_type='benchmarks')
    plt.close()

def main():
    # Define optimizers with default parameters for 2D
    dim = 2
    bounds = [(-5.0, 5.0)] * dim
    
    optimizers = [
        AntColonyOptimizer(dim=dim, bounds=bounds),
        GreyWolfOptimizer(dim=dim, bounds=bounds),
        EvolutionStrategyOptimizer(dim=dim, bounds=bounds),
        DifferentialEvolutionOptimizer(dim=dim, bounds=bounds)
    ]
    
    # Create benchmark runner
    runner = BenchmarkRunner(
        optimizers=optimizers,
        n_runs=5,  # Use fewer runs for testing
        max_evaluations=1000  # Use fewer evaluations for testing
    )
    
    # Run theoretical benchmarks
    print("Running theoretical benchmarks...")
    
    # Run benchmarks
    results = runner.run_theoretical_benchmarks()
    
    # Save results
    os.makedirs('results/benchmarks', exist_ok=True)
    results.to_csv('results/benchmarks/benchmark_results.csv', index=False)
    
    # Plot results
    plot_convergence(results, 'results/benchmarks')
    plot_boxplots(results, 'results/benchmarks')
    
    print("Benchmarks completed. Results saved to results/benchmarks/")
    
    # 2. Run ML benchmarks with larger synthetic dataset
    print("\nGenerating larger synthetic dataset...")
    df = generate_synthetic_data(num_days=1000)
    df_clean = preprocess_data(df)
    df_feat = add_migraine_features(df_clean)
    
    # Prepare data
    features = [c for c in df_feat.columns if c not in ['migraine_occurred','severity']]
    X = df_feat[features].values
    y = df_feat['migraine_occurred'].values.astype(int)
    
    # Define RandomForest parameter space
    param_space = {
        'n_estimators': (10, 200),
        'max_depth': (3, 30),
        'min_samples_split': (2, 20),
        'max_features': (0.1, 1.0)
    }
    
    print("Running ML benchmarks...")
    ml_results = runner.run_ml_benchmarks(
        X, y, RandomForestClassifier, param_space, metric='f1'
    )
    ml_results.to_csv('benchmark_results/ml_results.csv')
    
    # 3. Compare with CMA-ES
    print("\nRunning CMA-ES comparison...")
    cmaes_results = runner.run_cmaes_comparison('sphere', dim=10)
    cmaes_results.to_csv('benchmark_results/cmaes_comparison.csv')
    
    # Print summary
    print("\nResults Summary:")
    print("\nTheoretical Benchmarks:")
    print(results.to_string())
    
    print("\nML Benchmarks:")
    print(ml_results.to_string())
    
    print("\nCMA-ES Comparison:")
    print(cmaes_results.groupby('algorithm').agg({
        'best_score': ['mean', 'std'],
        'time': ['mean', 'std']
    }).round(4).to_string())

if __name__ == '__main__':
    main()
