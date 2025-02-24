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
from optimizers.es import EvolutionStrategy
from optimizers.de import DifferentialEvolutionOptimizer
from data.generate_synthetic import generate_synthetic_data
from data.preprocessing import preprocess_data
from data.domain_knowledge import add_migraine_features

def plot_convergence(results_df, save_path='benchmark_results'):
    """Plot convergence curves for each optimizer"""
    os.makedirs(save_path, exist_ok=True)
    
    g = sns.FacetGrid(results_df, col="function", row="dimension", hue="optimizer")
    g.map(plt.plot, "evaluations", "best_score")
    g.add_legend()
    plt.savefig(f"{save_path}/convergence.png")
    plt.close()

def plot_boxplots(results_df, save_path='benchmark_results'):
    """Plot boxplots of final scores"""
    os.makedirs(save_path, exist_ok=True)
    
    plt.figure(figsize=(12, 6))
    sns.boxplot(data=results_df, x="optimizer", y="best_score", hue="function")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f"{save_path}/scores_boxplot.png")
    plt.close()

def main():
    # Define optimizers
    optimizers = {
        'ACO': AntColonyOptimizer,
        'GWO': GreyWolfOptimizer,
        'ES': EvolutionStrategy,
        'DE': DifferentialEvolutionOptimizer
    }
    
    # 1. Run theoretical benchmarks
    print("Running theoretical benchmarks...")
    runner = BenchmarkRunner(optimizers, dimensions=[2, 10, 30], n_trials=30)
    results_df, stats = runner.run_benchmarks()
    
    # Save results
    results_df.to_csv('benchmark_results/theoretical_results.csv')
    stats.to_csv('benchmark_results/theoretical_stats.csv')
    
    # Generate plots
    plot_convergence(results_df)
    plot_boxplots(results_df)
    
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
    print(stats.to_string())
    
    print("\nML Benchmarks:")
    print(ml_results.to_string())
    
    print("\nCMA-ES Comparison:")
    print(cmaes_results.groupby('algorithm').agg({
        'best_score': ['mean', 'std'],
        'time': ['mean', 'std']
    }).round(4).to_string())

if __name__ == '__main__':
    main()
