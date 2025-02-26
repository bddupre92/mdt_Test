"""Visualization utilities for meta-optimization and drift detection"""
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from typing import Dict, List, Any
import pandas as pd

def plot_parameter_importance(param_importance: Dict[str, float], save_path: str = None):
    """Plot parameter importance scores"""
    plt.figure(figsize=(10, 6))
    params = list(param_importance.keys())
    scores = list(param_importance.values())
    
    # Create bar plot
    bars = plt.bar(params, scores)
    
    # Customize plot
    plt.title('Parameter Importance Scores', fontsize=12)
    plt.xlabel('Parameters')
    plt.ylabel('Importance Score')
    plt.xticks(rotation=45)
    
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}',
                ha='center', va='bottom')
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.close()

def plot_optimization_progress(history: List[Dict[str, Any]], save_path: str = None):
    """Plot optimization progress over iterations"""
    iterations = [h['iteration'] for h in history]
    scores = [h['score'] for h in history]
    
    plt.figure(figsize=(10, 6))
    
    # Plot score progression
    plt.plot(iterations, scores, 'b-', label='Score')
    plt.plot(iterations, pd.Series(scores).rolling(window=5).mean(), 'r-', 
             label='Moving Average (window=5)')
    
    # Add best score line
    best_score = max(scores)
    plt.axhline(y=best_score, color='g', linestyle='--', 
                label=f'Best Score ({best_score:.3f})')
    
    # Customize plot
    plt.title('Optimization Progress', fontsize=12)
    plt.xlabel('Iteration')
    plt.ylabel('Score')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.close()

def plot_parameter_distributions(history: List[Dict[str, Any]], save_path: str = None):
    """Plot distribution of parameter values tried during optimization"""
    # Extract parameter values
    params_data = {}
    for h in history:
        for param, value in h['params'].items():
            if param not in params_data:
                params_data[param] = []
            params_data[param].append(value)
    
    # Create subplot grid
    n_params = len(params_data)
    n_cols = 2
    n_rows = (n_params + 1) // 2
    
    plt.figure(figsize=(12, 4*n_rows))
    
    for i, (param, values) in enumerate(params_data.items(), 1):
        plt.subplot(n_rows, n_cols, i)
        
        if isinstance(values[0], (int, float)):
            # Numerical parameter
            sns.histplot(values, kde=True)
            plt.axvline(values[np.argmax([h['score'] for h in history])], 
                       color='r', linestyle='--', label='Best Value')
        else:
            # Categorical parameter
            value_counts = pd.Series(values).value_counts()
            sns.barplot(x=value_counts.index, y=value_counts.values)
        
        plt.title(f'{param} Distribution')
        plt.xticks(rotation=45)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.close()

def create_optimization_report(meta_learner, save_dir: str = 'plots'):
    """Create comprehensive optimization report with all visualizations"""
    import os
    os.makedirs(save_dir, exist_ok=True)
    
    # Get optimization statistics
    stats = meta_learner.get_optimization_stats()
    
    # Create visualizations
    plot_parameter_importance(
        stats['param_importance'],
        os.path.join(save_dir, 'parameter_importance.png')
    )
    
    plot_optimization_progress(
        meta_learner.eval_history,
        os.path.join(save_dir, 'optimization_progress.png')
    )
    
    plot_parameter_distributions(
        meta_learner.eval_history,
        os.path.join(save_dir, 'parameter_distributions.png')
    )
    
    # Create summary text file
    with open(os.path.join(save_dir, 'optimization_summary.txt'), 'w') as f:
        f.write("Optimization Summary\n")
        f.write("===================\n\n")
        f.write(f"Best Score: {stats['best_score']:.3f}\n")
        f.write(f"Total Evaluations: {stats['evaluations']}\n\n")
        
        f.write("Parameter Importance:\n")
        for param, importance in stats['param_importance'].items():
            f.write(f"  {param}: {importance:.3f}\n")
        
        f.write("\nBest Configuration:\n")
        for param, value in meta_learner.get_best_configuration().items():
            f.write(f"  {param}: {value}\n")
