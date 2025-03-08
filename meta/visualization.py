"""Visualization utilities for meta-optimization and drift detection"""
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from scipy.stats import spearmanr

def plot_parameter_importance(param_importance: Dict[str, float], 
                            correlation_scores: Optional[Dict[str, float]] = None,
                            interaction_scores: Optional[Dict[Tuple[str, str], float]] = None,
                            save_path: str = None):
    """Plot comprehensive parameter importance analysis
    
    Args:
        param_importance: Dictionary of parameter importance scores
        correlation_scores: Optional dictionary of parameter-performance correlations
        interaction_scores: Optional dictionary of parameter interaction scores
        save_path: Optional path to save the plot
    """
    fig = plt.figure(figsize=(15, 10))
    
    # Create subplot grid
    gs = plt.GridSpec(2, 2, figure=fig)
    
    # 1. Main importance plot
    ax1 = fig.add_subplot(gs[0, :])
    params = list(param_importance.keys())
    scores = list(param_importance.values())
    
    # Create bar plot with gradient colors
    colors = plt.cm.viridis(np.linspace(0, 1, len(params)))
    bars = ax1.bar(params, scores, color=colors)
    
    # Customize plot
    ax1.set_title('Parameter Importance Scores', fontsize=12, pad=20)
    ax1.set_xlabel('Parameters')
    ax1.set_ylabel('Importance Score')
    ax1.tick_params(axis='x', rotation=45)
    
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}',
                ha='center', va='bottom')
    
    # 2. Correlation plot
    if correlation_scores:
        ax2 = fig.add_subplot(gs[1, 0])
        corr_params = list(correlation_scores.keys())
        corr_scores = list(correlation_scores.values())
        
        # Create horizontal bar plot
        colors = plt.cm.RdYlBu(np.linspace(0, 1, len(corr_params)))
        bars = ax2.barh(corr_params, corr_scores, color=colors)
        
        ax2.set_title('Parameter-Performance Correlations', fontsize=12)
        ax2.set_xlabel('Correlation Coefficient')
        ax2.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
        
        # Add value labels
        for bar in bars:
            width = bar.get_width()
            ax2.text(width, bar.get_y() + bar.get_height()/2.,
                    f'{width:.2f}',
                    ha='left' if width >= 0 else 'right', 
                    va='center')
    
    # 3. Interaction heatmap
    if interaction_scores:
        ax3 = fig.add_subplot(gs[1, 1])
        
        # Convert interaction scores to matrix form
        unique_params = list(set([p for pair in interaction_scores.keys() for p in pair]))
        n_params = len(unique_params)
        interaction_matrix = np.zeros((n_params, n_params))
        
        for (p1, p2), score in interaction_scores.items():
            i = unique_params.index(p1)
            j = unique_params.index(p2)
            interaction_matrix[i, j] = score
            interaction_matrix[j, i] = score
        
        # Create heatmap
        sns.heatmap(interaction_matrix, 
                   xticklabels=unique_params,
                   yticklabels=unique_params,
                   cmap='coolwarm',
                   center=0,
                   ax=ax3)
        
        ax3.set_title('Parameter Interactions', fontsize=12)
        ax3.tick_params(axis='x', rotation=45)
        ax3.tick_params(axis='y', rotation=0)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def calculate_parameter_metrics(history: List[Dict[str, Any]]) -> Tuple[Dict[str, float], Dict[str, float], Dict[Tuple[str, str], float]]:
    """Calculate comprehensive parameter importance metrics
    
    Returns:
        Tuple containing:
        - Parameter importance scores
        - Parameter-performance correlations
        - Parameter interaction scores
    """
    # Convert history to DataFrame
    df = pd.DataFrame(history)
    
    # Extract parameters
    params = [k for k in history[0].keys() if k not in ['iteration', 'score']]
    
    # 1. Calculate importance scores based on variance and performance impact
    importance_scores = {}
    scaler = MinMaxScaler()
    
    for param in params:
        # Normalize parameter values
        values = scaler.fit_transform(df[param].values.reshape(-1, 1)).flatten()
        
        # Calculate variance contribution
        variance_score = np.std(values)
        
        # Calculate performance correlation
        correlation = abs(np.corrcoef(values, df['score'])[0, 1])
        
        # Combine metrics
        importance_scores[param] = 0.4 * variance_score + 0.6 * correlation
    
    # 2. Calculate correlation scores
    correlation_scores = {}
    for param in params:
        correlation_scores[param] = spearmanr(df[param], df['score'])[0]
    
    # 3. Calculate interaction scores
    interaction_scores = {}
    for i, p1 in enumerate(params):
        for p2 in params[i+1:]:
            # Calculate interaction effect on performance
            interaction = df[p1] * df[p2]
            interaction_corr = abs(np.corrcoef(interaction, df['score'])[0, 1])
            individual_corr = max(abs(correlation_scores[p1]), abs(correlation_scores[p2]))
            
            # Interaction score is the difference between combined and individual effects
            interaction_scores[(p1, p2)] = interaction_corr - individual_corr
    
    return importance_scores, correlation_scores, interaction_scores

def plot_optimization_progress(history: List[Dict[str, Any]], save_path: str = None, show_optimizer_stats: bool = True):
    """Plot optimization progress over iterations with enhanced metrics"""
    if show_optimizer_stats:
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 15))
    else:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # Convert to DataFrame for easier manipulation
    df = pd.DataFrame(history)
    iterations = df['iteration']
    scores = df['score']
    
    # 1. Score progression plot
    ax1.plot(iterations, scores, 'b-', label='Score', alpha=0.6)
    ax1.plot(iterations, pd.Series(scores).rolling(window=5).mean(), 'r-', 
             label='Moving Average (window=5)')
    
    # Add best score line
    best_score = max(scores)
    ax1.axhline(y=best_score, color='g', linestyle='--', 
                label=f'Best Score ({best_score:.3f})')
    
    # Add improvement rate
    improvements = np.maximum.accumulate(scores)
    ax1.plot(iterations, improvements, 'k--', label='Best So Far', alpha=0.5)
    
    ax1.set_title('Optimization Progress')
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('Score')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Performance metrics plot
    window_size = 10
    metrics = {
        'Improvement Rate': [
            (improvements[i] - improvements[i-window_size])/window_size 
            if i >= window_size else 0 
            for i in range(len(improvements))
        ],
        'Score Volatility': pd.Series(scores).rolling(window=window_size).std(),
        'Relative Performance': scores / pd.Series(scores).rolling(window=window_size).mean()
    }
    
    for name, values in metrics.items():
        ax2.plot(iterations, values, label=name, alpha=0.7)
    
    ax2.set_title('Performance Metrics')
    ax2.set_xlabel('Iteration')
    ax2.set_ylabel('Metric Value')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
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
    """Create comprehensive optimization report with enhanced visualizations"""
    import os
    os.makedirs(save_dir, exist_ok=True)
    
    # Calculate parameter metrics
    importance_scores, correlation_scores, interaction_scores = calculate_parameter_metrics(meta_learner.history)
    
    # 1. Parameter importance analysis
    plot_parameter_importance(
        importance_scores,
        correlation_scores,
        interaction_scores,
        save_path=os.path.join(save_dir, 'parameter_importance.png')
    )
    
    # 2. Optimization progress
    plot_optimization_progress(
        meta_learner.history,
        save_path=os.path.join(save_dir, 'optimization_progress.png')
    )
    
    # 3. Parameter distributions
    plot_parameter_distributions(
        meta_learner.history,
        save_path=os.path.join(save_dir, 'parameter_distributions.png')
    )
    
    # Create summary report
    summary = {
        'best_score': max(h['score'] for h in meta_learner.history),
        'n_iterations': len(meta_learner.history),
        'top_parameters': dict(sorted(importance_scores.items(), key=lambda x: x[1], reverse=True)[:3]),
        'strongest_interaction': max(interaction_scores.items(), key=lambda x: x[1]),
        'performance_metrics': {
            'final_improvement_rate': meta_learner.history[-1]['score'] - meta_learner.history[-2]['score'],
            'score_volatility': np.std([h['score'] for h in meta_learner.history[-10:]]),
            'convergence_iteration': next(
                (i for i, h in enumerate(meta_learner.history) 
                 if h['score'] >= 0.95 * max(h['score'] for h in meta_learner.history)),
                len(meta_learner.history)
            )
        }
    }
    
    # Save summary report
    with open(os.path.join(save_dir, 'optimization_summary.txt'), 'w') as f:
        f.write('Optimization Summary\n')
        f.write('===================\n\n')
        f.write(f'Best Score: {summary["best_score"]:.3f}\n')
        f.write(f'Number of Iterations: {summary["n_iterations"]}\n\n')
        
        f.write('Top 3 Most Important Parameters:\n')
        for param, score in summary['top_parameters'].items():
            f.write(f'- {param}: {score:.3f}\n')
        
        f.write(f'\nStrongest Parameter Interaction:\n')
        f.write(f'- {summary["strongest_interaction"][0]}: {summary["strongest_interaction"][1]:.3f}\n\n')
        
        f.write('Performance Metrics:\n')
        f.write(f'- Final Improvement Rate: {summary["performance_metrics"]["final_improvement_rate"]:.3f}\n')
        f.write(f'- Score Volatility: {summary["performance_metrics"]["score_volatility"]:.3f}\n')
        f.write(f'- Convergence Iteration: {summary["performance_metrics"]["convergence_iteration"]}\n')
    
    return summary
