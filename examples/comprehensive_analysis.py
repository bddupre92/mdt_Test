"""
Comprehensive analysis of MetaOptimizer performance.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
import logging
from tqdm.auto import tqdm
import pandas as pd
from datetime import datetime
from typing import Dict, List, Any, Tuple
import warnings

# Filter warnings
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

# Import optimizers
from optimizers.de import DifferentialEvolutionOptimizer
from optimizers.es import EvolutionStrategyOptimizer
from optimizers.gwo import GreyWolfOptimizer
from optimizers.aco import AntColonyOptimizer
from optimizers.ml_optimizers.surrogate_optimizer import SurrogateOptimizer

# Meta-learning
from meta.meta_optimizer import MetaOptimizer

# Analysis
from analysis.theoretical_analysis import ConvergenceAnalyzer, StabilityAnalyzer
from benchmarking.test_functions import create_test_suite
from benchmarking.statistical_analysis import run_statistical_tests
from visualization.optimizer_analysis import OptimizerAnalyzer

def setup_output_dirs() -> Dict[str, Path]:
    """Create output directories."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_dir = Path('results') / f'comprehensive_analysis_{timestamp}'
    
    dirs = {
        'main': base_dir,
        'plots': base_dir / 'plots',
        'data': base_dir / 'data',
        'analysis': base_dir / 'analysis'
    }
    
    for dir_path in dirs.values():
        dir_path.mkdir(parents=True, exist_ok=True)
        
    return dirs

def setup_logging(output_dir: Path):
    """Configure logging."""
    log_file = output_dir / 'analysis.log'
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )

def run_benchmarks(test_func: callable, dim: int, bounds: List[Tuple[float, float]], 
                  n_trials: int = 5, max_evals: int = 1000) -> Dict[str, Any]:
    """Run benchmarks for all optimizers."""
    # Create bounds list for each dimension
    full_bounds = [(bounds[0][0], bounds[0][1]) for _ in range(dim)]
    
    # Create base optimizers
    base_optimizers = {
        'de': DifferentialEvolutionOptimizer(dim=dim, bounds=full_bounds),
        'es': EvolutionStrategyOptimizer(dim=dim, bounds=full_bounds),
        'gwo': GreyWolfOptimizer(dim=dim, bounds=full_bounds),
        'aco': AntColonyOptimizer(dim=dim, bounds=full_bounds),
        'surrogate': SurrogateOptimizer(dim=dim, bounds=full_bounds)
    }
    
    # Create meta-optimizer with base optimizers
    meta_opt = MetaOptimizer(
        dim=dim,
        bounds=full_bounds,
        optimizers={
            'de': base_optimizers['de'],
            'es': base_optimizers['es'],
            'gwo': base_optimizers['gwo'],
            'aco': base_optimizers['aco'],
            'surrogate': base_optimizers['surrogate']
        }
    )
    
    # Add meta-optimizer to the set
    optimizers = {**base_optimizers, 'meta': meta_opt}
    
    results = {}
    # Progress bar for optimizers
    optimizer_pbar = tqdm(optimizers.items(), desc="Testing optimizers", leave=False)
    for name, opt in optimizer_pbar:
        optimizer_pbar.set_description(f"Testing {name.upper()}")
        opt_results = []
        histories = []
        
        # Progress bar for trials
        trial_pbar = tqdm(range(n_trials), desc=f"Trial", leave=False)
        for trial in trial_pbar:
            # Reset optimizer state
            if hasattr(opt, 'reset'):
                opt.reset()
            
            # Run optimization with history tracking
            best_score = float('inf')
            history = []
            
            # Progress bar for evaluations
            eval_pbar = tqdm(range(max_evals), desc="Evaluations", leave=False)
            for eval_count in eval_pbar:
                if hasattr(opt, 'step'):
                    solution = opt.step(test_func)
                else:
                    solution = opt.optimize(test_func, max_evals=1)
                
                if isinstance(solution, np.ndarray):
                    score = test_func(solution)
                else:
                    score = solution.best_score if hasattr(solution, 'best_score') else test_func(solution)
                
                best_score = min(best_score, score)
                history.append((eval_count, best_score))
                eval_pbar.set_postfix({'best': f'{best_score:.2e}'})
            
            opt_results.append(best_score)
            histories.append(history)
            trial_pbar.set_postfix({'best': f'{best_score:.2e}'})
                
        results[name] = {
            'scores': opt_results,
            'histories': histories,
            'mean': np.mean(opt_results),
            'std': np.std(opt_results),
            'best': np.min(opt_results),
            'worst': np.max(opt_results),
            'optimizer': opt  # Store optimizer instance for analysis
        }
        
    return results

def plot_convergence(results: Dict[str, Any], output_dir: Path, title: str):
    """Plot convergence curves with confidence intervals."""
    plt.figure(figsize=(12, 8))
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(results)))
    for (name, data), color in zip(results.items(), colors):
        if 'histories' in data and data['histories']:
            # Extract evaluation counts and scores
            eval_counts = []
            scores = []
            for history in data['histories']:
                counts, hist_scores = zip(*history)
                eval_counts.append(counts)
                scores.append(hist_scores)
            
            # Convert to numpy arrays for calculations
            eval_counts = np.array(eval_counts)
            scores = np.array(scores)
            
            # Calculate mean and std across trials
            mean_scores = np.mean(scores, axis=0)
            std_scores = np.std(scores, axis=0)
            
            # Plot mean line with confidence interval
            plt.plot(eval_counts[0], mean_scores, label=name.upper(), color=color, linewidth=2)
            plt.fill_between(eval_counts[0], 
                           mean_scores - std_scores,
                           mean_scores + std_scores,
                           alpha=0.2, color=color)
    
    plt.yscale('log')
    plt.xlabel('Function Evaluations')
    plt.ylabel('Best Score (log scale)')
    plt.title(f'Convergence Analysis - {title}')
    plt.legend()
    plt.grid(True, which="both", ls="-", alpha=0.2)
    plt.tight_layout()
    plt.savefig(output_dir / f'convergence_{title.lower()}.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_performance_distribution(results: Dict[str, Any], output_dir: Path, title: str):
    """Plot performance distribution across trials."""
    plt.figure(figsize=(12, 6))
    
    data = []
    labels = []
    for name, res in results.items():
        data.append(res['scores'])
        labels.extend([name.upper()] * len(res['scores']))
    
    plt.boxplot(data, tick_labels=[name.upper() for name in results.keys()])
    plt.yscale('log')
    plt.ylabel('Best Score (log scale)')
    plt.title(f'Performance Distribution - {title}')
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_dir / f'distribution_{title.lower()}.png')
    plt.close()

def plot_feature_importance(meta_opt: MetaOptimizer, output_dir: Path):
    """Plot feature importance for optimizer selection."""
    if not hasattr(meta_opt, 'selection_tracker'):
        return
        
    # Get feature correlations from selection tracker
    correlations = meta_opt.selection_tracker.get_feature_correlations()
    if not correlations:
        return
        
    # Prepare data for plotting
    features = []
    importances = []
    
    for opt_name, feat_corrs in correlations.items():
        for feat, corr in feat_corrs.items():
            if feat not in features:
                features.append(feat)
                importances.append(abs(corr))
            else:
                idx = features.index(feat)
                importances[idx] = max(importances[idx], abs(corr))
    
    if not features:
        return
        
    plt.figure(figsize=(10, 6))
    y_pos = np.arange(len(features))
    plt.barh(y_pos, importances)
    plt.yticks(y_pos, features)
    plt.xlabel('Maximum Absolute Correlation')
    plt.title('Feature Importance in Optimizer Selection')
    plt.tight_layout()
    plt.savefig(output_dir / 'feature_importance.png')
    plt.close()

def plot_optimizer_selection(meta_opt: MetaOptimizer, output_dir: Path):
    """Plot optimizer selection patterns."""
    if not hasattr(meta_opt, 'selection_tracker'):
        return
        
    # Get selection statistics
    stats = meta_opt.selection_tracker.get_selection_stats()
    if stats is None:
        return
        
    plt.figure(figsize=(10, 6))
    if isinstance(stats, pd.Series):
        stats.plot(kind='bar')
    else:
        # Handle numpy array or dict
        if isinstance(stats, dict):
            names = list(stats.keys())
            values = list(stats.values())
        else:
            names = [f'Optimizer {i}' for i in range(len(stats))]
            values = stats
            
        plt.bar(range(len(values)), values)
        plt.xticks(range(len(names)), names, rotation=45)
    
    plt.title('Optimizer Selection Frequency')
    plt.xlabel('Optimizer')
    plt.ylabel('Selection Count')
    plt.tight_layout()
    plt.savefig(output_dir / 'optimizer_selection.png')
    plt.close()

def analyze_meta_optimizer_behavior(meta_opt: MetaOptimizer, output_dir: Path):
    """Analyze and visualize meta-optimizer behavior."""
    if not hasattr(meta_opt, 'history'):
        return
        
    # Plot exploration vs exploitation
    if hasattr(meta_opt, 'exploration_rates'):
        plt.figure(figsize=(10, 6))
        plt.plot(meta_opt.exploration_rates)
        plt.title('Exploration Rate Over Time')
        plt.xlabel('Iteration')
        plt.ylabel('Exploration Rate')
        plt.grid(True)
        plt.savefig(output_dir / 'exploration_rate.png')
        plt.close()
    
    # Plot confidence scores
    if hasattr(meta_opt, 'confidence_scores'):
        plt.figure(figsize=(10, 6))
        for opt, scores in meta_opt.confidence_scores.items():
            plt.plot(scores, label=opt)
        plt.title('Optimizer Confidence Scores')
        plt.xlabel('Iteration')
        plt.ylabel('Confidence Score')
        plt.legend()
        plt.grid(True)
        plt.savefig(output_dir / 'confidence_scores.png')
        plt.close()

def main():
    """Main entry point."""
    # Setup
    dirs = setup_output_dirs()
    setup_logging(dirs['main'])
    logging.info("Starting comprehensive analysis...")
    
    # Get test suite
    suite = create_test_suite()
    
    # Run analysis for each function type
    for suite_name, functions in tqdm(suite.items(), desc="Testing function suites"):
        logging.info(f"\nAnalyzing {suite_name} functions")
        
        for func_name, func_info in tqdm(functions.items(), desc="Testing functions", leave=False):
            logging.info(f"Testing {func_name}")
            
            # Run benchmarks
            results = run_benchmarks(
                test_func=func_info['func'],
                dim=func_info['dim'],
                bounds=func_info['bounds']
            )
            
            # Generate visualizations
            plot_convergence(results, dirs['plots'], func_name)
            plot_performance_distribution(results, dirs['plots'], func_name)
            
            # Save results
            with open(dirs['data'] / f'{func_name}_results.json', 'w') as f:
                json.dump({k: {
                    'mean': float(v['mean']),
                    'std': float(v['std']),
                    'best': float(v['best']),
                    'worst': float(v['worst'])
                } for k, v in results.items()}, f, indent=2)
            
            # Analyze meta-optimizer
            if 'meta' in results:
                meta_opt = results['meta']['optimizer']
                plot_feature_importance(meta_opt, dirs['plots'])
                plot_optimizer_selection(meta_opt, dirs['plots'])
                analyze_meta_optimizer_behavior(meta_opt, dirs['plots'])
            
            # Log summary
            logging.info("\nResults Summary:")
            for name, res in results.items():
                logging.info(f"{name:15} Mean: {res['mean']:.2e} ± {res['std']:.2e}")
    
    logging.info(f"\nAnalysis complete! Results saved to {dirs['main']}")

if __name__ == '__main__':
    main()
