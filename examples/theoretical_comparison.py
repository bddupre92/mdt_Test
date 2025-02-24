"""
theoretical_comparison.py
------------------------
Comprehensive comparison of meta-learning framework against traditional optimizers.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
import logging
import pandas as pd
import warnings
from typing import Dict, List, Any
from datetime import datetime
import time

# Filter sklearn warnings
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

# Optimizers
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

def setup_logging(output_dir: str):
    """Configure logging"""
    log_file = Path(output_dir) / 'comparison.log'
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )

def create_output_dirs():
    """Create timestamped output directories"""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    base_dir = Path(f'results/comparison_{timestamp}')
    
    dirs = {
        'main': base_dir,
        'plots': base_dir / 'plots',
        'data': base_dir / 'data',
        'analysis': base_dir / 'analysis',
        'results': base_dir / 'results'
    }
    
    for dir_path in dirs.values():
        dir_path.mkdir(parents=True, exist_ok=True)
    
    return dirs

def compare_optimizers(
    test_func,
    dim,
    bounds,
    n_trials=5,
    max_evals=1000,
    population_size=50,
    dirs=None,
    problem_type=None
):
    """Compare meta-optimizer against individual optimizers"""
    
    if dirs is None:
        dirs = {}
    
    # Create individual optimizers
    de_optimizer = DifferentialEvolutionOptimizer(dim, bounds, population_size=population_size)
    es_optimizer = EvolutionStrategyOptimizer(dim, bounds, population_size=population_size)
    gwo_optimizer = GreyWolfOptimizer(dim, bounds, population_size=population_size)
    aco_optimizer = AntColonyOptimizer(dim, bounds, population_size=population_size)
    surrogate_optimizer = SurrogateOptimizer(dim, bounds, pop_size=population_size)
    
    # Create meta-optimizer with improved settings
    history_file = None
    selection_file = None
    if 'results' in dirs:
        history_file = str(dirs['results'] / 'meta_optimizer_history.json')
        selection_file = str(dirs['results'] / 'optimizer_selections.json')
        
    meta_optimizer = MetaOptimizer(
        dim=dim,
        bounds=bounds,
        optimizers={
            'de': de_optimizer,
            'es': es_optimizer,
            'gwo': gwo_optimizer,
            'aco': aco_optimizer,
            'surrogate': surrogate_optimizer
        },
        history_file=history_file,
        selection_file=selection_file
    )
    
    # Run trials
    results = {
        'individual': {},
        'meta': {'bayesian': []}
    }
    
    # Test individual optimizers
    for name, optimizer in {
        'de': de_optimizer,
        'es': es_optimizer,
        'gwo': gwo_optimizer,
        'aco': aco_optimizer,
        'surrogate': surrogate_optimizer
    }.items():
        logging.info(f"Testing {name}")
        results['individual'][name] = []
        
        for trial in range(n_trials):
            optimizer.reset()
            start_time = time.time()
            solution = optimizer.optimize(test_func, max_evals=max_evals)
            runtime = time.time() - start_time
            
            if solution is not None:
                value = test_func(solution)
                logging.info(f"  Trial {trial + 1}: {value:.2e}")
                
                history = optimizer.history
                if isinstance(history, pd.DataFrame):
                    history = history.to_dict('records')
                
                results['individual'][name].append({
                    'solution': solution.tolist(),  # Convert to list
                    'value': value,
                    'runtime': runtime,
                    'history': history,
                    'n_evals': getattr(optimizer, 'n_evals', max_evals)
                })
    
    # Test meta-optimizer
    logging.info("Testing meta-optimizer")
    
    for trial in range(n_trials):
        meta_optimizer.reset()
        start_time = time.time()
        solution = meta_optimizer.optimize(test_func, max_evals=max_evals)
        runtime = time.time() - start_time
        
        if solution is not None:
            value = test_func(solution)
            logging.info(f"  Trial {trial + 1}: {value:.2e}")
            
            # Store trial results
            results['meta']['bayesian'].append({
                'solution': solution.tolist(),
                'value': value,
                'runtime': runtime,
                'history': meta_optimizer.optimization_history,
                'total_evaluations': meta_optimizer.total_evaluations
            })
    
    return results

def plot_convergence(results: dict, output_dir: Path, title: str):
    """Create detailed convergence plot"""
    plt.figure(figsize=(12, 8))
    
    # Plot individual optimizers
    for name, trials in results['individual'].items():
        if not trials:  # Skip if no successful trials
            continue
            
        all_scores = {}
        for trial in trials:
            if isinstance(trial['history'], list):
                for idx, record in enumerate(trial['history']):
                    if isinstance(record, dict) and 'score' in record:
                        score = record['score']
                    else:
                        score = record  # Assume it's a direct score value
                        
                    if idx not in all_scores:
                        all_scores[idx] = []
                    all_scores[idx].append(score)
        
        if all_scores:
            iterations = sorted(all_scores.keys())
            mean_scores = [np.mean(all_scores[i]) for i in iterations]
            std_scores = [np.std(all_scores[i]) for i in iterations]
            
            plt.plot(iterations, mean_scores, label=name.upper(), alpha=0.8)
            plt.fill_between(
                iterations,
                [m - s for m, s in zip(mean_scores, std_scores)],
                [m + s for m, s in zip(mean_scores, std_scores)],
                alpha=0.2
            )
    
    # Plot meta-optimizer
    if 'meta' in results:
        for mode, trials in results['meta'].items():
            if not trials:  # Skip if no successful trials
                continue
                
            all_scores = {}
            for trial in trials:
                if isinstance(trial['history'], list):
                    for idx, record in enumerate(trial['history']):
                        if isinstance(record, dict) and 'score' in record:
                            score = record['score']
                        else:
                            score = record  # Assume it's a direct score value
                            
                        if idx not in all_scores:
                            all_scores[idx] = []
                        all_scores[idx].append(score)
            
            if all_scores:
                iterations = sorted(all_scores.keys())
                mean_scores = [np.mean(all_scores[i]) for i in iterations]
                std_scores = [np.std(all_scores[i]) for i in iterations]
                
                plt.plot(iterations, mean_scores, label='META-OPTIMIZER', linewidth=2)
                plt.fill_between(
                    iterations,
                    [m - s for m, s in zip(mean_scores, std_scores)],
                    [m + s for m, s in zip(mean_scores, std_scores)],
                    alpha=0.2
                )
    
    plt.yscale('log')
    plt.xlabel('Iteration', fontsize=12)
    plt.ylabel('Objective Value', fontsize=12)
    plt.title(f'Convergence Plot - {title}', fontsize=14)
    plt.grid(True, which='both', linestyle='--', alpha=0.5)
    plt.legend(fontsize=10)
    plt.tight_layout()
    
    plt.savefig(output_dir / f'convergence_{title.lower().replace(" ", "_")}.png', dpi=300)
    plt.close()

def plot_optimizer_selection(results: dict, output_dir: Path, title: str):
    """Plot meta-optimizer selection patterns"""
    if 'meta' not in results or not results['meta']['bayesian']:
        logging.warning(f"No meta-optimizer results for {title}")
        return
        
    selections = []
    for trial in results['meta']['bayesian']:
        if isinstance(trial['history'], list):
            for record in trial['history']:
                if isinstance(record, dict) and 'optimizer' in record:
                    selections.append(record['optimizer'])
    
    if not selections:
        logging.warning(f"No optimizer selections recorded for {title}")
        return
    
    # Count selections
    selection_counts = pd.Series(selections).value_counts()
    
    plt.figure(figsize=(10, 6))
    selection_counts.plot(kind='bar')
    plt.title(f'Optimizer Selection Pattern - {title}', fontsize=14)
    plt.xlabel('Optimizer', fontsize=12)
    plt.ylabel('Times Selected', fontsize=12)
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    plt.savefig(output_dir / f'selection_pattern_{title.lower().replace(" ", "_")}.png', dpi=300)
    plt.close()

def create_summary_table(results: dict, output_dir: Path, title: str, success_threshold: float = 1e-4):
    """Create summary statistics table"""
    summary = {
        'Optimizer': [],
        'Mean': [],
        'Std': [],
        'Best': [],
        'Worst': [],
        'Success Rate': [],
        'Avg Iterations': []
    }
    
    # Individual optimizers
    for name, trials in results['individual'].items():
        if not trials:  # Skip if no successful trials
            continue
            
        values = [trial['value'] for trial in trials]
        iterations = [trial['n_evals'] for trial in trials]
        success_rate = sum(1 for v in values if v < success_threshold) / len(values)
        
        summary['Optimizer'].append(name.upper())
        summary['Mean'].append(np.mean(values))
        summary['Std'].append(np.std(values))
        summary['Best'].append(np.min(values))
        summary['Worst'].append(np.max(values))
        summary['Success Rate'].append(success_rate)
        summary['Avg Iterations'].append(np.mean(iterations))
    
    # Meta-optimizer
    if 'meta' in results:
        for mode, trials in results['meta'].items():
            if not trials:  # Skip if no successful trials
                continue
                
            values = [trial['value'] for trial in trials]
            iterations = [trial['total_evaluations'] for trial in trials]  # Use stored total_evaluations
            success_rate = sum(1 for v in values if v < success_threshold) / len(values)
            
            summary['Optimizer'].append('META-OPTIMIZER')
            summary['Mean'].append(np.mean(values))
            summary['Std'].append(np.std(values))
            summary['Best'].append(np.min(values))
            summary['Worst'].append(np.max(values))
            summary['Success Rate'].append(success_rate)
            summary['Avg Iterations'].append(np.mean(iterations))
    
    # Create DataFrame and format
    df = pd.DataFrame(summary)
    df = df.round({
        'Mean': 6,
        'Std': 6,
        'Best': 6,
        'Worst': 6,
        'Success Rate': 1,
        'Avg Iterations': 1
    })
    
    # Save to file
    df.to_csv(output_dir / f'summary_{title.lower().replace(" ", "_")}.csv', index=False)
    
    return df

def save_results(results: dict, dirs: dict):
    """Save results to file"""
    for problem, result in results.items():
        with open(dirs['data'] / f'{problem}_results.json', 'w') as f:
            json.dump(result, f, indent=2)

def main():
    """Main entry point"""
    # Setup directories
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_dir = Path('results') / f'comparison_{timestamp}'
    
    dirs = {
        'main': base_dir,
        'plots': base_dir / 'plots',
        'data': base_dir / 'data',
        'analysis': base_dir / 'analysis',
        'results': base_dir
    }
    
    for dir_path in dirs.values():
        dir_path.mkdir(parents=True, exist_ok=True)
        
    setup_logging(dirs['main'])
    
    logging.info("Starting comprehensive comparison...")
    
    # Get test suite
    suite = create_test_suite()
    
    # Run comparison on each function type
    all_results = {}
    summary_tables = {}
    
    for suite_name, functions in suite.items():
        logging.info(f"\nTesting {suite_name} functions")
        suite_results = {}
        
        for func_name, func_info in functions.items():
            logging.info(f"Testing {func_name}")
            
            # Run comparison
            results = compare_optimizers(
                test_func=func_info['func'],
                dim=func_info['dim'],
                bounds=func_info['bounds'] * func_info['dim'],
                n_trials=5,
                max_evals=1000,
                dirs=dirs,
                problem_type=func_name
            )
            
            # Save results
            save_results({func_name: results}, dirs)
            
            # Create visualizations
            plot_convergence(results, dirs['plots'], f"{func_name}")
            plot_optimizer_selection(results, dirs['plots'], f"{func_name}")
            
            # Create summary table
            summary_tables[func_name] = create_summary_table(
                results, 
                dirs['analysis'],
                func_name
            )
            
            suite_results[func_name] = results
            
            # Log summary
            logging.info(f"\nResults Summary for {func_name}:")
            print(summary_tables[func_name].to_string())
            
        all_results[suite_name] = suite_results
    
    # Save overall results
    with open(dirs['data'] / 'all_results.json', 'w') as f:
        json.dump(all_results, f, indent=2)
    
    # Print selection analysis
    if Path(dirs['results'] / 'optimizer_selections.json').exists():
        selection_tracker = SelectionTracker(str(dirs['results'] / 'optimizer_selections.json'))
        stats = selection_tracker.get_selection_stats()
        if not stats.empty:
            logging.info("\nOptimizer Selection Analysis:")
            logging.info("\nSelection Statistics:")
            logging.info(stats.to_string())
            
            for problem in all_results.keys():
                correlations = selection_tracker.get_feature_correlations(problem)
                if correlations:
                    logging.info(f"\nFeature Correlations for {problem}:")
                    for opt, feat_corrs in correlations.items():
                        logging.info(f"\n{opt}:")
                        for feat, corr in feat_corrs.items():
                            logging.info(f"  {feat}: {corr:.3f}")
    
    logging.info(f"\nAll results saved to {dirs['main']}")
    logging.info("Comparison complete!")

if __name__ == '__main__':
    main()
