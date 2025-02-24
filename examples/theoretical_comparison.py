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
        'analysis': base_dir / 'analysis'
    }
    
    for dir_path in dirs.values():
        dir_path.mkdir(parents=True, exist_ok=True)
    
    return dirs

def compare_optimizers(
    test_func: callable,
    dim: int,
    bounds: list,
    n_trials: int = 5,
    max_evals: int = 1000,
    population_size: int = 30,
    success_threshold: float = 1e-4  # Relaxed threshold
):
    """Compare meta-optimizer against individual optimizers"""
    
    # Create individual optimizers
    optimizers = {
        'de': DifferentialEvolutionOptimizer(
            dim=dim,
            bounds=bounds,
            population_size=population_size,
            max_evals=max_evals
        ),
        'es': EvolutionStrategyOptimizer(
            dim=dim,
            bounds=bounds,
            population_size=population_size,
            max_evals=max_evals
        ),
        'gwo': GreyWolfOptimizer(
            dim=dim,
            bounds=bounds,
            population_size=population_size,
            max_evals=max_evals
        ),
        'aco': AntColonyOptimizer(
            dim=dim,
            bounds=bounds,
            population_size=population_size,
            max_evals=max_evals
        )
    }
    
    # Add surrogate optimizer with optimized settings
    optimizers['surrogate'] = SurrogateOptimizer(
        dim=dim,
        bounds=bounds,
        pop_size=20,  # Fixed size
        n_initial=5,  # Minimal initial samples
        noise=1e-6,
        length_scale=1.0,
        exploitation_ratio=0.5,
        max_gp_size=100  # Limit GP model size
    )
    
    # Create meta-optimizer with improved settings
    meta_opt = MetaOptimizer(
        optimizers.copy(),
        mode='bayesian',
        gp_kwargs={
            'alpha': 1e-3,
            'n_restarts_optimizer': 1,
            'normalize_y': True
        }
    )
    
    # Run trials
    results = {
        'individual': {},
        'meta': {'bayesian': []}
    }
    
    # Test individual optimizers
    for name, opt in optimizers.items():
        logging.info(f"Testing {name}")
        trials = []
        
        for trial in range(n_trials):
            try:
                opt.reset()  # Reset optimizer state
                solution = opt.optimize(
                    test_func,
                    max_evals=max_evals,
                    record_history=True
                )
                if solution is None:
                    logging.error(f"Trial {trial} failed for {name}: No solution found")
                    continue
                    
                final_value = test_func(solution)
                
                history = opt.get_performance_history()
                if history is None or len(history) == 0:
                    history = pd.DataFrame({
                        'iteration': [0],
                        'score': [final_value]
                    })
                
                trials.append({
                    'solution': solution.tolist(),
                    'value': float(final_value),
                    'history': {
                        'scores': history['score'].tolist(),
                        'iterations': history['iteration'].tolist()
                    }
                })
                logging.info(f"  Trial {trial + 1}: {final_value:.2e}")
            except Exception as e:
                logging.error(f"Trial {trial} failed for {name}: {str(e)}")
                continue
    
        results['individual'][name] = trials
    
    # Test meta-optimizer
    logging.info("Testing meta-optimizer")
    meta_trials = []
    
    for trial in range(n_trials):
        try:
            solution = meta_opt.optimize(
                test_func,
                context={
                    'dim': dim,
                    'multimodal': 0,
                    'discrete_vars': 0
                }
            )
            
            if solution is None:
                logging.error(f"Trial {trial} failed for meta-optimizer: No solution found")
                continue
                
            final_value = test_func(solution)
            
            history = meta_opt.get_performance_history()
            if history is None or len(history) == 0:
                history = pd.DataFrame({
                    'iteration': [0],
                    'score': [final_value]
                })
            
            meta_trials.append({
                'solution': solution.tolist(),
                'value': float(final_value),
                'history': {
                    'scores': history['score'].tolist(),
                    'iterations': history['iteration'].tolist(),
                    'optimizers': history['optimizer'].tolist() if 'optimizer' in history.columns else []
                }
            })
            logging.info(f"  Trial {trial + 1}: {final_value:.2e}")
        except Exception as e:
            logging.error(f"Trial {trial} failed for meta-optimizer: {str(e)}")
            continue
    
    results['meta']['bayesian'] = meta_trials
    
    return results

def plot_convergence(results: dict, output_dir: Path, title: str):
    """Create detailed convergence plot"""
    plt.figure(figsize=(12, 8))
    
    # Color scheme
    colors = sns.color_palette("husl", n_colors=len(results['individual']) + 1)
    color_dict = {name: color for name, color in zip(results['individual'].keys(), colors)}
    color_dict['meta'] = colors[-1]
    
    # Plot individual optimizers
    for name, trials in results['individual'].items():
        if not trials:  # Skip if no successful trials
            continue
            
        # Collect all histories and align them by iteration
        all_scores = {}
        for trial in trials:
            history = trial['history']
            for iter_idx, score in zip(history['iterations'], history['scores']):
                if iter_idx not in all_scores:
                    all_scores[iter_idx] = []
                all_scores[iter_idx].append(score)
        
        # Convert to arrays and compute statistics
        iterations = sorted(all_scores.keys())
        mean_scores = [np.mean(all_scores[i]) for i in iterations]
        std_scores = [np.std(all_scores[i]) if len(all_scores[i]) > 1 else 0 for i in iterations]
        
        plt.plot(iterations, mean_scores, label=name.upper(), color=color_dict[name], linewidth=2)
        plt.fill_between(
            iterations,
            np.array(mean_scores) - np.array(std_scores),
            np.array(mean_scores) + np.array(std_scores),
            alpha=0.2,
            color=color_dict[name]
        )
    
    # Plot meta-optimizer
    if results['meta']['bayesian']:
        # Collect all histories and align them by iteration
        all_scores = {}
        for trial in results['meta']['bayesian']:
            history = trial['history']
            for iter_idx, score in zip(history['iterations'], history['scores']):
                if iter_idx not in all_scores:
                    all_scores[iter_idx] = []
                all_scores[iter_idx].append(score)
        
        # Convert to arrays and compute statistics
        iterations = sorted(all_scores.keys())
        mean_scores = [np.mean(all_scores[i]) for i in iterations]
        std_scores = [np.std(all_scores[i]) if len(all_scores[i]) > 1 else 0 for i in iterations]
        
        plt.plot(iterations, mean_scores, label='META-OPTIMIZER', 
                color=color_dict['meta'], linewidth=3, linestyle='--')
        plt.fill_between(
            iterations,
            np.array(mean_scores) - np.array(std_scores),
            np.array(mean_scores) + np.array(std_scores),
            alpha=0.2,
            color=color_dict['meta']
        )
    
    plt.yscale('log')
    plt.xlabel('Iteration', fontsize=12)
    plt.ylabel('Objective Value', fontsize=12)
    plt.title(f'Convergence Comparison - {title}', fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True, which="both", ls="-", alpha=0.2)
    plt.tight_layout()
    
    # Save plot
    plt.savefig(output_dir / f'convergence_{title.lower().replace(" ", "_")}.png', dpi=300)
    plt.close()

def plot_optimizer_selection(results: dict, output_dir: Path, title: str):
    """Plot meta-optimizer selection patterns"""
    if not results['meta']['bayesian']:
        logging.warning(f"No meta-optimizer results for {title}, skipping selection pattern plot")
        return
        
    selections = []
    for trial in results['meta']['bayesian']:
        if 'optimizers' in trial['history']:
            selections.extend(trial['history']['optimizers'])
    
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
        iterations = [len(trial['history']['scores']) for trial in trials]
        success_rate = sum(1 for v in values if v < success_threshold) / len(values)
        
        summary['Optimizer'].append(name.upper())
        summary['Mean'].append(np.mean(values))
        summary['Std'].append(np.std(values))
        summary['Best'].append(np.min(values))
        summary['Worst'].append(np.max(values))
        summary['Success Rate'].append(success_rate)
        summary['Avg Iterations'].append(np.mean(iterations))
    
    # Meta-optimizer
    if results['meta']['bayesian']:
        meta_values = [trial['value'] for trial in results['meta']['bayesian']]
        meta_iterations = [len(trial['history']['scores']) for trial in results['meta']['bayesian']]
        meta_success = sum(1 for v in meta_values if v < success_threshold) / len(meta_values)
        
        summary['Optimizer'].append('META-OPTIMIZER')
        summary['Mean'].append(np.mean(meta_values))
        summary['Std'].append(np.std(meta_values))
        summary['Best'].append(np.min(meta_values))
        summary['Worst'].append(np.max(meta_values))
        summary['Success Rate'].append(meta_success)
        summary['Avg Iterations'].append(np.mean(meta_iterations))
    
    # Create DataFrame and save
    df = pd.DataFrame(summary)
    df.to_csv(output_dir / f'summary_{title.lower().replace(" ", "_")}.csv', index=False)
    
    return df

def main():
    """Main entry point"""
    # Create output directories
    dirs = create_output_dirs()
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
                population_size=30
            )
            
            # Save results
            with open(dirs['data'] / f'{func_name}_results.json', 'w') as f:
                json.dump(results, f, indent=2)
            
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
    
    logging.info(f"\nAll results saved to {dirs['main']}")
    logging.info("Comparison complete!")

if __name__ == '__main__':
    main()
