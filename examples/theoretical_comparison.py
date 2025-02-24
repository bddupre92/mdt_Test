"""
theoretical_comparison.py
------------------------
Demonstrates the theoretical superiority of the meta-learning framework
compared to individual optimizers.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
import logging

from meta.meta_optimizer import MetaOptimizer
from optimizers.de import DifferentialEvolutionOptimizer
from optimizers.es import EvolutionStrategyOptimizer
from optimizers.gwo import GreyWolfOptimizer
from optimizers.ml_optimizers.surrogate_optimizer import SurrogateOptimizer

from benchmarking.test_functions import create_test_suite
from benchmarking.statistical_analysis import run_statistical_tests
from analysis.theoretical_analysis import ConvergenceAnalyzer, StabilityAnalyzer

def setup_logging():
    """Configure logging"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

def compare_optimizers(
    test_func: callable,
    dim: int,
    bounds: list,
    n_trials: int = 5,
    max_evals: int = 1000
):
    """Compare meta-optimizer against individual optimizers"""
    
    # Create individual optimizers
    optimizers = {
        'de': DifferentialEvolutionOptimizer(
            dim=dim,
            bounds=bounds,
            population_size=30
        ),
        'es': EvolutionStrategyOptimizer(
            dim=dim,
            bounds=bounds,
            population_size=30
        ),
        'gwo': GreyWolfOptimizer(
            dim=dim,
            bounds=bounds,
            population_size=30
        ),
        'surrogate': SurrogateOptimizer(
            dim=dim,
            bounds=bounds,
            pop_size=30,
            n_initial=10
        )
    }
    
    # Create meta-optimizer
    meta_opt = MetaOptimizer(optimizers.copy(), mode='bayesian')
    
    # Run trials
    results = {
        'individual': {},
        'meta': {'bayesian': {}}
    }
    
    # Test individual optimizers
    for name, opt in optimizers.items():
        logging.info(f"Testing {name}")
        trials = []
        
        for trial in range(n_trials):
            try:
                history = []
                best_solution = opt.optimize(
                    test_func,
                    max_evals=max_evals,
                    record_history=True
                )
                final_value = test_func(best_solution)
                
                trials.append({
                    'solution': best_solution.tolist(),
                    'value': float(final_value),
                    'history': {
                        'scores': opt.performance_history['score'].tolist(),
                        'optimizers': [name] * len(opt.performance_history)
                    }
                })
            except Exception as e:
                logging.error(f"Trial {trial} failed for {name}: {str(e)}")
        
        results['individual'][name] = trials
    
    # Test meta-optimizer
    logging.info("Testing meta-optimizer")
    meta_trials = []
    for trial in range(n_trials):
        try:
            solution = meta_opt.optimize(
                test_func,
                max_evals=max_evals,
                record_history=True
            )
            final_value = test_func(solution)
            
            meta_trials.append({
                'solution': solution.tolist(),
                'value': float(final_value),
                'history': {
                    'scores': meta_opt.performance_history['score'].tolist(),
                    'optimizers': meta_opt.performance_history['optimizer'].tolist()
                }
            })
        except Exception as e:
            logging.error(f"Trial {trial} failed for meta-optimizer: {str(e)}")
    
    results['meta']['bayesian'] = {'test_func': meta_trials}
    
    return results

def analyze_results(results: dict, output_dir: str = 'results/theoretical'):
    """Analyze and visualize comparison results"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Run statistical analysis
    stats = run_statistical_tests(results)
    
    # Save results
    with open(output_dir / 'comparison_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    with open(output_dir / 'statistical_analysis.json', 'w') as f:
        json.dump(stats, f, indent=2)
    
    # Create visualization
    plt.figure(figsize=(12, 6))
    
    # Plot convergence curves
    for approach, approach_results in results.items():
        if approach == 'individual':
            for opt_name, trials in approach_results.items():
                histories = [trial['history']['scores'] for trial in trials]
                mean_history = np.mean(histories, axis=0)
                std_history = np.std(histories, axis=0)
                x = np.arange(len(mean_history))
                
                plt.plot(x, mean_history, label=opt_name)
                plt.fill_between(
                    x,
                    mean_history - std_history,
                    mean_history + std_history,
                    alpha=0.2
                )
        else:
            for mode, suite_results in approach_results.items():
                for func_name, trials in suite_results.items():
                    histories = [trial['history']['scores'] for trial in trials]
                    mean_history = np.mean(histories, axis=0)
                    std_history = np.std(histories, axis=0)
                    x = np.arange(len(mean_history))
                    
                    plt.plot(x, mean_history, label=f'meta-{mode}', linewidth=2)
                    plt.fill_between(
                        x,
                        mean_history - std_history,
                        mean_history + std_history,
                        alpha=0.2
                    )
    
    plt.yscale('log')
    plt.xlabel('Iteration')
    plt.ylabel('Objective Value')
    plt.title('Convergence Comparison')
    plt.legend()
    plt.grid(True)
    plt.savefig(output_dir / 'convergence_comparison.png')
    plt.close()
    
    return stats

def main():
    """Main entry point"""
    setup_logging()
    
    # Get test suite
    suite = create_test_suite()
    
    # Run comparison on each function type
    all_results = {}
    
    for suite_name, functions in suite.items():
        logging.info(f"\nTesting {suite_name} functions")
        suite_results = {}
        
        for func_name, func_info in functions.items():
            logging.info(f"Testing {func_name}")
            
            results = compare_optimizers(
                test_func=func_info['func'],
                dim=func_info['dim'],
                bounds=func_info['bounds'] * func_info['dim'],
                n_trials=5,
                max_evals=1000
            )
            
            suite_results[func_name] = results
            
            # Analyze results
            stats = analyze_results(
                results,
                output_dir=f'results/theoretical/{suite_name}/{func_name}'
            )
            
            # Log summary
            logging.info("\nResults Summary:")
            for approach in results:
                if approach == 'individual':
                    for opt_name, trials in results[approach].items():
                        values = [trial['value'] for trial in trials]
                        logging.info(
                            f"{opt_name}: "
                            f"mean={np.mean(values):.2e} ± {np.std(values):.2e}"
                        )
                else:
                    for mode in results[approach]:
                        for func, trials in results[approach][mode].items():
                            values = [trial['value'] for trial in trials]
                            logging.info(
                                f"meta-{mode}: "
                                f"mean={np.mean(values):.2e} ± {np.std(values):.2e}"
                            )
        
        all_results[suite_name] = suite_results
    
    # Save overall results
    with open('results/theoretical/all_results.json', 'w') as f:
        json.dump(all_results, f, indent=2)

if __name__ == '__main__':
    main()
