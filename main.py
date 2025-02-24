"""
main.py
-------
Demonstrates the usage of the meta-optimization framework for
both standard optimization problems and real-world applications.
"""

import numpy as np
from typing import Dict, Any, Optional
import logging
import json
from pathlib import Path

# Optimizers
from optimizers.de import DifferentialEvolutionOptimizer
from optimizers.es import EvolutionStrategyOptimizer
from optimizers.gwo import GreyWolfOptimizer
from optimizers.ml_optimizers.surrogate_optimizer import SurrogateOptimizer

# Meta-learning
from meta.meta_optimizer import MetaOptimizer
from meta.meta_learner import MetaLearner

# Analysis
from analysis.theoretical_analysis import ConvergenceAnalyzer, StabilityAnalyzer
from benchmarking.test_functions import create_test_suite
from benchmarking.statistical_analysis import run_statistical_tests

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

def setup_logging():
    """Configure logging"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('optimization.log'),
            logging.StreamHandler()
        ]
    )

def create_optimizers(dim: int, bounds: list) -> Dict[str, Any]:
    """Create optimizer instances"""
    return {
        'surrogate': SurrogateOptimizer(
            dim=dim,
            bounds=bounds,
            pop_size=30,
            n_initial=10
        ),
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
        )
    }

def optimize_problem(
    objective_func: callable,
    dim: int,
    bounds: list,
    mode: str = 'bayesian',
    max_evals: Optional[int] = None,
    context: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Optimize a problem using the meta-optimization framework.
    
    Args:
        objective_func: Function to minimize
        dim: Problem dimension
        bounds: List of (min, max) bounds for each dimension
        mode: Meta-optimizer mode ('bayesian' or 'bandit')
        max_evals: Maximum number of function evaluations
        context: Problem context for meta-learner
        
    Returns:
        Dictionary containing optimization results
    """
    # Create optimizers
    optimizers = create_optimizers(dim, bounds * dim)
    
    # Create meta-optimizer
    meta_opt = MetaOptimizer(optimizers, mode=mode)
    
    # Set up analysis
    conv_analyzer = ConvergenceAnalyzer()
    stab_analyzer = StabilityAnalyzer()
    
    # Run optimization
    logging.info(f"Starting optimization with {mode} meta-learner")
    solution = meta_opt.optimize(objective_func, context)
    final_value = objective_func(solution)
    
    # Analyze results
    convergence = conv_analyzer.analyze_convergence_rate(
        meta_opt.performance_history['optimizer'].iloc[-1],
        dim,
        list(range(len(meta_opt.performance_history))),
        meta_opt.performance_history['score'].values
    )
    
    stability = stab_analyzer.analyze_selection_stability(
        [
            {'selected_optimizer': opt} 
            for opt in meta_opt.performance_history['optimizer']
        ]
    )
    
    return {
        'solution': solution.tolist(),
        'value': float(final_value),
        'history': {
            'scores': meta_opt.performance_history['score'].tolist(),
            'optimizers': meta_opt.performance_history['optimizer'].tolist()
        },
        'analysis': {
            'convergence': convergence,
            'stability': stability
        }
    }

def run_benchmark_suite(
    output_dir: str = 'results/benchmarks',
    n_trials: int = 5,
    modes: list = ['bayesian', 'bandit']
):
    """
    Run complete benchmark suite with different meta-learning modes.
    
    Args:
        output_dir: Directory to save results
        n_trials: Number of trials per function
        modes: List of meta-learning modes to test
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Get test suite
    suite = create_test_suite()
    
    # Run benchmarks
    results = {}
    for mode in modes:
        results[mode] = {}
        for suite_name, functions in suite.items():
            results[mode][suite_name] = {}
            for func_name, func_info in functions.items():
                logging.info(f"Testing {func_name} with {mode}")
                
                trials = []
                for trial in range(n_trials):
                    try:
                        result = optimize_problem(
                            objective_func=func_info['func'],
                            dim=func_info['dim'],
                            bounds=func_info['bounds'],
                            mode=mode,
                            context={
                                'dim': func_info['dim'],
                                'multimodal': func_info['multimodal'],
                                'discrete_vars': func_info['discrete_vars']
                            }
                        )
                        trials.append(result)
                    except Exception as e:
                        logging.error(f"Trial {trial} failed: {str(e)}")
                
                results[mode][suite_name][func_name] = trials
    
    # Save results
    with open(output_dir / 'benchmark_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    # Run statistical analysis
    stats = run_statistical_tests(results)
    with open(output_dir / 'statistical_analysis.json', 'w') as f:
        json.dump(stats, f, indent=2)
    
    return results, stats

def main():
    """Main entry point demonstrating framework usage"""
    setup_logging()
    logging.info("Starting meta-optimization framework demonstration")
    
    # Run benchmarks
    results, stats = run_benchmark_suite()
    
    # Example: Optimize a specific problem
    def custom_objective(x):
        """Example objective function"""
        return np.sum(x**2)  # Simple sphere function
    
    result = optimize_problem(
        objective_func=custom_objective,
        dim=10,
        bounds=[(-5.12, 5.12)],
        mode='bayesian',
        context={'dim': 10, 'multimodal': 0}
    )
    
    logging.info(f"Optimization complete. Final value: {result['value']}")
    
    # Plot results
    plt.figure(figsize=(10, 6))
    plt.plot(result['history']['scores'])
    plt.yscale('log')
    plt.xlabel('Iteration')
    plt.ylabel('Objective Value')
    plt.title('Optimization Progress')
    plt.grid(True)
    plt.savefig('results/optimization_progress.png')
    plt.close()

if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        logging.error(f"Error in main: {str(e)}", exc_info=True)
