"""Example optimization using the meta-optimizer framework."""
import numpy as np
import logging
from typing import Dict, Any
import matplotlib.pyplot as plt
from pathlib import Path

from meta.meta_optimizer import MetaOptimizer
from optimizers.differential_evolution import DifferentialEvolutionOptimizer
from optimizers.evolution_strategy import EvolutionStrategyOptimizer


def setup_logging():
    """Configure logging."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )


def sphere(x: np.ndarray) -> float:
    """Sphere test function."""
    return np.sum(x**2)


def rastrigin(x: np.ndarray) -> float:
    """Rastrigin test function."""
    return 10 * len(x) + np.sum(x**2 - 10 * np.cos(2 * np.pi * x))


def plot_convergence(optimizer_results: Dict[str, Any], save_path: str):
    """Plot convergence curves."""
    plt.figure(figsize=(10, 6))
    
    # Plot best solution's convergence curve
    if 'convergence' in optimizer_results['best_solution']:
        plt.plot(
            optimizer_results['best_solution']['convergence'],
            label='Best Solution',
            linewidth=2
        )
    
    # Plot all optimizers' convergence curves
    for result in optimizer_results['history']:
        if 'convergence' in result:
            plt.plot(result['convergence'], alpha=0.7)
            
    plt.yscale('log')
    plt.xlabel('Iteration')
    plt.ylabel('Best Score')
    plt.title('Convergence Curves')
    plt.legend()
    plt.grid(True)
    
    # Save plot
    plt.savefig(save_path)
    plt.close()


def main():
    """Run optimization example."""
    # Setup logging
    setup_logging()
    logger = logging.getLogger(__name__)
    
    # Create results directory
    results_dir = Path('results')
    results_dir.mkdir(exist_ok=True)
    
    # Problem parameters
    dim = 30
    bounds = [(-5.12, 5.12)] * dim
    max_evals = 10000
    
    # Initialize optimizers
    optimizers = {
        'DifferentialEvolution': DifferentialEvolutionOptimizer(
            dim=dim,
            bounds=bounds,
            population_size=50,
            F=0.8,
            CR=0.5
        ),
        'EvolutionStrategy': EvolutionStrategyOptimizer(
            dim=dim,
            bounds=bounds,
            population_size=100,
            mu=20,
            sigma=0.1
        )
    }
    
    # Initialize meta-optimizer
    meta_opt = MetaOptimizer(
        dim=dim,
        bounds=bounds,
        optimizers=optimizers,
        history_file=str(results_dir / 'history.json'),
        selection_file=str(results_dir / 'selection.json')
    )
    
    # Test functions
    test_functions = {
        'sphere': sphere,
        'rastrigin': rastrigin
    }
    
    # Run optimization for each test function
    for func_name, func in test_functions.items():
        logger.info(f"Optimizing {func_name} function")
        
        # Run meta-optimizer
        results = meta_opt.optimize(
            objective_func=func,
            max_evals=max_evals,
            context={'problem_type': func_name}
        )
        
        # Log results
        logger.info(f"Best score: {results['best_score']:.3e}")
        logger.info(f"Total evaluations: {sum(r['evaluations'] for r in results['history'])}")
        
        # Plot convergence
        plot_convergence(
            results,
            str(results_dir / f'{func_name}_convergence.png')
        )
        
        # Reset for next function
        meta_opt.reset()


if __name__ == '__main__':
    main()
