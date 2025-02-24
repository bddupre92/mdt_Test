"""
quick_test.py
------------
Quick test of the meta-optimization framework with minimal settings.
"""

import numpy as np
import logging
from pathlib import Path

# Core components
from meta.meta_optimizer import MetaOptimizer
from optimizers.de import DifferentialEvolutionOptimizer
from optimizers.gwo import GreyWolfOptimizer

def setup_logging():
    """Configure minimal logging"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(message)s'  # Simplified format
    )

def sphere(x: np.ndarray) -> float:
    """Simple sphere function"""
    return np.sum(x**2)

def quick_test():
    """Run a minimal test"""
    # Setup
    setup_logging()
    logging.info("Starting quick test...")
    
    # Problem setup - minimal dimension and evaluations
    dim = 2
    bounds = [(-1, 1)]  # Small search space
    
    # Create optimizers with small populations
    optimizers = {
        'de': DifferentialEvolutionOptimizer(
            dim=dim,
            bounds=bounds * dim,
            population_size=10  # Small population
        ),
        'gwo': GreyWolfOptimizer(
            dim=dim,
            bounds=bounds * dim,
            population_size=10  # Small population
        )
    }
    
    # Create meta-optimizer
    meta_opt = MetaOptimizer(optimizers, mode='bayesian')
    
    try:
        # Run optimization
        logging.info("Running optimization...")
        solution = meta_opt.optimize(
            sphere,
            context={
                'dim': dim,
                'multimodal': 0,
                'discrete_vars': 0
            }
        )
        final_value = sphere(solution)
        
        # Log results
        logging.info("\nResults:")
        logging.info(f"Final value: {final_value:.2e}")
        logging.info("\nOptimizer usage:")
        
        # Count optimizer usage
        optimizer_counts = meta_opt.performance_history['optimizer'].value_counts()
        for opt, count in optimizer_counts.items():
            logging.info(f"- {opt}: {count} times")
        
        # Save minimal results
        results_dir = Path('results/quick_test')
        results_dir.mkdir(parents=True, exist_ok=True)
        
        np.save(
            results_dir / 'convergence.npy',
            meta_opt.performance_history['score'].values
        )
        
        logging.info("\nQuick test completed successfully!")
        
    except Exception as e:
        logging.error(f"Error in quick test: {str(e)}")
        raise

if __name__ == '__main__':
    quick_test()
