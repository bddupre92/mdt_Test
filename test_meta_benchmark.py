"""
Test script for verifying Meta-Optimizer works in benchmark system
"""
import logging
import numpy as np
from typing import Dict, Any

# Set up logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('test_meta_benchmark')

# Import necessary modules
from meta.meta_optimizer import MetaOptimizer
from visualization.optimizer_analysis import OptimizerAnalyzer
from optimizers.differential_evolution import DifferentialEvolutionOptimizer
from optimizers.evolution_strategy import EvolutionStrategyOptimizer

# Test function
def sphere(x):
    """Simple sphere function for testing"""
    return np.sum(x**2)

def setup_test_function(dim=10):
    """Create a test function with specified dimension"""
    bounds = [(-5, 5)] * dim
    return sphere, bounds

def create_optimizers(dim, bounds):
    """Create optimizers for benchmark comparison"""
    # Create base optimizers
    de_opt = DifferentialEvolutionOptimizer(dim=dim, bounds=bounds)
    es_opt = EvolutionStrategyOptimizer(dim=dim, bounds=bounds)
    
    # Create dictionary of optimizers
    optimizers = {
        'DE': de_opt,
        'ES': es_opt
    }
    
    # Create Meta-Optimizer
    meta_opt = MetaOptimizer(
        dim=dim,
        bounds=bounds,
        optimizers=optimizers,
        verbose=True
    )
    
    # Create final optimizers dictionary including Meta-Optimizer
    all_optimizers = optimizers.copy()
    all_optimizers['Meta-Optimizer'] = meta_opt
    
    return all_optimizers

def run_benchmark():
    """Run the benchmark test"""
    logger.info("Starting benchmark test")
    
    # Setup test environment
    dim = 10
    func, bounds = setup_test_function(dim)
    
    # Create optimizers
    optimizers = create_optimizers(dim, bounds)
    
    # Create optimizer analyzer
    analyzer = OptimizerAnalyzer(optimizers)
    
    # Run comparison with test function
    test_functions = {'sphere': func}
    
    logger.info("Running benchmark comparison")
    results = analyzer.run_comparison(
        test_functions,
        n_runs=3,  # Small number for quick testing
        max_evals=200,  # Limit evaluations to speed up test
        record_convergence=True
    )
    
    # Log results
    for func_name, func_results in results.items():
        logger.info(f"Results for {func_name}:")
        for optimizer_name, opt_results in func_results.items():
            scores = [r.best_score for r in opt_results]
            mean_score = np.mean(scores)
            logger.info(f"  {optimizer_name}: Mean score = {mean_score:.6f}")
    
    # Check that Meta-Optimizer completed successfully
    meta_results = results.get('sphere', {}).get('Meta-Optimizer', [])
    if meta_results:
        logger.info(f"Meta-Optimizer completed {len(meta_results)} runs successfully")
        return True
    else:
        logger.error("Meta-Optimizer failed to complete benchmark")
        return False

if __name__ == "__main__":
    success = run_benchmark()
    if success:
        print("\nBenchmark SUCCESS: Meta-Optimizer completed successfully")
    else:
        print("\nBenchmark FAILURE: Meta-Optimizer had issues")
