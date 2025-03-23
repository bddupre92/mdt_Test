#!/usr/bin/env python3
"""
Baseline Verification Tests
--------------------------
Tests to verify the functionality of the baseline comparison framework
and its components.
"""

import os
import sys
import logging
import numpy as np
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add parent directory to path
parent_dir = str(Path(__file__).parent.parent.absolute())
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

try:
    # Try different import paths based on module structure
    try:
        # Direct import from module
        from baseline_comparison.baseline_algorithms.satzilla_inspired import SatzillaInspiredSelector
        from baseline_comparison.comparison_runner import BaselineComparison
        from baseline_comparison.benchmark_utils import get_benchmark_function
        logger.info("Successfully imported directly from baseline_comparison")
    except ImportError:
        # Try alternate import paths if direct import fails
        logger.warning("Direct import failed, trying alternate import paths")
        
        # Try with mdt_Test prefix
        from mdt_Test.baseline_comparison.baseline_algorithms.satzilla_inspired import SatzillaInspiredSelector
        from mdt_Test.baseline_comparison.comparison_runner import BaselineComparison
        from mdt_Test.baseline_comparison.benchmark_utils import get_benchmark_function
        logger.info("Successfully imported with mdt_Test prefix")
    
    # Import mock Meta Optimizer if needed
    try:
        # Try to import MetaOptimizer
        logger.info("Attempting to import MetaOptimizer...")
        try:
            from meta.meta_optimizer import MetaOptimizer
            META_OPTIMIZER_AVAILABLE = True
            logger.info("Imported MetaOptimizer from meta.meta_optimizer")
        except ImportError:
            try:
                from meta_optimizer.meta.meta_optimizer import MetaOptimizer
                META_OPTIMIZER_AVAILABLE = True
                logger.info("Imported MetaOptimizer from meta_optimizer.meta.meta_optimizer")
            except ImportError:
                logger.warning("Could not import MetaOptimizer, using mock instead")
                META_OPTIMIZER_AVAILABLE = False
                
        if not META_OPTIMIZER_AVAILABLE:
            # Create a mock MetaOptimizer for testing
            class MetaOptimizer:
                def __init__(self, *args, **kwargs):
                    self.name = "MockMetaOptimizer"
                    
                def optimize(self, problem, *args, **kwargs):
                    # Simple random optimization for testing
                    best_x = np.random.uniform(-5, 5, problem.dims if hasattr(problem, 'dims') else 2)
                    best_y = problem.evaluate(best_x) if hasattr(problem, 'evaluate') else np.sum(best_x**2)
                    return best_x, best_y, 100
            
            logger.info("Using mock MetaOptimizer")
    
    except ImportError as e:
        logger.error(f"Import error: {e}")
        sys.exit(1)

except ImportError as e:
    logger.error(f"Import error: {e}")
    sys.exit(1)


def test_satzilla_selector():
    """Test the SATzilla-inspired selector functionality."""
    
    logger.info("Testing SATzilla-inspired selector...")
    
    try:
        # Initialize the selector with a list of algorithms
        algorithms = ['DE', 'ES', 'PSO', 'GWO', 'ACO']
        selector = SatzillaInspiredSelector(algorithms)
        
        # Generate simple test data
        features = [
            {'dimension': 2, 'modality': 1, 'separability': 0.8},
            {'dimension': 10, 'modality': 3, 'separability': 0.2},
            {'dimension': 5, 'modality': 2, 'separability': 0.5}
        ]
        
        performances = {
            'DE': [0.1, 0.5, 0.3],
            'ES': [0.2, 0.4, 0.2],
            'PSO': [0.3, 0.3, 0.1],
            'GWO': [0.4, 0.2, 0.5],
            'ACO': [0.5, 0.1, 0.4]
        }
        
        # Train the selector
        selector.train(features, performances)
        
        # Test prediction on known feature
        test_feature = {'dimension': 2, 'modality': 1, 'separability': 0.8}
        selected_algo = selector.select_algorithm(test_feature)
        
        logger.info(f"Selected algorithm for test feature: {selected_algo}")
        
        # Test prediction on unknown feature
        new_feature = {'dimension': 3, 'modality': 2, 'separability': 0.6}
        selected_algo = selector.select_algorithm(new_feature)
        
        logger.info(f"Selected algorithm for new feature: {selected_algo}")
        
        return True
    
    except Exception as e:
        logger.error(f"SATzilla selector test failed: {e}")
        return False


def test_baseline_comparison_initialization():
    """Test the initialization of the BaselineComparison class."""
    
    logger.info("Testing BaselineComparison initialization...")
    
    try:
        # Get the expected parameters for BaselineComparison
        import inspect
        params = inspect.signature(BaselineComparison.__init__).parameters
        logger.info(f"BaselineComparison initialization parameters: {list(params.keys())[1:]}")  # Skip self
        
        # Create simple test objects based on parameters
        algorithms = ['DE', 'ES', 'PSO', 'GWO', 'ACO']
        simple_baseline = SatzillaInspiredSelector(algorithms)
        meta_learner = MetaOptimizer()
        enhanced_meta = MetaOptimizer()
        satzilla_selector = SatzillaInspiredSelector(algorithms)
        
        # Create the BaselineComparison object
        comparison = BaselineComparison(
            simple_baseline=simple_baseline,
            meta_learner=meta_learner,
            enhanced_meta=enhanced_meta,
            satzilla_selector=satzilla_selector,
            max_evaluations=1000,
            num_trials=3,
            verbose=True
        )
        
        logger.info("Successfully created BaselineComparison object")
        return True
    
    except Exception as e:
        logger.error(f"BaselineComparison initialization test failed: {e}")
        return False


def test_benchmark_functions():
    """Test the benchmark function utilities."""
    
    logger.info("Testing benchmark functions...")
    
    try:
        # Test getting benchmark functions
        dimensions = 2
        
        sphere = get_benchmark_function("sphere", dimensions)
        result = sphere(np.zeros(dimensions))
        logger.info(f"Sphere function at origin: {result}")
        assert result == 0, f"Expected sphere(0,0) = 0, got {result}"
        
        rosenbrock = get_benchmark_function("rosenbrock", dimensions)
        result = rosenbrock(np.ones(dimensions))
        logger.info(f"Rosenbrock function at (1,1): {result}")
        assert result == 0, f"Expected rosenbrock(1,1) = 0, got {result}"
        
        return True
    
    except Exception as e:
        logger.error(f"Benchmark functions test failed: {e}")
        return False


def run_baseline_verification():
    """Run all baseline verification tests."""
    
    results = {}
    
    # Test SATzilla-inspired selector
    results["satzilla_selector"] = test_satzilla_selector()
    
    # Test BaselineComparison initialization
    results["baseline_comparison_init"] = test_baseline_comparison_initialization()
    
    # Test benchmark functions
    results["benchmark_functions"] = test_benchmark_functions()
    
    # Report results
    logger.info("\n=== Baseline Verification Results ===")
    
    all_passed = True
    for name, result in results.items():
        status = "PASSED" if result else "FAILED"
        logger.info(f"{name}: {status}")
        if not result:
            all_passed = False
    
    logger.info(f"\nOverall result: {'PASSED' if all_passed else 'FAILED'}")
    return all_passed


if __name__ == "__main__":
    success = run_baseline_verification()
    sys.exit(0 if success else 1)
