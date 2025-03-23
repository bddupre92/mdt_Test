#!/usr/bin/env python3
"""
Meta-Integration Test
--------------------
Tests the integration between Meta_Optimizer, Meta_Learner and explainability components
to ensure they are functioning correctly before MoE implementation.
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

# Add the parent directory to the path
parent_dir = str(Path(__file__).parent.parent.absolute())
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

# Import necessary components
try:
    # Use absolute imports instead of relative imports
    sys.path.append(parent_dir)
    from mdt_Test.meta.meta_optimizer import MetaOptimizer
    from mdt_Test.meta.meta_learner import MetaLearner
    from mdt_Test.explainability.explainer_factory import ExplainerFactory
    from mdt_Test.optimizers.de import DifferentialEvolutionOptimizer
    from mdt_Test.optimizers.gwo import GreyWolfOptimizer
    
    # Check if OptimizerExplainer is available
    try:
        from mdt_Test.explainability.optimizer_explainer import OptimizerExplainer
        OPTIMIZER_EXPLAINER_AVAILABLE = True
    except ImportError:
        logger.warning("OptimizerExplainer not available. Some tests will be skipped.")
        OPTIMIZER_EXPLAINER_AVAILABLE = False

except ImportError as e:
    logger.error(f"Failed to import required components: {e}")
    sys.exit(1)


def test_meta_optimizer():
    """Test basic functionality of the MetaOptimizer."""
    
    logger.info("Testing MetaOptimizer...")
    
    # Define a simple test function
    def sphere(x):
        return np.sum(x**2)
    
    # Create optimizers
    dim = 2
    bounds = [(-5, 5)] * dim
    
    optimizers = {
        "DE": DifferentialEvolutionOptimizer(dim=dim, bounds=bounds),
        "GWO": GreyWolfOptimizer(dim=dim, bounds=bounds)
    }
    
    # Create MetaOptimizer
    meta_opt = MetaOptimizer(
        dim=dim,
        bounds=bounds,
        optimizers=optimizers,
        verbose=True
    )
    
    # Run optimization
    try:
        result = meta_opt.run(sphere, max_evals=100)
        
        # Check result format
        assert "solution" in result, "Result should contain 'solution'"
        assert "score" in result, "Result should contain 'score'"
        
        # Check solution quality
        assert result["score"] < 1.0, f"Expected score < 1.0, got {result['score']}"
        
        logger.info(f"MetaOptimizer test passed. Score: {result['score']:.6f}")
        return True
    
    except Exception as e:
        logger.error(f"MetaOptimizer test failed: {e}")
        return False


def test_meta_learner():
    """Test basic functionality of the MetaLearner."""
    
    logger.info("Testing MetaLearner...")
    
    # Create a simple dataset
    features = [
        {"dimension": 2, "modality": 1},
        {"dimension": 10, "modality": 3},
        {"dimension": 5, "modality": 2}
    ]
    
    performances = {
        "DE": [0.1, 0.5, 0.3],
        "GWO": [0.2, 0.1, 0.4]
    }
    
    # Create MetaLearner
    try:
        meta_learner = MetaLearner(method='bayesian')
        
        # Train the meta-learner
        meta_learner.train(features, performances)
        
        # Make a prediction
        new_feature = {"dimension": 3, "modality": 1}
        prediction = meta_learner.predict(new_feature)
        
        # Check prediction format
        assert isinstance(prediction, dict), "Prediction should be a dictionary"
        assert "DE" in prediction, "Prediction should include 'DE'"
        assert "GWO" in prediction, "Prediction should include 'GWO'"
        
        logger.info(f"MetaLearner test passed. Prediction: {prediction}")
        return True
    
    except Exception as e:
        logger.error(f"MetaLearner test failed: {e}")
        return False


def test_optimizer_explainer():
    """Test the OptimizerExplainer with the MetaOptimizer."""
    
    if not OPTIMIZER_EXPLAINER_AVAILABLE:
        logger.warning("Skipping OptimizerExplainer test as it's not available")
        return None
    
    logger.info("Testing OptimizerExplainer...")
    
    # Define test function
    def rosenbrock(x):
        return sum(100.0 * (x[i+1] - x[i]**2)**2 + (1 - x[i])**2 for i in range(len(x)-1))
    
    # Create optimizers and run optimization
    dim = 2
    bounds = [(-2, 2)] * dim
    
    # Create and run optimizer to collect data
    de_opt = DifferentialEvolutionOptimizer(dim=dim, bounds=bounds, track_state=True)
    de_result = de_opt.run(rosenbrock, max_evals=100)
    
    try:
        # Create explainer
        explainer = ExplainerFactory.create_explainer('optimizer', optimizer=de_opt)
        
        # Generate explanation
        explanation = explainer.explain()
        
        # Check explanation format
        assert explanation is not None, "Explanation should not be None"
        
        # Generate plot (if available)
        try:
            explainer.plot("convergence")
            logger.info("Generated convergence plot successfully")
        except Exception as e:
            logger.warning(f"Plotting failed: {e}")
        
        logger.info("OptimizerExplainer test passed")
        return True
    
    except Exception as e:
        logger.error(f"OptimizerExplainer test failed: {e}")
        return False


def run_integration_tests():
    """Run all integration tests and report results."""
    
    results = {}
    
    # Test MetaOptimizer
    results["meta_optimizer"] = test_meta_optimizer()
    
    # Test MetaLearner
    results["meta_learner"] = test_meta_learner()
    
    # Test OptimizerExplainer
    results["optimizer_explainer"] = test_optimizer_explainer()
    
    # Report overall results
    logger.info("\n=== Integration Test Results ===")
    
    all_passed = True
    for name, result in results.items():
        if result is None:
            logger.info(f"{name}: SKIPPED")
        elif result:
            logger.info(f"{name}: PASSED")
        else:
            logger.info(f"{name}: FAILED")
            all_passed = False
    
    logger.info(f"\nOverall result: {'PASSED' if all_passed else 'FAILED'}")
    return all_passed


if __name__ == "__main__":
    success = run_integration_tests()
    sys.exit(0 if success else 1)
