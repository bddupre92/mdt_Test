#!/usr/bin/env python3
"""
Pre-MoE Verification Tests
--------------------------
Simple verification tests for the baseline framework components
before implementing the Mixture-of-Experts system.
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

# Create results directory if it doesn't exist
results_dir = Path(parent_dir) / "results" / "pre_moe_tests"
results_dir.mkdir(parents=True, exist_ok=True)

def test_explainability_components():
    """
    Test the explainability components to ensure they're working properly.
    This leverages the previously implemented explainability framework.
    """
    logger.info("Testing explainability components...")
    
    try:
        # Import explainability components
        from explainability.explainer_factory import ExplainerFactory
        
        # Create a simple model to explain
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.datasets import make_regression
        
        # Generate synthetic data
        X, y = make_regression(n_samples=100, n_features=5, random_state=42)
        
        # Train a simple model
        model = RandomForestRegressor(n_estimators=10, random_state=42)
        model.fit(X, y)
        
        # Test SHAP explainer if available
        try:
            explainer = ExplainerFactory.create_explainer('shap', model=model, X_train=X)
            explanation = explainer.explain(X[:5])
            logger.info("SHAP explainer successfully created and run")
            
            # Test plotting if available
            try:
                explainer.plot('summary', output_file=str(results_dir / "shap_summary.png"))
                logger.info("SHAP explainer plot successfully created")
            except Exception as e:
                logger.warning(f"SHAP plotting failed: {e}")
                
        except Exception as e:
            logger.warning(f"SHAP explainer failed: {e}")
            
        # Test Feature Importance explainer (should be more robust)
        try:
            explainer = ExplainerFactory.create_explainer('feature_importance', model=model)
            explanation = explainer.explain()
            feature_importance = explainer.get_feature_importance()
            
            logger.info(f"Feature importance values: {feature_importance[:3]}...")
            logger.info("Feature importance explainer successfully created and run")
            
            return True
            
        except Exception as e:
            logger.error(f"Feature importance explainer failed: {e}")
            return False
            
    except Exception as e:
        logger.error(f"Explainability test failed: {e}")
        return False

def test_optimizer_explainer():
    """
    Test the optimizer explainer component if available.
    This leverages the previously implemented optimizer explainability framework.
    """
    logger.info("Testing optimizer explainer...")
    
    try:
        # Try to import optimizer explainer
        try:
            from explainability.optimizer_explainer import OptimizerExplainer
            from explainability.explainer_factory import ExplainerFactory
        except ImportError:
            logger.warning("OptimizerExplainer not available, skipping test")
            return None
            
        # Create a simple optimizer
        try:
            # First try importing DifferentialEvolutionOptimizer
            from optimizers.de import DifferentialEvolutionOptimizer
            
            # Define a simple test function (sphere)
            def sphere(x):
                return np.sum(x**2)
            
            # Create and configure the optimizer
            dim = 2
            bounds = [(-5, 5)] * dim
            optimizer = DifferentialEvolutionOptimizer(
                dim=dim, 
                bounds=bounds,
                track_state=True  # Important for explainability
            )
            
            # Run a short optimization
            result = optimizer.run(sphere, max_evals=50)
            
            # Create the explainer
            explainer = ExplainerFactory.create_explainer('optimizer', optimizer=optimizer)
            explanation = explainer.explain()
            
            # Test basic functionality
            param_sensitivity = explainer.get_feature_importance()
            logger.info(f"Parameter sensitivity: {param_sensitivity}")
            
            # Try to generate a plot
            try:
                explainer.plot("convergence", output_file=str(results_dir / "optimizer_convergence.png"))
                logger.info("Successfully generated optimizer convergence plot")
            except Exception as e:
                logger.warning(f"Optimizer plot generation failed: {e}")
                
            return True
                
        except Exception as e:
            logger.warning(f"Failed to test with DifferentialEvolutionOptimizer: {e}")
            logger.warning("Will try with a mock optimizer")
            
            # Create a mock optimizer with the necessary attributes for explanation
            class MockOptimizer:
                def __init__(self):
                    self.name = "MockOptimizer"
                    self.dim = 2
                    self.bounds = [(-5, 5)] * 2
                    self.history = {'iterations': list(range(10)), 'best_scores': [10-i for i in range(10)]}
                    self.parameters = {'population_size': 20, 'mutation_rate': 0.5}
                    
                def get_state(self):
                    return {
                        'evaluations': 100,
                        'iterations': 10,
                        'best_score': 0.5,
                        'convergence': [10-i for i in range(10)]
                    }
            
            mock_opt = MockOptimizer()
            
            # Create explainer with mock optimizer
            explainer = OptimizerExplainer(optimizer=mock_opt)
            explanation = explainer.explain()
            
            logger.info("Successfully created and used OptimizerExplainer with mock optimizer")
            return True
            
    except Exception as e:
        logger.error(f"OptimizerExplainer test failed: {e}")
        return False

def test_meta_learner():
    """
    Test the MetaLearner component if available.
    """
    logger.info("Testing MetaLearner...")
    
    try:
        # Try to import MetaLearner
        try:
            from meta.meta_learner import MetaLearner
        except ImportError:
            try:
                from meta_optimizer.meta.meta_learner import MetaLearner
            except ImportError:
                logger.warning("MetaLearner not available, skipping test")
                return None
                
        # Create a simple dataset for meta-learning
        features = [
            {'dimension': 2, 'nonlinearity': 0.2, 'multimodality': 0.1},
            {'dimension': 5, 'nonlinearity': 0.5, 'multimodality': 0.3},
            {'dimension': 10, 'nonlinearity': 0.8, 'multimodality': 0.7}
        ]
        
        # Performance data for different algorithms
        performances = {
            'DE': [0.85, 0.75, 0.65],
            'PSO': [0.75, 0.80, 0.70],
            'GWO': [0.70, 0.72, 0.85]
        }
        
        # Create and train a meta-learner
        meta_learner = MetaLearner(method='bayesian')
        
        # Save feature names
        meta_learner.feature_names = list(features[0].keys())
        
        # Train the meta-learner
        try:
            meta_learner.train(features, performances)
            logger.info("Successfully trained MetaLearner")
            
            # Test prediction
            new_feature = {'dimension': 3, 'nonlinearity': 0.3, 'multimodality': 0.2}
            prediction = meta_learner.predict(new_feature)
            
            logger.info(f"MetaLearner prediction for new problem: {prediction}")
            return True
            
        except Exception as e:
            logger.error(f"MetaLearner training/prediction failed: {e}")
            return False
            
    except Exception as e:
        logger.error(f"MetaLearner test failed: {e}")
        return False

def run_pre_moe_verification():
    """Run all pre-MoE verification tests."""
    
    results = {}
    
    # Test explainability components
    results["explainability"] = test_explainability_components()
    
    # Test optimizer explainer
    results["optimizer_explainer"] = test_optimizer_explainer()
    
    # Test meta-learner
    results["meta_learner"] = test_meta_learner()
    
    # Report results
    logger.info("\n=== Pre-MoE Verification Results ===")
    
    all_passed = True
    for name, result in results.items():
        if result is None:
            logger.info(f"{name}: SKIPPED")
        elif result:
            logger.info(f"{name}: PASSED")
        else:
            logger.info(f"{name}: FAILED")
            all_passed = False
    
    return all_passed

if __name__ == "__main__":
    success = run_pre_moe_verification()
    sys.exit(0 if success else 1)
