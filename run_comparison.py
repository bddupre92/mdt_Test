import os
import logging
import sys
import numpy as np
from baseline_comparison.comparison_runner import BaselineComparison
from baseline_comparison.baseline_algorithms.simple_baseline import SimpleBaselineSelector
from baseline_comparison.baseline_algorithms.satzilla_inspired import SatzillaInspiredSelector
from meta_optimizer.benchmark.test_functions import create_test_suite
from cli.problem_wrapper import ProblemWrapper

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    # Create output directory
    output_dir = "results/detailed_comparison"
    os.makedirs(output_dir, exist_ok=True)
    
    # Path to the trained SATzilla model
    model_path = "results/satzilla_training/20250320_213623/models/satzilla_selector.joblib"
    
    # Check if the model exists
    if not os.path.exists(model_path):
        logger.error(f"SATzilla model not found at {model_path}")
        sys.exit(1)
    
    # Initialize the selectors
    simple_baseline = SimpleBaselineSelector()
    
    # Create SATzilla selector and load model
    satzilla = SatzillaInspiredSelector()
    try:
        satzilla.load_model(model_path)
        logger.info(f"Loaded SATzilla model from {model_path}")
    except Exception as e:
        logger.error(f"Error loading SATzilla model: {e}")
        sys.exit(1)
    
    # Initialize comparison framework
    comparison = BaselineComparison(
        simple_baseline=simple_baseline,
        meta_learner=satzilla,  # Use SATzilla as meta_learner
        enhanced_meta=satzilla,  # Use SATzilla as enhanced_meta
        satzilla_selector=satzilla,
        max_evaluations=1000,
        num_trials=5,  # 5 trials for faster execution, increase for more robust results
        output_dir=output_dir,
        model_path=model_path
    )
    
    # Get test functions from test suite
    test_suite = create_test_suite()
    dimensions = 10
    
    # Define default bounds for test functions
    default_bounds = {
        "sphere": (-5.12, 5.12),
        "rosenbrock": (-5.0, 10.0),
        "rastrigin": (-5.12, 5.12),
        "ackley": (-32.768, 32.768),
        "griewank": (-600, 600),
        "schwefel": (-500, 500),
        "levy": (-10, 10)
    }
    
    # Run comparison for selected benchmark functions
    benchmark_functions = [
        "sphere",
        "rosenbrock",
        "schwefel",
        "levy",
        "ackley",
        "rastrigin"
    ]
    
    for func_name in benchmark_functions:
        if func_name in test_suite:
            logger.info(f"Running comparison for {func_name} function")
            
            # Get bounds for this function
            bound = default_bounds.get(func_name, (-5, 5))
            bounds = [bound] * dimensions
            
            # Create test function with specified dimensions and bounds
            func_class = test_suite[func_name]
            func = func_class(dimensions, bounds)
            problem = ProblemWrapper(func, dimensions)
            
            # Run comparison
            comparison.run_comparison(
                problem_name=func_name,
                problem_func=problem,
                dimensions=dimensions,
                max_evaluations=1000,
                num_trials=5
            )
        else:
            logger.warning(f"Function {func_name} not found in test suite")
    
    # Generate summary visualizations
    from baseline_comparison.visualization import ComparisonVisualizer
    visualizer = ComparisonVisualizer(comparison.results, export_dir=output_dir)
    visualizer.create_all_visualizations()
    
    logger.info("Comparison completed successfully")
    logger.info(f"Results and visualizations saved to {output_dir}")
    logger.info(f"Per-trial visualizations can be found in each function's directory")

if __name__ == "__main__":
    main() 