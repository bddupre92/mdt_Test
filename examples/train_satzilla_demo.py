#!/usr/bin/env python3
"""
Example script for training the SATzilla-inspired algorithm selector

This script demonstrates how to train the SATzilla-inspired algorithm selector,
generate problem variations, and analyze feature importance. It provides a
simple example of using the training pipeline programmatically.
"""

import os
import logging
import numpy as np
from pathlib import Path
from datetime import datetime

# Import the necessary modules
from baseline_comparison.baseline_algorithms.satzilla_inspired import SatzillaInspiredSelector
from baseline_comparison.benchmark_utils import get_benchmark_function, get_all_benchmark_functions
from baseline_comparison.training import train_selector, feature_analysis

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    """Main function to demonstrate SATzilla training"""
    
    # Parameters
    dimensions = 2
    max_evaluations = 500  # Use a smaller value for quick demo
    num_problems = 10      # Use a smaller value for quick demo
    random_seed = 42
    
    # Setup output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(f"results/satzilla_training/demo_{timestamp}")
    output_dir.mkdir(exist_ok=True, parents=True)
    
    logger.info(f"Training SATzilla-inspired selector - Demo")
    logger.info(f"Output directory: {output_dir}")
    
    # Get benchmark functions
    functions = ['sphere', 'rosenbrock', 'rastrigin', 'ackley', 'griewank']
    benchmark_functions = [
        get_benchmark_function(name, dimensions) 
        for name in functions
    ]
    
    logger.info(f"Using benchmark functions: {[func.name for func in benchmark_functions]}")
    
    # Initialize selector
    selector = SatzillaInspiredSelector()
    
    # Generate problem variations
    np.random.seed(random_seed)
    
    logger.info(f"Generating {num_problems} problem variations")
    training_problems = train_selector.generate_problem_variations(
        benchmark_functions,
        num_problems,
        dimensions,
        random_seed=random_seed
    )
    
    logger.info(f"Generated {len(training_problems)} training problems")
    
    # Train the selector
    logger.info("Training selector...")
    start_time = datetime.now()
    
    trained_selector = train_selector.train_satzilla_selector(
        selector,
        training_problems,
        max_evaluations=max_evaluations,
        export_features=True,
        export_dir=output_dir
    )
    
    training_time = (datetime.now() - start_time).total_seconds()
    logger.info(f"Training completed in {training_time:.2f} seconds")
    
    # Save the trained selector
    model_path = output_dir / "models/satzilla_selector.pkl"
    train_selector.save_trained_selector(trained_selector, model_path)
    logger.info(f"Saved trained selector to {model_path}")
    
    # Analyze feature importance
    logger.info("Analyzing feature importance...")
    importance = feature_analysis.analyze_feature_importance(
        trained_selector,
        export_dir=output_dir
    )
    
    # Analyze feature correlation
    logger.info("Analyzing feature correlation...")
    correlation = feature_analysis.analyze_feature_correlation(
        trained_selector,
        export_dir=output_dir
    )
    
    # Analyze features with PCA
    logger.info("Analyzing features with PCA...")
    pca_results = feature_analysis.analyze_features_with_pca(
        trained_selector,
        export_dir=output_dir
    )
    
    # Test the trained selector on a new problem
    logger.info("Testing trained selector on a new problem...")
    
    # Create a new problem
    test_problem = get_benchmark_function('schwefel', dimensions)
    
    # Extract features
    features = trained_selector.extract_features(test_problem)
    logger.info(f"Extracted features: {features}")
    
    # Select algorithm
    selected_algorithm = trained_selector.select_algorithm(test_problem)
    logger.info(f"Selected algorithm: {selected_algorithm}")
    
    # Run optimization
    solution, fitness, evals = trained_selector.optimize(
        test_problem,
        algorithm=selected_algorithm,
        max_evaluations=max_evaluations
    )
    
    logger.info(f"Optimization result - Fitness: {fitness}, Evaluations: {evals}")
    
    logger.info("Demo completed successfully!")
    logger.info(f"Results saved to {output_dir}")

if __name__ == "__main__":
    main() 