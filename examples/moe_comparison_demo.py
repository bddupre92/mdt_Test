"""
MoE Baseline Comparison Demo

This script demonstrates how to use the MoE Baseline Comparison framework
to compare the performance of MoE against other algorithm selection approaches.
"""

import os
import sys
import numpy as np
import pandas as pd
import logging
import matplotlib.pyplot as plt
from sklearn.datasets import make_regression

# Add the parent directory to the path to ensure imports work
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import baseline comparison components
from baseline_comparison.moe_comparison import MoEBaselineComparison, create_moe_adapter
from baseline_comparison.comparison_runner import SimpleBaseline, MetaLearner, EnhancedMeta, SatzillaSelector
from moe_framework.workflow.moe_pipeline import MoEPipeline

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def generate_synthetic_benchmark_data(n_samples=1000, n_features=10, n_informative=5, test_size=0.2):
    """
    Generate a synthetic dataset for benchmarking.
    
    Args:
        n_samples: Number of samples to generate
        n_features: Total number of features
        n_informative: Number of informative features
        test_size: Proportion of data to use for testing
        
    Returns:
        Train/test split data (X_train, X_test, y_train, y_test)
    """
    # Generate synthetic data
    X, y = make_regression(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=n_informative,
        random_state=42
    )
    
    # Add timestamp and patient ID columns for time series validation
    dates = pd.date_range(start='2025-01-01', periods=n_samples)
    patient_ids = np.random.randint(1, 100, size=n_samples)
    
    # Create DataFrame
    data = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(n_features)])
    data['timestamp'] = dates
    data['patient_id'] = patient_ids
    data['target'] = y
    
    # Split into training and test sets
    split_idx = int(n_samples * (1 - test_size))
    
    train_data = data.iloc[:split_idx]
    test_data = data.iloc[split_idx:]
    
    X_train = train_data.drop(columns=['target'])
    y_train = train_data['target']
    X_test = test_data.drop(columns=['target'])
    y_test = test_data['target']
    
    return X_train, X_test, y_train, y_test


def create_algorithm_pool(n_algorithms=5):
    """
    Create a pool of algorithms for the baseline comparison.
    
    In a real-world scenario, these would be actual algorithm implementations.
    For this demo, we'll create placeholder algorithms with different performance levels.
    
    Args:
        n_algorithms: Number of algorithms in the pool
        
    Returns:
        Dictionary of algorithm names to algorithm objects
    """
    algorithms = {}
    
    for i in range(n_algorithms):
        # Create algorithms with different error levels
        # In a real scenario, these would be actual implementation classes
        algorithms[f'algorithm_{i}'] = {
            'name': f'algorithm_{i}',
            'error_level': 0.5 + np.random.rand() * 0.5,  # Random error between 0.5 and 1.0
            'complexity': np.random.randint(1, 10)       # Random complexity between 1 and 10
        }
    
    return algorithms


def run_moe_comparison_demo():
    """
    Run a demonstration of the MoE baseline comparison framework.
    """
    logger.info("Starting MoE Baseline Comparison Demo")
    
    # Generate synthetic data
    X_train, X_test, y_train, y_test = generate_synthetic_benchmark_data()
    logger.info(f"Generated synthetic data: {X_train.shape[0]} training samples, {X_test.shape[0]} test samples")
    
    # Create algorithm pool
    algorithms = create_algorithm_pool()
    logger.info(f"Created algorithm pool with {len(algorithms)} algorithms")
    
    # Create the MoE configuration
    moe_config = {
        'target_column': 'target',                 # Column name for the target variable
        'time_column': 'timestamp',               # Column name for the timestamp
        'patient_column': 'patient_id',           # Column name for the patient ID
        'expert_selection': 'weighted_confidence', # Strategy for selecting experts
        'output_dir': 'results/moe_demo',         # Output directory for MoE results
        'experts': {                              # Expert configurations
            f'expert_{i}': {
                'model_type': 'sklearn',
                'model_class': 'RandomForestRegressor',
                'model_kwargs': {'n_estimators': 100, 'random_state': 42},
            } for i in range(3)
        },
        'gating_network': {                       # Gating network configuration
            'model_type': 'sklearn',
            'model_class': 'GradientBoostingClassifier',
            'model_kwargs': {'n_estimators': 50, 'random_state': 42},
        }
    }
    
    # Create the MoE adapter
    moe_adapter = create_moe_adapter(config=moe_config, verbose=True)
    logger.info("Created MoE adapter")
    
    # Create the baseline selectors
    simple_baseline = SimpleBaseline(algorithms)
    meta_learner = MetaLearner(algorithms)
    enhanced_meta = EnhancedMeta(algorithms)
    satzilla_selector = SatzillaSelector(algorithms)
    
    # Create the comparison framework
    comparison = MoEBaselineComparison(
        simple_baseline=simple_baseline,
        meta_learner=meta_learner,
        enhanced_meta=enhanced_meta,
        satzilla_selector=satzilla_selector,
        moe_adapter=moe_adapter,
        verbose=True,
        output_dir="results/moe_comparison_demo"
    )
    logger.info("Created MoE baseline comparison framework")
    
    # Train MoE and other selectors on the training data
    # For this demo, we'll use cross-validation
    logger.info("Running cross-validation for all selectors")
    cv_results = comparison.cross_validate_all(
        X_train, y_train,
        n_splits=5,
        method='patient_aware'  # MoE will use patient-aware splitting
    )
    
    # Print cross-validation results
    for selector, scores in cv_results.items():
        if isinstance(scores, dict) and 'mean_scores' in scores:
            logger.info(f"{selector.upper()} cross-validation mean scores:")
            for metric, value in scores['mean_scores'].items():
                logger.info(f"  {metric}: {value:.4f}")
    
    # Now let's run a simple comparison on test data
    logger.info("Running comparison on test data")
    
    # Combine test data for simplicity
    test_data = X_test.copy()
    test_data['target'] = y_test
    
    # Define a simple problem function for the comparison
    def problem_func(dimensions=None):
        return test_data
    
    # Run the comparison
    results = comparison.run_comparison(
        problem_name="regression_benchmark",
        problem_func=problem_func,
        dimensions=X_test.shape[1],
        max_evaluations=1000,
        num_trials=3
    )
    
    # Get a summary of the results
    summary = comparison.get_summary_with_moe()
    logger.info("\nComparison Results Summary:")
    logger.info(summary.to_string())
    
    # Plot the results
    plt.figure(figsize=(10, 6))
    
    selectors = list(comparison.results.keys())
    mean_fitness = [np.mean(comparison.results[s]['best_fitness']) for s in selectors]
    std_fitness = [np.std(comparison.results[s]['best_fitness']) for s in selectors]
    
    # Create a bar chart
    bars = plt.bar(selectors, mean_fitness, yerr=std_fitness, alpha=0.7)
    
    # Add labels and title
    plt.xlabel('Selection Method')
    plt.ylabel('Mean Best Fitness (lower is better)')
    plt.title('Performance Comparison of Selection Methods')
    plt.xticks(rotation=45)
    
    # Highlight the MoE bar if it exists
    if 'moe' in selectors:
        moe_index = selectors.index('moe')
        bars[moe_index].set_color('green')
    
    plt.tight_layout()
    
    # Save the figure
    os.makedirs("results/moe_comparison_demo", exist_ok=True)
    plt.savefig("results/moe_comparison_demo/comparison_results.png")
    logger.info("Saved comparison results chart to results/moe_comparison_demo/comparison_results.png")
    
    # Show the figure if running interactively
    plt.show()
    
    logger.info("MoE Baseline Comparison Demo completed")


if __name__ == "__main__":
    run_moe_comparison_demo()
