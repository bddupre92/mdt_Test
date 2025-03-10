#!/usr/bin/env python3
# Test script for enhanced visualization functions

import os
import numpy as np
import logging
from core.meta_learning import (
    _create_radar_charts,
    _create_selection_frequency_chart,
    _create_feature_correlation_viz,
    _create_problem_clustering_viz,
    _create_convergence_plots,
    _create_performance_comparison,
    _create_feature_importance_viz
)

# Configure logging
logging.basicConfig(level=logging.INFO)

# Directory for saving visualizations
SAVE_DIR = "results/visualizations"
os.makedirs(SAVE_DIR, exist_ok=True)

# Create sample test data
np.random.seed(42)  # For reproducibility

# Create test results data
results = {}
algorithms = ["DE", "PSO", "GA", "ES", "ACO"]
problems = ["Sphere", "Rastrigin", "Rosenbrock", "Ackley", "Griewank"]

# Generate scores and history for each problem-algorithm combination
for problem in problems:
    results[problem] = {}
    
    # Generate random "best algorithm" with non-uniform distribution
    best_algo_idx = np.random.choice(range(len(algorithms)), p=[0.4, 0.3, 0.15, 0.1, 0.05])
    best_algo = algorithms[best_algo_idx]
    
    # Add algorithm scores
    algorithm_scores = {}
    for idx, algo in enumerate(algorithms):
        # Base score (lower is better)
        base_score = 0.1 + np.random.rand() * 0.5
        
        # Better score for the "best" algorithm
        if algo == best_algo:
            score = base_score * 0.5
        else:
            score = base_score * (1 + 0.5 * np.random.rand())
            
        algorithm_scores[algo] = score
    
    results[problem]["algorithm_scores"] = algorithm_scores
    results[problem]["best_algorithm"] = best_algo
    
    # Add algorithm-specific data
    for algo in algorithms:
        # Create decreasing history (convergence data)
        history_length = 100 + np.random.randint(-20, 20)
        start_value = 1.0 + np.random.rand() * 2.0
        
        # Different convergence rates
        if algo == "DE":
            rate = 0.95  # Fast initial convergence
        elif algo == "PSO":
            rate = 0.97
        elif algo == "GA":
            rate = 0.98
        elif algo == "ES":
            rate = 0.99
        else:  # ACO
            rate = 0.96
            
        # Generate convergence history
        history = start_value * (rate ** np.arange(history_length))
        
        # Add some noise
        history += np.random.randn(history_length) * 0.01
        
        # Ensure non-negative
        history = np.maximum(0.001, history)
        
        # Add to results
        results[problem][algo] = {
            "score": algorithm_scores[algo],
            "history": history
        }

# Create sample problem features data
problem_features = {}
feature_names = [
    "dimensionality", "multimodality", "separability", 
    "regularity", "basin_ratio", "global_structure",
    "variable_scaling", "constraints"
]

for problem in problems:
    # Generate random feature values
    features = {}
    for feature in feature_names:
        features[feature] = np.random.rand()
        
    # Add some correlation between problems and algorithms
    if problem == "Sphere":
        features["multimodality"] = 0.1  # Low multimodality
        features["separability"] = 0.9   # High separability
    elif problem == "Rastrigin":
        features["multimodality"] = 0.9  # High multimodality
        features["basin_ratio"] = 0.3    # Narrow basins
    elif problem == "Rosenbrock":
        features["variable_scaling"] = 0.8  # High variable scaling
        features["separability"] = 0.2     # Low separability
    
    problem_features[problem] = features

# Create sample selection data
selection_data = {
    "algorithm_frequencies": {
        "DE": 12,
        "PSO": 8,
        "GA": 5,
        "ES": 3,
        "ACO": 2
    },
    "feature_frequencies": {
        "multimodality": {
            "DE": 2,
            "PSO": 5,
            "GA": 3,
            "ES": 1,
            "ACO": 0
        },
        "separability": {
            "DE": 5,
            "PSO": 1,
            "GA": 2,
            "ES": 1,
            "ACO": 1
        },
        "dimensionality": {
            "DE": 3,
            "PSO": 2,
            "GA": 0,
            "ES": 1,
            "ACO": 1
        }
    },
    "feature_importance": {
        "multimodality": 0.35,
        "separability": 0.25,
        "dimensionality": 0.15,
        "regularity": 0.1,
        "basin_ratio": 0.08,
        "global_structure": 0.03,
        "variable_scaling": 0.03,
        "constraints": 0.01
    },
    "feature_algorithm_correlation": {
        "multimodality": {
            "DE": -0.3,
            "PSO": 0.5,
            "GA": 0.4,
            "ES": 0.1,
            "ACO": 0.2
        },
        "separability": {
            "DE": 0.6,
            "PSO": -0.2,
            "GA": 0.1,
            "ES": 0.3,
            "ACO": -0.1
        }
    },
    "algorithm_feature_importance": {
        "DE": {
            "separability": 0.5,
            "dimensionality": 0.3,
            "regularity": 0.2
        },
        "PSO": {
            "multimodality": 0.6,
            "basin_ratio": 0.4
        }
    }
}

# Run visualization functions
print("Generating visualizations...")

print("1. Creating radar charts...")
_create_radar_charts(results, SAVE_DIR)

print("2. Creating selection frequency charts...")
_create_selection_frequency_chart(selection_data, SAVE_DIR)

print("3. Creating feature correlation visualizations...")
_create_feature_correlation_viz(problem_features, SAVE_DIR)

print("4. Creating problem clustering visualizations...")
_create_problem_clustering_viz(problem_features, SAVE_DIR)

print("5. Creating convergence plots...")
_create_convergence_plots(results, SAVE_DIR)

print("6. Creating performance comparison...")
_create_performance_comparison(results, SAVE_DIR)

print("7. Creating feature importance visualizations...")
_create_feature_importance_viz(problem_features, selection_data, SAVE_DIR)

print(f"All visualizations generated and saved to {SAVE_DIR}")
print("Check visualization_guide.md for details on each visualization.") 