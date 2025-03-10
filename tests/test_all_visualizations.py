#!/usr/bin/env python3
# Comprehensive test script for all visualizations

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
    _create_feature_importance_viz,
    _create_algorithm_ranking_viz,
    _create_pipeline_performance_viz,
    _create_drift_detection_viz,
    _create_algorithm_selection_dashboard,
    visualize_meta_learning_results
)

# Configure logging
logging.basicConfig(level=logging.INFO)

# Directory for saving visualizations
SAVE_DIR = "results/visualizations"
os.makedirs(SAVE_DIR, exist_ok=True)

# Set random seed for reproducibility
np.random.seed(42)

def create_test_data():
    """Create comprehensive test data for all visualizations"""
    # Define algorithms and problems
    algorithms = ["DE", "PSO", "GA", "ES", "GWO", "ACO", "DE-Adaptive", "ES-Adaptive"]
    problems = ["Sphere", "Rastrigin", "Rosenbrock", "Ackley", "Griewank"]
    features = [
        "dimensionality", "multimodality", "separability", 
        "regularity", "basin_ratio", "global_structure",
        "variable_scaling", "constraints"
    ]
    
    # Create results data
    results = {}
    
    # Add problem results with history
    for problem in problems:
        results[problem] = {}
        
        # Choose best algorithm with non-uniform distribution
        best_algo_idx = np.random.choice(range(len(algorithms)))
        best_algo = algorithms[best_algo_idx]
        
        # Add algorithm scores
        algorithm_scores = {}
        for idx, algo in enumerate(algorithms):
            # Base score (lower is better)
            base_score = 0.1 + np.random.rand() * 0.5
            
            # Better score for best algorithm
            if algo == best_algo:
                score = base_score * 0.5
            else:
                score = base_score * (1 + 0.5 * np.random.rand())
                
            algorithm_scores[algo] = score
        
        results[problem]["algorithm_scores"] = algorithm_scores
        results[problem]["best_algorithm"] = best_algo
        
        # Add algorithm-specific data with history and standard deviation
        for algo in algorithms:
            # Create decreasing history with different convergence rates
            history_length = 100 + np.random.randint(-20, 20)
            start_value = 1.0 + np.random.rand() * 2.0
            
            # Different convergence rates per algorithm
            if "DE" in algo:
                rate = 0.95  # Fast initial convergence
            elif "PSO" in algo:
                rate = 0.97
            elif "GA" in algo:
                rate = 0.96
            elif "ES" in algo:
                rate = 0.98
            elif "GWO" in algo:
                rate = 0.94
            else:  # ACO
                rate = 0.97
                
            # Generate main convergence history
            history = start_value * (rate ** np.arange(history_length))
            
            # Add some noise
            history += np.random.randn(history_length) * 0.01
            
            # Ensure non-negative
            history = np.maximum(0.001, history)
            
            # Create standard deviation data
            std_data = history * (0.1 + 0.05 * np.random.rand(history_length))
            
            # Create multiple runs for testing
            num_runs = 5
            runs = []
            for _ in range(num_runs):
                run_noise = np.random.randn(history_length) * 0.05
                run = history + run_noise
                runs.append(run)
            
            # Add to results
            results[problem][algo] = {
                "score": algorithm_scores[algo],
                "history": history
            }
            
            # Add standard deviation data
            results[problem][f"{algo}_std"] = {
                "history": std_data
            }
            
            # Add runs data
            results[problem][f"{algo}_runs"] = {
                "history": runs
            }
    
    # Create problem features
    problem_features = {}
    
    for problem in problems:
        features_dict = {}
        for feature in features:
            features_dict[feature] = np.random.rand()
            
        # Add some correlation between problems and algorithms
        if problem == "Sphere":
            features_dict["multimodality"] = 0.1
            features_dict["separability"] = 0.9
        elif problem == "Rastrigin":
            features_dict["multimodality"] = 0.9
            features_dict["basin_ratio"] = 0.3
        elif problem == "Rosenbrock":
            features_dict["variable_scaling"] = 0.8
            features_dict["separability"] = 0.2
            
        problem_features[problem] = features_dict
    
    # Create selection data
    selection_data = {
        "algorithm_frequencies": {
            "DE-Adaptive": 8,
            "ES-Adaptive": 5,
            "GWO": 4,
            "DE": 2,
            "ACO": 1
        },
        "feature_frequencies": {
            "multimodality": {
                "DE-Adaptive": 2,
                "PSO": 5,
                "GWO": 3,
                "ES": 1,
                "ACO": 0
            },
            "separability": {
                "DE": 5,
                "PSO": 1,
                "GWO": 2,
                "ES": 1,
                "ACO": 1
            },
            "dimensionality": {
                "DE": 3,
                "ES-Adaptive": 2,
                "GWO": 0,
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
                "GWO": 0.4,
                "ES": 0.1,
                "ACO": 0.2
            },
            "separability": {
                "DE": 0.6,
                "PSO": -0.2,
                "GWO": 0.1,
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
            "GWO": {
                "multimodality": 0.6,
                "basin_ratio": 0.4
            }
        }
    }
    
    # Add pipeline performance data
    results["pipeline_performance"] = {
        "drift_scores": np.clip(0.1 + 0.05 * np.cumsum(np.random.randn(150)), 0, 0.3),
        "drift_threshold": 0.25,
        "feature_severities": {
            "temperature": 0.05 + 0.1 * np.abs(np.sin(np.linspace(0, 10, 150))),
            "pressure": 0.04 + 0.12 * np.abs(np.sin(np.linspace(0, 8, 150))),
            "stress_level": 0.05 + 0.15 * np.abs(np.sin(np.linspace(0, 12, 150))),
            "sleep_hours": 0.04 + 0.15 * np.abs(np.sin(np.linspace(0, 15, 150))),
            "screen_time": 0.04 + 0.12 * np.abs(np.sin(np.linspace(0, 9, 150)))
        },
        "model_confidence": 0.7 + 0.05 * np.cumsum(np.random.randn(150) * 0.01),
        "confidence_threshold": 0.8
    }
    
    # Add drift detection data
    # Create a signal with drift points
    t = np.linspace(0, 10, 1000)
    base_signal = np.sin(t * 2 * np.pi / 2)
    
    # Add a drift at specific points
    drift_points = [2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0, 7.5, 8.0, 8.5, 9.0, 9.5]
    
    # Create true signal with drift
    true_signal = base_signal.copy()
    drift_severity = np.zeros_like(t)
    
    for i, point in enumerate(t):
        if i > 250:  # First drift starts after 250 points
            for drift_point in drift_points:
                if point >= drift_point:
                    level = min(1.0, (point - 2.5) * 0.5) if point <= 4.0 else 1.0
                    true_signal[i] += level * 2
                    drift_severity[i] = 0.3 + 0.5 * np.sin(point)
    
    # Add noise to signal
    noisy_signal = true_signal + np.random.randn(len(t)) * 0.2
    
    # Create trend data
    trend = np.zeros_like(t)
    for i, point in enumerate(t):
        if 2 < point < 4:
            trend[i] = 5 * np.sin(point * 5)
            if 2.8 < point < 3.0:
                trend[i] = -25 * (point - 2.9)**2  # Sharp drop
    
    # Calculate drift scores and p-values
    drift_scores = np.zeros(20)
    p_values = np.zeros(20)
    
    for i in range(20):
        if i == 6 or i == 12:  # Drift detected at these points
            drift_scores[i] = 1.0
            p_values[i] = 0.001
        else:
            drift_scores[i] = 0.1 + 0.2 * np.random.rand()
            p_values[i] = 0.5 + 0.4 * np.random.rand()
    
    # Create feature values
    feature_values = {
        "Feature 0": np.random.randn(1000),
        "Feature 1": np.random.randn(1000),
        "Feature 2": np.random.randn(1000)
    }
    
    # Modify feature values at drift points
    for feature in feature_values:
        for drift_point in drift_points:
            idx = int(drift_point * 100)
            if 250 < idx < 600:  # Apply drift in middle section
                feature_values[feature][idx:] += 2.0
    
    results["drift_detection"] = {
        "signal": true_signal,
        "noisy_signal": noisy_signal,
        "drift_points": drift_points,
        "drift_severity": drift_severity,
        "severity_threshold": 0.5,
        "trend": trend,
        "drift_scores": drift_scores,
        "p_values": p_values,
        "significance_level": 0.05,
        "feature_values": feature_values,
        "feature_drift_scores": {
            "Feature 0": np.random.rand(20) * 0.5,
            "Feature 1": np.random.rand(20) * 0.5,
            "Feature 2": np.random.rand(20) * 0.5
        },
        "feature_contributions": {
            "point": 325,
            "values": [0.75, 0.73, 0.76, 0.93, 1.0]
        }
    }
    
    # Create algorithm selection dashboard data
    results["algorithm_selection"] = {
        "frequencies": selection_data["algorithm_frequencies"],
        "problem_types": {
            "multimodal": 15,
            "unimodal": 5
        },
        "timeline": [
            {"algorithm": "DE-Adaptive", "iteration": 1},
            {"algorithm": "GWO", "iteration": 1},
            {"algorithm": "DE-Adaptive", "iteration": 2},
            {"algorithm": "ES-Adaptive", "iteration": 1},
            {"algorithm": "ACO", "iteration": 2},
            {"algorithm": "DE-Adaptive", "iteration": 3},
            {"algorithm": "GWO", "iteration": 2},
            {"algorithm": "ES-Adaptive", "iteration": 2},
            {"algorithm": "ES-Adaptive", "iteration": 3}
        ],
        "performance": {
            "DE-Adaptive": {
                "Sphere": {"score": 3.625, "type": "multimodal"},
                "Rastrigin": {"score": 5.123, "type": "multimodal"},
                "Rosenbrock": {"score": 8.456, "type": "multimodal"}
            },
            "GWO": {
                "Sphere": {"score": 7.923, "type": "multimodal"},
                "Rastrigin": {"score": 6.235, "type": "multimodal"}
            },
            "ACO": {
                "Rosenbrock": {"score": 31.685, "type": "multimodal"}
            },
            "ES-Adaptive": {
                "Sphere": {"score": 3.625, "type": "multimodal"},
                "Rastrigin": {"score": 9.872, "type": "multimodal"}
            },
            "DE": {
                "Rastrigin": {"score": 14955.601, "type": "multimodal"}
            }
        },
        "improvement_rates": {
            "ACO": -0.05,
            "GWO": 0.0,
            "DE-Adaptive": 0.0,
            "ES-Adaptive": 0.0,
            "DE": 0.0
        },
        "statistics": {
            "DE-Adaptive": {
                "selections": 8,
                "selection_percentage": 0.4,
                "best_score": 3.625,
                "avg_score": float('inf'),
                "success_rate": 0.714,
                "avg_improvement": float('nan')
            },
            "GWO": {
                "selections": 4,
                "selection_percentage": 0.2,
                "best_score": 7.923,
                "avg_score": float('inf'),
                "success_rate": 0.667,
                "avg_improvement": float('nan')
            },
            "ACO": {
                "selections": 1,
                "selection_percentage": 0.05,
                "best_score": 31.685,
                "avg_score": 31.685,
                "success_rate": 0.0,
                "avg_improvement": 0.0
            },
            "ES-Adaptive": {
                "selections": 5,
                "selection_percentage": 0.25,
                "best_score": 3.625,
                "avg_score": float('inf'),
                "success_rate": 0.5,
                "avg_improvement": float('nan')
            },
            "DE": {
                "selections": 2,
                "selection_percentage": 0.1,
                "best_score": 14955.601,
                "avg_score": float('inf'),
                "success_rate": 0.0,
                "avg_improvement": float('inf')
            }
        }
    }
    
    return results, selection_data, problem_features

def main():
    # Generate test data
    print("Generating test data...")
    results, selection_data, problem_features = create_test_data()
    
    # Run all visualizations
    print("\nTesting all visualizations...")
    
    print("1. Testing radar charts...")
    _create_radar_charts(results, SAVE_DIR)
    
    print("2. Testing selection frequency charts...")
    _create_selection_frequency_chart(selection_data, SAVE_DIR)
    
    print("3. Testing feature correlation visualizations...")
    _create_feature_correlation_viz(problem_features, SAVE_DIR)
    
    print("4. Testing problem clustering visualizations...")
    _create_problem_clustering_viz(problem_features, SAVE_DIR)
    
    print("5. Testing convergence plots...")
    _create_convergence_plots(results, SAVE_DIR)
    
    print("6. Testing performance comparison...")
    _create_performance_comparison(results, SAVE_DIR)
    
    print("7. Testing feature importance visualizations...")
    _create_feature_importance_viz(problem_features, selection_data, SAVE_DIR)
    
    print("8. Testing algorithm ranking visualization...")
    _create_algorithm_ranking_viz(results, SAVE_DIR)
    
    print("9. Testing pipeline performance visualization...")
    _create_pipeline_performance_viz(results, SAVE_DIR)
    
    print("10. Testing drift detection visualization...")
    _create_drift_detection_viz(results, SAVE_DIR)
    
    print("11. Testing algorithm selection dashboard...")
    _create_algorithm_selection_dashboard(results, SAVE_DIR)
    
    print("\nTesting the comprehensive visualization function...")
    visualize_meta_learning_results(results, selection_data, problem_features, SAVE_DIR)
    
    print(f"\nAll visualizations have been generated and saved to {SAVE_DIR}")
    print("Check the following key files:")
    print("- pipeline_performance.png")
    print("- drift_detection_results.png")
    print("- algorithm_selection_dashboard.png")
    print("- convergence_Rastrigin.png (now with standard deviation bands)")
    print("- selection_frequency_heatmap.png (now includes GWO)")

if __name__ == "__main__":
    main() 