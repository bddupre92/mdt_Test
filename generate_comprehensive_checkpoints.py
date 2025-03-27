#!/usr/bin/env python
"""
Generate comprehensive checkpoint data for the Performance Analysis Dashboard.

This script creates rich checkpoint files with a complete set of metrics and data
needed for all dashboard visualizations covering the complete MoE workflow:
- Expert Model Benchmarks
- Gating Network Analysis
- End-to-End Metrics
- Temporal Analysis
- Baseline Comparisons
- Statistical Tests
- Expert-Optimizer Integration
- Adaptive Weighting
- Integration Strategies
- Pipeline Configuration
"""
import json
import os
import random
import numpy as np
import datetime
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import stats

# Configuration
OUTPUT_DIR = Path('moe_tests/dashboard/data')
TEMP_VISUALS_DIR = Path('moe_tests/dashboard/visuals')
TIMESTAMP = datetime.datetime.now().strftime('%Y_%m_%d')
NUM_TIMESTEPS = 20  # Number of time steps for temporal data
NUM_EXPERTS = 4     # Number of experts in the MoE system (all expert types)
BASELINE_MODELS = ["simple_linear", "simple_tree", "simple_average"]
OPTIMIZERS = ["differential_evolution", "particle_swarm", "bayes_opt", "grid_search"]
INTEGRATION_STRATEGIES = ["weighted_average", "confidence_based", "adaptive", "expert_select"]

# Ensure output directories exist
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)
TEMP_VISUALS_DIR.mkdir(exist_ok=True, parents=True)

def generate_timestamp_data(num_points=NUM_TIMESTEPS):
    """Generate evenly spaced timestamps for temporal analysis."""
    now = datetime.datetime.now()
    timestamps = []
    for i in range(num_points):
        # Create timestamps with 1-hour intervals
        ts = now - datetime.timedelta(hours=num_points-i-1)
        timestamps.append(ts.isoformat())
    return timestamps

def generate_metric_time_series(base_value, std_dev=0.02, trend_factor=0.001, num_points=NUM_TIMESTEPS):
    """Generate a time series for a metric with optional trend."""
    values = []
    for i in range(num_points):
        # Add some random variation and a slight improvement trend over time
        value = base_value - (i * trend_factor) + random.normalvariate(0, std_dev)
        # Ensure values make sense (e.g., RMSE/MAE shouldn't be negative)
        value = max(value, 0.001)
        values.append(round(value, 4))
    return values

def generate_expert_benchmarks(num_experts=NUM_EXPERTS):
    """Generate comprehensive benchmark data for each expert."""
    experts = {}
    expert_types = ["behavioral_expert", "environmental_expert", "medication_history_expert", "physiological_expert"]
    
    for i in range(min(num_experts, len(expert_types))):
        expert_name = expert_types[i]
        
        # Base metrics with slight variation between experts
        base_rmse = 0.15 + random.uniform(-0.05, 0.05)
        base_mae = 0.8 * base_rmse  # MAE is typically lower than RMSE
        base_r2 = 0.7 + random.uniform(-0.1, 0.2)
        
        # Generate time series for each metric
        timestamps = generate_timestamp_data()
        rmse_series = generate_metric_time_series(base_rmse)
        mae_series = generate_metric_time_series(base_mae)
        r2_series = generate_metric_time_series(1 - base_r2, trend_factor=-0.001)  # R² increases over time
        
        # Create comprehensive expert data
        experts[expert_name] = {
            # Basic performance metrics
            "rmse": round(base_rmse, 4),
            "mae": round(base_mae, 4),
            "r2": round(base_r2, 4),
            "auc": round(0.75 + random.uniform(-0.05, 0.15), 4),
            
            # Temporal performance
            "performance_over_time": {
                "timestamps": timestamps,
                "metrics": {
                    "rmse": rmse_series,
                    "mae": mae_series,
                    "r2": [round(1-x, 4) for x in r2_series]  # Convert back to proper R² values
                }
            },
            
            # Confidence metrics
            "confidence_metrics": {
                "mean_confidence": round(0.7 + random.uniform(-0.1, 0.2), 4),
                "calibration_error": round(0.05 + random.uniform(-0.02, 0.05), 4),
                "confidence_bins": [
                    {"bin_start": 0.0, "bin_end": 0.2, "accuracy": round(0.1 + random.uniform(0, 0.1), 4)},
                    {"bin_start": 0.2, "bin_end": 0.4, "accuracy": round(0.3 + random.uniform(0, 0.1), 4)},
                    {"bin_start": 0.4, "bin_end": 0.6, "accuracy": round(0.5 + random.uniform(0, 0.1), 4)},
                    {"bin_start": 0.6, "bin_end": 0.8, "accuracy": round(0.7 + random.uniform(0, 0.1), 4)},
                    {"bin_start": 0.8, "bin_end": 1.0, "accuracy": round(0.9 + random.uniform(0, 0.1), 4)}
                ]
            },
            
            # Timing information
            "timing": {
                "training_time": round(120 + random.uniform(-20, 50), 2),  # seconds
                "inference_time": round(0.05 + random.uniform(-0.01, 0.02), 4),  # seconds per sample
                "total_predictions": 1000 + random.randint(-100, 200)
            }
        }
    
    return experts

def generate_gating_evaluation(expert_names):
    """Generate comprehensive gating network evaluation data."""
    # Calculate random but plausible expert selection probabilities
    num_experts = len(expert_names)
    raw_weights = [random.uniform(0.5, 1.5) for _ in range(num_experts)]
    total = sum(raw_weights)
    selection_probs = [w/total for w in raw_weights]
    
    # Generate timestamps for temporal analysis
    timestamps = generate_timestamp_data()
    
    # Create selection frequency time series data
    selection_time_series = {}
    for i, expert in enumerate(expert_names):
        # Generate slightly varying selection frequencies over time
        base_freq = selection_probs[i]
        freq_series = []
        for _ in range(len(timestamps)):
            freq = base_freq + random.uniform(-0.05, 0.05)
            freq = min(max(freq, 0.05), 0.95)  # Keep between reasonable bounds
            freq_series.append(round(freq, 4))
        selection_time_series[expert] = freq_series
    
    # Create simulated decision boundaries data
    feature_ranges = {
        "clinical_score": {"min": 0, "max": 100},
        "environmental_score": {"min": 0, "max": 50}
    }
    
    # Generate weight distribution data
    samples = 100
    weight_distributions = {}
    for expert in expert_names:
        # Create a beta distribution for the weights with different params per expert
        alpha = random.uniform(1.5, 4.0)
        beta = random.uniform(1.5, 4.0)
        weights = stats.beta.rvs(alpha, beta, size=samples)
        weight_distributions[expert] = [round(w, 4) for w in weights]
    
    return {
        # Basic metrics
        "selection_accuracy": round(0.7 + random.uniform(-0.1, 0.2), 4),
        "optimal_selection_rate": round(0.65 + random.uniform(-0.1, 0.2), 4),
        "mean_regret": round(0.08 + random.uniform(-0.03, 0.05), 4),
        "max_regret": round(0.2 + random.uniform(-0.05, 0.1), 4),
        "selection_entropy": round(0.6 + random.uniform(-0.2, 0.2), 4),
        
        # Expert utilization
        "expert_utilization": {expert: round(prob, 4) for expert, prob in zip(expert_names, selection_probs)},
        
        # Temporal selection patterns
        "selection_over_time": {
            "timestamps": timestamps,
            "expert_selection": selection_time_series
        },
        
        # Weight analysis
        "weight_analysis": {
            "weight_concentration": round(0.65 + random.uniform(-0.15, 0.15), 4),
            "weight_correlation": round(0.4 + random.uniform(-0.3, 0.3), 4),
            "weight_distributions": weight_distributions
        },
        
        # Decision boundaries approximation
        "decision_boundaries": {
            "feature_ranges": feature_ranges,
            "boundaries": [
                {
                    "region_id": 1,
                    "primary_expert": expert_names[0],
                    "bounds": {"clinical_score": {"min": 0, "max": 40}, "environmental_score": {"min": 0, "max": 25}}
                },
                {
                    "region_id": 2,
                    "primary_expert": expert_names[1],
                    "bounds": {"clinical_score": {"min": 40, "max": 100}, "environmental_score": {"min": 0, "max": 25}}
                },
                {
                    "region_id": 3,
                    "primary_expert": expert_names[2] if len(expert_names) > 2 else expert_names[0],
                    "bounds": {"clinical_score": {"min": 0, "max": 100}, "environmental_score": {"min": 25, "max": 50}}
                }
            ]
        }
    }

def generate_baseline_comparisons(moe_metrics, baselines=BASELINE_MODELS):
    """Generate comparison data between MoE and baseline models."""
    comparisons = {}
    
    for baseline in baselines:
        # Make baseline slightly worse than MoE (realistic scenario)
        rmse_factor = random.uniform(1.1, 1.4)  # Baseline RMSE 10%-40% worse
        mae_factor = random.uniform(1.1, 1.4)   # Similar for MAE
        r2_factor = random.uniform(0.7, 0.9)    # R² correspondingly lower
        
        # Calculate metrics
        rmse = round(moe_metrics["rmse"] * rmse_factor, 4)
        mae = round(moe_metrics["mae"] * mae_factor, 4)
        r2 = round(moe_metrics["r2"] * r2_factor, 4)
        
        # Calculate percentage improvements
        rmse_improvement = round(((rmse - moe_metrics["rmse"]) / rmse) * 100, 2)
        mae_improvement = round(((mae - moe_metrics["mae"]) / mae) * 100, 2)
        r2_improvement = round(((moe_metrics["r2"] - r2) / (1 - r2)) * 100, 2)
        
        # Generate p-values for statistical significance
        p_value_rmse = round(random.uniform(0.001, 0.05), 4)
        p_value_mae = round(random.uniform(0.001, 0.05), 4)
        p_value_r2 = round(random.uniform(0.001, 0.05), 4)
        
        comparisons[baseline] = {
            "metrics": {
                "rmse": rmse,
                "mae": mae,
                "r2": r2,
                "auc": round(0.7 + random.uniform(-0.1, 0.1), 4)
            },
            "improvements": {
                "rmse": f"{rmse_improvement}%",
                "mae": f"{mae_improvement}%",
                "r2": f"{r2_improvement}%"
            },
            "statistical_significance": {
                "rmse": {"p_value": p_value_rmse, "significant": p_value_rmse < 0.05},
                "mae": {"p_value": p_value_mae, "significant": p_value_mae < 0.05},
                "r2": {"p_value": p_value_r2, "significant": p_value_r2 < 0.05}
            }
        }
    
    return comparisons

def generate_statistical_tests(moe_metrics):
    """Generate statistical test results for the MoE model performance."""
    return {
        "normality_tests": {
            "shapiro_wilk": {
                "test_statistic": round(0.95 + random.uniform(-0.1, 0.05), 4),
                "p_value": round(0.3 + random.uniform(-0.2, 0.5), 4),
                "is_normal": True
            },
            "anderson_darling": {
                "test_statistic": round(0.4 + random.uniform(-0.2, 0.3), 4),
                "critical_values": [0.576, 0.656, 0.787, 0.918, 1.092],
                "significance_levels": [15, 10, 5, 2.5, 1],
                "is_normal": True
            }
        },
        "confidence_intervals": {
            "rmse": {
                "lower_bound": round(moe_metrics["rmse"] - 0.02, 4),
                "upper_bound": round(moe_metrics["rmse"] + 0.02, 4),
                "confidence_level": 0.95
            },
            "mae": {
                "lower_bound": round(moe_metrics["mae"] - 0.015, 4),
                "upper_bound": round(moe_metrics["mae"] + 0.015, 4),
                "confidence_level": 0.95
            },
            "r2": {
                "lower_bound": round(moe_metrics["r2"] - 0.05, 4),
                "upper_bound": round(moe_metrics["r2"] + 0.05, 4),
                "confidence_level": 0.95
            }
        },
        "hypothesis_tests": {
            "one_sample_t_test": {
                "test_statistic": round(random.uniform(2.0, 5.0), 4),
                "p_value": round(random.uniform(0.0001, 0.02), 4),
                "is_significant": True
            }
        }
    }

def generate_end_to_end_metrics(expert_benchmarks):
    """Generate end-to-end metrics for the MoE model."""
    # Base the end-to-end performance on a weighted average of expert performances
    # (MoE should perform better than individual experts due to specialization)
    expert_metrics = {expert: data for expert, data in expert_benchmarks.items() 
                      if isinstance(data, dict) and "rmse" in data}
    
    if not expert_metrics:
        # Fallback if no valid experts found
        base_rmse = 0.15
        base_mae = 0.12
        base_r2 = 0.75
    else:
        # Calculate weighted average of expert metrics
        rmse_values = [data["rmse"] for data in expert_metrics.values()]
        mae_values = [data["mae"] for data in expert_metrics.values()]
        r2_values = [data["r2"] for data in expert_metrics.values()]
        
        # MoE typically performs better than the average expert
        improvement_factor = 0.85  # 15% better than average expert
        
        base_rmse = sum(rmse_values) / len(rmse_values) * improvement_factor
        base_mae = sum(mae_values) / len(mae_values) * improvement_factor
        base_r2 = sum(r2_values) / len(r2_values) / improvement_factor  # R² increases
    
    # Generate timestamps for temporal analysis
    timestamps = generate_timestamp_data()
    
    # Create time series for each metric
    rmse_series = generate_metric_time_series(base_rmse)
    mae_series = generate_metric_time_series(base_mae)
    r2_series = generate_metric_time_series(1 - base_r2, trend_factor=-0.001)  # R² increases over time
    
    # Overall metrics
    overall_metrics = {
        "rmse": round(base_rmse, 4),
        "mae": round(base_mae, 4),
        "r2": round(base_r2, 4),
        "auc": round(0.85 + random.uniform(-0.05, 0.1), 4)
    }
    
    # Generate temporal analysis data
    temporal_analysis = {
        "timestamps": timestamps,
        "metrics": {
            "rmse": rmse_series,
            "mae": mae_series,
            "r2": [round(1-x, 4) for x in r2_series]  # Convert back to proper R² values
        }
    }
    
    # Generate baseline comparisons
    baseline_comparisons = generate_baseline_comparisons(overall_metrics)
    
    # Generate statistical tests
    statistical_tests = generate_statistical_tests(overall_metrics)
    
    return {
        "overall": overall_metrics,
        "temporal": temporal_analysis,
        "baseline_comparisons": baseline_comparisons,
        "statistical_tests": statistical_tests
    }

def generate_expert_optimizer_integration():
    """Generate data about how experts worked with their optimizers."""
    integration_data = {}
    
    for optimizer in OPTIMIZERS:
        optimizer_results = {}
        
        # Create convergence curves for each expert with this optimizer
        expert_types = ["behavioral_expert", "environmental_expert", "medication_history_expert", "physiological_expert"]
        for expert in expert_types:
            # Simulate convergence curve - starts high and decreases
            iterations = 50
            base_error = 0.4 + random.uniform(-0.1, 0.1)
            convergence_rate = 0.02 + random.uniform(-0.01, 0.01)
            
            error_curve = []
            for i in range(iterations):
                # Error decays exponentially with occasional noise
                error = base_error * np.exp(-convergence_rate * i) + random.uniform(0, 0.02)
                error_curve.append(round(error, 4))
            
            # Generate time per iteration (milliseconds)
            time_per_iteration = round(50 + random.uniform(-10, 100), 1)
            
            # Generate hyperparameter paths
            hyperparams = {
                "learning_rate": [round(0.1 - 0.001 * i + random.uniform(-0.01, 0.01), 4) for i in range(10)],
                "regularization": [round(0.01 + 0.001 * i + random.uniform(-0.002, 0.002), 4) for i in range(10)]
            }
            
            optimizer_results[expert] = {
                "convergence_curve": error_curve,
                "best_iteration": random.randint(30, 45),
                "total_iterations": iterations,
                "time_per_iteration_ms": time_per_iteration,
                "total_time_seconds": round(time_per_iteration * iterations / 1000, 2),
                "hyperparameter_path": hyperparams,
                "final_error": error_curve[-1]
            }
        
        integration_data[optimizer] = optimizer_results
    
    return integration_data

def generate_adaptive_weighting_data():
    """Generate data about how weights adapted over time and conditions."""
    expert_types = ["behavioral_expert", "environmental_expert", "medication_history_expert", "physiological_expert"]
    timestamps = generate_timestamp_data(num_points=30)
    
    # Create patient IDs for demonstrating personalization
    patient_ids = ["P001", "P002", "P003", "P004", "P005"]
    
    # Generate adaptive weighting data
    adaptive_data = {
        "temporal_adaptation": {
            "timestamps": timestamps,
            "weights": {}
        },
        "quality_based_adaptation": {
            "data_quality_levels": [0.1, 0.3, 0.5, 0.7, 0.9],
            "weight_adjustments": {}
        },
        "drift_based_adaptation": {
            "drift_magnitudes": [0.0, 0.2, 0.4, 0.6, 0.8],
            "weight_adjustments": {}
        },
        "personalized_adaptation": {
            "patient_ids": patient_ids,
            "personalized_weights": {}
        }
    }
    
    # Generate temporal weight changes for each expert
    for expert in expert_types:
        # Create base weight with slight trend
        base_weight = 1.0/len(expert_types) + random.uniform(-0.1, 0.1)
        weights = []
        
        for i in range(len(timestamps)):
            # Add trend and noise
            weight = base_weight + i * 0.002 * random.choice([-1, 1]) + random.uniform(-0.05, 0.05)
            weight = max(0.05, min(0.9, weight))  # Keep within reasonable bounds
            weights.append(round(weight, 4))
        
        adaptive_data["temporal_adaptation"]["weights"][expert] = weights
        
        # Generate quality-based adjustments
        adaptive_data["quality_based_adaptation"]["weight_adjustments"][expert] = []
        for quality in adaptive_data["quality_based_adaptation"]["data_quality_levels"]:
            # Lower quality data should reduce weight
            adjustment = round((quality - 0.5) * random.uniform(0.3, 0.7), 4)
            adaptive_data["quality_based_adaptation"]["weight_adjustments"][expert].append(adjustment)
        
        # Generate drift-based adjustments
        adaptive_data["drift_based_adaptation"]["weight_adjustments"][expert] = []
        for drift in adaptive_data["drift_based_adaptation"]["drift_magnitudes"]:
            # Higher drift should reduce weight
            adjustment = round(-drift * random.uniform(0.3, 0.7), 4)
            adaptive_data["drift_based_adaptation"]["weight_adjustments"][expert].append(adjustment)
        
        # Generate personalized weights
        adaptive_data["personalized_adaptation"]["personalized_weights"][expert] = {}
        for patient in patient_ids:
            # Different weight for each patient
            weight = 1.0/len(expert_types) + random.uniform(-0.25, 0.25)
            weight = max(0.05, min(0.9, weight))
            adaptive_data["personalized_adaptation"]["personalized_weights"][expert][patient] = round(weight, 4)
    
    return adaptive_data

def generate_integration_strategies_comparison():
    """Generate comparison data for different integration strategies."""
    strategies = INTEGRATION_STRATEGIES
    
    # Create metrics for each strategy
    metrics = ["rmse", "mae", "r2", "auc"]
    scenarios = ["normal_operation", "missing_data", "concept_drift", "noisy_data"]
    
    integration_comparison = {
        "overall_metrics": {},
        "scenario_metrics": {scenario: {} for scenario in scenarios},
        "time_series_performance": {
            "timestamps": generate_timestamp_data(),
            "metrics": {strategy: {} for strategy in strategies}
        },
        "adaptation_speed": {strategy: round(random.uniform(0.5, 5.0), 2) for strategy in strategies}  # seconds
    }
    
    # Generate overall metrics for each strategy
    for strategy in strategies:
        # Base performance with variation by strategy
        strategy_factor = 1.0
        if strategy == "adaptive":
            strategy_factor = 0.85  # Adaptive performs better
        elif strategy == "confidence_based":
            strategy_factor = 0.9   # Confidence-based also does well
            
        integration_comparison["overall_metrics"][strategy] = {
            "rmse": round(0.15 * strategy_factor + random.uniform(-0.02, 0.02), 4),
            "mae": round(0.12 * strategy_factor + random.uniform(-0.015, 0.015), 4),
            "r2": round(0.8 / strategy_factor + random.uniform(-0.05, 0.05), 4),
            "auc": round(0.85 / strategy_factor + random.uniform(-0.03, 0.03), 4)
        }
        
        # Generate time series data for each metric
        for metric in metrics:
            base_value = integration_comparison["overall_metrics"][strategy][metric]
            
            if metric in ["rmse", "mae"]:  # Lower is better, so generate decreasing trend
                integration_comparison["time_series_performance"]["metrics"][strategy][metric] = \
                    generate_metric_time_series(base_value, trend_factor=0.001)
            else:  # Higher is better, so generate increasing trend
                integration_comparison["time_series_performance"]["metrics"][strategy][metric] = \
                    [round(1 - x, 4) if metric == "r2" else round(x, 4) 
                     for x in generate_metric_time_series(1 - base_value if metric == "r2" else base_value, 
                                                         trend_factor=-0.001)]
    
    # Generate scenario-specific metrics
    for scenario in scenarios:
        for strategy in strategies:
            base_metrics = integration_comparison["overall_metrics"][strategy].copy()
            
            # Adjust metrics based on scenario and strategy
            if scenario == "missing_data":
                if strategy in ["adaptive", "confidence_based"]:
                    # These strategies handle missing data better
                    adjustment = 1.1
                else:
                    adjustment = 1.3
            elif scenario == "concept_drift":
                if strategy == "adaptive":
                    # Adaptive handles drift best
                    adjustment = 1.05
                else:
                    adjustment = 1.25
            elif scenario == "noisy_data":
                if strategy == "confidence_based":
                    # Confidence-based handles noise best
                    adjustment = 1.1
                else:
                    adjustment = 1.2
            else:  # normal_operation
                adjustment = 1.0
            
            # Apply adjustments (worse for challenging scenarios)
            integration_comparison["scenario_metrics"][scenario][strategy] = {
                "rmse": round(base_metrics["rmse"] * adjustment, 4),
                "mae": round(base_metrics["mae"] * adjustment, 4),
                "r2": round(base_metrics["r2"] / adjustment, 4),
                "auc": round(base_metrics["auc"] / adjustment, 4)
            }
    
    return integration_comparison

def generate_pipeline_configuration_data():
    """Generate data about different pipeline configurations and their impact."""
    # Define different pipeline configurations
    pipeline_configs = {
        "standard": {
            "preprocessing": "standard",
            "feature_selection": "auto",
            "expert_training": "parallel",
            "optimization": "default"
        },
        "optimized": {
            "preprocessing": "enhanced",
            "feature_selection": "recursive",
            "expert_training": "sequential_tuned",
            "optimization": "hyperband"
        },
        "minimal": {
            "preprocessing": "minimal",
            "feature_selection": "none",
            "expert_training": "quick",
            "optimization": "minimal"
        },
        "production": {
            "preprocessing": "production",
            "feature_selection": "stability_based",
            "expert_training": "distributed",
            "optimization": "practical"
        }
    }
    
    # Generate performance metrics for each configuration
    pipeline_performance = {}
    
    for config_name, config in pipeline_configs.items():
        # Base performance varies by configuration
        if config_name == "optimized":
            base_factor = 0.85  # Optimized performs best
        elif config_name == "minimal":
            base_factor = 1.15  # Minimal performs worst
        elif config_name == "production":
            base_factor = 0.9   # Production is good but prioritizes stability over raw performance
        else:  # standard
            base_factor = 1.0
        
        # Generate metrics
        pipeline_performance[config_name] = {
            "metrics": {
                "rmse": round(0.15 * base_factor + random.uniform(-0.02, 0.02), 4),
                "mae": round(0.12 * base_factor + random.uniform(-0.015, 0.015), 4),
                "r2": round(0.8 / base_factor + random.uniform(-0.05, 0.05), 4),
                "training_time_minutes": round(20 * (2 - base_factor) + random.uniform(-5, 5), 1),
                "inference_time_ms": round(50 * base_factor + random.uniform(-10, 10), 1)
            },
            "configuration": config,
            "memory_usage_mb": round(200 * (2 - base_factor) + random.uniform(-20, 50), 1),
            "scalability_factor": round(1.0 / base_factor + random.uniform(-0.1, 0.1), 2)
        }
    
    return pipeline_performance

def generate_comprehensive_checkpoint():
    """Generate a complete checkpoint with all necessary data structures."""
    # Generate expert benchmarks
    expert_benchmarks = generate_expert_benchmarks()
    
    # Get expert names for gating evaluation
    expert_names = list(expert_benchmarks.keys())
    
    # Generate gating evaluation data
    gating_evaluation = generate_gating_evaluation(expert_names)
    
    # Generate end-to-end metrics
    end_to_end_metrics = generate_end_to_end_metrics(expert_benchmarks)
    
    # Generate expert-optimizer integration data
    expert_optimizer_integration = generate_expert_optimizer_integration()
    
    # Generate adaptive weighting data
    adaptive_weighting = generate_adaptive_weighting_data()
    
    # Generate integration strategies comparison
    integration_strategies = generate_integration_strategies_comparison()
    
    # Generate pipeline configuration data
    pipeline_configurations = generate_pipeline_configuration_data()
    
    # Assemble the complete checkpoint structure
    checkpoint_data = {
        "version": "2.0",
        "timestamp": datetime.datetime.now().isoformat(),
        "metadata": {
            "model_version": "MoE-v3.0",
            "dataset": "migraine-clinical-dataset-v3",
            "experiment_id": f"comprehensive-test-{TIMESTAMP}",
            "description": "Comprehensive MoE evaluation checkpoint with complete metrics"
        },
        "expert_benchmarks": expert_benchmarks,
        "gating_evaluation": gating_evaluation,
        "end_to_end_metrics": end_to_end_metrics,
        "expert_optimizer_integration": expert_optimizer_integration,
        "adaptive_weighting": adaptive_weighting,
        "integration_strategies": integration_strategies,
        "pipeline_configurations": pipeline_configurations
    }
    
    return checkpoint_data

def create_checkpoints():
    """Create comprehensive checkpoints for dashboard testing."""
    # Generate standard comprehensive checkpoint
    comprehensive_data = generate_comprehensive_checkpoint()
    checkpoint_name = f'checkpoint_comprehensive_{TIMESTAMP}.json'
    
    # Write the checkpoint file
    with open(OUTPUT_DIR / checkpoint_name, 'w') as f:
        json.dump(comprehensive_data, f, indent=2)
    
    print(f'Created comprehensive checkpoint: {checkpoint_name}')
    
    # Create variations with different data structures
    # 1. Flat format - move some nested metrics to top level
    flat_data = comprehensive_data.copy()
    flat_data.update({
        "overall_rmse": flat_data["end_to_end_metrics"]["overall"]["rmse"],
        "overall_mae": flat_data["end_to_end_metrics"]["overall"]["mae"],
        "overall_r2": flat_data["end_to_end_metrics"]["overall"]["r2"],
        "selection_accuracy": flat_data["gating_evaluation"]["selection_accuracy"]
    })
    flat_checkpoint_name = f'checkpoint_comprehensive_flat_{TIMESTAMP}.json'
    with open(OUTPUT_DIR / flat_checkpoint_name, 'w') as f:
        json.dump(flat_data, f, indent=2)
    print(f'Created flat format checkpoint: {flat_checkpoint_name}')
    
    # 2. Attribute style - Convert to a more object-style representation
    # This is for testing purposes only, to verify our helpers work with attribute access
    # The file is still JSON but when loaded in Python could be converted to SimpleNamespace
    attribute_checkpoint_name = f'checkpoint_comprehensive_attribute_{TIMESTAMP}.json'
    with open(OUTPUT_DIR / attribute_checkpoint_name, 'w') as f:
        json.dump(comprehensive_data, f, indent=2)
    print(f'Created attribute-style checkpoint: {attribute_checkpoint_name}')
    
    return [checkpoint_name, flat_checkpoint_name, attribute_checkpoint_name]

def create_sample_visualizations():
    """Create sample visualizations to validate the data before dashboard integration."""
    # Generate data to visualize
    checkpoint_data = generate_comprehensive_checkpoint()
    
    # 1. Expert Performance Comparison
    plt.figure(figsize=(10, 6))
    experts = list(checkpoint_data["expert_benchmarks"].keys())
    metrics = ["rmse", "mae", "r2"]
    
    # Create a different subplot for each metric
    for i, metric in enumerate(metrics):
        plt.subplot(1, 3, i+1)
        values = [checkpoint_data["expert_benchmarks"][expert][metric] for expert in experts]
        plt.bar(experts, values)
        plt.title(f"Expert {metric.upper()}")
        plt.xticks(rotation=45)
        plt.tight_layout()
    
    plt.savefig(TEMP_VISUALS_DIR / "expert_performance_comparison.png")
    plt.close()
    
    # 2. Gating Network Selection Frequency
    plt.figure(figsize=(8, 6))
    selection_freq = checkpoint_data["gating_evaluation"]["expert_utilization"]
    plt.pie(list(selection_freq.values()), labels=list(selection_freq.keys()), autopct='%1.1f%%')
    plt.title("Expert Selection Frequency")
    plt.savefig(TEMP_VISUALS_DIR / "expert_selection_frequency.png")
    plt.close()
    
    # 3. Temporal Performance
    plt.figure(figsize=(10, 6))
    timestamps = checkpoint_data["end_to_end_metrics"]["temporal"]["timestamps"]
    rmse_values = checkpoint_data["end_to_end_metrics"]["temporal"]["metrics"]["rmse"]
    
    # Convert timestamps to datetime objects and then to relative days
    dates = [datetime.datetime.fromisoformat(ts) for ts in timestamps]
    days = [(date - dates[0]).total_seconds() / (24 * 3600) for date in dates]
    
    plt.plot(days, rmse_values)
    plt.xlabel("Days")
    plt.ylabel("RMSE")
    plt.title("MoE Performance Over Time")
    plt.savefig(TEMP_VISUALS_DIR / "temporal_performance.png")
    plt.close()
    
    # 4. Optimizer Convergence Comparison
    plt.figure(figsize=(10, 6))
    optimizer = list(checkpoint_data["expert_optimizer_integration"].keys())[0]
    expert_data = checkpoint_data["expert_optimizer_integration"][optimizer]
    
    for expert_name, data in expert_data.items():
        plt.plot(data["convergence_curve"], label=expert_name)
    
    plt.xlabel("Iteration")
    plt.ylabel("Error")
    plt.title(f"Optimizer Convergence for {optimizer}")
    plt.legend()
    plt.savefig(TEMP_VISUALS_DIR / "optimizer_convergence.png")
    plt.close()
    
    # 5. Integration Strategies Comparison
    plt.figure(figsize=(10, 6))
    strategies = list(checkpoint_data["integration_strategies"]["overall_metrics"].keys())
    rmse_values = [checkpoint_data["integration_strategies"]["overall_metrics"][s]["rmse"] for s in strategies]
    
    plt.bar(strategies, rmse_values)
    plt.title("Integration Strategies RMSE Comparison")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(TEMP_VISUALS_DIR / "integration_strategies.png")
    plt.close()
    
    print(f"Created 5 sample visualizations in {TEMP_VISUALS_DIR}")
    return TEMP_VISUALS_DIR

def create_visualizations(checkpoint_data, output_dir=None):
    """
    Create visualizations from a checkpoint data dictionary.
    
    Args:
        checkpoint_data: Dictionary containing checkpoint data
        output_dir: Output directory for visualizations (default: TEMP_VISUALS_DIR)
    
    Returns:
        Dictionary of visualization paths
    """
    if output_dir is None:
        output_dir = TEMP_VISUALS_DIR
    
    # Convert output_dir to Path if it's a string
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Dictionary to store visualization paths
    vis_paths = {}
    
    # 1. Expert Benchmarks - Performance Comparison
    if 'expert_benchmarks' in checkpoint_data:
        try:
            plt.figure(figsize=(10, 6))
            experts = checkpoint_data['expert_benchmarks']
            metrics = ['rmse', 'mae', 'r2']
            
            x = np.arange(len(experts))
            width = 0.25
            
            for i, metric in enumerate(metrics):
                values = []
                labels = []
                
                for expert_name, expert_data in experts.items():
                    # Handle R² scaling for better visualization
                    value = expert_data.get(metric, 0)
                    if metric == 'r2':
                        value = value * 100  # Scale to 0-100
                    values.append(value)
                    labels.append(expert_name.replace('_expert', ''))
                
                plt.bar(x + i*width, values, width, label=metric.upper())
            
            plt.xlabel('Expert')
            plt.ylabel('Metric Value')
            plt.title('Expert Model Performance Comparison')
            plt.xticks(x + width, labels, rotation=45)
            plt.legend()
            plt.tight_layout()
            
            # Save the figure
            benchmark_path = output_dir / 'expert_benchmarks.png'
            plt.savefig(benchmark_path)
            vis_paths['expert_benchmarks'] = str(benchmark_path)
            plt.close()
        except Exception as e:
            print(f"Error creating expert benchmarks visualization: {str(e)}")
    
    # 2. Gating Network - Expert Selection
    if 'gating_evaluation' in checkpoint_data:
        try:
            plt.figure(figsize=(10, 6))
            gating_data = checkpoint_data['gating_evaluation']
            
            if 'expert_utilization' in gating_data:
                expert_util = gating_data['expert_utilization']
                
                # Create pie chart
                labels = []
                sizes = []
                
                for expert_name, util in expert_util.items():
                    labels.append(expert_name.replace('_expert', ''))
                    sizes.append(util)
                
                plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90)
                plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle
                plt.title('Expert Utilization Distribution')
                
                # Save the figure
                gating_path = output_dir / 'gating_expert_utilization.png'
                plt.savefig(gating_path)
                vis_paths['gating_expert_utilization'] = str(gating_path)
                plt.close()
        except Exception as e:
            print(f"Error creating gating visualization: {str(e)}")
    
    # 3. End-to-End Metrics - Temporal Performance
    if 'end_to_end_metrics' in checkpoint_data:
        try:
            end_to_end = checkpoint_data['end_to_end_metrics']
            
            if 'temporal_performance' in end_to_end and 'timestamps' in end_to_end['temporal_performance']:
                plt.figure(figsize=(12, 6))
                
                timestamps = end_to_end['temporal_performance']['timestamps']
                metrics = end_to_end['temporal_performance']['metrics']
                
                # Simplify timestamps for display
                x_labels = []
                for ts in timestamps:
                    try:
                        dt = datetime.datetime.fromisoformat(ts)
                        x_labels.append(dt.strftime('%H:%M'))
                    except:
                        x_labels.append(str(ts))
                
                # Plot each metric
                for metric_name, values in metrics.items():
                    plt.plot(range(len(values)), values, label=metric_name.upper())
                
                # Only show a subset of timestamps to avoid overcrowding
                step = max(1, len(x_labels) // 10)
                plt.xticks(range(0, len(x_labels), step), x_labels[::step], rotation=45)
                
                plt.xlabel('Time')
                plt.ylabel('Metric Value')
                plt.title('End-to-End Performance Over Time')
                plt.legend()
                plt.grid(True, linestyle='--', alpha=0.7)
                plt.tight_layout()
                
                # Save the figure
                temporal_path = output_dir / 'temporal_performance.png'
                plt.savefig(temporal_path)
                vis_paths['temporal_performance'] = str(temporal_path)
                plt.close()
        except Exception as e:
            print(f"Error creating temporal performance visualization: {str(e)}")
    
    # 4. Create overall metrics summary visualization
    try:
        plt.figure(figsize=(8, 6))
        
        # Get end-to-end metrics
        if 'end_to_end_metrics' in checkpoint_data:
            end_to_end = checkpoint_data['end_to_end_metrics']
            metrics = {'RMSE': end_to_end.get('rmse', 0), 
                      'MAE': end_to_end.get('mae', 0), 
                      'R²': end_to_end.get('r2', 0)}
            
            # Create bar chart for overall metrics
            plt.bar(metrics.keys(), metrics.values())
            plt.title('Overall Performance Metrics')
            plt.ylabel('Value')
            plt.grid(True, linestyle='--', alpha=0.7, axis='y')
            
            # Set different y-axis scale for R² (typically 0-1) vs error metrics
            if 'R²' in metrics and metrics['R²'] < 1.5:
                # Add a second y-axis for R²
                ax1 = plt.gca()
                ax2 = ax1.twinx()
                r2_pos = list(metrics.keys()).index('R²')
                ax2.bar([r2_pos], [metrics['R²']], color='green', alpha=0.7)
                ax2.set_ylabel('R² Value')
                ax2.set_ylim(0, 1)
            
            plt.tight_layout()
            
            # Save the figure
            summary_path = output_dir / 'overall_metrics.png'
            plt.savefig(summary_path)
            vis_paths['overall_metrics'] = str(summary_path)
            plt.close()
    except Exception as e:
        print(f"Error creating overall metrics visualization: {str(e)}")
    
    print(f"Created {len(vis_paths)} visualizations in {output_dir}")
    return vis_paths

if __name__ == "__main__":
    # Generate checkpoints
    created_checkpoints = create_checkpoints()
    print(f"\nSuccessfully created {len(created_checkpoints)} checkpoints:")
    
    # Create visualizations for the first checkpoint
    if created_checkpoints:
        with open(created_checkpoints[0], 'r') as f:
            checkpoint_data = json.load(f)
        
        create_visualizations(checkpoint_data)
    for checkpoint in created_checkpoints:
        print(f"- {checkpoint}")
    
    # Create sample visualizations
    visuals_dir = create_sample_visualizations()
    print(f"\nSample visualizations created in: {visuals_dir}")
    print("\nThese resources can be used to validate the data before dashboard integration.")
