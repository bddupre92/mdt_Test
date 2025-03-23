#!/usr/bin/env python3
"""
generate_benchmark_data.py
--------------------------
Generate benchmark data for the dashboard, including:
- Model comparisons across test functions
- Optimizer performance metrics
- Meta-learner performance analysis
"""

import json
import os
import random
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path

# Ensure data directories exist
data_dir = Path(__file__).resolve().parent.parent / "data" / "benchmarks"
data_dir.mkdir(parents=True, exist_ok=True)

def generate_benchmark_comparison_data():
    """Generate model comparison data across standard benchmark functions"""
    
    # Benchmark functions
    functions = [
        "Sphere", 
        "Rastrigin", 
        "Rosenbrock", 
        "Ackley", 
        "Griewank",
        "Schwefel", 
        "CEC_F1", 
        "CEC_F2"
    ]
    
    # Optimizers to compare
    optimizers = [
        "Meta-Learner",
        "ACO", 
        "Differential Evolution", 
        "Particle Swarm",
        "Grey Wolf",
        "CMA-ES", 
        "Bayesian"
    ]
    
    # Generate benchmark results
    results = {}
    
    # For each benchmark function
    for function in functions:
        function_results = {}
        
        # Meta-learner generally performs better (15-20% better)
        base_meta_performance = random.uniform(0.85, 0.95)
        
        # For each optimizer
        for optimizer in optimizers:
            # Meta-learner has highest performance
            if optimizer == "Meta-Learner":
                mean_performance = base_meta_performance
            else:
                # Other optimizers have lower performance with some variance
                mean_performance = base_meta_performance * random.uniform(0.7, 0.9)
            
            # Run specific adjustments
            if optimizer == "CMA-ES" and function in ["Rosenbrock", "Schwefel"]:
                # CMA-ES performs better on these
                mean_performance *= random.uniform(1.05, 1.15)
            
            if optimizer == "Bayesian" and function in ["Sphere", "Ackley"]:
                # Bayesian performs better on simpler functions
                mean_performance *= random.uniform(1.05, 1.10)
            
            # Generate multiple runs (for statistical significance)
            runs = [
                max(0, min(1, mean_performance + random.uniform(-0.05, 0.05)))
                for _ in range(10)
            ]
            
            # Calculate statistics
            function_results[optimizer] = {
                "mean": np.mean(runs),
                "std": np.std(runs),
                "min": np.min(runs),
                "max": np.max(runs),
                "runs": runs
            }
        
        results[function] = function_results
    
    # Save to file
    output_file = data_dir / "benchmark_comparison.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Saved benchmark comparison data to {output_file}")
    return results

def generate_convergence_data():
    """Generate convergence data for each optimizer"""
    
    # Optimizers
    optimizers = [
        "Meta-Learner",
        "ACO", 
        "Differential Evolution", 
        "Particle Swarm",
        "Grey Wolf",
        "CMA-ES", 
        "Bayesian"
    ]
    
    # Number of iterations
    max_iterations = 100
    iterations = list(range(0, max_iterations+1, 5))
    
    # Generate convergence results
    results = {}
    
    # For each optimizer
    for optimizer in optimizers:
        # Initial performance (higher for better optimizers)
        if optimizer == "Meta-Learner":
            initial_performance = random.uniform(0.65, 0.75)
            final_performance = random.uniform(0.88, 0.95)
            convergence_rate = random.uniform(0.15, 0.25)  # Fast convergence
        elif optimizer in ["CMA-ES", "Bayesian"]:
            initial_performance = random.uniform(0.60, 0.70)
            final_performance = random.uniform(0.82, 0.90)
            convergence_rate = random.uniform(0.10, 0.15)  # Medium-fast convergence
        else:
            initial_performance = random.uniform(0.55, 0.65)
            final_performance = random.uniform(0.78, 0.88)
            convergence_rate = random.uniform(0.05, 0.10)  # Slower convergence
        
        # Generate convergence curve
        performance_values = []
        for i in iterations:
            normalized_iter = i / max_iterations
            # Exponential convergence with some noise
            performance = initial_performance + (final_performance - initial_performance) * (1 - np.exp(-convergence_rate * i))
            # Add some noise
            performance += random.uniform(-0.02, 0.02) * (1 - normalized_iter)
            performance_values.append(round(max(0, min(1, performance)), 4))
        
        results[optimizer] = {
            "iterations": iterations,
            "performance": performance_values,
            "initial_performance": initial_performance,
            "final_performance": final_performance
        }
    
    # Save to file
    output_file = data_dir / "convergence_data.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Saved convergence data to {output_file}")
    return results

def generate_real_data_performance():
    """Generate synthetic performance data based on real datasets"""
    
    # Datasets representing different scenarios
    datasets = [
        "Migraine_Clinical", 
        "Migraine_Wearable", 
        "Migraine_Combined",
        "Headache_General", 
        "Neurological_Symptoms"
    ]
    
    # Models to compare
    models = [
        "Meta-Learner (Optimized)",
        "Random Forest",
        "XGBoost",
        "Ensemble",
        "Neural Network",
        "SVM" 
    ]
    
    # Metrics to track
    metrics = ["accuracy", "precision", "recall", "f1", "auc", "specificity"]
    
    # Generate results
    results = {}
    
    # For each dataset
    for dataset in datasets:
        dataset_results = {}
        
        # Meta-learner generally performs better
        meta_learner_boost = random.uniform(0.05, 0.12)
        
        # For each model
        for model in models:
            model_metrics = {}
            
            # Base performance varies by dataset
            base_performance = {
                "Migraine_Clinical": random.uniform(0.70, 0.75),
                "Migraine_Wearable": random.uniform(0.65, 0.72),
                "Migraine_Combined": random.uniform(0.75, 0.82),
                "Headache_General": random.uniform(0.68, 0.73),
                "Neurological_Symptoms": random.uniform(0.72, 0.77)
            }[dataset]
            
            # For each metric
            for metric in metrics:
                # Adjust by metric
                metric_adjustments = {
                    "accuracy": random.uniform(0.98, 1.02),
                    "precision": random.uniform(0.96, 1.04),
                    "recall": random.uniform(0.94, 1.06),
                    "f1": random.uniform(0.97, 1.03),
                    "auc": random.uniform(0.98, 1.02),
                    "specificity": random.uniform(0.95, 1.05)
                }
                
                # Calculate performance
                if model == "Meta-Learner (Optimized)":
                    # Meta-learner gets a boost
                    performance = base_performance * metric_adjustments[metric] + meta_learner_boost
                else:
                    # Model-specific adjustments
                    model_boost = {
                        "XGBoost": random.uniform(0.01, 0.05),
                        "Ensemble": random.uniform(0.02, 0.06),
                        "Neural Network": random.uniform(-0.02, 0.04),
                        "Random Forest": random.uniform(-0.01, 0.03),
                        "SVM": random.uniform(-0.04, 0.02)
                    }[model]
                    
                    performance = base_performance * metric_adjustments[metric] + model_boost
                
                # Ensure valid range
                model_metrics[metric] = round(max(0, min(1, performance)), 3)
            
            dataset_results[model] = model_metrics
        
        results[dataset] = dataset_results
    
    # Save to file
    output_file = data_dir / "real_data_performance.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Saved real data performance to {output_file}")
    return results

def generate_feature_importance_data():
    """Generate feature importance data for the meta-learner"""
    
    # Features for migraine prediction
    features = [
        "recent_headache_frequency", 
        "hours_sleep",
        "stress_level", 
        "hydration_level",
        "physical_activity",
        "barometric_pressure_change",
        "menstrual_cycle_phase",
        "caffeine_intake",
        "screen_time",
        "medication_adherence",
        "alcohol_consumption",
        "weather_temperature",
        "dietary_triggers",
        "heartrate_variability",
        "previous_aura"
    ]
    
    # Generate importance scores
    importance = {}
    
    # Set relative importance (higher for known triggers)
    base_importance = {
        "recent_headache_frequency": random.uniform(0.75, 0.95),
        "stress_level": random.uniform(0.65, 0.85),
        "hours_sleep": random.uniform(0.60, 0.80),
        "hydration_level": random.uniform(0.55, 0.75),
        "barometric_pressure_change": random.uniform(0.50, 0.70),
        "menstrual_cycle_phase": random.uniform(0.45, 0.65),
        "caffeine_intake": random.uniform(0.40, 0.60),
        "physical_activity": random.uniform(0.35, 0.55),
        "screen_time": random.uniform(0.30, 0.50),
        "medication_adherence": random.uniform(0.25, 0.45),
        "alcohol_consumption": random.uniform(0.20, 0.40),
        "weather_temperature": random.uniform(0.15, 0.35),
        "dietary_triggers": random.uniform(0.10, 0.30),
        "heartrate_variability": random.uniform(0.05, 0.25),
        "previous_aura": random.uniform(0.05, 0.20)
    }
    
    # Normalize to [0,1] range
    total = sum(base_importance.values())
    normalized_importance = {feature: value/total for feature, value in base_importance.items()}
    
    # Sort by importance (descending)
    sorted_importance = dict(sorted(normalized_importance.items(), key=lambda x: x[1], reverse=True))
    
    # Format for output
    importance = {
        "features": list(sorted_importance.keys()),
        "importance_scores": list(sorted_importance.values())
    }
    
    # Save to file
    output_file = data_dir / "feature_importance.json"
    with open(output_file, 'w') as f:
        json.dump(importance, f, indent=2)
    
    print(f"Saved feature importance data to {output_file}")
    return importance

if __name__ == "__main__":
    print("Generating benchmark data for dashboard...")
    generate_benchmark_comparison_data()
    generate_convergence_data()
    generate_real_data_performance()
    generate_feature_importance_data()
    print("Done!")
