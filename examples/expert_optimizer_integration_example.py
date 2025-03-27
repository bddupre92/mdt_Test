"""
Expert-Optimizer Integration Example

This example demonstrates how to use the expert-optimizer integration to optimize
hyperparameters for different expert models in the MoE framework.
"""

import numpy as np
import pandas as pd
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split

from moe_framework.experts.physiological_expert import PhysiologicalExpert
from moe_framework.experts.environmental_expert import EnvironmentalExpert
from moe_framework.experts.behavioral_expert import BehavioralExpert
from moe_framework.experts.medication_history_expert import MedicationHistoryExpert
from meta_optimizer.evaluation.expert_evaluation_functions import create_evaluation_function

# Create synthetic data for demonstration
def create_synthetic_data(n_samples=100, n_features=20, random_state=42):  # Reduced sample size for debugging
    """Create synthetic data for example purposes"""
    X, y = make_regression(
        n_samples=n_samples, 
        n_features=n_features,
        noise=0.5,
        random_state=random_state
    )
    
    # Create feature names
    feature_names = [
        # Physiological features
        'heart_rate', 'blood_pressure', 'respiration_rate', 'body_temperature', 'skin_conductance',
        # Environmental features
        'temperature', 'humidity', 'barometric_pressure', 'air_quality', 'light_intensity',
        # Behavioral features
        'sleep_hours', 'physical_activity', 'stress_level', 'caffeine_intake', 'screen_time',
        # Medication features
        'medication_a_dose', 'medication_b_dose', 'days_since_last_dose', 'treatment_duration', 'medication_adherence'
    ]
    
    # Convert to DataFrame
    df = pd.DataFrame(X, columns=feature_names)
    target = pd.Series(y, name='migraine_intensity')
    
    return df, target

# Main example function
def run_example():
    print("Expert-Optimizer Integration Example")
    print("------------------------------------")
    
    # Generate synthetic data
    print("Generating synthetic data...")
    X, y = create_synthetic_data()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Create expert instances
    print("\nInitializing experts...")
    
    # Physiological expert
    physio_cols = ['heart_rate', 'blood_pressure', 'respiration_rate', 
                  'body_temperature', 'skin_conductance']
    physio_expert = PhysiologicalExpert(
        vital_cols=physio_cols,
        normalize_vitals=True,
        extract_variability=True
    )
    
    # Environmental expert
    env_cols = ['temperature', 'humidity', 'barometric_pressure', 
               'air_quality', 'light_intensity']
    env_expert = EnvironmentalExpert(
        env_cols=env_cols,
        include_weather=True,
        include_pollution=True
    )
    
    # Behavioral expert
    behavior_cols = ['sleep_hours', 'physical_activity', 'stress_level', 
                    'caffeine_intake', 'screen_time']
    behavior_expert = BehavioralExpert(
        behavior_cols=behavior_cols,
        include_sleep=True,
        include_activity=True,
        include_stress=True
    )
    
    # Medication history expert
    med_cols = ['medication_a_dose', 'medication_b_dose', 'days_since_last_dose', 
               'treatment_duration', 'medication_adherence']
    med_expert = MedicationHistoryExpert(
        medication_cols=med_cols,
        include_dosage=True,
        include_frequency=True,
        include_interactions=True
    )
    
    # Define hyperparameter optimization spaces for each expert
    print("\nDefining hyperparameter spaces...")
    
    # Physiological expert hyperparameters - REDUCED FOR DEBUGGING
    physio_param_bounds = {
        'n_estimators': (50, 200),
        'max_depth': (3, 10)  # Reduced dimensions for debugging
    }
    physio_param_types = {
        'n_estimators': 'int',
        'max_depth': 'int'
    }
    
    # Environmental expert hyperparameters - REDUCED FOR DEBUGGING
    env_param_bounds = {
        'learning_rate': (0.01, 0.3),
        'n_estimators': (50, 200)  # Reduced dimensions for debugging
    }
    env_param_types = {
        'learning_rate': 'float',
        'n_estimators': 'int'
    }
    
    # Behavioral expert hyperparameters - REDUCED FOR DEBUGGING
    behavior_param_bounds = {
        'n_estimators': (50, 200),
        'max_features': (0.3, 1.0)  # Reduced dimensions for debugging
    }
    behavior_param_types = {
        'n_estimators': 'int',
        'max_features': 'float'
    }
    
    # Medication expert hyperparameters - REDUCED FOR DEBUGGING
    med_param_bounds = {
        'learning_rate': (0.01, 0.3),
        'max_iter': (50, 200)  # Reduced dimensions for debugging
    }
    med_param_types = {
        'learning_rate': 'float',
        'max_iter': 'int'
    }
    
    # Run hyperparameter optimization for each expert
    print("\nOptimizing hyperparameters for physiological expert...")
    physio_opt_params = physio_expert.optimize_hyperparameters(
        X_train[physio_cols], 
        y_train,
        param_bounds=physio_param_bounds,
        param_types=physio_param_types,
        max_iterations=1,  # Extremely reduced for quick debugging
        cv_folds=2
    )
    print(f"Optimized parameters: {physio_opt_params}")
    print(f"Performance metrics: {physio_expert.quality_metrics}")
    
    print("\nOptimizing hyperparameters for environmental expert...")
    env_opt_params = env_expert.optimize_hyperparameters(
        X_train[env_cols], 
        y_train,
        param_bounds=env_param_bounds,
        param_types=env_param_types,
        max_iterations=1,  # Extremely reduced for quick debugging
        cv_folds=2
    )
    print(f"Optimized parameters: {env_opt_params}")
    print(f"Performance metrics: {env_expert.quality_metrics}")
    
    print("\nOptimizing hyperparameters for behavioral expert...")
    behavior_opt_params = behavior_expert.optimize_hyperparameters(
        X_train[behavior_cols], 
        y_train,
        param_bounds=behavior_param_bounds,
        param_types=behavior_param_types,
        max_iterations=1,  # Extremely reduced for quick debugging
        cv_folds=2
    )
    print(f"Optimized parameters: {behavior_opt_params}")
    print(f"Performance metrics: {behavior_expert.quality_metrics}")
    
    print("\nOptimizing hyperparameters for medication history expert...")
    med_opt_params = med_expert.optimize_hyperparameters(
        X_train[med_cols], 
        y_train,
        param_bounds=med_param_bounds,
        param_types=med_param_types,
        max_iterations=1,  # Extremely reduced for quick debugging
        cv_folds=2
    )
    print(f"Optimized parameters: {med_opt_params}")
    print(f"Performance metrics: {med_expert.quality_metrics}")
    
    # Demonstrate early stopping setup
    print("\nSetting up early stopping...")
    physio_expert.setup_early_stopping(monitor_metric='val_loss', patience=5)
    env_expert.setup_early_stopping(monitor_metric='val_mae', patience=7)
    behavior_expert.setup_early_stopping(monitor_metric='val_score', patience=10)
    med_expert.setup_early_stopping(monitor_metric='val_response_score', patience=8)
    
    print("\nExample complete!")


if __name__ == "__main__":
    run_example()
