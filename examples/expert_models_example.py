"""
Expert Models Example

This script demonstrates how to use the expert models in the MoE framework.
It shows how to:
1. Load and preprocess data
2. Create and configure expert models
3. Train and evaluate the models
4. Make predictions with confidence scores
5. Analyze feature importance

This is a complete workflow example that can be used as a reference for
implementing expert models in your own applications.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import logging
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

from moe_framework.experts.physiological_expert import PhysiologicalExpert
from moe_framework.experts.environmental_expert import EnvironmentalExpert
from moe_framework.experts.behavioral_expert import BehavioralExpert
from moe_framework.experts.medication_history_expert import MedicationHistoryExpert

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_sample_data(n_samples=500):
    """
    Load or generate sample data for demonstration.
    
    In a real application, you would load your data from a file or database.
    Here we generate synthetic data for demonstration purposes.
    
    Args:
        n_samples: Number of samples to generate
        
    Returns:
        DataFrame with sample data
    """
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Create timestamps
    timestamps = pd.date_range(start='2023-01-01', periods=n_samples, freq='D')
    
    # Create patient IDs and locations
    patient_ids = np.random.choice(['P001', 'P002', 'P003', 'P004', 'P005'], n_samples)
    locations = np.random.choice(['New York', 'Boston', 'Chicago', 'Los Angeles', 'Seattle'], n_samples)
    
    # Create physiological features
    heart_rate = np.random.normal(75, 10, n_samples)
    blood_pressure_sys = np.random.normal(120, 15, n_samples)
    blood_pressure_dia = np.random.normal(80, 10, n_samples)
    temperature = np.random.normal(37, 0.5, n_samples)
    
    # Create environmental features
    env_temperature = np.random.normal(20, 8, n_samples)
    humidity = np.random.normal(60, 15, n_samples)
    pressure = np.random.normal(1013, 10, n_samples)
    aqi = np.random.normal(50, 20, n_samples)
    
    # Create behavioral features
    sleep_duration = np.random.normal(7, 1.5, n_samples)
    sleep_quality = np.random.normal(70, 15, n_samples)
    activity_level = np.random.normal(60, 20, n_samples)
    stress_level = np.random.normal(50, 25, n_samples)
    
    # Create medication features
    medication_a = np.random.choice(['', 'Low', 'Medium', 'High'], n_samples)
    medication_b = np.random.choice(['', 'Low', 'Medium', 'High'], n_samples)
    medication_c = np.random.choice(['', 'Low', 'Medium', 'High'], n_samples)
    
    # Create target variable (migraine severity)
    # Each domain contributes to the target with some noise
    physio_effect = 0.3 * heart_rate + 0.2 * blood_pressure_sys
    env_effect = 0.25 * env_temperature + 0.15 * humidity + 0.1 * aqi
    behavior_effect = -0.2 * sleep_quality + 0.3 * stress_level
    
    # Medication effects (higher medication levels reduce severity)
    med_a_effect = np.where(medication_a == '', 0, 
                   np.where(medication_a == 'Low', -5, 
                   np.where(medication_a == 'Medium', -10, -15)))
    
    med_b_effect = np.where(medication_b == '', 0, 
                   np.where(medication_b == 'Low', -3, 
                   np.where(medication_b == 'Medium', -7, -12)))
    
    med_c_effect = np.where(medication_c == '', 0, 
                   np.where(medication_c == 'Low', -2, 
                   np.where(medication_c == 'Medium', -5, -8)))
    
    # Combine effects with noise
    migraine_severity = (
        50 +  # baseline
        physio_effect + 
        env_effect + 
        behavior_effect + 
        med_a_effect + 
        med_b_effect + 
        med_c_effect + 
        np.random.normal(0, 10, n_samples)  # random noise
    )
    
    # Ensure severity is in a reasonable range (0-100)
    migraine_severity = np.clip(migraine_severity, 0, 100)
    
    # Create DataFrame
    data = pd.DataFrame({
        # Identifiers
        'patient_id': patient_ids,
        'location': locations,
        'date': timestamps,
        
        # Physiological features
        'heart_rate': heart_rate,
        'blood_pressure_sys': blood_pressure_sys,
        'blood_pressure_dia': blood_pressure_dia,
        'temperature': temperature,
        
        # Environmental features
        'env_temperature': env_temperature,
        'humidity': humidity,
        'pressure': pressure,
        'aqi': aqi,
        
        # Behavioral features
        'sleep_duration': sleep_duration,
        'sleep_quality': sleep_quality,
        'activity_level': activity_level,
        'stress_level': stress_level,
        
        # Medication features
        'medication_a': medication_a,
        'medication_b': medication_b,
        'medication_c': medication_c,
        
        # Target
        'migraine_severity': migraine_severity
    })
    
    return data


def create_expert_models():
    """
    Create and configure the expert models.
    
    Returns:
        Dictionary of expert models
    """
    # Create physiological expert
    physiological_expert = PhysiologicalExpert(
        vital_cols=['heart_rate', 'blood_pressure_sys', 'blood_pressure_dia', 'temperature'],
        patient_id_col='patient_id',
        timestamp_col='date',
        normalize_vitals=True,
        extract_variability=True
    )
    
    # Create environmental expert
    environmental_expert = EnvironmentalExpert(
        env_cols=['env_temperature', 'humidity', 'pressure', 'aqi'],
        location_col='location',
        timestamp_col='date',
        include_weather=True,
        include_pollution=True
    )
    
    # Create behavioral expert
    behavioral_expert = BehavioralExpert(
        behavior_cols=['sleep_duration', 'sleep_quality', 'activity_level', 'stress_level'],
        patient_id_col='patient_id',
        timestamp_col='date',
        include_sleep=True,
        include_activity=True,
        include_stress=True
    )
    
    # Create medication history expert
    medication_expert = MedicationHistoryExpert(
        medication_cols=['medication_a', 'medication_b', 'medication_c'],
        patient_id_col='patient_id',
        timestamp_col='date',
        include_dosage=True,
        include_frequency=True,
        include_interactions=True
    )
    
    # Return dictionary of experts
    return {
        'physiological': physiological_expert,
        'environmental': environmental_expert,
        'behavioral': behavioral_expert,
        'medication': medication_expert
    }


def train_and_evaluate_experts(experts, data, test_size=0.2, random_state=42):
    """
    Train and evaluate the expert models.
    
    Args:
        experts: Dictionary of expert models
        data: DataFrame with training data
        test_size: Fraction of data to use for testing
        random_state: Random seed for reproducibility
        
    Returns:
        Dictionary of evaluation results
    """
    # Split data into training and testing sets
    X = data.drop('migraine_severity', axis=1)
    y = data['migraine_severity']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    logger.info(f"Training data shape: {X_train.shape}")
    logger.info(f"Testing data shape: {X_test.shape}")
    
    # Train and evaluate each expert
    results = {}
    
    for name, expert in experts.items():
        logger.info(f"Training {name} expert...")
        
        # Train the expert
        expert.fit(X_train, y_train)
        
        # Evaluate on test data
        metrics = expert.evaluate(X_test, y_test)
        
        # Make predictions
        predictions = expert.predict(X_test)
        
        # Store results
        results[name] = {
            'expert': expert,
            'metrics': metrics,
            'predictions': predictions,
            'true_values': y_test
        }
        
        # Log evaluation results
        logger.info(f"{name.capitalize()} Expert - RMSE: {metrics['rmse']:.2f}, R²: {metrics['r2']:.2f}")
    
    return results


def analyze_feature_importance(experts_results):
    """
    Analyze and visualize feature importance for each expert.
    
    Args:
        experts_results: Dictionary of evaluation results
    """
    # Create directory for plots if it doesn't exist
    os.makedirs('plots', exist_ok=True)
    
    # Analyze feature importance for each expert
    for name, results in experts_results.items():
        expert = results['expert']
        
        # Get feature importance
        importance = expert.feature_importances
        
        # Sort by importance
        importance = dict(sorted(importance.items(), key=lambda x: x[1], reverse=True))
        
        # Take top 10 features
        top_features = dict(list(importance.items())[:10])
        
        # Plot feature importance
        plt.figure(figsize=(10, 6))
        plt.barh(list(top_features.keys()), list(top_features.values()))
        plt.xlabel('Importance')
        plt.ylabel('Feature')
        plt.title(f'Top 10 Features - {name.capitalize()} Expert')
        plt.tight_layout()
        plt.savefig(f'plots/{name}_feature_importance.png')
        plt.close()
        
        # Log top features
        logger.info(f"Top 5 features for {name} expert: {list(top_features.keys())[:5]}")


def predict_with_confidence(experts_results, X_new):
    """
    Make predictions with confidence scores for new data.
    
    Args:
        experts_results: Dictionary of evaluation results
        X_new: New data to predict
        
    Returns:
        Dictionary of predictions and confidence scores
    """
    predictions = {}
    
    for name, results in experts_results.items():
        expert = results['expert']
        
        # Make predictions with confidence
        preds, confidence = expert.predict_with_confidence(X_new)
        
        # Store results
        predictions[name] = {
            'predictions': preds,
            'confidence': confidence
        }
        
        # Log average confidence
        logger.info(f"{name.capitalize()} Expert - Avg Confidence: {np.mean(confidence):.2f}")
    
    return predictions


def save_experts(experts_results, save_dir='models'):
    """
    Save trained expert models to disk.
    
    Args:
        experts_results: Dictionary of evaluation results
        save_dir: Directory to save models
    """
    # Create directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    
    # Save each expert
    for name, results in experts_results.items():
        expert = results['expert']
        
        # Save to disk
        filepath = os.path.join(save_dir, f"{name}_expert.pkl")
        expert.save(filepath)
        
        logger.info(f"Saved {name} expert to {filepath}")


def main():
    """Main function to run the example."""
    # Load sample data
    logger.info("Loading sample data...")
    data = load_sample_data(n_samples=500)
    
    # Create expert models
    logger.info("Creating expert models...")
    experts = create_expert_models()
    
    # Train and evaluate experts
    logger.info("Training and evaluating experts...")
    results = train_and_evaluate_experts(experts, data)
    
    # Analyze feature importance
    logger.info("Analyzing feature importance...")
    analyze_feature_importance(results)
    
    # Make predictions for new data
    logger.info("Making predictions for new data...")
    # In a real application, you would load new data
    # Here we just use a subset of the test data
    X_new = data.iloc[-10:].drop('migraine_severity', axis=1)
    predictions = predict_with_confidence(results, X_new)
    
    # Save trained models
    logger.info("Saving trained models...")
    save_experts(results)
    
    logger.info("Example completed successfully!")


if __name__ == "__main__":
    main()
