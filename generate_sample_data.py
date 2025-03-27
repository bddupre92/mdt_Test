#!/usr/bin/env python
"""
Generate sample data for the MoE framework.

This script creates a sample dataset similar to what the pytest fixture would generate,
but without requiring pytest to run.
"""

import os
import sys
import argparse
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Add the project root to the Python path
project_root = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, project_root)

def generate_sample_data(n_samples=100, output_path=None):
    """Generate sample data for testing."""
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Create timestamps
    timestamps = pd.date_range(start='2023-01-01', periods=n_samples, freq='D')
    
    # Create patient IDs and locations
    patient_ids = np.random.choice(['P001', 'P002', 'P003'], n_samples)
    locations = np.random.choice(['New York', 'Boston', 'Chicago'], n_samples)
    
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
    
    # Save to file if path provided
    if output_path:
        data.to_csv(output_path, index=False)
        print(f"Sample data saved to {output_path}")
    
    return data

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate sample data for MoE framework")
    parser.add_argument("--samples", type=int, default=100, help="Number of samples to generate")
    parser.add_argument("--output", type=str, default="sample_data.csv", help="Output file path")
    
    args = parser.parse_args()
    
    # Generate and save data
    generate_sample_data(n_samples=args.samples, output_path=args.output) 