"""
Synthetic data generator for MoE framework testing and benchmarking.

This script creates a synthetic dataset with features relevant to different
expert domains (physiological, behavioral, environmental, medication).
"""

import numpy as np
import pandas as pd
import os
from pathlib import Path

def generate_synthetic_medical_data(n_samples=1000, seed=42):
    """
    Generate synthetic medical data for MoE framework testing.
    
    Args:
        n_samples: Number of samples to generate
        seed: Random seed for reproducibility
        
    Returns:
        DataFrame with synthetic medical data
    """
    np.random.seed(seed)
    
    # Generate patient IDs
    patient_ids = np.array([f"P{i:04d}" for i in range(100)])
    
    # Generate timestamps (one per day for the past n_samples/100 days)
    days_per_patient = n_samples // 100
    base_date = pd.Timestamp('2024-01-01')
    timestamps = []
    all_patient_ids = []
    
    for patient_id in patient_ids:
        for day in range(days_per_patient):
            timestamps.append(base_date + pd.Timedelta(days=day))
            all_patient_ids.append(patient_id)
    
    # Generate physiological features
    heart_rate = 70 + 15 * np.random.randn(n_samples)
    blood_pressure_sys = 120 + 10 * np.random.randn(n_samples)
    blood_pressure_dia = 80 + 8 * np.random.randn(n_samples)
    temperature = 37 + 0.5 * np.random.randn(n_samples)
    
    # Generate behavioral features
    sleep_hours = 7 + 1.5 * np.random.randn(n_samples)
    steps = 8000 + 2000 * np.random.randn(n_samples)
    exercise_minutes = 30 + 15 * np.random.randn(n_samples)
    stress_level = np.random.randint(1, 11, n_samples)
    
    # Generate environmental features
    temperature_env = 22 + 5 * np.random.randn(n_samples)
    humidity = 50 + 10 * np.random.randn(n_samples)
    barometric_pressure = 1013 + 5 * np.random.randn(n_samples)
    air_quality = 50 + 20 * np.random.randn(n_samples)
    
    # Generate medication features
    medication_a_dose = np.random.choice([0, 5, 10, 15], n_samples)
    medication_b_dose = np.random.choice([0, 25, 50, 75], n_samples)
    days_since_last_medication = np.random.randint(0, 8, n_samples)
    
    # Generate target variable (migraine intensity: 0-10)
    # Influenced by all domains with some noise
    target = (
        0.3 * (blood_pressure_sys - 120) / 10 +
        0.2 * (temperature - 37) / 0.5 +
        -0.25 * (sleep_hours - 7) / 1.5 +
        0.15 * (stress_level - 5) / 5 +
        0.2 * (barometric_pressure - 1013) / 5 +
        0.1 * (humidity - 50) / 10 +
        -0.3 * (medication_a_dose) / 15 +
        -0.2 * (medication_b_dose) / 75 +
        0.15 * days_since_last_medication / 7 +
        np.random.randn(n_samples) * 0.5  # Add noise
    )
    
    # Scale to 0-10 range
    target = 5 + 3 * target
    target = np.clip(target, 0, 10)
    
    # Create DataFrame
    data = pd.DataFrame({
        'patient_id': all_patient_ids,
        'timestamp': timestamps,
        
        # Physiological features
        'heart_rate': heart_rate,
        'blood_pressure_sys': blood_pressure_sys,
        'blood_pressure_dia': blood_pressure_dia,
        'temperature': temperature,
        
        # Behavioral features
        'sleep_hours': sleep_hours,
        'steps': steps,
        'exercise_minutes': exercise_minutes,
        'stress_level': stress_level,
        
        # Environmental features
        'temperature_env': temperature_env,
        'humidity': humidity,
        'barometric_pressure': barometric_pressure,
        'air_quality': air_quality,
        
        # Medication features
        'medication_a_dose': medication_a_dose,
        'medication_b_dose': medication_b_dose,
        'days_since_last_medication': days_since_last_medication,
        
        # Target variable
        'migraine_intensity': target
    })
    
    # Add some missing values to simulate real-world data
    for col in data.columns:
        if col not in ['patient_id', 'timestamp', 'migraine_intensity']:
            mask = np.random.random(n_samples) < 0.05  # 5% missing values
            data.loc[mask, col] = np.nan
    
    return data

if __name__ == "__main__":
    # Generate synthetic data
    data = generate_synthetic_medical_data(n_samples=1000)
    
    # Create data directory if it doesn't exist
    data_dir = Path(__file__).parent.parent / 'data'
    os.makedirs(data_dir, exist_ok=True)
    
    # Save to CSV
    data_path = data_dir / 'synthetic_medical_data.csv'
    data.to_csv(data_path, index=False)
    
    print(f"Generated synthetic data with {len(data)} samples and saved to {data_path}")
    print(f"Feature columns: {[col for col in data.columns if col != 'migraine_intensity']}")
    print(f"Target column: migraine_intensity")
