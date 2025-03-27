#!/usr/bin/env python
"""
Domain-Specific Preprocessing Pipeline Example

This script demonstrates how to use the domain-specific preprocessing operations
in a complete preprocessing pipeline for migraine data analysis.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
import os
import sys

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.preprocessing_pipeline import PreprocessingPipeline
from data.domain_specific_preprocessing import (
    MedicationNormalizer,
    SymptomExtractor,
    TemporalPatternExtractor,
    PhysiologicalSignalProcessor,
    ComorbidityAnalyzer,
    EnvironmentalTriggerAnalyzer,
    AdvancedFeatureEngineer
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def generate_sample_data(n_patients=10, records_per_patient=30):
    """
    Generate sample migraine data for demonstration purposes.
    
    Args:
        n_patients: Number of patients to generate data for
        records_per_patient: Number of records per patient
        
    Returns:
        DataFrame with sample migraine data
    """
    np.random.seed(42)
    
    # Create patient IDs and timestamps
    patient_ids = np.repeat(range(1, n_patients + 1), records_per_patient)
    start_date = datetime(2023, 1, 1)
    dates = []
    
    for patient_id in range(1, n_patients + 1):
        patient_dates = [start_date + timedelta(days=i) for i in range(records_per_patient)]
        dates.extend(patient_dates)
    
    # Create physiological features
    heart_rate = np.random.normal(75, 10, size=n_patients * records_per_patient)
    blood_pressure_sys = np.random.normal(120, 15, size=n_patients * records_per_patient)
    blood_pressure_dia = np.random.normal(80, 10, size=n_patients * records_per_patient)
    body_temperature = np.random.normal(36.8, 0.3, size=n_patients * records_per_patient)
    
    # Create environmental features
    temperature = np.random.normal(22, 5, size=n_patients * records_per_patient)
    humidity = np.random.normal(50, 15, size=n_patients * records_per_patient)
    pressure = np.random.normal(1013, 5, size=n_patients * records_per_patient)
    aqi = np.random.normal(50, 20, size=n_patients * records_per_patient)
    light_exposure = np.random.normal(500, 150, size=n_patients * records_per_patient)
    
    # Create behavioral features
    sleep_hours = np.random.normal(7, 1.5, size=n_patients * records_per_patient)
    stress_level = np.random.randint(1, 11, size=n_patients * records_per_patient)
    physical_activity = np.random.normal(30, 15, size=n_patients * records_per_patient)  # minutes
    
    # Create medication data
    medications = np.random.choice(
        ['Sumatriptan', 'Rizatriptan', 'Topiramate', 'Propranolol', 'Amitriptyline', 'None'],
        size=n_patients * records_per_patient,
        p=[0.2, 0.15, 0.1, 0.1, 0.05, 0.4]
    )
    dosages = []
    for med in medications:
        if med == 'Sumatriptan':
            dosages.append(f"{np.random.choice([25, 50, 100])} mg")
        elif med == 'Rizatriptan':
            dosages.append(f"{np.random.choice([5, 10])} mg")
        elif med == 'Topiramate':
            dosages.append(f"{np.random.choice([25, 50, 100, 200])} mg")
        elif med == 'Propranolol':
            dosages.append(f"{np.random.choice([20, 40, 80])} mg")
        elif med == 'Amitriptyline':
            dosages.append(f"{np.random.choice([10, 25, 50])} mg")
        else:
            dosages.append('')
    
    frequencies = np.random.choice(
        ['once daily', 'twice daily', 'as needed', 'three times daily', ''],
        size=n_patients * records_per_patient
    )
    
    # Create symptom data
    headache_severity = np.random.choice([0, 1, 2, 3, 4, 5], size=n_patients * records_per_patient)
    nausea = np.random.choice([0, 1], size=n_patients * records_per_patient, p=[0.7, 0.3])
    photophobia = np.random.choice([0, 1], size=n_patients * records_per_patient, p=[0.6, 0.4])
    phonophobia = np.random.choice([0, 1], size=n_patients * records_per_patient, p=[0.7, 0.3])
    aura = np.random.choice([0, 1], size=n_patients * records_per_patient, p=[0.8, 0.2])
    
    # Create symptom notes
    symptom_notes = []
    for i in range(n_patients * records_per_patient):
        notes = []
        if headache_severity[i] > 0:
            location = np.random.choice(['frontal', 'temporal', 'occipital', 'whole head'])
            quality = np.random.choice(['throbbing', 'pressure', 'stabbing', 'dull'])
            notes.append(f"Headache: {location}, {quality}, severity {headache_severity[i]}/5")
        
        if nausea[i]:
            notes.append("Nausea present")
        
        if photophobia[i]:
            notes.append("Light sensitivity")
        
        if phonophobia[i]:
            notes.append("Sound sensitivity")
        
        if aura[i]:
            aura_type = np.random.choice(['visual', 'sensory', 'speech', 'motor'])
            notes.append(f"Aura: {aura_type}")
        
        symptom_notes.append("; ".join(notes) if notes else "No symptoms")
    
    # Create comorbidity data
    anxiety = np.random.choice([0, 1], size=n_patients * records_per_patient, p=[0.8, 0.2])
    depression = np.random.choice([0, 1], size=n_patients * records_per_patient, p=[0.85, 0.15])
    insomnia = np.random.choice([0, 1], size=n_patients * records_per_patient, p=[0.9, 0.1])
    
    # Create target variable (migraine occurrence)
    # Higher probability of migraine when:
    # - Physiological factors are abnormal
    # - Environmental triggers are present
    # - Behavioral factors are poor
    # - Symptoms are severe
    migraine_prob = (
        0.05 +  # Base probability
        0.01 * (heart_rate - 75) +  # Heart rate effect
        0.005 * (blood_pressure_sys - 120) +  # Blood pressure effect
        0.05 * (7 - sleep_hours) +  # Sleep effect
        0.02 * stress_level +  # Stress effect
        0.03 * headache_severity +  # Headache effect
        0.1 * nausea +  # Nausea effect
        0.1 * photophobia +  # Photophobia effect
        0.1 * aura  # Aura effect
    )
    migraine_prob = np.clip(migraine_prob, 0, 1)
    migraine = np.random.binomial(1, migraine_prob)
    
    # Create location data
    locations = np.random.choice(['home', 'work', 'outside'], size=n_patients * records_per_patient)
    
    # Create the dataframe
    data = pd.DataFrame({
        'patient_id': patient_ids,
        'date': dates,
        # Physiological
        'heart_rate': heart_rate,
        'blood_pressure_sys': blood_pressure_sys,
        'blood_pressure_dia': blood_pressure_dia,
        'body_temperature': body_temperature,
        # Environmental
        'temperature': temperature,
        'humidity': humidity,
        'pressure': pressure,
        'aqi': aqi,
        'light_exposure': light_exposure,
        'location': locations,
        # Behavioral
        'sleep_hours': sleep_hours,
        'stress_level': stress_level,
        'physical_activity': physical_activity,
        # Medication
        'medication_name': medications,
        'dosage': dosages,
        'frequency': frequencies,
        # Symptoms
        'headache_severity': headache_severity,
        'nausea': nausea,
        'photophobia': photophobia,
        'phonophobia': phonophobia,
        'aura': aura,
        'symptom_notes': symptom_notes,
        # Comorbidities
        'anxiety': anxiety,
        'depression': depression,
        'insomnia': insomnia,
        # Target
        'migraine': migraine
    })
    
    return data


def create_preprocessing_pipeline():
    """
    Create a complete preprocessing pipeline for migraine data.
    
    Returns:
        PreprocessingPipeline: The configured preprocessing pipeline
    """
    # Create medication mappings
    medication_mappings = {
        'sumatriptan': ['Sumatriptan', 'sumatriptan', 'Imitrex', 'imitrex'],
        'rizatriptan': ['Rizatriptan', 'rizatriptan', 'Maxalt', 'maxalt'],
        'topiramate': ['Topiramate', 'topiramate', 'Topamax', 'topamax'],
        'propranolol': ['Propranolol', 'propranolol', 'Inderal', 'inderal'],
        'amitriptyline': ['Amitriptyline', 'amitriptyline', 'Elavil', 'elavil']
    }
    
    # Create symptom dictionary
    symptom_dict = {
        'headache': ['headache', 'head pain', 'head ache', 'cephalgia'],
        'nausea': ['nausea', 'nauseated', 'sick to stomach', 'queasy'],
        'vomiting': ['vomit', 'vomiting', 'threw up', 'emesis'],
        'photophobia': ['photophobia', 'light sensitivity', 'sensitive to light'],
        'phonophobia': ['phonophobia', 'sound sensitivity', 'sensitive to sound', 'noise sensitivity'],
        'aura': ['aura', 'visual aura', 'sensory aura', 'speech aura', 'motor aura'],
        'dizziness': ['dizzy', 'dizziness', 'vertigo', 'lightheaded'],
        'fatigue': ['fatigue', 'tired', 'exhausted', 'lethargy']
    }
    
    # Create feature groups for advanced feature engineering
    feature_groups = {
        'vitals': ['heart_rate', 'blood_pressure_sys', 'blood_pressure_dia', 'body_temperature'],
        'environment': ['temperature', 'humidity', 'pressure', 'aqi', 'light_exposure'],
        'behavior': ['sleep_hours', 'stress_level', 'physical_activity'],
        'symptoms': ['headache_severity', 'nausea', 'photophobia', 'phonophobia', 'aura']
    }
    
    # Create interaction pairs for advanced feature engineering
    interaction_pairs = [
        ('sleep_hours', 'stress_level'),
        ('heart_rate', 'physical_activity'),
        ('temperature', 'humidity'),
        ('headache_severity', 'stress_level')
    ]
    
    # Initialize preprocessing operations
    medication_normalizer = MedicationNormalizer(
        medication_cols=['medication_name'],
        dosage_cols=['dosage'],
        frequency_cols=['frequency'],
        medication_mappings=medication_mappings
    )
    
    symptom_extractor = SymptomExtractor(
        text_cols=['symptom_notes'],
        symptom_dict=symptom_dict,
        extract_severity=True
    )
    
    temporal_extractor = TemporalPatternExtractor(
        timestamp_col='date',
        target_col='migraine',
        patient_id_col='patient_id',
        extract_cyclical=True,
        extract_trends=True
    )
    
    physiological_processor = PhysiologicalSignalProcessor(
        vital_cols=['heart_rate', 'blood_pressure_sys', 'blood_pressure_dia', 'body_temperature'],
        patient_id_col='patient_id',
        timestamp_col='date',
        normalize_vitals=True,
        extract_variability=True
    )
    
    comorbidity_analyzer = ComorbidityAnalyzer(
        comorbidity_cols=['anxiety', 'depression', 'insomnia'],
        patient_id_col='patient_id',
        calculate_burden=True,
        extract_interactions=True
    )
    
    environmental_analyzer = EnvironmentalTriggerAnalyzer(
        weather_cols=['temperature', 'humidity', 'pressure'],
        pollution_cols=['aqi'],
        light_cols=['light_exposure'],
        timestamp_col='date',
        location_col='location'
    )
    
    feature_engineer = AdvancedFeatureEngineer(
        feature_groups=feature_groups,
        interaction_pairs=interaction_pairs,
        temporal_cols=['heart_rate', 'blood_pressure_sys', 'sleep_hours', 'stress_level'],
        timestamp_col='date',
        target_col='migraine',
        patient_id_col='patient_id',
        polynomial_degree=2,
        create_clusters=True,
        n_clusters=3
    )
    
    # Create the pipeline
    pipeline = PreprocessingPipeline(
        operations=[
            medication_normalizer,
            symptom_extractor,
            temporal_extractor,
            physiological_processor,
            comorbidity_analyzer,
            environmental_analyzer,
            feature_engineer
        ],
        name="MigraineDT_Preprocessing_Pipeline"
    )
    
    return pipeline


def main():
    """Main function to demonstrate the preprocessing pipeline."""
    logger.info("Generating sample migraine data...")
    data = generate_sample_data(n_patients=10, records_per_patient=30)
    logger.info(f"Generated dataset with {len(data)} records")
    
    logger.info("Creating preprocessing pipeline...")
    pipeline = create_preprocessing_pipeline()
    
    logger.info("Fitting preprocessing pipeline to data...")
    pipeline.fit(data)
    
    logger.info("Transforming data using preprocessing pipeline...")
    transformed_data = pipeline.transform(data)
    
    logger.info(f"Original data shape: {data.shape}")
    logger.info(f"Transformed data shape: {transformed_data.shape}")
    
    # Display sample of transformed data
    logger.info("\nSample of transformed data:")
    print(transformed_data.head())
    
    # Display feature summary
    logger.info("\nFeature summary:")
    feature_groups = {
        'Original features': data.columns.tolist(),
        'Medication features': [col for col in transformed_data.columns if 'medication' in col or 'dosage' in col or 'frequency' in col],
        'Symptom features': [col for col in transformed_data.columns if 'symptom' in col or 'severity' in col],
        'Temporal features': [col for col in transformed_data.columns if 'day' in col or 'month' in col or 'hour' in col or 'weekend' in col],
        'Physiological features': [col for col in transformed_data.columns if 'heart' in col or 'blood' in col or 'temperature' in col or 'vital' in col],
        'Comorbidity features': [col for col in transformed_data.columns if 'comorbidity' in col or 'anxiety' in col or 'depression' in col or 'insomnia' in col],
        'Environmental features': [col for col in transformed_data.columns if 'temperature' in col or 'humidity' in col or 'pressure' in col or 'aqi' in col or 'light' in col or 'env' in col],
        'Advanced features': [col for col in transformed_data.columns if 'interaction' in col or 'cluster' in col or 'polynomial' in col or 'lag' in col or 'rolling' in col]
    }
    
    for group_name, feature_list in feature_groups.items():
        print(f"\n{group_name} ({len(feature_list)}):")
        for feature in feature_list[:10]:  # Show only first 10 features per group
            print(f"  - {feature}")
        if len(feature_list) > 10:
            print(f"  - ... and {len(feature_list) - 10} more")
    
    # Save the pipeline
    pipeline_path = "migraine_preprocessing_pipeline.pkl"
    logger.info(f"\nSaving pipeline to {pipeline_path}...")
    pipeline.save(pipeline_path)
    logger.info("Pipeline saved successfully")
    
    # Save the transformed data
    data_path = "transformed_migraine_data.csv"
    logger.info(f"Saving transformed data to {data_path}...")
    transformed_data.to_csv(data_path, index=False)
    logger.info("Transformed data saved successfully")


if __name__ == "__main__":
    main()
