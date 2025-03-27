#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test script for the Enhanced Patient Data Generator
"""

import os
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

from utils.enhanced_synthetic_data import EnhancedPatientDataGenerator

def test_all_drift_types():
    """Test generator with all drift types"""
    print("Testing Enhanced Patient Data Generator with various drift types...")
    
    # Set output directory
    output_dir = Path("./test_data/enhanced_output")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize generator
    generator = EnhancedPatientDataGenerator(output_dir=output_dir)
    
    # Test configurations
    drift_types = ['none', 'sudden', 'gradual', 'recurring']
    
    all_patients = []
    
    for drift_type in drift_types:
        print(f"\nGenerating patient data with {drift_type} drift...")
        
        # Generate 3 patients for each drift type
        patient_ids = generator.generate_enhanced_patient_set(
            num_patients=3,
            time_periods=60,  # 60 days 
            samples_per_period=6,  # 6 samples per day
            drift_type=drift_type,
            drift_start_time=0.5,  # Drift starts halfway
            output_format='llif',
            include_evaluation=True,
            include_visualization=True
        )
        
        all_patients.extend(patient_ids)
    
    print(f"\nGenerated {len(all_patients)} patient datasets")
    print(f"Output saved to: {output_dir}")
    
    # Generate a patient summary
    summary = generator.create_patient_summary()
    print(f"Patient summary created with {len(summary['patients'])} patients")
    
    return all_patients, summary

def test_multimodal_data():
    """Test generator with multiple data modalities"""
    print("\nTesting Enhanced Patient Data Generator with multiple data modalities...")
    
    # Set output directory
    output_dir = Path("./test_data/multimodal_output")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize generator
    generator = EnhancedPatientDataGenerator(output_dir=output_dir)
    
    # Generate patients with all data modalities
    patient_ids = generator.generate_enhanced_patient_set(
        num_patients=5,
        time_periods=45,
        samples_per_period=8,
        drift_type='gradual',
        drift_start_time=0.55,  # Drift starts a bit past halfway
        output_format='llif',
        include_evaluation=True,
        include_visualization=True
    )
    
    print(f"Generated {len(patient_ids)} patient datasets with all data modalities")
    print(f"Output saved to: {output_dir}")
    
    return patient_ids

def test_evaluation_metrics():
    """Test the evaluation metrics functionality"""
    print("\nTesting evaluation metrics calculation...")
    
    # Set output directory
    output_dir = Path("./test_data/metrics_output")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize generator
    generator = EnhancedPatientDataGenerator(output_dir=output_dir)
    
    # Generate patients with different drift types for comparison
    configurations = [
        {'name': 'No Drift', 'drift_type': 'none'},
        {'name': 'Sudden Drift', 'drift_type': 'sudden'},
        {'name': 'Gradual Drift', 'drift_type': 'gradual'}
    ]
    
    results = {}
    
    for config in configurations:
        print(f"\nGenerating data with {config['name']}...")
        
        patient_ids = generator.generate_enhanced_patient_set(
            num_patients=10,
            time_periods=40,
            samples_per_period=6,
            drift_type=config['drift_type'],
            drift_start_time=0.5,
            output_format='llif',
            include_evaluation=True,
            include_visualization=True
        )
        
        # Get summary with metrics
        summary = generator.create_patient_summary()
        results[config['name']] = summary['overall_metrics']
    
    # Display comparison
    print("\nEvaluation Metrics Comparison:")
    metrics_df = pd.DataFrame(results).T
    print(metrics_df)
    
    return metrics_df

if __name__ == "__main__":
    print("Starting Enhanced Patient Data Generator Tests")
    
    # Test with different drift types
    all_patients, summary = test_all_drift_types()
    
    # Test with multiple data modalities
    multimodal_patients = test_multimodal_data()
    
    # Test evaluation metrics
    metrics_df = test_evaluation_metrics()
    
    print("\nAll tests completed successfully!")
