#!/usr/bin/env python
"""
MoE Framework Phase 1 Demo

This script demonstrates the use of the Phase 1 components of the MoE framework:
1. Universal Data Connector
2. Data Quality Assessment
3. Simple Upload Interface
4. One-Click Execution workflow

It shows how to load a sample dataset, assess its quality, and run the execution pipeline.
"""

import os
import sys
import logging
import pandas as pd
import numpy as np
import json
from pathlib import Path

# Add the project root to the path so we can import the modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from moe_framework.data_connectors.file_connector import FileDataConnector
from moe_framework.data_connectors.data_quality import DataQualityAssessment
from moe_framework.upload.upload_manager import UploadManager
from moe_framework.execution.execution_pipeline import ExecutionPipeline

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def create_sample_dataset(output_path):
    """Create a sample dataset for demonstration purposes."""
    # Create a directory for sample data if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Generate sample data
    np.random.seed(42)
    n_samples = 1000
    
    # Features
    age = np.random.normal(45, 15, n_samples).astype(int)
    age = np.clip(age, 18, 90)
    
    # Physiological features
    blood_pressure_systolic = np.random.normal(120, 15, n_samples).astype(int)
    blood_pressure_diastolic = np.random.normal(80, 10, n_samples).astype(int)
    heart_rate = np.random.normal(75, 10, n_samples).astype(int)
    
    # Environmental features
    temperature = np.random.normal(22, 5, n_samples)
    humidity = np.random.normal(60, 15, n_samples)
    pressure = np.random.normal(1013, 10, n_samples)
    
    # Behavioral features
    sleep_hours = np.random.normal(7, 1.5, n_samples)
    stress_level = np.random.normal(5, 2, n_samples)
    stress_level = np.clip(stress_level, 0, 10)
    
    # Medication features
    medication_dose = np.random.choice([0, 50, 100, 150, 200], n_samples)
    
    # Target: migraine (0 = no, 1 = yes)
    # Higher probability of migraine with:
    # - Higher stress
    # - Lower sleep
    # - Higher blood pressure
    # - Extreme temperatures
    migraine_prob = (
        0.1 +
        0.02 * stress_level +
        0.02 * (10 - sleep_hours) +
        0.01 * (blood_pressure_systolic - 120) / 10 +
        0.01 * np.abs(temperature - 22) / 2
    )
    migraine_prob = np.clip(migraine_prob, 0.05, 0.95)
    migraine = np.random.binomial(1, migraine_prob)
    
    # Create DataFrame
    df = pd.DataFrame({
        'age': age,
        'blood_pressure_systolic': blood_pressure_systolic,
        'blood_pressure_diastolic': blood_pressure_diastolic,
        'heart_rate': heart_rate,
        'temperature': temperature,
        'humidity': humidity,
        'pressure': pressure,
        'sleep_hours': sleep_hours,
        'stress_level': stress_level,
        'medication_dose': medication_dose,
        'migraine': migraine
    })
    
    # Add some missing values
    for col in df.columns:
        if col != 'migraine':  # Don't add missing values to target
            mask = np.random.random(n_samples) < 0.05  # 5% missing values
            df.loc[mask, col] = np.nan
    
    # Save to CSV
    df.to_csv(output_path, index=False)
    logger.info(f"Created sample dataset with {n_samples} samples at {output_path}")
    
    return output_path

def demo_file_connector(file_path):
    """Demonstrate the use of the FileDataConnector."""
    logger.info("\n=== Demonstrating FileDataConnector ===")
    
    # Initialize the connector
    connector = FileDataConnector(verbose=True)
    
    # Connect to the file
    connection_params = {'file_path': file_path}
    if connector.connect(connection_params):
        logger.info("Successfully connected to the file")
    else:
        logger.error("Failed to connect to the file")
        return
    
    # Load the data
    data = connector.load_data()
    logger.info(f"Loaded data with shape: {data.shape}")
    
    # Get schema information
    schema = connector.get_schema()
    logger.info(f"Schema: {json.dumps(schema, indent=2)}")
    
    # Get metadata
    metadata = connector.get_metadata()
    logger.info(f"Metadata: {json.dumps(metadata, indent=2)}")
    
    # Save data to a different format
    json_path = os.path.splitext(file_path)[0] + '.json'
    connector.save_data(data, json_path, 'json')
    logger.info(f"Saved data to JSON format at {json_path}")
    
    return data

def demo_data_quality(data, target_column='migraine'):
    """Demonstrate the use of the DataQualityAssessment."""
    logger.info("\n=== Demonstrating DataQualityAssessment ===")
    
    # Initialize the quality assessment
    quality = DataQualityAssessment(verbose=True)
    
    # Assess data quality
    quality_results = quality.assess_quality(data, target_column)
    
    # Print quality results
    logger.info(f"Quality Score: {quality_results['quality_score']:.2f}")
    logger.info(f"Missing Data: {quality_results['missing_data_score']:.2f}")
    logger.info(f"Outliers: {quality_results['outlier_score']:.2f}")
    logger.info(f"Class Imbalance: {quality_results['class_imbalance_score']:.2f}")
    logger.info(f"Feature Correlation: {quality_results['feature_correlation_score']:.2f}")
    
    # Get EC algorithm recommendations
    recommendations = quality.get_algorithm_recommendations(quality_results)
    logger.info(f"Algorithm Recommendations: {json.dumps(recommendations, indent=2)}")
    
    return quality_results

def demo_upload_manager(file_path):
    """Demonstrate the use of the UploadManager."""
    logger.info("\n=== Demonstrating UploadManager ===")
    
    # Initialize the upload manager
    upload_manager = UploadManager(verbose=True)
    
    # Validate the file
    validation_results = upload_manager.validate_file(file_path)
    logger.info(f"Validation Results: {json.dumps(validation_results, indent=2)}")
    
    # Upload the file
    upload_results = upload_manager.upload_file(file_path)
    logger.info(f"Upload Results: {json.dumps(upload_results, indent=2)}")
    
    # Process the uploaded file
    if upload_results['success']:
        processing_results = upload_manager.process_uploaded_file(
            upload_results['upload_path'],
            target_column='migraine'
        )
        logger.info(f"Processing Results: {json.dumps(processing_results, indent=2)}")
    
    # Get upload history
    history = upload_manager.get_upload_history()
    logger.info(f"Upload History: {len(history)} records")
    
    # Save upload history
    upload_manager.save_upload_history()
    
    return upload_results

def demo_execution_pipeline(file_path):
    """Demonstrate the use of the ExecutionPipeline."""
    logger.info("\n=== Demonstrating ExecutionPipeline ===")
    
    # Initialize the execution pipeline
    pipeline = ExecutionPipeline(verbose=True)
    
    # Execute the pipeline
    execution_results = pipeline.execute(
        file_path,
        target_column='migraine'
    )
    
    # Print execution results
    if execution_results['success']:
        logger.info(f"Execution completed successfully")
        logger.info(f"Execution ID: {execution_results['execution_id']}")
        logger.info(f"Results Path: {execution_results['results_path']}")
        logger.info(f"Summary: {json.dumps(execution_results['summary'], indent=2)}")
    else:
        logger.error(f"Execution failed: {execution_results['message']}")
    
    return execution_results

def main():
    """Main function to run the demo."""
    logger.info("Starting MoE Framework Phase 1 Demo")
    
    # Create a sample dataset
    sample_data_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        'data',
        'sample_migraine_data.csv'
    )
    create_sample_dataset(sample_data_path)
    
    # Demonstrate FileDataConnector
    data = demo_file_connector(sample_data_path)
    
    # Demonstrate DataQualityAssessment
    quality_results = demo_data_quality(data)
    
    # Demonstrate UploadManager
    upload_results = demo_upload_manager(sample_data_path)
    
    # Demonstrate ExecutionPipeline
    execution_results = demo_execution_pipeline(sample_data_path)
    
    logger.info("MoE Framework Phase 1 Demo completed successfully")

if __name__ == "__main__":
    main()
