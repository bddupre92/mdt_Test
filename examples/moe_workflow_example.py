#!/usr/bin/env python
"""
MoE Framework End-to-End Workflow Example

This script demonstrates how to use the Mixture of Experts (MoE) pipeline
for a complete end-to-end workflow from data loading to prediction.
"""

import os
import logging
import numpy as np
import pandas as pd
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Import the MoE pipeline
from moe_framework.workflow.moe_pipeline import MoEPipeline

def generate_synthetic_data(n_samples=1000, n_features=20, n_informative=10, noise=0.5, seed=42):
    """Generate synthetic data for demonstration."""
    X, y = make_regression(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=n_informative,
        noise=noise,
        random_state=seed
    )
    
    # Create a pandas DataFrame for easier handling
    feature_names = [f'feature_{i+1}' for i in range(n_features)]
    df = pd.DataFrame(X, columns=feature_names)
    df['target'] = y
    
    # Add domain-specific features
    # Physiological
    df['heart_rate'] = np.random.normal(70, 10, size=n_samples)
    df['blood_pressure'] = np.random.normal(120, 15, size=n_samples)
    df['temperature'] = np.random.normal(98.6, 0.7, size=n_samples)
    
    # Behavioral
    df['sleep_hours'] = np.random.normal(7, 1.5, size=n_samples)
    df['stress_level'] = np.random.normal(5, 2, size=n_samples)
    df['physical_activity'] = np.random.normal(30, 15, size=n_samples)
    
    # Environmental
    df['temperature_outside'] = np.random.normal(75, 15, size=n_samples)
    df['humidity'] = np.random.normal(60, 20, size=n_samples)
    df['barometric_pressure'] = np.random.normal(1013, 10, size=n_samples)
    
    # Medication
    df['medication_dose'] = np.random.choice([0, 5, 10, 15, 20], size=n_samples)
    df['days_since_last_dose'] = np.random.randint(0, 30, size=n_samples)
    
    # Add patient_id column for demonstration
    df['patient_id'] = np.random.choice(['P001', 'P002', 'P003', 'P004', 'P005'], size=n_samples)
    
    # Save to CSV file
    os.makedirs('data', exist_ok=True)
    csv_path = os.path.join('data', 'synthetic_migraine_data.csv')
    df.to_csv(csv_path, index=False)
    
    return csv_path

def main():
    """Run the MoE workflow example."""
    print("\n=== MoE Framework End-to-End Workflow Example ===\n")
    
    # Step 1: Generate synthetic data
    print("Generating synthetic data...")
    data_path = generate_synthetic_data()
    print(f"Data saved to {data_path}")
    
    # Step 2: Configure and initialize the MoE pipeline
    print("\nInitializing MoE pipeline...")
    config = {
        'output_dir': 'results',
        'environment': 'dev',
        'experts': {
            'use_physiological': True,
            'use_behavioral': True,
            'use_environmental': True,
            'use_medication_history': True,
            'physiological': {
                'feature_patterns': ['heart_rate', 'blood_pressure', 'temperature']
            },
            'behavioral': {
                'feature_patterns': ['sleep_hours', 'stress_level', 'physical_activity']
            },
            'environmental': {
                'feature_patterns': ['temperature_outside', 'humidity', 'barometric_pressure']
            },
            'medication_history': {
                'feature_patterns': ['medication_dose', 'days_since_last_dose']
            }
        },
        'gating': {
            'type': 'quality_aware',
            'params': {
                'quality_weight': 0.7
            }
        },
        'meta_learner': {
            'method': 'bayesian',
            'exploration_factor': 0.2,
            'history_weight': 0.7,
            'quality_impact': 0.4,
            'drift_impact': 0.3,
            'memory_storage_dir': 'patient_memory',
            'enable_personalization': True
        }
    }
    
    pipeline = MoEPipeline(config=config, verbose=True)
    print("MoE pipeline initialized successfully.")
    
    # Step 3: Load data
    print("\nLoading data...")
    load_result = pipeline.load_data(data_path, target_column='target')
    if load_result.get('success', False):
        print(f"Data loaded successfully. Shape: {load_result.get('data_shape')}")
        print(f"Quality score: {load_result.get('quality_score', 0.0):.2f}")
    else:
        print(f"Data loading failed: {load_result.get('message', 'Unknown error')}")
        return
        
    # Step 4: Train the MoE system
    print("\nTraining the MoE system...")
    train_result = pipeline.train(validation_split=0.2, random_state=42)
    if train_result.get('success', False):
        print("Training completed successfully.")
        
        # Print expert results
        print("\nExpert training results:")
        for expert_id, result in train_result.get('expert_results', {}).items():
            status = "Success" if result.get('success', False) else "Failed"
            score = result.get('train_score', 'N/A')
            print(f"  {expert_id}: {status}, Score: {score}")
            
        # Print gating result
        gating_result = train_result.get('gating_result', {})
        gating_status = "Success" if gating_result.get('success', False) else "Failed"
        print(f"\nGating network training: {gating_status}")
        print(f"  Message: {gating_result.get('message', 'N/A')}")
    else:
        print(f"Training failed: {train_result.get('message', 'Unknown error')}")
        return
        
    # Step 5: Set patient for personalization
    print("\nSetting patient context for personalization...")
    patient_id = 'P001'
    pipeline.set_patient(patient_id)
    print(f"Patient context set to {patient_id}")
    
    # Step 6: Generate predictions
    print("\nGenerating predictions...")
    predict_result = pipeline.predict(use_loaded_data=True)
    if predict_result.get('success', False):
        print("Prediction completed successfully.")
        
        # Print expert weights
        print("\nExpert weights:")
        for expert_id, weight in predict_result.get('expert_weights', {}).items():
            print(f"  {expert_id}: {weight:.4f}")
            
        # Plot expert weights
        plt.figure(figsize=(10, 6))
        weights = predict_result.get('expert_weights', {})
        plt.bar(weights.keys(), weights.values())
        plt.title('Expert Weights for Prediction')
        plt.xlabel('Expert')
        plt.ylabel('Weight')
        plt.ylim(0, 1)
        plt.tight_layout()
        plt.savefig('expert_weights.png')
        print("Expert weights plot saved to 'expert_weights.png'")
    else:
        print(f"Prediction failed: {predict_result.get('message', 'Unknown error')}")
        return
        
    # Step 7: Evaluate the MoE system
    print("\nEvaluating the MoE system...")
    eval_result = pipeline.evaluate()
    if eval_result.get('success', False):
        print("Evaluation completed successfully.")
        
        # Print overall metrics
        print("\nOverall metrics:")
        for metric_name, value in eval_result.get('metrics', {}).items():
            print(f"  {metric_name}: {value:.4f}")
            
        # Print expert metrics
        print("\nExpert metrics:")
        for expert_id, metrics in eval_result.get('expert_metrics', {}).items():
            print(f"  {expert_id}:")
            for metric_name, value in metrics.items():
                print(f"    {metric_name}: {value:.4f}")
                
        # Plot comparison of expert performance
        plt.figure(figsize=(12, 8))
        
        # Extract MSE values for comparison
        expert_mse = {}
        for expert_id, metrics in eval_result.get('expert_metrics', {}).items():
            expert_mse[expert_id] = metrics.get('mse', 0)
        
        # Add combined model MSE
        expert_mse['combined'] = eval_result.get('metrics', {}).get('mse', 0)
        
        # Sort by MSE value (lower is better)
        expert_mse = {k: v for k, v in sorted(expert_mse.items(), key=lambda item: item[1])}
        
        # Create the bar plot
        plt.bar(expert_mse.keys(), expert_mse.values())
        plt.title('MSE Comparison: Individual Experts vs Combined MoE')
        plt.xlabel('Expert / Combined Model')
        plt.ylabel('Mean Squared Error (lower is better)')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig('expert_performance_comparison.png')
        print("Expert performance comparison plot saved to 'expert_performance_comparison.png'")
    else:
        print(f"Evaluation failed: {eval_result.get('message', 'Unknown error')}")
        
    print("\n=== MoE Workflow Example Completed ===\n")

if __name__ == "__main__":
    main()
