"""
Demonstration of optimizer features in the migraine prediction package.
"""
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from migraine_model import MigrainePredictor

def run_optimizer_demo():
    """Run the optimizer demonstration."""
    # Check if data files exist, if not, create them
    if not os.path.exists("train_data.csv") or not os.path.exists("test_data.csv"):
        print("Data files not found. Please run sample_usage.py first to generate the data.")
        return
    
    # Load the data
    train_data = pd.read_csv("train_data.csv")
    test_data = pd.read_csv("test_data.csv")
    
    print(f"Loaded {len(train_data)} training samples and {len(test_data)} test samples")
    
    # Create a predictor
    print("\nInitializing migraine predictor...")
    predictor = MigrainePredictor()
    
    # Try to use the optimize method
    print("\nAttempting to train model with optimization...")
    try:
        model_id = predictor.optimize(
            train_data,
            model_name="optimized_model",
            description="Model trained with optimization",
            optimizer="meta",  # Try meta-optimizer first
            max_evals=100
        )
        print(f"Model trained with optimization! Model ID: {model_id}")
        
        # Check if optimization was successful
        metadata = predictor.get_model_metadata()
        if 'optimizer_used' in metadata:
            print(f"Optimization successful using: {metadata['optimizer_used']}")
            print(f"Optimization score: {metadata.get('optimization_score', 'N/A')}")
            
            # Print feature weights if available
            if 'feature_weights' in metadata:
                print("\nOptimized feature weights:")
                feature_weights = metadata['feature_weights']
                for i, feature in enumerate(predictor.feature_columns):
                    print(f"  {feature}: {feature_weights[i]:.4f}")
        else:
            print("Optimization metadata not available - likely fell back to standard training.")
        
    except Exception as e:
        print(f"Error during optimization: {e}")
        print("Falling back to standard training...")
        model_id = predictor.train(train_data, model_name="standard_model")
        print(f"Model trained with standard approach. Model ID: {model_id}")
    
    # Evaluate the model on test data
    print("\nEvaluating the model...")
    metrics = predictor.evaluate(test_data)
    print("Model performance:")
    print(f"  Accuracy: {metrics['accuracy']:.4f}")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall: {metrics['recall']:.4f}")
    print(f"  F1 Score: {metrics['f1']:.4f}")
    
    # Try different optimizers if available
    optimizers = ["de", "es", "gwo", "aco"]
    
    for opt in optimizers:
        print(f"\nTrying optimizer: {opt.upper()}")
        try:
            model_id = predictor.optimize(
                train_data,
                model_name=f"{opt}_model",
                description=f"Model trained with {opt.upper()} optimizer",
                optimizer=opt,
                max_evals=50  # Fewer evaluations for demo purposes
            )
            print(f"Model trained with {opt.upper()} optimizer! Model ID: {model_id}")
            
            # Check if optimization was successful
            metadata = predictor.get_model_metadata()
            if 'optimizer_used' in metadata:
                print(f"Optimization successful using: {metadata['optimizer_used']}")
                print(f"Optimization score: {metadata.get('optimization_score', 'N/A')}")
            else:
                print("Optimization metadata not available - likely fell back to standard training.")
                
        except Exception as e:
            print(f"Error using {opt.upper()} optimizer: {e}")
    
    print("\nOptimizer demo completed!")

if __name__ == "__main__":
    run_optimizer_demo()
