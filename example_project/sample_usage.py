"""
Sample usage of the migraine prediction package.
"""
import pandas as pd
import numpy as np
from migraine_model import MigrainePredictor

def create_sample_data(num_samples=100):
    """Create sample data for demonstration."""
    np.random.seed(42)
    
    # Create features
    data = {
        'sleep_hours': np.random.normal(7, 1.5, num_samples),  # Mean 7, std 1.5
        'stress_level': np.random.randint(0, 11, num_samples),  # 0-10
        'weather_pressure': np.random.normal(1013, 10, num_samples),  # Normal atmospheric pressure
        'heart_rate': np.random.normal(75, 8, num_samples),  # Normal heart rate
        'hormonal_level': np.random.normal(5, 1, num_samples),  # Arbitrary units
    }
    
    # Create target: more likely to have migraine with high stress, low sleep, high pressure
    migraine_probability = (
        (10 - data['sleep_hours']) * 0.1 +  # Less sleep -> more migraines
        data['stress_level'] * 0.08 +        # More stress -> more migraines
        (data['weather_pressure'] - 1013) * 0.02 +  # Higher pressure -> more migraines
        (data['heart_rate'] - 75) * 0.01      # Higher heart rate -> slightly more migraines
    )
    
    # Normalize to 0-1 range
    migraine_probability = (migraine_probability - migraine_probability.min()) / (migraine_probability.max() - migraine_probability.min())
    
    # Generate binary outcome
    data['migraine_occurred'] = (migraine_probability > 0.5).astype(int)
    
    # Convert to DataFrame
    df = pd.DataFrame(data)
    
    return df

def main():
    """Run the full demo."""
    print("Creating sample migraine data...")
    data = create_sample_data(200)
    
    # Split into train/test
    train_data = data.iloc[:150]
    test_data = data.iloc[150:]
    
    print(f"Created {len(train_data)} training samples and {len(test_data)} test samples")
    print(f"Sample data:\n{train_data.head()}")
    
    # Save the data for later use
    train_data.to_csv("train_data.csv", index=False)
    test_data.to_csv("test_data.csv", index=False)
    
    # Create a predictor
    print("\nInitializing migraine predictor...")
    predictor = MigrainePredictor()
    
    # Train a model
    print("\nTraining model...")
    model_id = predictor.train(train_data, model_name="sample_model", description="Sample migraine model")
    print(f"Model trained with ID: {model_id}")
    
    # Evaluate the model
    print("\nEvaluating model...")
    metrics = predictor.evaluate(test_data)
    print(f"Model performance:")
    print(f"  Accuracy: {metrics['accuracy']:.4f}")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall: {metrics['recall']:.4f}")
    print(f"  F1 Score: {metrics['f1']:.4f}")
    
    # Make predictions
    print("\nMaking predictions on new data...")
    predictions = predictor.predict(test_data.iloc[:5])
    
    # Print predictions
    print("Predictions for the first 5 test samples:")
    for i, pred in enumerate(predictions):
        actual = test_data.iloc[i]['migraine_occurred']
        sample = test_data.iloc[i].drop('migraine_occurred').to_dict()
        print(f"Sample {i+1}:")
        for feature, value in sample.items():
            print(f"  {feature}: {value:.2f}")
        print(f"  Predicted: {'Migraine' if pred == 1 else 'No Migraine'}")
        print(f"  Actual: {'Migraine' if actual == 1 else 'No Migraine'}")
        print()
    
    # Get more detailed predictions
    print("\nGetting detailed predictions...")
    detailed_preds = predictor.predict_with_details(test_data.iloc[:2])
    print(f"Detailed predictions: {detailed_preds}")
    
    print("\nDemo completed successfully!")

if __name__ == "__main__":
    main()
