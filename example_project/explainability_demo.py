"""
Demonstration of explainability features in the migraine prediction package.
"""
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from migraine_model import MigrainePredictor

def run_explanation_demo():
    """Run the explainability demonstration."""
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
    
    # Train a model if one doesn't exist
    try:
        # Try to load the default model
        predictor.load_model()
        print("Loaded existing model")
    except:
        # Train a new model if one doesn't exist
        print("No model found. Training a new model...")
        model_id = predictor.train(train_data, model_name="explanation_model", 
                                   description="Model for explainability demo")
        print(f"Model trained with ID: {model_id}")
    
    # Check if the model has feature_importance in its metadata
    metadata = predictor.get_model_metadata()
    
    # Get feature importance from the model
    print("\nModel Feature Importance:")
    
    # Create output directory for plots
    os.makedirs("explanations", exist_ok=True)
    
    # Get feature importance from model evaluation
    metrics = predictor.evaluate(test_data)
    
    if 'feature_importance' in metrics:
        # Sort features by importance
        feature_importance = metrics['feature_importance']
        sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
        
        # Print feature importance
        for feature, importance in sorted_features:
            print(f"  {feature}: {importance:.4f}")
        
        # Create a bar plot of feature importance
        features, importances = zip(*sorted_features)
        
        plt.figure(figsize=(10, 6))
        plt.bar(features, importances)
        plt.xlabel('Features')
        plt.ylabel('Importance')
        plt.title('Feature Importance')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig('explanations/feature_importance.png')
        print(f"\nFeature importance plot saved to explanations/feature_importance.png")
    else:
        print("Feature importance not available in the model.")
    
    # Try to use SHAP explainer if available
    try:
        import shap
        
        print("\nGenerating SHAP explanations...")
        
        # Create a small dataset for explanation
        explain_data = test_data.sample(min(5, len(test_data)))
        
        # Get the model from the predictor
        model = predictor.model
        
        # Create a SHAP explainer
        explainer = shap.TreeExplainer(model)
        
        # Calculate SHAP values
        X = explain_data[predictor.feature_columns].values
        shap_values = explainer.shap_values(X)
        
        # Create SHAP summary plot
        plt.figure(figsize=(12, 8))
        shap.summary_plot(shap_values, explain_data[predictor.feature_columns], show=False)
        plt.tight_layout()
        plt.savefig('explanations/shap_summary.png')
        print(f"SHAP summary plot saved to explanations/shap_summary.png")
        
        # Create SHAP force plot for a single prediction
        plt.figure(figsize=(20, 3))
        shap.initjs()
        force_plot = shap.force_plot(explainer.expected_value[1], 
                                    shap_values[1][0], 
                                    explain_data[predictor.feature_columns].iloc[0],
                                    matplotlib=True,
                                    show=False)
        plt.savefig('explanations/shap_force_plot.png', bbox_inches='tight')
        print(f"SHAP force plot saved to explanations/shap_force_plot.png")
        
    except ImportError:
        print("\nSHAP package not installed. Install with: pip install shap")
    except Exception as e:
        print(f"\nError generating SHAP explanations: {e}")
    
    # Try to use LIME explainer if available
    try:
        from lime import lime_tabular
        
        print("\nGenerating LIME explanations...")
        
        # Create a LIME explainer
        lime_explainer = lime_tabular.LimeTabularExplainer(
            train_data[predictor.feature_columns].values,
            feature_names=predictor.feature_columns,
            class_names=['No Migraine', 'Migraine'],
            mode='classification'
        )
        
        # Get a sample for explanation
        explain_instance = test_data[predictor.feature_columns].iloc[0].values
        
        # Define a prediction function for LIME
        def predict_fn(x):
            return predictor.model.predict_proba(x)
        
        # Generate an explanation
        explanation = lime_explainer.explain_instance(
            explain_instance,
            predict_fn,
            num_features=len(predictor.feature_columns)
        )
        
        # Save explanation as HTML
        explanation.save_to_file('explanations/lime_explanation.html')
        print(f"LIME explanation saved to explanations/lime_explanation.html")
        
        # Plot the explanation
        plt.figure(figsize=(10, 6))
        explanation.as_pyplot_figure()
        plt.tight_layout()
        plt.savefig('explanations/lime_explanation.png')
        print(f"LIME explanation plot saved to explanations/lime_explanation.png")
        
    except ImportError:
        print("\nLIME package not installed. Install with: pip install lime")
    except Exception as e:
        print(f"\nError generating LIME explanations: {e}")
    
    print("\nExplainability demo completed!")

if __name__ == "__main__":
    run_explanation_demo()
