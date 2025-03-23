"""
Main entry point for the migraine prediction package.
"""

import argparse
import sys
import os
import pandas as pd
import pickle
import numpy as np
import json
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

# Add original project path to Python path
original_project_path = os.path.dirname(project_root)
sys.path.append(original_project_path)

# Import package modules
from src.migraine_model.migraine_predictor import MigrainePredictor
from src.pipeline.data_ingestion import DataIngestion

# Try to import explainability modules from original codebase
try:
    from explainability.explainer_factory import ExplainerFactory
    EXPLAINABILITY_AVAILABLE = True
except ImportError:
    logger.warning("Explainability modules not available. Explainability features will be disabled.")
    EXPLAINABILITY_AVAILABLE = False


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Migraine Prediction CLI")
    
    # Add subparsers for different commands
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Train command
    train_parser = subparsers.add_parser("train", help="Train a new model")
    train_parser.add_argument("--data", type=str, required=True, help="Path to training data CSV")
    train_parser.add_argument("--model-name", type=str, default="migraine_model", help="Name for the model")
    train_parser.add_argument("--description", type=str, default="", help="Model description")
    train_parser.add_argument("--model-dir", type=str, default="models", help="Directory to save model")
    train_parser.add_argument("--summary", action="store_true", help="Print summary of training results")
    train_parser.add_argument("--test-size", type=float, default=0.2, help="Proportion of data to use for testing")
    
    # Predict command
    predict_parser = subparsers.add_parser("predict", help="Make predictions")
    predict_parser.add_argument("--data", type=str, required=True, help="Path to prediction data CSV")
    predict_parser.add_argument("--model-id", type=str, help="Model ID to use (default: use default model)")
    predict_parser.add_argument("--output", type=str, help="Path to save prediction results")
    predict_parser.add_argument("--summary", action="store_true", help="Print summary of prediction results")
    
    # Export command
    export_parser = subparsers.add_parser("export", help="Export model to pickle")
    export_parser.add_argument("--model-id", type=str, help="Model ID to export (default: use default model)")
    export_parser.add_argument("--output", type=str, required=True, help="Path to save pickle file")
    
    # List command
    list_parser = subparsers.add_parser("list", help="List available models")
    list_parser.add_argument("--model-dir", type=str, default="models", help="Models directory")
    
    # Explainability command if available
    if EXPLAINABILITY_AVAILABLE:
        explain_parser = subparsers.add_parser("explain", help="Explain model predictions")
        explain_parser.add_argument("--data", type=str, required=True, help="Path to data CSV")
        explain_parser.add_argument("--model-id", type=str, help="Model ID to explain (default: use default model)")
        explain_parser.add_argument("--explainer", type=str, default="feature_importance", 
                                   choices=["feature_importance", "optimizer", "shap", "lime"], 
                                   help="Explainer type to use")
        explain_parser.add_argument("--explain-plots", action="store_true", help="Generate explanation plots")
        explain_parser.add_argument("--explain-plot-types", type=str, nargs="+", 
                                   default=["bar", "summary"], 
                                   help="Types of plots to generate")
        explain_parser.add_argument("--explain-samples", type=int, default=5, 
                                   help="Number of samples to use for explanation")
        explain_parser.add_argument("--output-dir", type=str, default="explanations", 
                                   help="Directory to save explanation plots")
        explain_parser.add_argument("--summary", action="store_true", help="Print summary of explanation results")
    
    # Load test data command
    load_parser = subparsers.add_parser("load", help="Load and combine test data")
    load_parser.add_argument("--data-dir", type=str, default="test_data", help="Directory containing test data")
    load_parser.add_argument("--output", type=str, default="combined_test_data.csv", 
                            help="Path to save combined data")
    load_parser.add_argument("--summary", action="store_true", help="Print summary of loaded data")
    
    # Parse arguments
    args = parser.parse_args()
    
    # Handle commands
    if args.command == "train":
        train_model(args)
    elif args.command == "predict":
        make_predictions(args)
    elif args.command == "export":
        export_model(args)
    elif args.command == "list":
        list_models(args)
    elif args.command == "explain" and EXPLAINABILITY_AVAILABLE:
        explain_model(args)
    elif args.command == "load":
        load_and_combine_data(args)
    else:
        parser.print_help()


def train_model(args):
    """Train a new model."""
    print(f"Loading data from {args.data}...")
    data_ingestion = DataIngestion()
    data = data_ingestion.load_csv(args.data)
    
    # Clean data
    data = data_ingestion.clean_data(data)
    
    # Split data
    train_data, test_data = data_ingestion.split_data(data, test_size=args.test_size)
    
    print(f"Training model with {len(train_data)} samples...")
    predictor = MigrainePredictor(model_dir=args.model_dir)
    model_id = predictor.train(
        data=train_data,
        model_name=args.model_name,
        description=args.description,
        make_default=True
    )
    
    print(f"Model trained successfully! Model ID: {model_id}")
    
    # Evaluate on test data
    y_true = test_data['migraine_occurred'].values
    predictions = []
    for _, row in test_data.iterrows():
        prediction = predictor.predict(pd.DataFrame([row]))
        predictions.append(prediction['prediction'])
    
    # Calculate accuracy
    accuracy = np.mean(np.array(predictions) == y_true)
    print(f"Test accuracy: {accuracy:.4f}")
    
    # Print feature importance
    feature_importance = predictor.get_feature_importance()
    print("\nFeature Importance:")
    for feature, importance in sorted(feature_importance.items(), key=lambda x: x[1], reverse=True):
        print(f"  {feature}: {importance:.4f}")
    
    # Print detailed summary if requested
    if args.summary:
        print("\nTraining Summary:")
        print(f"  Model ID: {model_id}")
        print(f"  Model Name: {args.model_name}")
        print(f"  Description: {args.description}")
        print(f"  Training samples: {len(train_data)}")
        print(f"  Test samples: {len(test_data)}")
        print(f"  Test accuracy: {accuracy:.4f}")
        print(f"  Feature importance: {json.dumps(feature_importance, indent=2)}")
        
        # Print model metadata
        models = predictor.list_models()
        current_model = next((m for m in models if m['id'] == model_id), None)
        if current_model:
            print(f"  Created: {pd.to_datetime(current_model['created_at'], unit='s')}")
            print(f"  Version: {current_model['version']}")


def make_predictions(args):
    """Make predictions using a trained model."""
    print(f"Loading data from {args.data}...")
    data_ingestion = DataIngestion()
    data = data_ingestion.load_csv(args.data)
    
    # Clean data
    data = data_ingestion.clean_data(data)
    
    print(f"Making predictions for {len(data)} samples...")
    predictor = MigrainePredictor()
    
    if args.model_id:
        print(f"Using model: {args.model_id}")
        predictor.load_model(args.model_id)
    else:
        print("Using default model")
        predictor.load_model()
    
    # Make predictions for each row
    results = []
    for _, row in data.iterrows():
        prediction = predictor.predict(pd.DataFrame([row]))
        results.append({
            **{col: row[col] for col in data.columns},
            "predicted_probability": prediction["probability"],
            "predicted_migraine": prediction["prediction"]
        })
    
    # Create results DataFrame
    results_df = pd.DataFrame(results)
    
    # Save results if output specified
    if args.output:
        results_df.to_csv(args.output, index=False)
        print(f"Predictions saved to {args.output}")
    
    # Print summary
    if args.summary or not args.output:
        total = len(results_df)
        predicted_positive = results_df["predicted_migraine"].sum()
        print(f"\nPrediction Summary:")
        print(f"  Total samples: {total}")
        print(f"  Predicted migraines: {predicted_positive} ({predicted_positive/total:.1%})")
        
        if "migraine_occurred" in results_df.columns:
            actual_positive = results_df["migraine_occurred"].sum()
            correct_predictions = (results_df["predicted_migraine"] == results_df["migraine_occurred"]).sum()
            accuracy = correct_predictions / total
            print(f"  Actual migraines: {actual_positive} ({actual_positive/total:.1%})")
            print(f"  Accuracy: {accuracy:.1%}")


def export_model(args):
    """Export model to pickle file."""
    predictor = MigrainePredictor()
    
    if args.model_id:
        print(f"Exporting model: {args.model_id}")
        predictor.load_model(args.model_id)
    else:
        print("Exporting default model")
        predictor.load_model()
    
    print(f"Saving model to {args.output}...")
    predictor.save_as_pickle(args.output)
    print(f"Model exported successfully to {args.output}")


def list_models(args):
    """List available models."""
    predictor = MigrainePredictor(model_dir=args.model_dir)
    models = predictor.list_models()
    
    if not models:
        print("No models found.")
        return
    
    print(f"Found {len(models)} models:")
    for i, model in enumerate(models, 1):
        is_default = model.get("is_default", False)
        default_marker = " (default)" if is_default else ""
        created_date = pd.to_datetime(model["created_at"], unit="s").strftime("%Y-%m-%d %H:%M:%S")
        
        print(f"{i}. {model['name']}{default_marker}")
        print(f"   ID: {model['id']}")
        print(f"   Created: {created_date}")
        if model.get("description"):
            print(f"   Description: {model['description']}")
        print()


def explain_model(args):
    """Explain model predictions using the original explainability framework."""
    if not EXPLAINABILITY_AVAILABLE:
        print("Explainability modules not available. Please ensure the original explainability framework is installed.")
        return
    
    print(f"Loading data from {args.data}...")
    data_ingestion = DataIngestion()
    data = data_ingestion.load_csv(args.data)
    
    # Clean data
    data = data_ingestion.clean_data(data)
    
    # Load model
    predictor = MigrainePredictor()
    if args.model_id:
        print(f"Using model: {args.model_id}")
        predictor.load_model(args.model_id)
    else:
        print("Using default model")
        predictor.load_model()
    
    print(f"Generating explanations using {args.explainer} explainer...")
    
    # Prepare data for explanation
    X = data[predictor.feature_columns].values
    if hasattr(predictor.scaler, 'mean_'):
        X = predictor.scaler.transform(X)
    y = data['migraine_occurred'].values if 'migraine_occurred' in data.columns else None
    
    # Create explainer using factory
    explainer_factory = ExplainerFactory()
    
    if args.explainer == 'optimizer':
        # Special case for optimizer explainer
        explainer = explainer_factory.create_explainer(
            explainer_type='optimizer',
            optimizer=predictor.meta_optimizer,
            feature_names=predictor.feature_columns
        )
    else:
        # Other explainers
        explainer = explainer_factory.create_explainer(
            explainer_type=args.explainer,
            model=predictor.meta_optimizer,
            feature_names=predictor.feature_columns
        )
    
    # Generate explanations
    explanation = explainer.explain(X, y, n_samples=args.explain_samples)
    
    # Generate plots if requested
    if args.explain_plots:
        os.makedirs(args.output_dir, exist_ok=True)
        plot_paths = []
        
        for plot_type in args.explain_plot_types:
            try:
                plot_path = os.path.join(args.output_dir, f"{args.explainer}_{plot_type}.png")
                explainer.plot(explanation, plot_type=plot_type, save_path=plot_path)
                plot_paths.append(plot_path)
                print(f"Generated plot: {plot_path}")
            except Exception as e:
                print(f"Error generating {plot_type} plot: {str(e)}")
    
    # Get feature importance
    feature_importance = explainer.get_feature_importance(explanation)
    
    # Print summary
    print("\nExplanation Summary:")
    print("Feature Importance:")
    for feature, importance in sorted(feature_importance.items(), key=lambda x: x[1], reverse=True):
        print(f"  {feature}: {importance:.4f}")
    
    if args.summary:
        print("\nDetailed Explanation Summary:")
        
        # Print explainer-specific information
        if args.explainer == 'optimizer':
            print("\nOptimizer Explanation:")
            print(f"  Performance metrics: {explanation.get('performance_metrics', {})}")
            print(f"  Parameter adaptations: {explanation.get('parameter_adaptations', {})}")
            print(f"  Problem characteristics: {explanation.get('problem_characteristics', {})}")
        elif args.explainer == 'shap':
            print("\nSHAP Explanation:")
            print(f"  Base value: {explanation.get('base_value', 'N/A')}")
            print(f"  Number of features: {len(predictor.feature_columns)}")
        elif args.explainer == 'lime':
            print("\nLIME Explanation:")
            print(f"  Number of features: {len(predictor.feature_columns)}")
            print(f"  Number of samples: {args.explain_samples}")
        
        # Print additional details
        print("\nModel information:")
        print(f"  Model version: {predictor.get_model_version()}")
        print(f"  Feature columns: {predictor.feature_columns}")


def load_and_combine_data(args):
    """Load and combine test data."""
    print(f"Loading test data from {args.data_dir}...")
    data_ingestion = DataIngestion()
    
    try:
        combined_data = data_ingestion.load_test_data(args.data_dir)
        
        # Save combined data if output specified
        if args.output:
            combined_data.to_csv(args.output, index=False)
            print(f"Combined data saved to {args.output}")
        
        # Print summary
        if args.summary or not args.output:
            total = len(combined_data)
            
            print("\nCombined Data Summary:")
            print(f"  Total samples: {total}")
            
            if 'patient_id' in combined_data.columns:
                patients = combined_data['patient_id'].nunique()
                print(f"  Unique patients: {patients}")
            
            if 'migraine_occurred' in combined_data.columns:
                migraine_count = combined_data['migraine_occurred'].sum()
                print(f"  Migraine events: {migraine_count} ({migraine_count/total:.1%})")
            
            print("\nFeature statistics:")
            stats = combined_data.describe().T
            print(stats[['count', 'mean', 'std', 'min', 'max']])
            
            print("\nMissing values:")
            missing = combined_data.isna().sum()
            if missing.sum() > 0:
                for col, count in missing[missing > 0].items():
                    print(f"  {col}: {count} ({count/total:.1%})")
            else:
                print("  No missing values")
    
    except Exception as e:
        print(f"Error loading test data: {str(e)}")


if __name__ == "__main__":
    main()
