"""
Command-line interface for the migraine prediction model.
"""

import argparse
import logging
import sys
import json
from pathlib import Path

from migraine_model.migraine_predictor import MigrainePredictor

logger = logging.getLogger(__name__)


def parse_args(args=None):
    """
    Parse command-line arguments for the migraine prediction CLI.
    
    Args:
        args: List of command-line arguments
        
    Returns:
        Parsed arguments
    """
    parser = argparse.ArgumentParser(description="Migraine Prediction CLI")
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Train command
    train_parser = subparsers.add_parser("train", help="Train a new model")
    train_parser.add_argument("--data", required=True, help="Path to training data CSV")
    train_parser.add_argument("--model-name", default="migraine_model", help="Name for the trained model")
    train_parser.add_argument("--description", default="", help="Description for the model")
    train_parser.add_argument("--summary", action="store_true", help="Print summary of training results")
    
    # Optimize command
    optimize_parser = subparsers.add_parser("optimize", help="Train model with meta-optimization")
    optimize_parser.add_argument("--data", required=True, help="Path to training data CSV")
    optimize_parser.add_argument("--model-name", default="optimized_model", help="Name for the trained model")
    optimize_parser.add_argument("--description", default="Model trained with meta-optimization", 
                              help="Description for the model")
    optimize_parser.add_argument("--optimizer", default="meta", choices=["meta", "de", "es", "gwo", "aco"],
                              help="Type of optimizer to use (meta, de, es, gwo, aco)")
    optimize_parser.add_argument("--max-evals", type=int, default=500, 
                              help="Maximum number of function evaluations")
    optimize_parser.add_argument("--summary", action="store_true", help="Print summary of optimization results")
    
    # Predict command
    predict_parser = subparsers.add_parser("predict", help="Make predictions")
    predict_parser.add_argument("--data", required=True, help="Path to data for prediction")
    predict_parser.add_argument("--model-id", help="ID of model to use (default: use default model)")
    predict_parser.add_argument("--output", help="Path to save predictions (default: print to stdout)")
    predict_parser.add_argument("--detailed", action="store_true", help="Get detailed predictions")
    
    # Export command
    export_parser = subparsers.add_parser("export", help="Export model to pickle")
    export_parser.add_argument("--model-id", help="ID of model to export (default: use default model)")
    export_parser.add_argument("--output", required=True, help="Path to save exported model")
    
    # List command
    list_parser = subparsers.add_parser("list", help="List available models")
    
    # Explain command
    explain_parser = subparsers.add_parser("explain", help="Explain model predictions")
    explain_parser.add_argument("--data", required=True, help="Path to data for explanation")
    explain_parser.add_argument("--model-id", help="ID of model to explain (default: use default model)")
    explain_parser.add_argument("--explainer", default="feature_importance", 
                                choices=["feature_importance", "shap", "lime"], 
                                help="Type of explainer to use")
    explain_parser.add_argument("--explain-plots", action="store_true", help="Generate and save plots")
    explain_parser.add_argument("--explain-samples", type=int, default=5, 
                                help="Number of samples to use for local explanations")
    explain_parser.add_argument("--output-dir", default="./explanations", 
                                help="Directory to save explanation outputs")
    explain_parser.add_argument("--summary", action="store_true", help="Print summary of explanation results")
    
    # Generate command for synthetic data
    generate_parser = subparsers.add_parser("generate", help="Generate synthetic data for testing")
    generate_parser.add_argument("--num-samples", type=int, default=100, 
                                help="Number of samples to generate")
    generate_parser.add_argument("--train-ratio", type=float, default=0.8, 
                                help="Ratio of training data (0.0-1.0)")
    generate_parser.add_argument("--output-train", default="train_data.csv", 
                                help="Path to save training data")
    generate_parser.add_argument("--output-test", default="test_data.csv", 
                                help="Path to save test data")
    generate_parser.add_argument("--seed", type=int, default=42, 
                                help="Random seed for reproducibility")
    
    # Load command
    load_parser = subparsers.add_parser("load", help="Load and combine test data")
    load_parser.add_argument("--data-dir", default="./data", help="Directory containing data files")
    load_parser.add_argument("--output", default="combined_test_data.csv", help="Path to save combined data")
    
    return parser.parse_args(args)


def main(args=None):
    """
    Main entry point for the migraine prediction CLI.
    
    Args:
        args: List of command-line arguments
    """
    # Parse arguments
    args = parse_args(args)
    
    # Create predictor
    predictor = MigrainePredictor()
    
    # Execute command
    if args.command == "train":
        # Load data
        print(f"Loading data from {args.data}...")
        import pandas as pd
        data = pd.read_csv(args.data)
        
        # Train model
        print(f"Training model with {len(data)} samples...")
        model_id = predictor.train(data, model_name=args.model_name, description=args.description)
        print(f"Model trained successfully! Model ID: {model_id}")
        
        # Evaluate model
        metrics = predictor.evaluate(data)
        print(f"Test accuracy: {metrics['accuracy']:.4f}")
        
        # Print feature importance
        feature_importance = metrics.get("feature_importance", {})
        if feature_importance:
            print("\nFeature Importance:")
            for feature, importance in sorted(feature_importance.items(), key=lambda x: x[1], reverse=True):
                print(f"  {feature}: {importance:.4f}")
        
        # Print summary
        if args.summary:
            metadata = predictor.get_model_metadata()
            print("\nTraining Summary:")
            print(f"  Model ID: {metadata.get('id', 'Unknown')}")
            print(f"  Model Name: {metadata.get('name', 'Unknown')}")
            print(f"  Description: {metadata.get('description', '')}")
            print(f"  Training samples: {len(data)}")
            print(f"  Test samples: {metrics.get('test_samples', 0)}")
            print(f"  Test accuracy: {metrics.get('accuracy', 0):.4f}")
            print(f"  Feature importance: {json.dumps(feature_importance, indent=2)}")
            print(f"  Created: {metadata.get('created_at', '')}")
            print(f"  Version: {metadata.get('version', 'Unknown')}")
    
    elif args.command == "predict":
        # Load data
        print(f"Loading data from {args.data}...")
        try:
            data = pd.read_csv(args.data)
            print(f"Loaded {len(data)} samples")
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            sys.exit(1)
        
        # Make predictions
        if args.model_id:
            print(f"Loading model {args.model_id}...")
            predictor.load_model(args.model_id)
        else:
            print("Using default model...")
        
        print("Making predictions...")
        try:
            if args.detailed:
                # Use predict_with_details for detailed output
                predictions = predictor.predict_with_details(data)
                
                # Write to CSV or print to console
                if args.output:
                    # Create dataframe with predictions and features
                    result_df = data.copy()
                    for i, pred in enumerate(predictions):
                        result_df.loc[i, 'prediction'] = pred['prediction']
                        result_df.loc[i, 'probability'] = pred['probability']
                    result_df.to_csv(args.output, index=False)
                    print(f"Predictions saved to {args.output}")
                else:
                    for i, pred in enumerate(predictions):
                        print(f"Sample {i+1}: Prediction={pred['prediction']}, Probability={pred['probability']:.4f}")
            else:
                # Use simple predict for just predictions
                predictions = predictor.predict(data)
                
                # Write to CSV or print to console
                if args.output:
                    # Create dataframe with predictions
                    result_df = data.copy()
                    result_df['prediction'] = predictions
                    result_df.to_csv(args.output, index=False)
                    print(f"Predictions saved to {args.output}")
                else:
                    for i, pred in enumerate(predictions):
                        print(f"Sample {i+1}: Prediction={pred}")
        except Exception as e:
            logger.error(f"Error making predictions: {e}")
            sys.exit(1)
    
    elif args.command == "export":
        # Load model
        if args.model_id:
            predictor.load_model(args.model_id)
        else:
            print("Using default model")
            predictor.load_model()
        
        # Export model
        import pickle
        with open(args.output, "wb") as f:
            pickle.dump(predictor.model, f)
        print(f"Model exported to {args.output}")
    
    elif args.command == "list":
        # List models
        models = predictor.model_manager.list_models()
        if not models:
            print("No models found")
        else:
            print(f"Found {len(models)} models:")
            for model in models:
                print(f"  ID: {model['id']}")
                print(f"    Name: {model['name']}")
                print(f"    Description: {model['description']}")
                print(f"    Created: {model['created_at']}")
                print()
    
    elif args.command == "explain":
        # Load data
        print(f"Loading data from {args.data}...")
        import pandas as pd
        data = pd.read_csv(args.data)
        
        # Load model
        if args.model_id:
            predictor.load_model(args.model_id)
        else:
            print("Using default model")
            predictor.load_model()
        
        # Create output directory if it doesn't exist
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Check if explainability is available in the main project
        try:
            from explainability import ExplainerFactory, BaseExplainer
            explain_available = True
        except ImportError:
            print("Warning: Explainability components not available from main project.")
            explain_available = False
        
        if explain_available:
            # Get feature names
            feature_names = predictor.feature_names
            
            # Create explainer
            explainer = ExplainerFactory.create_explainer(
                explainer_type=args.explainer,
                model=predictor.model,
                feature_names=feature_names
            )
            
            # Generate explanations
            explanations = explainer.explain(data.values)
            
            # Get feature importance
            feature_importance = explainer.get_feature_importance()
            
            # Print feature importance
            print("\nFeature Importance:")
            for feature, importance in sorted(feature_importance.items(), key=lambda x: x[1], reverse=True):
                print(f"  {feature}: {importance:.4f}")
            
            # Generate and save plots
            if args.explain_plots:
                plot_types = ["bar", "horizontal_bar"]
                for plot_type in plot_types:
                    try:
                        plot_path = output_dir / f"{args.explainer}_{plot_type}.png"
                        explainer.save_plot(str(plot_path), plot_type=plot_type)
                        print(f"Plot saved to {plot_path}")
                    except Exception as e:
                        print(f"Error generating {plot_type} plot: {e}")
            
            # Print summary
            if args.summary:
                metadata = predictor.get_model_metadata()
                print("\nExplanation Summary:")
                print(f"  Model ID: {metadata.get('id', 'Unknown')}")
                print(f"  Model Name: {metadata.get('name', 'Unknown')}")
                print(f"  Explainer: {args.explainer}")
                print(f"  Feature importance: {json.dumps(feature_importance, indent=2)}")
        else:
            # Fallback to simple feature importance from the predictor
            metrics = predictor.evaluate(data)
            feature_importance = metrics.get("feature_importance", {})
            
            print("\nFeature Importance (from model evaluation):")
            for feature, importance in sorted(feature_importance.items(), key=lambda x: x[1], reverse=True):
                print(f"  {feature}: {importance:.4f}")
    
    elif args.command == "generate":
        # Generate synthetic data
        import numpy as np
        import pandas as pd
        np.random.seed(args.seed)
        num_samples = args.num_samples
        train_ratio = args.train_ratio
        num_train = int(num_samples * train_ratio)
        num_test = num_samples - num_train
        
        print(f"Generating {num_samples} samples of synthetic migraine data...")
        
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
        migraine_probability = (migraine_probability - np.min(migraine_probability)) / (np.max(migraine_probability) - np.min(migraine_probability))
        
        # Generate binary outcome
        data['migraine_occurred'] = (migraine_probability > 0.5).astype(int)
        
        # Convert to DataFrame
        df = pd.DataFrame(data)
        
        # Split into train/test
        train_df = df.iloc[:num_train]
        test_df = df.iloc[num_train:]
        
        # Save data to CSV
        train_df.to_csv(args.output_train, index=False)
        test_df.to_csv(args.output_test, index=False)
        
        print(f"Generated {num_samples} samples with migraine-specific features")
        print(f"Training data saved to {args.output_train} with {num_train} samples")
        print(f"Test data saved to {args.output_test} with {num_test} samples")
    
    elif args.command == "optimize":
        # Load data
        print(f"Loading data from {args.data}...")
        import pandas as pd
        data = pd.read_csv(args.data)
        
        # Train model using meta-optimization
        print(f"Training model with {len(data)} samples using {args.optimizer} optimizer...")
        model_id = predictor.optimize(data, model_name=args.model_name, description=args.description, 
                                      optimizer=args.optimizer, max_evals=args.max_evals)
        print(f"Model trained successfully! Model ID: {model_id}")
        
        # Print summary
        if args.summary:
            metadata = predictor.get_model_metadata()
            print("\nOptimization Summary:")
            print(f"  Model ID: {metadata.get('id', 'Unknown')}")
            print(f"  Model Name: {metadata.get('name', 'Unknown')}")
            print(f"  Optimizer: {args.optimizer}")
            print(f"  Maximum evaluations: {args.max_evals}")
            print(f"  Created: {metadata.get('created_at', '')}")
            print(f"  Version: {metadata.get('version', 'Unknown')}")
    
    elif args.command == "load":
        # Load and combine test data
        try:
            from pipeline.data_ingestion import DataIngestion
            data_ingestion = DataIngestion()
            combined_data = data_ingestion.load_and_combine_data(args.data_dir)
            combined_data.to_csv(args.output, index=False)
            print(f"Combined data saved to {args.output} with {len(combined_data)} samples")
        except ImportError:
            print("Error: Could not import DataIngestion. Make sure the pipeline module is installed.")
            sys.exit(1)
    
    else:
        logger.error("No command specified")
        print("Please specify a command: train, predict, export, list, explain, optimize, generate, or load")
        sys.exit(1)


if __name__ == "__main__":
    main()
