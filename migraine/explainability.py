def run_migraine_explainability(args):
    """
    Run explainability analysis on migraine prediction model.
    
    Args:
        args: Command-line arguments containing explanation parameters
        
    Returns:
        Dictionary with explanation results
    """
    try:
        logging.info(f"Running migraine explainability analysis with {args.explainer} explainer")
        
        # Initialize predictor with proper directories
        from migraine_prediction_project.src.migraine_model.new_data_migraine_predictor import MigrainePredictorV2
        
        predictor = MigrainePredictorV2(
            model_dir=args.model_dir,
            data_dir=args.data_dir
        )
        
        # Load model if model_id provided, otherwise use default
        if hasattr(args, 'model_id') and args.model_id:
            predictor.load_model(model_id=args.model_id)
            logging.info(f"Loaded model with ID: {args.model_id}")
        else:
            predictor.load_model()  # This will load the default model
            logging.info(f"Loaded default model with ID: {predictor.model_id}")
        
        # Import data for explanation
        data_path = args.prediction_data if hasattr(args, 'prediction_data') and args.prediction_data else None
        
        if not data_path:
            logging.error("No prediction data provided. Please specify with --prediction-data")
            return {"success": False, "error": "No prediction data provided"}
            
        # Import data
        data = predictor.import_data(
            data_path=data_path,
            add_new_columns=False
        )
        
        # Run explainability analysis
        explainer_type = args.explainer if hasattr(args, 'explainer') else 'shap'
        n_samples = args.explain_samples if hasattr(args, 'explain_samples') else 5
        generate_plots = args.explain_plots if hasattr(args, 'explain_plots') else True
        plot_types = args.explain_plot_types if hasattr(args, 'explain_plot_types') else None
        
        # Generate explanations
        explanation_results = predictor.explain_predictions(
            data=data,
            explainer_type=explainer_type,
            n_samples=n_samples,
            generate_plots=generate_plots,
            plot_types=plot_types
        )
        
        # If successful, print summary
        if explanation_results.get("success", False):
            if args.summary:
                print("\nMigraine Explainability Summary:")
                print(f"Explainer Type: {explanation_results['explainer_type']}")
                
                # Print top feature importance
                if "feature_importance" in explanation_results:
                    print("\nTop Features by Importance:")
                    feature_importance = explanation_results["feature_importance"]
                    
                    # Convert numpy arrays to scalars if needed
                    processed_importance = {}
                    for feature, importance in feature_importance.items():
                        # Handle numpy arrays
                        import numpy as np
                        if isinstance(importance, np.ndarray):
                            # Use absolute mean value for arrays
                            processed_importance[feature] = float(np.abs(importance).mean())
                        else:
                            processed_importance[feature] = float(importance)
                    
                    # Sort features by importance (absolute value)
                    sorted_features = sorted(
                        processed_importance.items(),
                        key=lambda x: abs(x[1]) if isinstance(x[1], (int, float)) else 0,
                        reverse=True
                    )
                    
                    # Print top 10 features
                    for i, (feature, importance) in enumerate(sorted_features[:10]):
                        print(f"  {i+1}. {feature}: {importance:.6f}")
                
                # Print plot paths
                if "plot_paths" in explanation_results and explanation_results["plot_paths"]:
                    print("\nGenerated Plots:")
                    for plot_type, path in explanation_results["plot_paths"].items():
                        print(f"  {plot_type}: {path}")
                
            return explanation_results
        else:
            error_msg = explanation_results.get("error", "Unknown error")
            logging.error(f"Error running migraine explainability: {error_msg}")
            return {"success": False, "error": error_msg}
            
    except Exception as e:
        import traceback
        logging.error(f"Error running migraine explainability: {str(e)}")
        logging.error(traceback.format_exc())
        return {"success": False, "error": str(e)}
