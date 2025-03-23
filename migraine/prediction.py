def run_migraine_prediction(args):
    """
    Run prediction using a migraine model on new data, handling missing features.
    
    Args:
        args: Command-line arguments containing prediction parameters
        
    Returns:
        Dictionary with prediction results
    """
    if not MIGRAINE_MODULES_AVAILABLE:
        logging.error("Migraine prediction modules are not available. Please install the package first.")
        return {"success": False, "error": "Migraine modules not available"}
    
    try:
        logging.info(f"Running migraine prediction using data from {args.prediction_data}")
        
        # Initialize the predictor
        from migraine_prediction_project.src.migraine_model import MigrainePredictor
        predictor = MigrainePredictor()
        
        # Load the model if specified
        if args.model_id:
            predictor.load_model(args.model_id)
            logging.info(f"Loaded model with ID: {args.model_id}")
        else:
            # Load the default model if no model ID is specified
            predictor.load_model()  # This will load the default model
            logging.info(f"Loaded default model with ID: {predictor.model_id}")
        
        # Import prediction data
        pred_data = predictor.import_data(
            data_path=args.prediction_data,
            add_new_columns=False  # Don't add new columns for prediction
        )
        
        # Make predictions with missing features
        predictions = predictor.predict_with_missing_features(pred_data)
        
        # Format results
        results = []
        for i, pred in enumerate(predictions):
            sample_result = {
                "index": i,
                "prediction": pred["prediction"],
                "probability": pred["probability"],
                "top_features": sorted(
                    pred["feature_importances"].items(),
                    key=lambda x: x[1],
                    reverse=True
                )[:5] if "feature_importances" in pred else []
            }
            results.append(sample_result)
        
        # Save results if requested
        if args.save_predictions:
            # Create DataFrame with predictions
            pred_df = pd.DataFrame({
                "prediction": [p["prediction"] for p in predictions],
                "probability": [p["probability"] for p in predictions]
            })
            # Add original data
            pred_df = pd.concat([pred_data.reset_index(drop=True), pred_df], axis=1)
            output_path = os.path.join(args.data_dir, "predictions.csv")
            pred_df.to_csv(output_path, index=False)
            logging.info(f"Saved predictions to {output_path}")
        
        # If summary is requested, print it
        if args.summary:
            print("\nMigraine Prediction Summary:")
            print(f"Number of samples predicted: {len(predictions)}")
            migraine_count = sum(1 for p in predictions if p["prediction"] == 1)
            print(f"Predicted migraines: {migraine_count} ({migraine_count/len(predictions)*100:.2f}%)")
            print(f"Average probability: {sum(p['probability'] for p in predictions)/len(predictions):.4f}")
            print("\nSample predictions:")
            for i, result in enumerate(results[:5]):  # Show first 5 predictions
                print(f"Sample {i}: {'Migraine' if result['prediction'] == 1 else 'No Migraine'} " +
                      f"(Probability: {result['probability']:.4f})")
        
        return {
            "success": True,
            "predictions": results,
            "summary": {
                "total_samples": len(predictions),
                "predicted_migraines": sum(1 for p in predictions if p["prediction"] == 1),
                "average_probability": sum(p["probability"] for p in predictions)/len(predictions)
            }
        }
        
    except Exception as e:
        logging.error(f"Error running migraine prediction: {str(e)}")
        import traceback
        traceback.print_exc()
        return {"success": False, "error": str(e)}
