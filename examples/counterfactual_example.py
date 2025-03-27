"""
counterfactual_example.py
-----------------------
Example demonstrating the use of CounterfactualExplainer
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import logging

# Add parent directory to path to import explainability modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from explainability.explainer_factory import ExplainerFactory
from tests.moe_interactive_report import generate_interactive_report

# Configure logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    """Run counterfactual explanation example"""
    
    # Load sample dataset (diabetes dataset)
    logger.info("Loading diabetes dataset...")
    diabetes = load_diabetes()
    X = diabetes.data
    y = diabetes.target
    feature_names = diabetes.feature_names
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )
    
    # Train a simple model
    logger.info("Training random forest model...")
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Evaluate model
    train_score = model.score(X_train, y_train)
    test_score = model.score(X_test, y_test)
    logger.info(f"Model R² on training data: {train_score:.4f}")
    logger.info(f"Model R² on test data: {test_score:.4f}")
    
    # Create counterfactual explainer
    logger.info("Creating counterfactual explainer...")
    explainer = ExplainerFactory.create_explainer(
        'counterfactual',
        model=model,
        feature_names=feature_names,
        method='alibi',  # Use 'alibi' or 'dice'
        kappa=0.1,       # Alibi-specific parameter
        beta=0.1         # Alibi-specific parameter
    )
    
    # Select a test instance to explain
    instance_idx = 0
    instance = X_test[instance_idx:instance_idx+1]
    actual_value = y_test[instance_idx]
    predicted_value = model.predict(instance)[0]
    
    logger.info(f"Explaining instance {instance_idx}")
    logger.info(f"Actual value: {actual_value:.2f}")
    logger.info(f"Predicted value: {predicted_value:.2f}")
    
    # Generate counterfactual explanation
    logger.info("Generating counterfactual explanation...")
    try:
        explanation = explainer.explain(
            X_test,
            instance_idx=instance_idx,
            # For regression, we can specify a target outcome
            # If not specified, the explainer will try to find a counterfactual
            # that changes the prediction significantly
            target_outcome=predicted_value + 50  # Aim for a higher prediction
        )
        
        # Check if counterfactual generation was successful
        if explanation.get('success', False):
            logger.info("Counterfactual generation successful!")
            
            # Get original and counterfactual predictions
            orig_pred = explanation['original_prediction']
            cf_pred = explanation['counterfactual_prediction']
            
            logger.info(f"Original prediction: {orig_pred:.2f}")
            logger.info(f"Counterfactual prediction: {cf_pred:.2f}")
            logger.info(f"Prediction difference: {cf_pred - orig_pred:.2f}")
            
            # Display feature changes
            logger.info("Feature changes:")
            for i, feature in enumerate(feature_names):
                orig_val = explanation['original_instance'][i]
                cf_val = explanation['counterfactual_instance'][i]
                change = cf_val - orig_val
                logger.info(f"  {feature}: {orig_val:.4f} -> {cf_val:.4f} (change: {change:.4f})")
            
            # Generate plots
            logger.info("Generating plots...")
            
            # Feature change plot
            feature_change_plot = explainer.plot('feature_change')
            
            # Comparison plot
            comparison_plot = explainer.plot('comparison')
            
            # Radar plot
            radar_plot = explainer.plot('radar')
            
            # Parallel coordinates plot
            parallel_plot = explainer.plot('parallel')
            
            # Create output directory
            results_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'output', 'counterfactual_example')
            os.makedirs(results_dir, exist_ok=True)
            
            # Save explanation to JSON
            json_path = os.path.join(results_dir, 'counterfactual_explanation.json')
            explainer.save_explanation(json_path, format='json')
            logger.info(f"Saved explanation to {json_path}")
            
            # Save explanation to HTML
            html_path = os.path.join(results_dir, 'counterfactual_explanation.html')
            explainer.save_explanation(html_path, format='html')
            logger.info(f"Saved HTML report to {html_path}")
            
            # Create test results dictionary for interactive report
            test_results = {
                'counterfactual_explanation': {
                    'feature_change_plot': feature_change_plot,
                    'comparison_plot': comparison_plot,
                    'radar_plot': radar_plot,
                    'parallel_plot': parallel_plot,
                    'original_instance': explanation['original_instance'],
                    'counterfactual_instance': explanation['counterfactual_instance'],
                    'feature_names': feature_names,
                    'original_prediction': explanation['original_prediction'],
                    'counterfactual_prediction': explanation['counterfactual_prediction'],
                    'feature_changes': explanation['feature_changes']
                }
            }
            
            # Generate interactive report
            report_path = generate_interactive_report(test_results, results_dir)
            logger.info(f"Generated interactive report at {report_path}")
            
        else:
            logger.warning("Counterfactual generation was not successful.")
    
    except Exception as e:
        logger.error(f"Error generating counterfactual explanation: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
