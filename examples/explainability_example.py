"""
Example script demonstrating the explainability framework
"""

import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from pathlib import Path

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import explainability framework
from explainability.explainer_factory import ExplainerFactory
from explainability.base_explainer import BaseExplainer
from utils.plot_utils import save_plot

def main():
    """Run explainability example"""
    print("Running explainability example...")
    
    # Create results directory
    results_dir = Path('results/explainability_example')
    results_dir.mkdir(exist_ok=True, parents=True)
    
    # Load dataset
    print("Loading diabetes dataset...")
    diabetes = load_diabetes()
    X = diabetes.data
    y = diabetes.target
    feature_names = diabetes.feature_names
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Train model
    print("Training RandomForestRegressor...")
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Evaluate model
    train_score = model.score(X_train, y_train)
    test_score = model.score(X_test, y_test)
    print(f"Model R² score - Train: {train_score:.4f}, Test: {test_score:.4f}")
    
    # Run explainability analysis with all available explainers
    explainer_types = ExplainerFactory.get_available_explainers()
    
    for explainer_type in explainer_types:
        print(f"\nRunning {explainer_type} explainer...")
        
        # Create explainer
        explainer = ExplainerFactory.create_explainer(
            explainer_type=explainer_type,
            model=model,
            feature_names=feature_names
        )
        
        # Generate explanation
        explanation = explainer.explain(X_test, y_test)
        
        # Get feature importance
        feature_importance = explainer.get_feature_importance()
        
        # Print top 5 features
        print("Top 5 important features:")
        for feature, importance in sorted(
            feature_importance.items(), 
            key=lambda x: x[1], 
            reverse=True
        )[:5]:
            print(f"  {feature}: {importance:.4f}")
        
        # Generate and save plots
        for plot_type in explainer.get_supported_plot_types()[:2]:  # Limit to first 2 plot types
            try:
                print(f"  Generating {plot_type} plot...")
                fig = explainer.plot(plot_type=plot_type)
                # Use save_plot function
                save_plot(
                    fig,
                    f"{explainer_type}_{plot_type}.png", 
                    plot_type='explainability'
                )
                plt.close(fig)
            except Exception as e:
                print(f"  Error generating {plot_type} plot: {str(e)}")
        
        # Save explanation data
        explainer.save_explanation(results_dir / f"{explainer_type}_explanation.json")
    
    print("\nAll results have been saved to the 'results/explainability_example' directory.")

if __name__ == '__main__':
    main()
