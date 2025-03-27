"""
MoE Counterfactual Example
--------------------------

This example demonstrates how to use the CounterfactualExplainer with a Mixture of Experts model
to generate counterfactual explanations that show how feature changes affect model predictions.

The example:
1. Creates a simple synthetic dataset
2. Trains a set of expert models (Random Forest models with different parameters)
3. Creates a simple MoE ensemble
4. Generates counterfactual explanations for the MoE model
5. Visualizes the counterfactual explanations in the interactive HTML report
"""

import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional, Union
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

# Add parent directory to path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import model factory and explainer components
from models.model_factory import ModelFactory
from explainability.explainer_factory import ExplainerFactory
from tests.moe_interactive_report import generate_interactive_report

class SimpleMoEModel:
    """
    A simple Mixture of Experts model that uses a weighted average of expert predictions.
    
    This is a simplified version for demonstration purposes. In a real-world scenario,
    you would use a more sophisticated gating mechanism.
    """
    
    def __init__(self, experts: List[Any], expert_weights: Optional[List[float]] = None):
        """
        Initialize the MoE model with experts and weights
        
        Parameters:
        -----------
        experts : List[Any]
            List of expert models
        expert_weights : Optional[List[float]]
            Weights for each expert. If None, equal weights are used.
        """
        self.experts = experts
        
        # Initialize weights (equal by default)
        if expert_weights is None:
            self.expert_weights = [1.0 / len(experts)] * len(experts)
        else:
            # Normalize weights to sum to 1
            total = sum(expert_weights)
            self.expert_weights = [w / total for w in expert_weights]
            
        # Feature names for explainability
        self.feature_names = None
            
    def fit(self, X, y, feature_names: Optional[List[str]] = None):
        """
        Train all expert models
        
        Parameters:
        -----------
        X : array-like
            Features
        y : array-like
            Target values
        feature_names : Optional[List[str]]
            Names of the features
        """
        self.feature_names = feature_names
        
        # Train each expert
        for expert in self.experts:
            expert.fit(X, y, feature_names)
            
        return self
    
    def predict(self, X):
        """
        Make predictions using a weighted average of expert predictions
        
        Parameters:
        -----------
        X : array-like
            Features
            
        Returns:
        --------
        array-like
            Predictions
        """
        # Get predictions from each expert
        predictions = np.array([expert.predict(X) for expert in self.experts])
        
        # Weighted average of predictions
        weighted_preds = np.sum([pred * weight for pred, weight in 
                                zip(predictions, self.expert_weights)], axis=0)
        
        return weighted_preds
    
    def get_feature_importance(self):
        """
        Get aggregated feature importance from all experts
        
        Returns:
        --------
        Dict[str, float]
            Feature importance dictionary
        """
        # Initialize feature importance dictionary
        feature_importance = {}
        
        # Aggregate feature importance from all experts
        for expert, weight in zip(self.experts, self.expert_weights):
            if hasattr(expert, 'get_feature_importance'):
                expert_importance = expert.get_feature_importance()
                
                for feature, importance in expert_importance.items():
                    if feature in feature_importance:
                        feature_importance[feature] += importance * weight
                    else:
                        feature_importance[feature] = importance * weight
        
        return feature_importance


def generate_synthetic_data(n_samples=1000, n_features=10, noise=0.1):
    """
    Generate synthetic regression data with non-linear relationships
    
    Parameters:
    -----------
    n_samples : int
        Number of samples
    n_features : int
        Number of features
    noise : float
        Noise level
        
    Returns:
    --------
    Tuple[pd.DataFrame, pd.Series]
        Features and target
    """
    # Generate random features
    X = np.random.randn(n_samples, n_features)
    
    # Generate target with non-linear relationships
    y = (
        5 * np.sin(X[:, 0]) + 
        2 * np.exp(X[:, 1]) + 
        3 * (X[:, 2] ** 2) - 
        2 * X[:, 3] * X[:, 4] +
        np.random.randn(n_samples) * noise
    )
    
    # Create feature names
    feature_names = [f'feature_{i}' for i in range(n_features)]
    
    # Convert to DataFrame
    X_df = pd.DataFrame(X, columns=feature_names)
    y_series = pd.Series(y, name='target')
    
    return X_df, y_series


def main():
    """Main function to run the example"""
    print("MoE Counterfactual Example")
    print("=========================")
    
    # Create results directory
    results_dir = Path("results/moe_counterfactual_example")
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate synthetic data
    print("\nGenerating synthetic data...")
    X, y = generate_synthetic_data(n_samples=1000, n_features=10, noise=0.1)
    feature_names = X.columns.tolist()
    
    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    print(f"Training data shape: {X_train.shape}")
    print(f"Test data shape: {X_test.shape}")
    
    # Create expert models with different configurations
    print("\nCreating expert models...")
    model_factory = ModelFactory()
    
    expert_configs = [
        # Expert 1: Deep trees, fewer features
        {
            'task_type': 'regression',
            'n_estimators': 100,
            'max_depth': 10,
            'max_features': 0.6,
        },
        # Expert 2: Shallow trees, more features
        {
            'task_type': 'regression',
            'n_estimators': 100,
            'max_depth': 5,
            'max_features': 0.8,
        },
        # Expert 3: Medium trees, balanced features
        {
            'task_type': 'regression',
            'n_estimators': 100,
            'max_depth': 7,
            'max_features': 0.7,
        }
    ]
    
    experts = [model_factory.create_model(config) for config in expert_configs]
    
    # Create MoE model
    print("\nCreating MoE model...")
    moe_model = SimpleMoEModel(experts, expert_weights=[0.4, 0.3, 0.3])
    
    # Train MoE model
    print("\nTraining MoE model...")
    moe_model.fit(X_train.values, y_train.values, feature_names)
    
    # Evaluate MoE model
    print("\nEvaluating MoE model...")
    y_pred = moe_model.predict(X_test.values)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print(f"Mean Squared Error: {mse:.4f}")
    print(f"R² Score: {r2:.4f}")
    
    # Get feature importance
    feature_importance = moe_model.get_feature_importance()
    
    # Sort features by importance
    sorted_features = sorted(
        feature_importance.items(), 
        key=lambda x: abs(x[1]), 
        reverse=True
    )
    
    print("\nTop 5 important features:")
    for feature, importance in sorted_features[:5]:
        print(f"  {feature}: {importance:.4f}")
    
    # Create counterfactual explainer
    print("\nCreating counterfactual explainer...")
    explainer_factory = ExplainerFactory()
    try:
        counterfactual_explainer = explainer_factory.create_explainer(
            'counterfactual',
            model=moe_model,
            feature_names=feature_names,
            method='alibi',  # Use Alibi for counterfactual generation
            categorical_features=None,  # No categorical features in our synthetic data
            # Additional parameters for the Alibi counterfactual generator
            kappa=0.1,  # Regularization parameter for the counterfactual loss
            beta=0.1,   # Regularization parameter for the prototype loss
            # Limit the number of iterations to speed up the example
            max_iterations=500
        )
    except Exception as e:
        print(f"Error creating counterfactual explainer: {e}")
        print("Falling back to default parameters...")
        # Try with minimal parameters
        counterfactual_explainer = explainer_factory.create_explainer(
            'counterfactual',
            model=moe_model,
            feature_names=feature_names,
            method='alibi'
        )
    
    # Select a test instance for explanation
    instance_idx = 0
    test_instance = X_test.iloc[instance_idx].values.reshape(1, -1)
    original_prediction = moe_model.predict(test_instance)[0]
    
    print(f"\nExplaining prediction for test instance {instance_idx}:")
    print(f"Original prediction: {original_prediction:.4f}")
    
    # Generate counterfactual explanation
    print("\nGenerating counterfactual explanation...")
    try:
        # First try with TensorFlow compatibility layer if available
        explanation = counterfactual_explainer.explain(
            test_instance, 
            desired_outcome=original_prediction + 2.0,  # Aim for a +2 change
            feature_names=feature_names  # Explicitly pass feature names
        )
    except Exception as e:
        print(f"Error generating counterfactual explanation: {e}")
        print("Trying with a smaller desired change...")
        try:
            # Try with a smaller desired change
            explanation = counterfactual_explainer.explain(
                test_instance, 
                desired_outcome=original_prediction + 0.5,  # Aim for a smaller change
                feature_names=feature_names  # Explicitly pass feature names
            )
        except Exception as e:
            print(f"Still encountering error: {e}")
            # Create a more complete explanation structure for demonstration purposes
            explanation = {
                'original_instance': test_instance[0],
                'counterfactuals': [],
                'feature_names': feature_names,  # Explicitly include feature names
                'original_prediction': original_prediction,
                'success': False,
                'error_message': str(e)
            }
    
    # Extract counterfactual instances
    counterfactuals = explanation.get('counterfactuals', [])
    
    if counterfactuals and len(counterfactuals) > 0:
        # Get the first counterfactual
        counterfactual = counterfactuals[0]
        cf_prediction = moe_model.predict(counterfactual.reshape(1, -1))[0]
        
        print(f"Counterfactual prediction: {cf_prediction:.4f}")
        print(f"Prediction difference: {cf_prediction - original_prediction:.4f}")
        
        # Show feature changes
        print("\nFeature changes:")
        for i, (orig, cf) in enumerate(zip(test_instance[0], counterfactual)):
            if abs(orig - cf) > 1e-6:  # Only show significant changes
                print(f"  {feature_names[i]}: {orig:.4f} -> {cf:.4f} (Δ: {cf - orig:.4f})")
    else:
        print("No counterfactual found.")
        
        # Create a distinctive synthetic counterfactual for demonstration purposes
        print("\nCreating a synthetic counterfactual for demonstration purposes...")
        synthetic_cf = test_instance[0].copy()
        
        # Create a more dramatic counterfactual with clear differences
        # We'll make some features increase and others decrease to show the full range of visualizations
        for i, (feature, importance) in enumerate(sorted_features):
            feature_idx = feature_names.index(feature)
            
            # Apply different changes to different features to create a varied example
            if i < 3:  # Top 3 features get large changes
                if i % 2 == 0:  # Even indices increase
                    synthetic_cf[feature_idx] *= 1.8  # 80% increase
                else:  # Odd indices decrease
                    synthetic_cf[feature_idx] *= 0.5  # 50% decrease
            elif i < 6:  # Next 3 features get moderate changes
                if i % 2 == 0:  # Even indices decrease
                    synthetic_cf[feature_idx] *= 0.7  # 30% decrease
                else:  # Odd indices increase
                    synthetic_cf[feature_idx] *= 1.4  # 40% increase
            else:  # Remaining features get small changes
                if i % 2 == 0:  # Even indices increase slightly
                    synthetic_cf[feature_idx] *= 1.1  # 10% increase
                else:  # Odd indices decrease slightly
                    synthetic_cf[feature_idx] *= 0.9  # 10% decrease
        
        # Calculate the prediction for the synthetic counterfactual
        synthetic_prediction = float(moe_model.predict(synthetic_cf.reshape(1, -1))[0])
        
        # Calculate feature changes
        feature_changes = synthetic_cf - test_instance[0]
        
        # Add the synthetic counterfactual to the explanation
        explanation['counterfactuals'] = [synthetic_cf]
        explanation['feature_changes'] = feature_changes
        explanation['counterfactual_prediction'] = synthetic_prediction
        explanation['is_synthetic'] = True  # Mark as synthetic for visualization
    
    # Visualize counterfactual explanation
    print("\nVisualizing counterfactual explanation...")
    try:
        # Ensure feature_names are in the explanation
        if 'feature_names' not in explanation:
            explanation['feature_names'] = feature_names
            
        visualization = counterfactual_explainer.visualize(
            explanation, 
            plot_type='comparison'
        )
        
        # Save visualization if it's a matplotlib figure
        if hasattr(visualization, 'savefig'):
            viz_path = results_dir / "counterfactual_visualization.png"
            visualization.savefig(viz_path)
            plt.close()
            print(f"Visualization saved to {viz_path}")
        else:
            # For Plotly figures or other visualization types
            try:
                # Try to save Plotly figure
                if hasattr(visualization, 'write_image'):
                    viz_path = results_dir / "counterfactual_visualization.png"
                    visualization.write_image(str(viz_path))
                    print(f"Visualization saved to {viz_path}")
                else:
                    print("Visualization generated successfully but not saved as image")
            except Exception as viz_save_error:
                print(f"Could not save visualization: {viz_save_error}")
    except Exception as e:
        print(f"Error visualizing counterfactual explanation: {e}")
        print("Creating a simple visualization instead...")
        
        # Create a simple visualization showing the original instance
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Plot original instance values
        x = np.arange(len(feature_names))
        ax.bar(x, test_instance[0], color='blue', alpha=0.7, label='Original Instance')
        
        # If we have a synthetic counterfactual, plot it too
        if 'counterfactuals' in explanation and len(explanation['counterfactuals']) > 0:
            counterfactual = explanation['counterfactuals'][0]
            ax.bar(x, counterfactual, color='red', alpha=0.5, label='Synthetic Counterfactual')
        
        # Add labels and title
        ax.set_xlabel('Features')
        ax.set_ylabel('Value')
        ax.set_title('Original Instance vs Counterfactual')
        ax.set_xticks(x)
        ax.set_xticklabels(feature_names, rotation=45, ha='right')
        ax.legend()
        
        # Save the simple visualization
        viz_path = results_dir / "original_instance_visualization.png"
        plt.tight_layout()
        plt.savefig(viz_path)
        plt.close()
        print(f"Simple visualization saved to {viz_path}")
    
    # Generate interactive report
    print("\nGenerating interactive report...")
    
    # Prepare data for interactive report following the standardized format
    report_data = {
        # Model performance section
        'model_performance': {
            'metrics': {
                'mse': mse,
                'r2': r2
            },
            'feature_importance': {
                'values': feature_importance,
                'feature_names': feature_names
            }
        },
        
        # Counterfactual explanation section
        'counterfactual_explanations': {
            'original_instance': {
                'values': test_instance[0].tolist(),
                'feature_names': feature_names,
                'prediction': float(original_prediction)
            },
            'counterfactuals': []
        }
    }
    
    # Add counterfactuals if available
    if 'counterfactuals' in explanation and explanation['counterfactuals']:
        for cf in explanation['counterfactuals']:
            # Ensure cf is a numpy array
            cf_array = np.array(cf) if not isinstance(cf, np.ndarray) else cf
            
            # Calculate prediction if not already in the explanation
            if 'counterfactual_prediction' in explanation:
                cf_prediction = explanation['counterfactual_prediction']
            else:
                cf_prediction = float(moe_model.predict(cf_array.reshape(1, -1))[0])
            
            # Add to the report data
            report_data['counterfactual_explanations']['counterfactuals'].append({
                'values': cf_array.tolist(),
                'feature_names': feature_names,
                'prediction': cf_prediction,
                'is_synthetic': explanation.get('is_synthetic', False)
            })
    
    # Add feature changes if available
    if 'feature_changes' in explanation and explanation['feature_changes'] is not None:
        feature_changes = explanation['feature_changes']
        if isinstance(feature_changes, np.ndarray):
            report_data['counterfactual_explanations']['feature_changes'] = feature_changes.tolist()
        else:
            report_data['counterfactual_explanations']['feature_changes'] = feature_changes
    else:
        # If no feature changes in explanation but we have counterfactuals, calculate them
        if 'counterfactuals' in explanation and explanation['counterfactuals'] and len(explanation['counterfactuals']) > 0:
            cf = explanation['counterfactuals'][0]
            feature_changes = np.array(cf) - test_instance[0]
            report_data['counterfactual_explanations']['feature_changes'] = feature_changes.tolist()
    
    # Add visualization data for the interactive report
    report_data['visualizations'] = {
        'counterfactual_comparison': {
            'type': 'bar_comparison',
            'title': 'Original vs Counterfactual Instance',
            'x_label': 'Features',
            'y_label': 'Value',
            'data': {
                'original': test_instance[0].tolist(),
                'counterfactual': explanation.get('counterfactuals', [])[0].tolist() if explanation.get('counterfactuals', []) else [],
                'feature_names': feature_names
            }
        },
        'feature_importance': {
            'type': 'bar',
            'title': 'Feature Importance',
            'x_label': 'Importance',
            'y_label': 'Feature',
            'data': {
                'values': [importance for _, importance in sorted_features],
                'labels': [feature for feature, _ in sorted_features]
            }
        }
    }
    
    # Generate the report using the standardized interactive HTML report framework
    report_path = generate_interactive_report(
        report_data,
        str(results_dir)
    )
    
    print(f"Interactive report generated at {report_path}")
    print("\nExample completed successfully!")


if __name__ == "__main__":
    main()
