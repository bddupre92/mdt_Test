import os
import numpy as np
import pandas as pd
import logging
import matplotlib.pyplot as plt
from typing import Dict, List, Any, Tuple, Optional, Union, Callable
import argparse
from pathlib import Path
import time
import warnings

from utils.json_utils import save_json
from utils.plotting import save_plot, setup_plot_style

class BaseExplainer:
    """Base class for all explainers"""
    def __init__(self, model, feature_names=None):
        """
        Initialize the explainer
        
        Parameters:
        -----------
        model : object
            Trained model to explain
        feature_names : List[str], optional
            Names of features
        """
        self.model = model
        self.feature_names = feature_names
        self.explanation = None
    
    def explain(self, X, y=None):
        """
        Generate explanation for the model
        
        Parameters:
        -----------
        X : np.ndarray or pd.DataFrame
            Input features
        y : np.ndarray or pd.Series, optional
            Target values
            
        Returns:
        --------
        Dict[str, Any]
            Explanation results
        """
        raise NotImplementedError("Subclasses must implement explain()")
    
    def plot(self, plot_type='summary', **kwargs):
        """
        Generate plots from the explanation
        
        Parameters:
        -----------
        plot_type : str, optional
            Type of plot to generate
        **kwargs : dict
            Additional parameters for the plot
            
        Returns:
        --------
        matplotlib.figure.Figure
            The generated plot
        """
        raise NotImplementedError("Subclasses must implement plot()")

class ShapExplainer(BaseExplainer):
    """SHAP explainer for model interpretability"""
    def __init__(self, model, feature_names=None):
        """
        Initialize the SHAP explainer
        
        Parameters:
        -----------
        model : object
            Trained model to explain
        feature_names : List[str], optional
            Names of features
        """
        super().__init__(model, feature_names)
        
        # Import SHAP here to avoid dependency if not used
        try:
            import shap
            self.shap = shap
            self.explainer = None
        except ImportError:
            logging.error("SHAP not installed. Install with: pip install shap")
            raise
    
    def explain(self, X, y=None):
        """
        Generate SHAP values for the model
        
        Parameters:
        -----------
        X : np.ndarray or pd.DataFrame
            Input features
        y : np.ndarray or pd.Series, optional
            Target values
            
        Returns:
        --------
        Dict[str, Any]
            Explanation results with SHAP values
        """
        try:
            # Convert to numpy if pandas
            if isinstance(X, pd.DataFrame):
                if self.feature_names is None:
                    self.feature_names = X.columns.tolist()
                X_np = X.values
            else:
                X_np = X
                if self.feature_names is None:
                    self.feature_names = [f'Feature {i}' for i in range(X.shape[1])]
            
            # Create explainer based on model type
            model_type = type(self.model).__name__
            
            if 'Tree' in model_type or 'Forest' in model_type:
                # Tree-based models
                self.explainer = self.shap.TreeExplainer(self.model)
            elif 'Boost' in model_type:
                # Boosting models
                self.explainer = self.shap.TreeExplainer(self.model)
            elif 'Linear' in model_type:
                # Linear models
                self.explainer = self.shap.LinearExplainer(self.model, X_np)
            else:
                # Generic Kernel explainer for any model
                self.explainer = self.shap.KernelExplainer(self.model.predict, X_np[:100])
            
            # Compute SHAP values
            shap_values = self.explainer.shap_values(X_np)
            
            # Store explanation
            self.explanation = {
                'shap_values': shap_values,
                'feature_names': self.feature_names,
                'X': X_np,
                'expected_value': getattr(self.explainer, 'expected_value', 0)
            }
            
            # Generate feature importance
            feature_importance = self._get_feature_importance()
            
            return {
                'shap_values': shap_values,
                'feature_importance': feature_importance,
                'feature_names': self.feature_names
            }
            
        except Exception as e:
            logging.error(f"Error generating SHAP explanation: {str(e)}")
            raise
    
    def _get_feature_importance(self):
        """
        Get feature importance from SHAP values
        
        Returns:
        --------
        Dict[str, float]
            Feature importance scores
        """
        if self.explanation is None:
            return {}
        
        # Calculate absolute mean SHAP values for each feature
        shap_values = self.explanation['shap_values']
        
        # Handle different shapes of SHAP values
        if isinstance(shap_values, list):
            # Multi-output models return a list of arrays
            importance_values = np.abs(np.array(shap_values)).mean(axis=1).mean(axis=0)
        else:
            # Single-output models return a single array
            importance_values = np.abs(shap_values).mean(axis=0)
        
        # Create feature importance dictionary
        feature_importance = {}
        for i, feature in enumerate(self.feature_names):
            feature_importance[feature] = float(importance_values[i])
        
        return feature_importance
    
    def plot(self, plot_type='summary', **kwargs):
        """
        Generate SHAP plots
        
        Parameters:
        -----------
        plot_type : str, optional
            Type of plot ('summary', 'bar', 'beeswarm', 'waterfall', 'force', 'dependence')
        **kwargs : dict
            Additional parameters for the plot
            
        Returns:
        --------
        matplotlib.figure.Figure
            The generated plot
        """
        if self.explanation is None:
            raise ValueError("No explanation generated. Call explain() first.")
        
        # Get data from explanation
        shap_values = self.explanation['shap_values']
        X = self.explanation['X']
        feature_names = self.explanation['feature_names']
        
        # Set up matplotlib
        setup_plot_style()
        
        # Create figure
        fig = plt.figure(figsize=(12, 8))
        
        # Generate different types of plots
        try:
            if plot_type == 'summary':
                # Summary plot (default)
                self.shap.summary_plot(shap_values, X, feature_names=feature_names, show=False)
            
            elif plot_type == 'bar':
                # Bar plot of feature importance
                self.shap.summary_plot(shap_values, X, feature_names=feature_names, plot_type='bar', show=False)
            
            elif plot_type == 'beeswarm':
                # Beeswarm plot (detailed summary)
                self.shap.summary_plot(shap_values, X, feature_names=feature_names, plot_type='violin', show=False)
            
            elif plot_type == 'waterfall':
                # Waterfall plot for a single prediction
                instance_idx = kwargs.get('instance_idx', 0)
                self.shap.waterfall_plot(self.shap.Explanation(
                    values=shap_values[instance_idx], 
                    base_values=self.explanation['expected_value'],
                    data=X[instance_idx],
                    feature_names=feature_names
                ), show=False)
            
            elif plot_type == 'force':
                # Force plot for a single prediction
                instance_idx = kwargs.get('instance_idx', 0)
                plt.close()  # Force plots create their own figure
                
                # Create a simplified force plot as matplotlib figure
                plt.figure(figsize=(12, 4))
                
                # Create a bar plot of SHAP values
                values = shap_values[instance_idx]
                sorted_idx = np.argsort(np.abs(values))
                sorted_values = values[sorted_idx]
                sorted_names = [feature_names[i] for i in sorted_idx]
                
                plt.barh(range(len(sorted_values)), sorted_values)
                plt.yticks(range(len(sorted_values)), sorted_names)
                plt.xlabel('SHAP Value')
                plt.title('Feature Contributions')
                
                fig = plt.gcf()
            
            elif plot_type == 'dependence':
                # Dependence plot for a specific feature
                feature_idx = kwargs.get('feature_idx', 0)
                if isinstance(feature_idx, str) and feature_idx in feature_names:
                    feature_idx = feature_names.index(feature_idx)
                
                plt.clf()  # Clear the figure
                self.shap.dependence_plot(
                    feature_idx, shap_values, X, 
                    feature_names=feature_names, 
                    show=False
                )
            
            else:
                raise ValueError(f"Unknown plot type: {plot_type}")
            
            return fig
            
        except Exception as e:
            logging.error(f"Error generating SHAP plot: {str(e)}")
            plt.close(fig)
            
            # Create a simple error figure
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.text(0.5, 0.5, f"Error generating plot: {str(e)}", 
                    ha='center', va='center', fontsize=12)
            ax.set_title(f"Error: {type(e).__name__}")
            ax.axis('off')
            
            return fig

class LimeExplainer(BaseExplainer):
    """LIME explainer for model interpretability"""
    def __init__(self, model, feature_names=None):
        """
        Initialize the LIME explainer
        
        Parameters:
        -----------
        model : object
            Trained model to explain
        feature_names : List[str], optional
            Names of features
        """
        super().__init__(model, feature_names)
        
        # Import LIME here to avoid dependency if not used
        try:
            import lime
            import lime.lime_tabular
            self.lime = lime
            self.explainer = None
        except ImportError:
            logging.error("LIME not installed. Install with: pip install lime")
            raise
    
    def explain(self, X, y=None):
        """
        Generate LIME explanations for the model
        
        Parameters:
        -----------
        X : np.ndarray or pd.DataFrame
            Input features
        y : np.ndarray or pd.Series, optional
            Target values
            
        Returns:
        --------
        Dict[str, Any]
            Explanation results with LIME explanations
        """
        try:
            # Convert to numpy if pandas
            if isinstance(X, pd.DataFrame):
                if self.feature_names is None:
                    self.feature_names = X.columns.tolist()
                X_np = X.values
            else:
                X_np = X
                if self.feature_names is None:
                    self.feature_names = [f'Feature {i}' for i in range(X.shape[1])]
            
            # Determine if classification or regression
            predict_fn = self.model.predict_proba if hasattr(self.model, 'predict_proba') else self.model.predict
            mode = 'classification' if hasattr(self.model, 'predict_proba') else 'regression'
            
            # Create explainer
            self.explainer = self.lime.lime_tabular.LimeTabularExplainer(
                X_np,
                feature_names=self.feature_names,
                mode=mode
            )
            
            # Generate explanation for selected samples
            n_samples = min(5, X_np.shape[0])
            explanations = []
            
            for i in range(n_samples):
                exp = self.explainer.explain_instance(
                    X_np[i], 
                    predict_fn,
                    num_features=X_np.shape[1]
                )
                explanations.append(exp)
            
            # Calculate feature importance
            feature_importance = {}
            for feature in self.feature_names:
                feature_importance[feature] = 0.0
            
            # Aggregate importance across samples
            for exp in explanations:
                # Get the explanation as a list of (feature, weight) tuples
                if mode == 'classification':
                    explanation = exp.as_list(label=0)  # Use first class for classification
                else:
                    explanation = exp.as_list()
                
                # Add importance to feature
                for feature, weight in explanation:
                    for feat_name in self.feature_names:
                        if feat_name in feature:
                            feature_importance[feat_name] += abs(weight)
            
            # Normalize importance
            if explanations:
                for feature in feature_importance:
                    feature_importance[feature] /= len(explanations)
            
            # Store explanation
            self.explanation = {
                'explanations': explanations,
                'feature_names': self.feature_names,
                'feature_importance': feature_importance,
                'X': X_np[:n_samples]
            }
            
            return {
                'explanations': [exp.as_list() for exp in explanations],
                'feature_importance': feature_importance,
                'feature_names': self.feature_names
            }
            
        except Exception as e:
            logging.error(f"Error generating LIME explanation: {str(e)}")
            raise
    
    def plot(self, plot_type='summary', **kwargs):
        """
        Generate LIME plots
        
        Parameters:
        -----------
        plot_type : str, optional
            Type of plot ('summary', 'local', 'all_local')
        **kwargs : dict
            Additional parameters for the plot
            
        Returns:
        --------
        matplotlib.figure.Figure
            The generated plot
        """
        if self.explanation is None:
            raise ValueError("No explanation generated. Call explain() first.")
        
        # Get data from explanation
        explanations = self.explanation['explanations']
        feature_importance = self.explanation['feature_importance']
        
        # Set up matplotlib
        setup_plot_style()
        
        # Create different types of plots
        try:
            if plot_type == 'summary':
                # Summary of feature importance
                fig, ax = plt.subplots(figsize=(10, 6))
                
                # Sort features by importance
                sorted_features = sorted(
                    feature_importance.items(), 
                    key=lambda x: x[1], 
                    reverse=True
                )
                
                features = [f[0] for f in sorted_features]
                importance = [f[1] for f in sorted_features]
                
                # Plot feature importance
                ax.barh(range(len(features)), importance)
                ax.set_yticks(range(len(features)))
                ax.set_yticklabels(features)
                ax.set_xlabel('Importance')
                ax.set_title('LIME Feature Importance')
                
                plt.tight_layout()
                
            elif plot_type == 'local':
                # Plot for a single instance
                instance_idx = kwargs.get('instance_idx', 0)
                
                if instance_idx >= len(explanations):
                    raise ValueError(f"Instance index {instance_idx} out of range")
                
                exp = explanations[instance_idx]
                
                # Get explanation as a list
                if hasattr(exp, 'as_list'):
                    if hasattr(self.model, 'predict_proba'):
                        exp_list = exp.as_list(label=0)  # Use first class for classification
                    else:
                        exp_list = exp.as_list()
                else:
                    exp_list = exp
                
                # Sort by absolute weight
                exp_list = sorted(exp_list, key=lambda x: abs(x[1]), reverse=True)
                
                # Extract features and weights
                features = [x[0] for x in exp_list]
                weights = [x[1] for x in exp_list]
                
                # Create plot
                fig, ax = plt.subplots(figsize=(10, 6))
                bars = ax.barh(range(len(features)), weights)
                
                # Color positive and negative bars
                for i, w in enumerate(weights):
                    bars[i].set_color('green' if w > 0 else 'red')
                
                ax.set_yticks(range(len(features)))
                ax.set_yticklabels(features)
                ax.set_xlabel('Weight')
                ax.set_title(f'LIME Explanation for Instance {instance_idx}')
                
                # Add a vertical line at x=0
                ax.axvline(x=0, color='black', linestyle='-', alpha=0.3)
                
                plt.tight_layout()
                
            elif plot_type == 'all_local':
                # Plot all local explanations
                n_instances = len(explanations)
                n_cols = min(3, n_instances)
                n_rows = (n_instances + n_cols - 1) // n_cols
                
                fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
                
                # Handle single subplot case
                if n_rows == 1 and n_cols == 1:
                    axes = np.array([[axes]])
                elif n_rows == 1 or n_cols == 1:
                    axes = axes.reshape(-1, 1) if n_cols == 1 else axes.reshape(1, -1)
                
                # Create subplots for each instance
                for i, exp in enumerate(explanations):
                    if i >= n_rows * n_cols:
                        break
                        
                    row = i // n_cols
                    col = i % n_cols
                    ax = axes[row, col]
                    
                    # Get explanation as a list
                    if hasattr(exp, 'as_list'):
                        if hasattr(self.model, 'predict_proba'):
                            exp_list = exp.as_list(label=0)
                        else:
                            exp_list = exp.as_list()
                    else:
                        exp_list = exp
                    
                    # Sort by absolute weight
                    exp_list = sorted(exp_list, key=lambda x: abs(x[1]), reverse=True)
                    
                    # Extract features and weights
                    features = [x[0] for x in exp_list]
                    weights = [x[1] for x in exp_list]
                    
                    # Create plot
                    bars = ax.barh(range(len(features)), weights)
                    
                    # Color positive and negative bars
                    for j, w in enumerate(weights):
                        bars[j].set_color('green' if w > 0 else 'red')
                    
                    ax.set_yticks(range(len(features)))
                    ax.set_yticklabels(features)
                    ax.set_xlabel('Weight')
                    ax.set_title(f'Instance {i}')
                    
                    # Add a vertical line at x=0
                    ax.axvline(x=0, color='black', linestyle='-', alpha=0.3)
                
                # Hide unused subplots
                for i in range(n_instances, n_rows * n_cols):
                    row = i // n_cols
                    col = i % n_cols
                    axes[row, col].axis('off')
                
                plt.tight_layout()
                
            else:
                raise ValueError(f"Unknown plot type: {plot_type}")
            
            return fig
            
        except Exception as e:
            logging.error(f"Error generating LIME plot: {str(e)}")
            
            # Create a simple error figure
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.text(0.5, 0.5, f"Error generating plot: {str(e)}", 
                    ha='center', va='center', fontsize=12)
            ax.set_title(f"Error: {type(e).__name__}")
            ax.axis('off')
            
            return fig

class FeatureImportanceExplainer(BaseExplainer):
    """Feature importance explainer for model interpretability"""
    def __init__(self, model, feature_names=None):
        """
        Initialize the feature importance explainer
        
        Parameters:
        -----------
        model : object
            Trained model to explain
        feature_names : List[str], optional
            Names of features
        """
        super().__init__(model, feature_names)
    
    def explain(self, X, y=None):
        """
        Generate feature importance explanation
        
        Parameters:
        -----------
        X : np.ndarray or pd.DataFrame
            Input features
        y : np.ndarray or pd.Series, optional
            Target values
            
        Returns:
        --------
        Dict[str, Any]
            Explanation results with feature importance
        """
        try:
            # Convert to numpy if pandas
            if isinstance(X, pd.DataFrame):
                if self.feature_names is None:
                    self.feature_names = X.columns.tolist()
                X_np = X.values
            else:
                X_np = X
                if self.feature_names is None:
                    self.feature_names = [f'Feature {i}' for i in range(X.shape[1])]
            
            # Get feature importance
            if hasattr(self.model, 'feature_importances_'):
                # Tree-based models have feature_importances_
                importance = self.model.feature_importances_
            elif hasattr(self.model, 'coef_'):
                # Linear models have coef_
                importance = np.abs(self.model.coef_)
                if importance.ndim > 1:
                    importance = importance.mean(axis=0)
            else:
                # Compute permutation importance
                from sklearn.inspection import permutation_importance
                
                # Need target values for permutation importance
                if y is None:
                    raise ValueError("Target values (y) required for permutation importance")
                
                result = permutation_importance(self.model, X_np, y, n_repeats=10, random_state=42)
                importance = result.importances_mean
            
            # Normalize importance
            importance = importance / importance.sum() if importance.sum() > 0 else importance
            
            # Create feature importance dictionary
            feature_importance = {}
            for i, feature in enumerate(self.feature_names):
                feature_importance[feature] = float(importance[i])
            
            # Store explanation
            self.explanation = {
                'feature_importance': feature_importance,
                'feature_names': self.feature_names,
                'X': X_np
            }
            
            return {
                'feature_importance': feature_importance,
                'feature_names': self.feature_names
            }
            
        except Exception as e:
            logging.error(f"Error generating feature importance explanation: {str(e)}")
            raise
    
    def plot(self, plot_type='bar', **kwargs):
        """
        Generate feature importance plots
        
        Parameters:
        -----------
        plot_type : str, optional
            Type of plot ('bar', 'horizontal_bar', 'heatmap')
        **kwargs : dict
            Additional parameters for the plot
            
        Returns:
        --------
        matplotlib.figure.Figure
            The generated plot
        """
        if self.explanation is None:
            raise ValueError("No explanation generated. Call explain() first.")
        
        # Get data from explanation
        feature_importance = self.explanation['feature_importance']
        
        # Set up matplotlib
        setup_plot_style()
        
        try:
            if plot_type == 'bar':
                # Bar plot of feature importance
                fig, ax = plt.subplots(figsize=(10, 6))
                
                # Sort features by importance
                sorted_features = sorted(
                    feature_importance.items(), 
                    key=lambda x: x[1], 
                    reverse=True
                )
                
                features = [f[0] for f in sorted_features]
                importance = [f[1] for f in sorted_features]
                
                # Plot feature importance
                ax.bar(features, importance)
                ax.set_xlabel('Feature')
                ax.set_ylabel('Importance')
                ax.set_title('Feature Importance')
                
                # Rotate x labels if many features
                if len(features) > 5:
                    plt.xticks(rotation=45, ha='right')
                
                plt.tight_layout()
                
            elif plot_type == 'horizontal_bar':
                # Horizontal bar plot of feature importance
                fig, ax = plt.subplots(figsize=(10, 8))
                
                # Sort features by importance
                sorted_features = sorted(
                    feature_importance.items(), 
                    key=lambda x: x[1]
                )
                
                features = [f[0] for f in sorted_features]
                importance = [f[1] for f in sorted_features]
                
                # Plot feature importance
                ax.barh(features, importance)
                ax.set_xlabel('Importance')
                ax.set_ylabel('Feature')
                ax.set_title('Feature Importance')
                
                plt.tight_layout()
                
            elif plot_type == 'heatmap':
                # Heatmap of feature importance
                fig, ax = plt.subplots(figsize=(12, 10))
                
                # Sort features by importance
                sorted_features = sorted(
                    feature_importance.items(), 
                    key=lambda x: x[1], 
                    reverse=True
                )
                
                features = [f[0] for f in sorted_features]
                importance = [f[1] for f in sorted_features]
                
                # Create heatmap
                im = ax.imshow([importance], cmap='YlOrRd')
                
                # Add colorbar
                cbar = ax.figure.colorbar(im, ax=ax)
                cbar.ax.set_ylabel('Importance', rotation=-90, va='bottom')
                
                # Add feature names
                ax.set_yticks([0])
                ax.set_yticklabels(['Importance'])
                ax.set_xticks(np.arange(len(features)))
                ax.set_xticklabels(features)
                
                # Rotate x labels if many features
                if len(features) > 5:
                    plt.xticks(rotation=45, ha='right')
                
                ax.set_title('Feature Importance Heatmap')
                
                # Add text annotations
                for i in range(len(features)):
                    text = ax.text(i, 0, f'{importance[i]:.2f}',
                                ha='center', va='center', color='black')
                
                plt.tight_layout()
                
            else:
                raise ValueError(f"Unknown plot type: {plot_type}")
            
            return fig
            
        except Exception as e:
            logging.error(f"Error generating feature importance plot: {str(e)}")
            
            # Create a simple error figure
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.text(0.5, 0.5, f"Error generating plot: {str(e)}", 
                    ha='center', va='center', fontsize=12)
            ax.set_title(f"Error: {type(e).__name__}")
            ax.axis('off')
            
            return fig

class ExplainerFactory:
    """Factory class to create explainers"""
    def create_explainer(self, explainer_type, model, feature_names=None):
        """
        Create an explainer of the specified type
        
        Parameters:
        -----------
        explainer_type : str
            Type of explainer ('shap', 'lime', 'feature_importance')
        model : object
            Trained model to explain
        feature_names : List[str], optional
            Names of features
            
        Returns:
        --------
        BaseExplainer
            The created explainer
        """
        if explainer_type == 'shap':
            return ShapExplainer(model, feature_names)
        elif explainer_type == 'lime':
            return LimeExplainer(model, feature_names)
        elif explainer_type == 'feature_importance':
            return FeatureImportanceExplainer(model, feature_names)
        else:
            raise ValueError(f"Unknown explainer type: {explainer_type}")

def run_explainability_analysis(
    args=None, model=None, X=None, y=None, 
    explainer_type='shap', plot_types=None, 
    generate_plots=True, n_samples=5, **kwargs
):
    """
    Run explainability analysis on a trained model
    
    Parameters:
    -----------
    args : argparse.Namespace, optional
        Command-line arguments
    model : object, optional
        Trained model to explain
    X : np.ndarray or pd.DataFrame, optional
        Input features
    y : np.ndarray or pd.Series, optional
        Target values
    explainer_type : str, optional
        Type of explainer to use ('shap', 'lime', 'feature_importance')
    plot_types : List[str], optional
        List of plot types to generate
    generate_plots : bool, optional
        Whether to generate plots
    n_samples : int, optional
        Number of samples to use for explanation
    **kwargs : dict
        Additional parameters for the explainer
        
    Returns:
    --------
    Dict[str, Any]
        Dictionary containing explanation results
    """
    logging.info("Running explainability analysis")
    
    # Process args if provided
    if args is not None:
        explainer_type = getattr(args, 'explainer', explainer_type)
        generate_plots = getattr(args, 'explain_plots', generate_plots)
        n_samples = getattr(args, 'explain_samples', n_samples)
        
        # Parse plot types
        if hasattr(args, 'explain_plot_types'):
            if args.explain_plot_types == 'all':
                if explainer_type == 'shap':
                    plot_types = ['summary', 'bar', 'beeswarm', 'waterfall', 'dependence']
                elif explainer_type == 'lime':
                    plot_types = ['summary', 'local', 'all_local']
                elif explainer_type == 'feature_importance':
                    plot_types = ['bar', 'horizontal_bar', 'heatmap']
                else:
                    plot_types = None
            else:
                plot_types = args.explain_plot_types.split(',')
    
    # Create results directory
    results_dir = Path('results/explainability')
    results_dir.mkdir(exist_ok=True, parents=True)
    
    # Create or load model and data if not provided
    from core.evaluation import create_synthetic_data, create_default_model
    
    if model is None:
        model = create_default_model()
        
        if X is None or y is None:
            X, y = create_synthetic_data(n_samples=100, n_features=5)
            
            # Split data
            from sklearn.model_selection import train_test_split
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            # Train model
            model.fit(X_train, y_train)
            
            # Use test data for explanation
            X, y = X_test, y_test
    
    elif X is None or y is None:
        X, y = create_synthetic_data(n_samples=100, n_features=5)
    
    # Limit number of samples to explain
    if n_samples and n_samples < len(X):
        indices = np.random.choice(len(X), n_samples, replace=False)
        X_sample = X[indices]
        y_sample = y[indices] if y is not None else None
    else:
        X_sample = X
        y_sample = y
    
    # Determine feature names
    if hasattr(X, 'columns'):
        feature_names = X.columns.tolist()
    else:
        feature_names = [f'Feature {i}' for i in range(X.shape[1])]
    
    # Create explainer
    try:
        factory = ExplainerFactory()
        explainer = factory.create_explainer(explainer_type, model, feature_names)
        
        # Generate explanation
        explanation = explainer.explain(X_sample, y_sample)
        
        # Generate plots if requested
        plot_paths = {}
        
        if generate_plots and plot_types:
            for plot_type in plot_types:
                try:
                    # Generate plot
                    fig = explainer.plot(plot_type, **kwargs)
                    
                    # Save plot
                    plot_path = save_plot(
                        fig, 
                        f"explainability_{explainer_type}_{plot_type}", 
                        plot_type='explainability'
                    )
                    
                    plot_paths[plot_type] = str(plot_path)
                    
                    logging.info(f"Generated {plot_type} plot for {explainer_type} explainer")
                    
                except Exception as e:
                    logging.error(f"Error generating {plot_type} plot: {str(e)}")
        
        # Prepare results
        results = {
            'explainer_type': explainer_type,
            'feature_names': feature_names,
            'explanation': explanation,
            'plot_paths': plot_paths
        }
        
        # Save results
        timestamp = time.strftime('%Y%m%d_%H%M%S')
        results_file = results_dir / f'explainability_results_{explainer_type}_{timestamp}.json'
        
        # Clean explanation for JSON serialization
        if 'explanation' in results and 'shap_values' in results['explanation']:
            # Convert shap_values to list for JSON serialization
            results['explanation']['shap_values'] = 'large_array_omitted'
        
        save_json(results, results_file)
        logging.info(f"Explainability results saved to {results_file}")
        
        # Print summary
        print("\nExplainability Analysis Summary:")
        print("================================")
        print(f"Explainer: {explainer_type}")
        print(f"Number of samples: {len(X_sample)}")
        
        if 'feature_importance' in explanation:
            print("\nTop 5 Important Features:")
            sorted_features = sorted(
                explanation['feature_importance'].items(),
                key=lambda x: x[1],
                reverse=True
            )
            for i, (feature, importance) in enumerate(sorted_features[:5]):
                print(f"  {i+1}. {feature}: {importance:.4f}")
        
        if plot_paths:
            print("\nGenerated Plots:")
            for plot_type, path in plot_paths.items():
                print(f"  {plot_type}: {path}")
        
        return results
        
    except Exception as e:
        logging.error(f"Error in explainability analysis: {str(e)}")
        import traceback
        traceback.print_exc()
        
        return {
            'error': str(e),
            'explainer_type': explainer_type
        }
