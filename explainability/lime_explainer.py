"""
lime_explainer.py
---------------
LIME-based model explainability implementation
"""

from typing import Dict, Any, List, Optional, Union, Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from .base_explainer import BaseExplainer

class LimeExplainer(BaseExplainer):
    """
    Explainer using LIME (Local Interpretable Model-agnostic Explanations)
    
    This explainer uses the LIME library to generate explanations for machine learning models.
    It provides local explanations for individual predictions.
    """
    
    def __init__(self, model=None, feature_names: Optional[List[str]] = None, 
                 mode: str = 'regression', **kwargs):
        """
        Initialize LIME explainer
        
        Args:
            model: Pre-trained model to explain
            feature_names: List of feature names
            mode: Type of task ('classification' or 'regression')
            **kwargs: Additional parameters for LIME explainer
        """
        super().__init__('LIME', model, feature_names)
        self.mode = mode
        self.explainer_kwargs = kwargs
        self.explainer = None
        self.explanations = []
        
        # Define supported plot types
        self.supported_plot_types = ['local', 'all_local', 'summary']
    
    def _create_explainer(self, X: Union[np.ndarray, pd.DataFrame]):
        """
        Create LIME explainer based on data type
        
        Args:
            X: Input features to use for creating explainer
        """
        try:
            import lime
            from lime import lime_tabular
        except ImportError:
            raise ImportError("LIME package is required. Install with 'pip install lime'")
        
        # Determine if data is categorical or continuous
        categorical_features = self.explainer_kwargs.pop('categorical_features', None)
        if categorical_features is None and isinstance(X, pd.DataFrame):
            # Automatically detect categorical features
            categorical_features = []
            for i, col in enumerate(X.columns):
                if X[col].dtype == 'object' or X[col].dtype.name == 'category' or len(X[col].unique()) < 10:
                    categorical_features.append(i)
        
        # Remove parameters that should not be passed to the explainer constructor
        # but should be used in explain_instance
        self.explainer_kwargs.pop('n_samples', None)
        self.explainer_kwargs.pop('num_samples', None)
        
        # Create explainer
        self.explainer = lime_tabular.LimeTabularExplainer(
            X.values if isinstance(X, pd.DataFrame) else X,
            feature_names=self.feature_names,
            class_names=self.explainer_kwargs.pop('class_names', None),
            categorical_features=categorical_features,
            mode=self.mode,
            **self.explainer_kwargs
        )
    
    def explain(self, X: Union[np.ndarray, pd.DataFrame], y: Optional[np.ndarray] = None, 
               **kwargs) -> Dict[str, Any]:
        """
        Generate LIME explanation for the provided data
        
        Args:
            X: Input features to explain
            y: Optional target values (not used for LIME)
            **kwargs: Additional parameters:
                - num_features: Number of features to include in explanation
                - num_samples: Number of samples to use for LIME
                - instances: List of specific instance indices to explain
                - labels: List of specific class labels to explain
            
        Returns:
            Dictionary containing explanation data
        """
        if self.model is None:
            raise ValueError("Model not set. Call set_model() first.")
        
        # Create explainer if not already created
        if self.explainer is None:
            self._create_explainer(X)
        
        # Convert pandas DataFrame to numpy array if needed
        if isinstance(X, pd.DataFrame):
            if self.feature_names is None:
                self.feature_names = X.columns.tolist()
            X_values = X.values
        else:
            X_values = X
        
        # Extract parameters
        num_features = kwargs.get('num_features', 10)
        num_samples = kwargs.get('num_samples', 5000)
        instances = kwargs.get('instances', None)
        
        # For regression, we don't need labels
        if self.mode == 'regression':
            labels = [0]  # For regression, only one output
        else:
            labels = kwargs.get('labels', [1])  # Default to positive class for binary classification
        
        # Determine instances to explain
        if instances is None:
            # Default to explaining all instances
            instances = list(range(len(X_values)))
        elif isinstance(instances, int):
            instances = [instances]
        
        # Create predict function
        if self.mode == 'classification':
            if hasattr(self.model, 'predict_proba'):
                predict_fn = self.model.predict_proba
            else:
                predict_fn = self.model.predict
        else:  # regression
            predict_fn = self.model.predict
        
        # Generate explanations for each instance
        self.explanations = []
        feature_importance = {}
        
        for instance_idx in instances:
            instance = X_values[instance_idx]
            
            # Generate explanation
            exp = self.explainer.explain_instance(
                instance, 
                predict_fn,
                num_features=num_features,
                num_samples=num_samples,
                labels=labels
            )
            
            self.explanations.append(exp)
            
            # Extract feature importance
            for label_idx in labels:
                for feature, importance in exp.as_list(label=label_idx):
                    if feature not in feature_importance:
                        feature_importance[feature] = []
                    feature_importance[feature].append(abs(importance))
        
        # Calculate average feature importance
        avg_feature_importance = {
            feature: np.mean(values) 
            for feature, values in feature_importance.items()
        }
        
        # Store explanation data
        self.last_explanation = {
            'explanations': self.explanations,
            'instances': instances,
            'feature_importance': avg_feature_importance,
            'feature_names': self.feature_names,
            'mode': self.mode,
            'labels': labels
        }
        
        return self.last_explanation
    
    def plot(self, plot_type: str = 'local', **kwargs) -> plt.Figure:
        """
        Create LIME visualization
        
        Args:
            plot_type: Type of plot to generate:
                - 'local': Plot explanation for a single instance
                - 'all_local': Plot explanations for all instances
                - 'summary': Summary of feature importance across all instances
            **kwargs: Additional parameters:
                - instance_index: Index of instance to explain (for 'local')
                - label: Class label to explain
                - figsize: Figure size as (width, height) tuple
                - title: Plot title
                - top_n: Number of top features to display
            
        Returns:
            Matplotlib figure object
        """
        if self.last_explanation is None:
            raise ValueError("No explanation available. Run explain() first.")
        
        # Extract parameters
        instance_index = kwargs.get('instance_index', 0)
        label = kwargs.get('label', self.last_explanation['labels'][0])
        figsize = kwargs.get('figsize', (10, 6))
        title = kwargs.get('title', 'LIME Explanation')
        top_n = kwargs.get('top_n', 10)
        
        # Get explanations
        explanations = self.last_explanation['explanations']
        instances = self.last_explanation['instances']
        
        if not explanations:
            raise ValueError("No explanations available")
        
        # Create figure
        fig = plt.figure(figsize=figsize)
        
        # Generate appropriate plot based on plot_type
        if plot_type == 'local':
            if instance_index >= len(explanations):
                raise ValueError(f"Instance index {instance_index} out of range")
            
            # Get explanation for specific instance
            exp = explanations[instance_index]
            
            # Create subplot
            ax = fig.add_subplot(111)
            
            # Plot explanation
            exp.as_pyplot_figure(label=label)
            plt.title(f"{title} - Instance {instances[instance_index]}")
            plt.tight_layout()
        
        elif plot_type == 'all_local':
            # Create subplots
            for i, (exp, instance_idx) in enumerate(zip(explanations, instances)):
                plt.figure(figsize=(8, 4))
                exp.as_pyplot_figure(label=label)
                plt.title(f"Instance {instance_idx}")
                plt.savefig(f"lime_instance_{instance_idx}.png")
                plt.close()
            
            # Create a summary figure
            ax = fig.add_subplot(111)
            ax.text(0.5, 0.5, f"Generated {len(explanations)} individual plots\nSaved as lime_instance_*.png", 
                   horizontalalignment='center', verticalalignment='center', fontsize=12)
            ax.axis('off')
        
        elif plot_type == 'summary':
            # Get feature importance
            feature_importance = self.last_explanation['feature_importance']
            
            # Sort features by importance
            sorted_features = sorted(
                feature_importance.items(), 
                key=lambda x: x[1], 
                reverse=True
            )
            
            # Limit to top_n features
            if top_n is not None and top_n < len(sorted_features):
                sorted_features = sorted_features[:top_n]
            
            # Extract feature names and values
            feature_names = [item[0] for item in sorted_features]
            importance_values = [item[1] for item in sorted_features]
            
            # Create bar plot
            ax = fig.add_subplot(111)
            ax.barh(feature_names, importance_values, color='skyblue')
            ax.set_xlabel('Average Absolute Importance')
            ax.set_ylabel('Features')
            ax.set_title(f"{title} - Feature Importance Summary")
            plt.tight_layout()
        
        else:
            raise ValueError(f"Unsupported plot type: {plot_type}. Supported types: {self.supported_plot_types}")
        
        return fig
