"""
shap_explainer.py
----------------
SHAP-based model explainability implementation
"""

from typing import Dict, Any, List, Optional, Union, Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from .base_explainer import BaseExplainer

class ShapExplainer(BaseExplainer):
    """
    Explainer using SHAP (SHapley Additive exPlanations)
    
    This explainer uses the SHAP library to generate explanations for machine learning models.
    It provides both global and local explanations with various visualization options.
    """
    
    def __init__(self, model=None, feature_names: Optional[List[str]] = None, 
                 explainer_type: str = 'auto', **kwargs):
        """
        Initialize SHAP explainer
        
        Args:
            model: Pre-trained model to explain
            feature_names: List of feature names
            explainer_type: Type of SHAP explainer to use ('auto', 'tree', 'kernel', 'deep', etc.)
            **kwargs: Additional parameters for SHAP explainer
        """
        super().__init__('SHAP', model, feature_names)
        self.explainer_type = explainer_type
        self.explainer_kwargs = kwargs
        self.explainer = None
        self.shap_values = None
        self.background_data = None
        
        # Define supported plot types
        self.supported_plot_types = [
            'summary', 'bar', 'beeswarm', 'waterfall', 'force', 
            'decision', 'dependence', 'interaction'
        ]
    
    def _create_explainer(self, X: Union[np.ndarray, pd.DataFrame]):
        """
        Create appropriate SHAP explainer based on model type and data
        
        Args:
            X: Input features to use for creating explainer
        """
        try:
            import shap
        except ImportError:
            raise ImportError("SHAP package is required. Install with 'pip install shap'")
        
        # Convert pandas DataFrame to numpy array if needed
        if isinstance(X, pd.DataFrame):
            if self.feature_names is None:
                self.feature_names = X.columns.tolist()
            X_values = X.values
        else:
            X_values = X
        
        # Sample background data for kernel explainer
        if self.explainer_type in ['kernel', 'auto']:
            # Sample up to 100 background samples for efficiency
            if len(X_values) > 100:
                import sklearn.utils
                background = sklearn.utils.resample(X_values, n_samples=100, random_state=42)
            else:
                background = X_values
            self.background_data = background
        
        # Create appropriate explainer based on model type
        if self.explainer_type == 'auto':
            # Try to automatically determine the best explainer
            model_type = type(self.model).__module__ + '.' + type(self.model).__name__
            
            if 'sklearn' in model_type and 'tree' in model_type.lower():
                # For tree-based models (Random Forest, Decision Trees, etc.)
                self.explainer = shap.TreeExplainer(self.model, **self.explainer_kwargs)
            elif 'sklearn' in model_type:
                # For other sklearn models
                self.explainer = shap.KernelExplainer(
                    self.model.predict_proba if hasattr(self.model, 'predict_proba') 
                    else self.model.predict,
                    background,
                    **self.explainer_kwargs
                )
            elif 'xgboost' in model_type.lower():
                # For XGBoost models
                self.explainer = shap.TreeExplainer(self.model, **self.explainer_kwargs)
            elif 'lightgbm' in model_type.lower():
                # For LightGBM models
                self.explainer = shap.TreeExplainer(self.model, **self.explainer_kwargs)
            elif 'keras' in model_type.lower() or 'tensorflow' in model_type.lower():
                # For deep learning models
                self.explainer = shap.DeepExplainer(self.model, background, **self.explainer_kwargs)
            else:
                # Default to KernelExplainer for unknown models
                self.explainer = shap.KernelExplainer(
                    self.model.predict if hasattr(self.model, 'predict') else self.model,
                    background,
                    **self.explainer_kwargs
                )
        elif self.explainer_type == 'tree':
            self.explainer = shap.TreeExplainer(self.model, **self.explainer_kwargs)
        elif self.explainer_type == 'kernel':
            self.explainer = shap.KernelExplainer(
                self.model.predict if hasattr(self.model, 'predict') else self.model,
                background,
                **self.explainer_kwargs
            )
        elif self.explainer_type == 'deep':
            self.explainer = shap.DeepExplainer(self.model, background, **self.explainer_kwargs)
        elif self.explainer_type == 'gradient':
            self.explainer = shap.GradientExplainer(self.model, background, **self.explainer_kwargs)
        else:
            raise ValueError(f"Unsupported explainer type: {self.explainer_type}")
    
    def explain(self, X: Union[np.ndarray, pd.DataFrame], y: Optional[np.ndarray] = None, 
               **kwargs) -> Dict[str, Any]:
        """
        Generate SHAP explanation for the provided data
        
        Args:
            X: Input features to explain
            y: Optional target values (not used for SHAP)
            **kwargs: Additional parameters:
                - sample_size: Number of samples to use for explanation
                - nsamples: Number of samples for KernelExplainer
                - l1_reg: L1 regularization for KernelExplainer
            
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
        
        # Sample data if sample_size is provided
        sample_size = kwargs.get('sample_size', None)
        if sample_size is not None and sample_size < len(X_values):
            import sklearn.utils
            X_sample = sklearn.utils.resample(X_values, n_samples=sample_size, random_state=42)
        else:
            X_sample = X_values
        
        # Calculate SHAP values
        nsamples = kwargs.get('nsamples', 'auto')
        l1_reg = kwargs.get('l1_reg', 'auto')
        
        if self.explainer_type in ['kernel', 'auto'] and isinstance(self.explainer, object) and hasattr(self.explainer, 'shap_values'):
            self.shap_values = self.explainer.shap_values(X_sample, nsamples=nsamples, l1_reg=l1_reg)
        else:
            self.shap_values = self.explainer.shap_values(X_sample)
        
        # Calculate feature importance
        if isinstance(self.shap_values, list):
            # For multi-class models, use the mean absolute value across all classes
            importance_values = np.abs(np.array(self.shap_values)).mean(axis=0).mean(axis=0)
        else:
            importance_values = np.abs(self.shap_values).mean(axis=0)
        
        # Create feature importance dictionary
        if self.feature_names is not None:
            feature_importance = dict(zip(self.feature_names, importance_values))
        else:
            feature_importance = dict(enumerate(importance_values))
        
        # Store explanation data
        self.last_explanation = {
            'shap_values': self.shap_values,
            'data': X_sample,
            'feature_names': self.feature_names,
            'feature_importance': feature_importance,
            'explainer_type': self.explainer_type,
            'background_data': self.background_data
        }
        
        return self.last_explanation
    
    def plot(self, plot_type: str = 'summary', **kwargs) -> plt.Figure:
        """
        Create SHAP visualization
        
        Args:
            plot_type: Type of plot to generate:
                - 'summary': Summary plot of feature importance
                - 'bar': Bar plot of feature importance
                - 'beeswarm': Beeswarm plot of SHAP values
                - 'waterfall': Waterfall plot for a single prediction
                - 'force': Force plot for a single prediction
                - 'decision': Decision plot for a single prediction
                - 'dependence': Dependence plot for a specific feature
                - 'interaction': Interaction plot between two features
            **kwargs: Additional parameters:
                - instance_index: Index of instance for local explanations
                - feature_index: Feature index for dependence plot
                - interaction_index: Second feature index for interaction plot
                - max_display: Maximum number of features to display
                - class_index: Class index for multi-class models
            
        Returns:
            Matplotlib figure object
        """
        if self.last_explanation is None:
            raise ValueError("No explanation available. Run explain() first.")
        
        try:
            import shap
        except ImportError:
            raise ImportError("SHAP package is required. Install with 'pip install shap'")
        
        # Extract parameters
        max_display = kwargs.get('max_display', 20)
        instance_index = kwargs.get('instance_index', 0)
        feature_index = kwargs.get('feature_index', 0)
        interaction_index = kwargs.get('interaction_index', None)
        class_index = kwargs.get('class_index', 0)
        
        # Get data from explanation
        shap_values = self.last_explanation['shap_values']
        data = self.last_explanation['data']
        feature_names = self.last_explanation['feature_names']
        
        # Handle multi-class models
        if isinstance(shap_values, list):
            # Use specified class for multi-class models
            selected_shap_values = shap_values[class_index]
        else:
            selected_shap_values = shap_values
        
        # Create figure
        plt.figure(figsize=kwargs.get('figsize', (10, 6)))
        
        # Generate appropriate plot based on plot_type
        if plot_type == 'summary':
            shap.summary_plot(
                selected_shap_values, 
                data, 
                feature_names=feature_names,
                max_display=max_display,
                show=False
            )
        elif plot_type == 'bar':
            shap.summary_plot(
                selected_shap_values, 
                data, 
                feature_names=feature_names,
                max_display=max_display,
                plot_type='bar',
                show=False
            )
        elif plot_type == 'beeswarm':
            shap.summary_plot(
                selected_shap_values, 
                data, 
                feature_names=feature_names,
                max_display=max_display,
                plot_type='dot',
                show=False
            )
        elif plot_type == 'waterfall':
            shap.waterfall_plot(
                shap.Explanation(
                    values=selected_shap_values[instance_index], 
                    base_values=np.mean(selected_shap_values),
                    data=data[instance_index],
                    feature_names=feature_names
                ),
                show=False
            )
        elif plot_type == 'force':
            shap.force_plot(
                np.mean(selected_shap_values),
                selected_shap_values[instance_index],
                data[instance_index],
                feature_names=feature_names,
                matplotlib=True,
                show=False
            )
        elif plot_type == 'decision':
            shap.decision_plot(
                np.mean(selected_shap_values),
                selected_shap_values[instance_index:instance_index+1],
                data[instance_index:instance_index+1],
                feature_names=feature_names,
                show=False
            )
        elif plot_type == 'dependence':
            if isinstance(feature_index, str) and feature_names is not None:
                feature_index = feature_names.index(feature_index)
            
            shap.dependence_plot(
                feature_index,
                selected_shap_values,
                data,
                feature_names=feature_names,
                interaction_index=interaction_index,
                show=False
            )
        elif plot_type == 'interaction':
            if isinstance(feature_index, str) and feature_names is not None:
                feature_index = feature_names.index(feature_index)
            
            if isinstance(interaction_index, str) and feature_names is not None:
                interaction_index = feature_names.index(interaction_index)
            
            shap.dependence_plot(
                feature_index,
                selected_shap_values,
                data,
                feature_names=feature_names,
                interaction_index=interaction_index,
                show=False
            )
        else:
            raise ValueError(f"Unsupported plot type: {plot_type}. Supported types: {self.supported_plot_types}")
        
        fig = plt.gcf()
        return fig
