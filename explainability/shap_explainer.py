"""
shap_explainer.py
----------------
SHAP-based model explainability implementation
"""

from typing import Dict, Any, List, Optional, Union, Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import logging
from .base_explainer import BaseExplainer

# Set up logging
logger = logging.getLogger(__name__)

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
        
        # Create a copy of kwargs without n_samples for tree-based explainers
        tree_kwargs = {k: v for k, v in self.explainer_kwargs.items() if k != 'n_samples'}
        
        # Create appropriate explainer based on model type
        if self.explainer_type == 'auto':
            # Try to automatically determine the best explainer
            model_type = type(self.model).__module__ + '.' + type(self.model).__name__
            logger.info(f"Model type: {model_type}")
            
            if 'sklearn' in model_type and 'RandomForestClassifier' in model_type:
                # For Random Forest classifiers, specifically handle probability output
                try:
                    self.explainer = shap.TreeExplainer(self.model, model_output='probability', **tree_kwargs)
                    logger.info("Using TreeExplainer with RandomForestClassifier")
                except Exception as e:
                    logger.warning(f"Error creating TreeExplainer for RandomForest with probability output: {str(e)}")
                    # Fallback to standard TreeExplainer
                    self.explainer = shap.TreeExplainer(self.model, **tree_kwargs)
            elif 'sklearn' in model_type and 'tree' in model_type.lower():
                # For other tree-based models (Decision Trees, etc.)
                self.explainer = shap.TreeExplainer(self.model, **tree_kwargs)
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
                self.explainer = shap.TreeExplainer(self.model, **tree_kwargs)
            elif 'lightgbm' in model_type.lower():
                # For LightGBM models
                self.explainer = shap.TreeExplainer(self.model, **tree_kwargs)
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
            self.explainer = shap.TreeExplainer(self.model, **tree_kwargs)
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
                feature_names: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Calculate SHAP values for the provided input features
        
        Args:
            X: Input features
            y: Target values (optional)
            feature_names: Names of features (optional)
            
        Returns:
            Dictionary containing explanation and feature importance
        """
        # Set feature names if provided
        if feature_names is not None:
            self.feature_names = feature_names
            
        # Create explainer if not already created
        if self.explainer is None:
            self._create_explainer(X)
        
        # Try multiple approaches to get valid SHAP values
        # This handles different variations in SHAP API across versions
        shap_values = None
        error_message = None
        
        try:
            # First try standard shap_values() call
            if hasattr(self.explainer, 'shap_values'):
                try:
                    # Try with nsamples parameter first (for KernelExplainer)
                    if 'kernel' in str(type(self.explainer)).lower():
                        shap_values = self.explainer.shap_values(X, nsamples=100)
                    else:
                        # For tree-based models, we don't use nsamples
                        shap_values = self.explainer.shap_values(X)
                except Exception as e1:
                    error_message = str(e1)
                    # If that fails, try without nsamples
                    try:
                        shap_values = self.explainer.shap_values(X)
                    except Exception as e2:
                        # If still failing, we'll try alternative approaches
                        error_message += f" | {str(e2)}"
                        pass
        except Exception as e:
            error_message = str(e)
        
        # If shap_values is still None, try alternative approaches
        if shap_values is None:
            # Try using TreeExplainer with different configurations
            try:
                logger.warning(f"Error calculating SHAP values: {error_message}. Trying TreeExplainer with interventional approach.")
                import shap
                explainer = shap.TreeExplainer(self.model, data=X[:10] if len(X) > 10 else X, 
                                             feature_perturbation="interventional")
                shap_values = explainer.shap_values(X)
            except Exception as e:
                logger.warning(f"TreeExplainer with interventional approach failed: {str(e)}. Falling back to model feature importances.")
                # If that fails too, we'll use model's feature importances as a fallback
                shap_values = None
        
        # Initialize feature importance dictionary
        feature_importance = {}
        
        # Process SHAP values if available
        if shap_values is not None:
            # SHAP values can come in different formats
            if isinstance(shap_values, list):
                # For multi-class models, average across classes
                # We take absolute values because both positive and negative 
                # SHAP values indicate importance
                if self.feature_names is not None:
                    if isinstance(shap_values[0], np.ndarray):
                        # Common case: a list of arrays (one per class)
                        avg_abs_shap = np.mean([np.abs(sv) for sv in shap_values], axis=0)
                        for i, feature in enumerate(self.feature_names):
                            if i < len(avg_abs_shap):
                                if len(avg_abs_shap.shape) > 1:
                                    # If we have a 2D array, take mean across samples
                                    feature_importance[feature] = np.mean(avg_abs_shap[:, i])
                                else:
                                    # If we have a 1D array (just one sample)
                                    feature_importance[feature] = avg_abs_shap[i]
                    else:
                        # Handle other list formats if needed
                        logger.warning(f"Unexpected SHAP values format: {type(shap_values[0])}")
                else:
                    # If feature names not provided, use indices
                    feature_importance = dict(enumerate(np.mean([np.abs(sv) for sv in shap_values], axis=0)))
            elif isinstance(shap_values, np.ndarray):
                # For single-class models
                if self.feature_names is not None:
                    for i, feature in enumerate(self.feature_names):
                        if i < shap_values.shape[1]:
                            feature_importance[feature] = np.mean(np.abs(shap_values[:, i]))
                else:
                    feature_importance = dict(enumerate(np.mean(np.abs(shap_values), axis=0)))
            else:
                logger.warning(f"Unexpected SHAP values format: {type(shap_values)}")
                
            # Scale values for better interpretability 
            total = sum(feature_importance.values()) if feature_importance else 1
            if total > 0:
                feature_importance = {k: v / total for k, v in feature_importance.items()}
        
        # If feature importance is all zeros, use model coefficients or other importance metrics
        if all(np.abs(val).max() < 1e-10 if isinstance(val, np.ndarray) else abs(val) < 1e-10 
               for val in feature_importance.values()):
            logger.warning("SHAP values are effectively zero. Trying to use model-specific feature importance.")
            
            if hasattr(self.model, 'feature_importances_'):
                importance_values = self.model.feature_importances_
                if self.feature_names is not None:
                    feature_importance = dict(zip(self.feature_names, importance_values))
                else:
                    feature_importance = dict(enumerate(importance_values))
            elif hasattr(self.model, 'coef_'):
                coef = self.model.coef_
                if len(coef.shape) > 1:
                    # For multi-class models
                    importance_values = np.abs(coef).mean(axis=0)
                else:
                    importance_values = np.abs(coef)
                    
                if self.feature_names is not None:
                    feature_importance = dict(zip(self.feature_names, importance_values))
                else:
                    feature_importance = dict(enumerate(importance_values))
                    
            # If still no valid feature importance, create some synthetic values based on feature variance
            if all(np.abs(val).max() < 1e-10 if isinstance(val, np.ndarray) else abs(val) < 1e-10
                  for val in feature_importance.values()) and isinstance(X, pd.DataFrame):
                logger.warning("Using feature variance as a proxy for importance")
                for i, col in enumerate(X.columns):
                    if col in self.feature_names:
                        feature_importance[col] = X[col].var()
        
        # Store explanation data
        self.last_explanation = {
            'shap_values': shap_values if hasattr(self, 'shap_values') else None,
            'data': X,
            'feature_names': self.feature_names,
            'feature_importance': feature_importance,
            'explainer_type': self.explainer_type,
            'background_data': self.background_data
        }
        
        return self.last_explanation
    
    def get_feature_importance(self) -> Dict[str, float]:
        """
        Get feature importance from the last explanation
        
        Returns:
            Dictionary containing feature importance
        """
        if self.last_explanation is None:
            raise ValueError("No explanation available. Run explain() first.")
        
        return self.last_explanation['feature_importance']
    
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
        
        # Create a simple bar plot of feature importance as a fallback
        if plot_type == 'bar' or plot_type == 'summary':
            try:
                # Create a simple bar plot of feature importance
                feature_importance = self.get_feature_importance()
                
                # Convert to float and handle arrays
                processed_importance = {}
                for feature, importance in feature_importance.items():
                    # Handle numpy arrays
                    if isinstance(importance, np.ndarray):
                        # Use absolute mean value for arrays
                        processed_importance[feature] = float(np.abs(importance).mean())
                    else:
                        processed_importance[feature] = float(importance)
                
                # Sort by absolute importance
                sorted_importance = sorted(
                    processed_importance.items(),
                    key=lambda x: abs(x[1]),
                    reverse=True
                )[:max_display]
                
                # Create bar chart
                plt.figure(figsize=(10, 8))
                features = [x[0] for x in sorted_importance]
                importance_values = [x[1] for x in sorted_importance]
                
                # Plot horizontal bar chart
                plt.barh(range(len(features)), importance_values, align='center')
                plt.yticks(range(len(features)), features)
                plt.xlabel('Feature Importance')
                plt.title('Feature Importance (Absolute Values)')
                plt.tight_layout()
                
                return plt.gcf()
            except Exception as e:
                logger.error(f"Error generating bar plot: {str(e)}")
                # Fall through to try other plot types
        
        # Handle multi-class models
        if isinstance(shap_values, list):
            # Use specified class for multi-class models
            selected_shap_values = shap_values[class_index]
        else:
            selected_shap_values = shap_values
        
        # Create figure
        plt.figure(figsize=kwargs.get('figsize', (10, 6)))
        
        try:
            # Generate appropriate plot based on plot_type
            if plot_type == 'summary':
                # Try to use shap's summary plot with error handling
                try:
                    shap.summary_plot(
                        selected_shap_values, 
                        data, 
                        feature_names=feature_names,
                        max_display=max_display,
                        show=False
                    )
                except Exception as e:
                    logger.error(f"Error generating summary plot: {str(e)}")
                    # Fall back to simpler visualization
                    plt.clf()  # Clear the figure
                    
                    # Create simple bar plot of feature importance
                    feature_importance = self.get_feature_importance()
                    
                    # Sort by importance
                    sorted_importance = sorted(
                        [(k, float(np.mean(abs(v))) if isinstance(v, np.ndarray) else abs(float(v))) 
                         for k, v in feature_importance.items()],
                        key=lambda x: x[1],
                        reverse=True
                    )[:max_display]
                    
                    features = [x[0] for x in sorted_importance]
                    importance_values = [x[1] for x in sorted_importance]
                    
                    plt.barh(range(len(features)), importance_values, align='center')
                    plt.yticks(range(len(features)), features)
                    plt.xlabel('Feature Importance')
                    plt.title('Feature Importance (Absolute Values)')
            elif plot_type == 'bar':
                try:
                    shap.summary_plot(
                        selected_shap_values, 
                        data, 
                        feature_names=feature_names,
                        max_display=max_display,
                        plot_type='bar',
                        show=False
                    )
                except Exception as e:
                    logger.error(f"Error generating bar plot: {str(e)}")
                    # Fall back to simpler visualization
                    plt.clf()  # Clear the figure
                    
                    # Create simple bar plot of feature importance
                    feature_importance = self.get_feature_importance()
                    
                    # Sort by importance
                    sorted_importance = sorted(
                        [(k, float(np.mean(abs(v))) if isinstance(v, np.ndarray) else abs(float(v))) 
                         for k, v in feature_importance.items()],
                        key=lambda x: x[1],
                        reverse=True
                    )[:max_display]
                    
                    features = [x[0] for x in sorted_importance]
                    importance_values = [x[1] for x in sorted_importance]
                    
                    plt.barh(range(len(features)), importance_values, align='center')
                    plt.yticks(range(len(features)), features)
                    plt.xlabel('Feature Importance')
                    plt.title('Feature Importance (Absolute Values)')
            elif plot_type == 'beeswarm':
                try:
                    shap.summary_plot(
                        selected_shap_values, 
                        data, 
                        feature_names=feature_names,
                        max_display=max_display,
                        plot_type='dot',
                        show=False
                    )
                except Exception as e:
                    logger.error(f"Error generating beeswarm plot: {str(e)}")
                    # Fall back to bar plot
                    plt.clf()
                    self.plot('bar', **kwargs)
            else:
                # For other plot types, try with robust error handling
                try:
                    if plot_type == 'waterfall':
                        # Make sure instance_index is valid
                        if instance_index >= len(selected_shap_values):
                            instance_index = 0
                            
                        shap.waterfall_plot(
                            shap.Explanation(
                                values=selected_shap_values[instance_index], 
                                base_values=float(np.mean(selected_shap_values)),
                                data=data[instance_index] if instance_index < len(data) else data[0],
                                feature_names=feature_names
                            ),
                            show=False
                        )
                    elif plot_type == 'force':
                        # Make sure instance_index is valid
                        if instance_index >= len(selected_shap_values):
                            instance_index = 0
                            
                        shap.force_plot(
                            float(np.mean(selected_shap_values)),
                            selected_shap_values[instance_index],
                            data[instance_index] if instance_index < len(data) else data[0],
                            feature_names=feature_names,
                            matplotlib=True,
                            show=False
                        )
                    elif plot_type == 'decision':
                        # Make sure instance_index is valid
                        if instance_index >= len(selected_shap_values):
                            instance_index = 0
                            
                        shap.decision_plot(
                            float(np.mean(selected_shap_values)),
                            selected_shap_values[instance_index:instance_index+1],
                            data[instance_index:instance_index+1] if instance_index < len(data) else data[0:1],
                            feature_names=feature_names,
                            show=False
                        )
                    elif plot_type == 'dependence':
                        if isinstance(feature_index, str) and feature_names is not None:
                            feature_index = feature_names.index(feature_index)
                        
                        # Make sure feature_index is valid
                        if feature_index >= len(feature_names):
                            feature_index = 0
                            
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
                        
                        # Make sure indices are valid
                        if feature_index >= len(feature_names):
                            feature_index = 0
                        if interaction_index is not None and interaction_index >= len(feature_names):
                            interaction_index = 1 if feature_index != 1 else 0
                            
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
                except Exception as e:
                    logger.error(f"Error generating {plot_type} plot: {str(e)}")
                    # Fall back to bar plot
                    plt.clf()
                    self.plot('bar', **kwargs)
        except Exception as e:
            logger.error(f"Error creating plot: {str(e)}")
            # Create a simple fallback plot
            plt.clf()
            plt.figure(figsize=(10, 6))
            plt.text(0.5, 0.5, f"Error creating {plot_type} plot: {str(e)}", 
                     horizontalalignment='center', verticalalignment='center')
            plt.axis('off')
            
        fig = plt.gcf()
        return fig
