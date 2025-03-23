"""
feature_importance_explainer.py
------------------------------
Simple feature importance-based explainability
"""

from typing import Dict, Any, List, Optional, Union, Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from .base_explainer import BaseExplainer

class FeatureImportanceExplainer(BaseExplainer):
    """
    Explainer using built-in feature importance methods
    
    This explainer uses the feature_importances_ attribute of tree-based models
    or permutation importance for other models.
    """
    
    def __init__(self, model=None, feature_names: Optional[List[str]] = None, 
                 method: str = 'auto'):
        """
        Initialize feature importance explainer
        
        Args:
            model: Pre-trained model to explain
            feature_names: List of feature names
            method: Method to use for feature importance calculation:
                - 'auto': Automatically select the best method
                - 'built_in': Use model's built-in feature_importances_
                - 'permutation': Use permutation importance
                - 'coef': Use coefficients for linear models
        """
        super().__init__('FeatureImportance', model, feature_names)
        self.method = method
        
        # Define supported plot types
        self.supported_plot_types = ['bar', 'horizontal_bar', 'heatmap']
    
    def explain(self, X: Union[np.ndarray, pd.DataFrame], y: Optional[np.ndarray] = None, 
               **kwargs) -> Dict[str, Any]:
        """
        Generate feature importance explanation
        
        Args:
            X: Input features to explain
            y: Target values (required for permutation importance)
            **kwargs: Additional parameters:
                - n_repeats: Number of repeats for permutation importance
                - random_state: Random state for permutation importance
            
        Returns:
            Dictionary containing explanation data
        """
        if self.model is None:
            raise ValueError("Model not set. Call set_model() first.")
        
        # Convert pandas DataFrame to numpy array if needed
        if isinstance(X, pd.DataFrame):
            if self.feature_names is None:
                self.feature_names = X.columns.tolist()
            X_values = X.values
        else:
            X_values = X
        
        # Determine method to use
        method = self.method
        if method == 'auto':
            if hasattr(self.model, 'feature_importances_'):
                method = 'built_in'
            elif hasattr(self.model, 'coef_'):
                method = 'coef'
            else:
                method = 'permutation'
        
        # Calculate feature importance
        if method == 'built_in':
            if not hasattr(self.model, 'feature_importances_'):
                raise ValueError("Model does not have feature_importances_ attribute")
            importance_values = self.model.feature_importances_
        
        elif method == 'coef':
            if not hasattr(self.model, 'coef_'):
                raise ValueError("Model does not have coef_ attribute")
            
            # Handle different shapes of coefficients
            coef = self.model.coef_
            if coef.ndim > 1:
                # For multi-class models, use the mean absolute value
                importance_values = np.abs(coef).mean(axis=0)
            else:
                importance_values = np.abs(coef)
        
        elif method == 'permutation':
            if y is None:
                raise ValueError("Target values (y) are required for permutation importance")
            
            # Use scikit-learn's permutation importance
            try:
                from sklearn.inspection import permutation_importance
            except ImportError:
                raise ImportError("scikit-learn is required for permutation importance")
            
            n_repeats = kwargs.get('n_repeats', 10)
            random_state = kwargs.get('random_state', 42)
            
            result = permutation_importance(
                self.model, X_values, y, 
                n_repeats=n_repeats,
                random_state=random_state
            )
            importance_values = result.importances_mean
        
        else:
            raise ValueError(f"Unsupported method: {method}")
        
        # Create feature importance dictionary
        if self.feature_names is not None:
            feature_importance = dict(zip(self.feature_names, importance_values))
        else:
            feature_importance = dict(enumerate(importance_values))
        
        # Store explanation data
        self.last_explanation = {
            'feature_importance': feature_importance,
            'method': method,
            'raw_importance': importance_values,
            'feature_names': self.feature_names
        }
        
        return self.last_explanation
    
    def plot(self, plot_type: str = 'bar', **kwargs) -> plt.Figure:
        """
        Create feature importance visualization
        
        Args:
            plot_type: Type of plot to generate:
                - 'bar': Vertical bar plot of feature importance
                - 'horizontal_bar': Horizontal bar plot of feature importance
                - 'heatmap': Heatmap of feature importance
            **kwargs: Additional parameters:
                - top_n: Number of top features to display
                - figsize: Figure size as (width, height) tuple
                - title: Plot title
                - color: Bar color or colormap for heatmap
            
        Returns:
            Matplotlib figure object
        """
        if self.last_explanation is None:
            raise ValueError("No explanation available. Run explain() first.")
        
        # Extract parameters
        top_n = kwargs.get('top_n', 20)
        figsize = kwargs.get('figsize', (10, 6))
        title = kwargs.get('title', f"Feature Importance ({self.last_explanation['method']})")
        color = kwargs.get('color', 'skyblue')
        
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
        
        # Create figure
        fig, ax = plt.subplots(figsize=figsize)
        
        # Generate appropriate plot based on plot_type
        if plot_type == 'bar':
            ax.bar(feature_names, importance_values, color=color)
            ax.set_xlabel('Features')
            ax.set_ylabel('Importance')
            ax.set_title(title)
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
        
        elif plot_type == 'horizontal_bar':
            # Reverse order for horizontal bar plot to have highest at the top
            feature_names.reverse()
            importance_values.reverse()
            
            ax.barh(feature_names, importance_values, color=color)
            ax.set_xlabel('Importance')
            ax.set_ylabel('Features')
            ax.set_title(title)
            plt.tight_layout()
        
        elif plot_type == 'heatmap':
            try:
                import seaborn as sns
            except ImportError:
                raise ImportError("seaborn is required for heatmap plot")
            
            # Create DataFrame for heatmap
            importance_df = pd.DataFrame({
                'Feature': feature_names,
                'Importance': importance_values
            })
            
            # Reshape for heatmap
            heatmap_data = importance_df.set_index('Feature')
            
            # Create heatmap
            sns.heatmap(
                heatmap_data.T, 
                annot=True, 
                cmap=color if isinstance(color, str) else 'viridis',
                ax=ax
            )
            ax.set_title(title)
            plt.tight_layout()
        
        else:
            raise ValueError(f"Unsupported plot type: {plot_type}. Supported types: {self.supported_plot_types}")
        
        return fig
