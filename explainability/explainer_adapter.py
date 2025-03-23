"""
explainer_adapter.py
-------------------
Adapter to bridge between different explainer interfaces

This adapter allows explainers from the explainability framework to
be used by components that expect a different API, like the
PersonalizationLayer which expects an explain_model method.
"""

from typing import Dict, Any, List, Optional, Union, Tuple
import numpy as np
import pandas as pd
import logging
from .base_explainer import BaseExplainer

# Set up logging
logger = logging.getLogger(__name__)

class ExplainerAdapter:
    """
    Adapter that wraps explainers and provides compatibility with
    other modules that expect different method signatures.
    """
    
    def __init__(self, explainer: BaseExplainer):
        """
        Initialize the adapter with an explainer
        
        Args:
            explainer: An instance of BaseExplainer or derived class
        """
        self.explainer = explainer
        
    def explain_model(self, model, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Adapter method that provides the explain_model interface expected
        by the PersonalizationLayer
        
        Args:
            model: The model to explain
            data: DataFrame containing the data to explain
            
        Returns:
            Dictionary containing explanation data with feature importance
        """
        try:
            # Ensure the explainer has the model set
            if self.explainer.model is None and hasattr(self.explainer, 'set_model'):
                self.explainer.set_model(model)
            elif self.explainer.model is None:
                self.explainer.model = model
                
            # Call the explainer's explain method
            explanation = self.explainer.explain(data)
            
            # Extract feature importance
            feature_importance = {}
            
            # Try different keys/attributes where feature importance might be found
            if hasattr(explanation, 'get') and explanation.get('feature_importance') is not None:
                # If explanation has feature_importance dictionary
                feature_importance = explanation['feature_importance']
            elif hasattr(self.explainer, 'get_feature_importance'):
                # Use explainer's method to get feature importance
                feature_importance = self.explainer.get_feature_importance()
            elif hasattr(model, 'feature_importances_'):
                # Fall back to model's built-in feature importances
                importances = model.feature_importances_
                if hasattr(model, 'feature_names_in_'):
                    feature_names = model.feature_names_in_
                elif self.explainer.feature_names is not None:
                    feature_names = self.explainer.feature_names
                else:
                    feature_names = [f"feature_{i}" for i in range(len(importances))]
                    
                feature_importance = dict(zip(feature_names, importances))
            else:
                # Create a simple fallback with equal importance
                feature_names = data.columns.tolist()
                equal_importance = 1.0 / len(feature_names)
                feature_importance = {name: equal_importance for name in feature_names}
            
            return {
                'feature_importance': feature_importance,
                'explanation_type': self.explainer.name,
                'explanation_details': explanation
            }
            
        except Exception as e:
            logger.warning(f"Error in explain_model adapter: {str(e)}")
            # Provide a fallback result with minimal feature importance
            feature_names = data.columns.tolist()
            equal_importance = 1.0 / len(feature_names)
            return {
                'feature_importance': {name: equal_importance for name in feature_names},
                'explanation_type': 'fallback',
                'error': str(e)
            }

def adapt_explainer(explainer: BaseExplainer) -> ExplainerAdapter:
    """
    Factory function to create an adapter for an explainer
    
    Args:
        explainer: BaseExplainer instance to adapt
        
    Returns:
        ExplainerAdapter that wraps the provided explainer
    """
    return ExplainerAdapter(explainer)
