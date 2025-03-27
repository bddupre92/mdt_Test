"""
explainer_factory.py
------------------
Factory for creating explainers
"""

from typing import Dict, Any, List, Optional, Union
from .base_explainer import BaseExplainer
from .shap_explainer import ShapExplainer
from .lime_explainer import LimeExplainer
from .feature_importance_explainer import FeatureImportanceExplainer
from .optimizer_explainer import OptimizerExplainer
from .counterfactual_explainer import CounterfactualExplainer

class ExplainerFactory:
    """Factory for creating explainers"""
    
    @staticmethod
    def create_explainer(explainer_type: str, model=None, feature_names: Optional[List[str]] = None, 
                        **kwargs) -> BaseExplainer:
        """
        Create an explainer of the specified type
        
        Args:
            explainer_type: Type of explainer to create ('shap', 'lime', 'feature_importance', 'optimizer', 'counterfactual')
            model: Pre-trained model to explain
            feature_names: List of feature names
            **kwargs: Additional parameters for specific explainer
            
        Returns:
            Explainer instance
        """
        explainer_type = explainer_type.lower()
        
        if explainer_type == 'shap':
            return ShapExplainer(model, feature_names, **kwargs)
        elif explainer_type == 'lime':
            return LimeExplainer(model, feature_names, **kwargs)
        elif explainer_type == 'feature_importance':
            return FeatureImportanceExplainer(model, feature_names, **kwargs)
        elif explainer_type == 'optimizer':
            return OptimizerExplainer(model, feature_names, **kwargs)
        elif explainer_type == 'counterfactual':
            return CounterfactualExplainer(model, feature_names, **kwargs)
        else:
            raise ValueError(f"Unsupported explainer type: {explainer_type}")
    
    @staticmethod
    def get_available_explainers() -> List[str]:
        """
        Get list of available explainer types
        
        Returns:
            List of explainer type names
        """
        return ['shap', 'lime', 'feature_importance', 'optimizer', 'counterfactual']
    
    @staticmethod
    def get_explainer_info() -> Dict[str, Dict[str, Any]]:
        """
        Get information about available explainers
        
        Returns:
            Dictionary mapping explainer names to information about them
        """
        return {
            'shap': {
                'name': 'SHAP',
                'description': 'SHapley Additive exPlanations for global and local explanations',
                'dependencies': ['shap'],
                'supported_plot_types': [
                    'summary', 'bar', 'beeswarm', 'waterfall', 'force', 
                    'decision', 'dependence', 'interaction'
                ],
                'best_for': 'Both global and local explanations with strong theoretical foundation'
            },
            'lime': {
                'name': 'LIME',
                'description': 'Local Interpretable Model-agnostic Explanations for local explanations',
                'dependencies': ['lime'],
                'supported_plot_types': ['local', 'all_local', 'summary'],
                'best_for': 'Local explanations for individual predictions'
            },
            'feature_importance': {
                'name': 'Feature Importance',
                'description': 'Simple feature importance based on model attributes or permutation',
                'dependencies': ['scikit-learn'],
                'supported_plot_types': ['bar', 'horizontal_bar', 'heatmap'],
                'best_for': 'Global explanations with minimal dependencies'
            },
            'optimizer': {
                'name': 'Optimizer Explainer',
                'description': 'Explainability tools for optimization algorithms',
                'dependencies': ['matplotlib', 'seaborn'],
                'supported_plot_types': [
                    'convergence', 'parameter_adaptation', 'diversity', 
                    'landscape_analysis', 'decision_process', 'exploration_exploitation',
                    'gradient_estimation', 'performance_profile'
                ],
                'best_for': 'Understanding optimizer behavior and decision-making process'
            }
        }
