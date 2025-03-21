"""
explainability package
---------------------
Modular framework for model explainability
"""

from .base_explainer import BaseExplainer
from .explainer_factory import ExplainerFactory
from .shap_explainer import ShapExplainer
from .lime_explainer import LimeExplainer
from .feature_importance_explainer import FeatureImportanceExplainer
from .optimizer_explainer import OptimizerExplainer
from .explainer_adapter import ExplainerAdapter, adapt_explainer

__all__ = [
    'BaseExplainer',
    'ExplainerFactory',
    'ShapExplainer',
    'LimeExplainer',
    'FeatureImportanceExplainer',
    'OptimizerExplainer',
    'ExplainerAdapter',
    'adapt_explainer',
]
