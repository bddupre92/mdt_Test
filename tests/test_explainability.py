"""
Tests for the explainability framework
"""

import pytest
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from pathlib import Path
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import explainability framework
from explainability.explainer_factory import ExplainerFactory
from explainability.base_explainer import BaseExplainer
from explainability.shap_explainer import ShapExplainer
from explainability.lime_explainer import LimeExplainer
from explainability.feature_importance_explainer import FeatureImportanceExplainer

@pytest.fixture
def model_and_data():
    """Create model and data for testing"""
    # Load dataset
    diabetes = load_diabetes()
    X = diabetes.data
    y = diabetes.target
    feature_names = diabetes.feature_names
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Train model
    model = RandomForestRegressor(n_estimators=10, random_state=42)
    model.fit(X_train, y_train)
    
    return {
        'model': model,
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test,
        'feature_names': feature_names
    }

def test_explainer_factory():
    """Test ExplainerFactory"""
    # Test get_available_explainers
    explainers = ExplainerFactory.get_available_explainers()
    assert 'shap' in explainers
    assert 'lime' in explainers
    assert 'feature_importance' in explainers
    
    # Test get_explainer_info
    explainer_info = ExplainerFactory.get_explainer_info()
    assert 'shap' in explainer_info
    assert 'lime' in explainer_info
    assert 'feature_importance' in explainer_info
    
    # Test create_explainer
    model = RandomForestRegressor()
    
    shap_explainer = ExplainerFactory.create_explainer('shap', model)
    assert isinstance(shap_explainer, ShapExplainer)
    
    lime_explainer = ExplainerFactory.create_explainer('lime', model)
    assert isinstance(lime_explainer, LimeExplainer)
    
    fi_explainer = ExplainerFactory.create_explainer('feature_importance', model)
    assert isinstance(fi_explainer, FeatureImportanceExplainer)
    
    # Test invalid explainer type
    with pytest.raises(ValueError):
        ExplainerFactory.create_explainer('invalid', model)

def test_feature_importance_explainer(model_and_data):
    """Test FeatureImportanceExplainer"""
    # Create explainer
    explainer = FeatureImportanceExplainer(
        model=model_and_data['model'],
        feature_names=model_and_data['feature_names']
    )
    
    # Test explain
    explanation = explainer.explain(model_and_data['X_test'])
    
    # Check explanation results
    assert 'feature_importance' in explanation
    assert 'method' in explanation
    assert 'raw_importance' in explanation
    assert 'feature_names' in explanation
    
    # Test feature importance
    feature_importance = explainer.get_feature_importance()
    assert len(feature_importance) == len(model_and_data['feature_names'])
    
    # Test plotting
    for plot_type in explainer.get_supported_plot_types():
        fig = explainer.plot(plot_type=plot_type)
        assert fig is not None
        plt.close(fig)

@pytest.mark.skipif(not pytest.importorskip("shap", reason="shap not installed"), reason="shap not installed")
def test_shap_explainer(model_and_data):
    """Test ShapExplainer"""
    # Create explainer
    explainer = ShapExplainer(
        model=model_and_data['model'],
        feature_names=model_and_data['feature_names'],
        explainer_type='tree'  # Use tree explainer for faster tests
    )
    
    # Test explain with small sample
    sample_size = 5
    explanation = explainer.explain(
        model_and_data['X_test'][:sample_size], 
        sample_size=sample_size
    )
    
    # Check explanation results
    assert 'shap_values' in explanation
    assert 'feature_importance' in explanation
    
    # Test feature importance
    feature_importance = explainer.get_feature_importance()
    assert len(feature_importance) == len(model_and_data['feature_names'])
    
    # Test plotting (only test summary plot for speed)
    fig = explainer.plot(plot_type='summary')
    assert fig is not None
    plt.close(fig)

@pytest.mark.skipif(not pytest.importorskip("lime", reason="lime not installed"), reason="lime not installed")
def test_lime_explainer(model_and_data):
    """Test LimeExplainer"""
    # Create explainer
    explainer = LimeExplainer(
        model=model_and_data['model'],
        feature_names=model_and_data['feature_names'],
        mode='regression'
    )
    
    # Test explain with small sample and single instance
    explanation = explainer.explain(
        model_and_data['X_test'][:5],
        instances=[0]
    )
    
    # Check explanation results
    assert 'explanations' in explanation
    assert 'feature_importance' in explanation
    
    # Test feature importance
    feature_importance = explainer.get_feature_importance()
    assert len(feature_importance) > 0
    
    # Test plotting
    fig = explainer.plot(plot_type='local')
    assert fig is not None
    plt.close(fig)

def test_save_load_explanation(model_and_data, tmp_path):
    """Test saving and loading explanations"""
    # Create explainer
    explainer = FeatureImportanceExplainer(
        model=model_and_data['model'],
        feature_names=model_and_data['feature_names']
    )
    
    # Generate explanation
    explainer.explain(model_and_data['X_test'])
    
    # Save explanation
    save_path = tmp_path / "explanation.json"
    explainer.save_explanation(save_path)
    
    # Create new explainer
    new_explainer = FeatureImportanceExplainer(
        model=model_and_data['model'],
        feature_names=model_and_data['feature_names']
    )
    
    # Load explanation
    new_explainer.load_explanation(save_path)
    
    # Check that explanation was loaded
    assert new_explainer.last_explanation is not None
    assert 'feature_importance' in new_explainer.last_explanation
    
    # Compare feature importance
    orig_importance = explainer.get_feature_importance()
    loaded_importance = new_explainer.get_feature_importance()
    
    assert orig_importance.keys() == loaded_importance.keys()
    for feature in orig_importance:
        assert orig_importance[feature] == loaded_importance[feature]
