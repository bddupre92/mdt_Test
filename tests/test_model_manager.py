"""
Tests for model manager functionality.
"""
import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from app.core.models.model_manager import ModelManager
from app.core.config.test_config import TestConfig, load_config
from app.core.data.generators.test import TestDataGenerator

@pytest.fixture
def test_config():
    """Get test configuration."""
    return load_config()

@pytest.fixture
def test_data(test_config):
    """Generate test data."""
    generator = TestDataGenerator(seed=42)
    return generator.generate_time_series(n_days=30)

@pytest.fixture
def model_manager(test_config):
    """Get model manager instance."""
    return ModelManager(config=test_config)

def test_model_training(model_manager, test_data):
    """Test model training process."""
    # Prepare data
    X = test_data.drop(['migraine_occurred', 'migraine_probability', 'date'], axis=1)
    y = test_data['migraine_occurred']
    
    # Train model
    results = model_manager.train(X, y)
    
    # Verify results
    assert results['status'] == 'success'
    assert len(results['selected_features']) > 0
    assert all(f in X.columns for f in results['selected_features'])
    assert all(isinstance(v, float) for v in results['feature_importance'].values())
    assert isinstance(results['best_params'], dict)

def test_prediction(model_manager, test_data):
    """Test prediction functionality."""
    # Train model first
    X = test_data.drop(['migraine_occurred', 'migraine_probability', 'date'], axis=1)
    y = test_data['migraine_occurred']
    model_manager.train(X, y)
    
    # Make prediction
    features = {col: X.iloc[0][col] for col in X.columns}
    result = model_manager.predict(features)
    
    # Verify prediction
    assert 0 <= result['probability'] <= 1
    assert isinstance(result['prediction'], bool)
    assert isinstance(result['drift_detected'], bool)
    assert isinstance(result['feature_importance'], dict)
    assert all(f in X.columns for f in result['feature_importance'].keys())

def test_drift_handling(model_manager, test_data, test_config):
    """Test drift detection and handling."""
    # Train model
    X = test_data.drop(['migraine_occurred', 'migraine_probability', 'date'], axis=1)
    y = test_data['migraine_occurred']
    model_manager.train(X, y)
    
    # Generate drift data with stronger drift
    generator = TestDataGenerator(seed=43)
    drift_data = generator.generate_time_series(
        n_days=30,
        drift_start=0,
        drift_magnitude=2.0
    )
    
    # Make predictions with drift data
    X_drift = drift_data.drop(['migraine_occurred', 'migraine_probability', 'date'], axis=1)
    predictions = []
    drift_detected = False
    
    for _, row in X_drift.iterrows():
        features = row.to_dict()
        result = model_manager.predict(features)
        predictions.append(result)
        if result['drift_detected']:
            drift_detected = True
            break
    
    # Verify drift was detected
    assert drift_detected, "Drift should have been detected"

def test_model_persistence(model_manager, test_data, test_config, tmp_path):
    """Test model saving and loading."""
    # Train model
    X = test_data.drop(['migraine_occurred', 'migraine_probability', 'date'], axis=1)
    y = test_data['migraine_occurred']
    model_manager.train(X, y)
    
    # Save model
    model_path = tmp_path / "model.joblib"
    model_manager.save_model(str(model_path))
    
    # Create new manager and load model
    new_manager = ModelManager(config=test_config)
    new_manager.load_model(str(model_path))
    
    # Verify loaded model works
    features = {col: X.iloc[0][col] for col in X.columns}
    result = new_manager.predict(features)
    
    assert 0 <= result['probability'] <= 1
    assert new_manager.feature_names == model_manager.feature_names
    assert new_manager.feature_importance == model_manager.feature_importance

def test_feature_importance_calculation(model_manager, test_data):
    """Test feature importance calculation."""
    # Train model
    X = test_data.drop(['migraine_occurred', 'migraine_probability', 'date'], axis=1)
    y = test_data['migraine_occurred']
    model_manager.train(X, y)
    
    # Get feature importance for a prediction
    features = {col: X.iloc[0][col] for col in X.columns}
    result = model_manager.predict(features)
    
    # Verify importance values
    importance = result['feature_importance']
    assert len(importance) == len(model_manager.feature_names)
    assert all(isinstance(v, float) for v in importance.values())
    assert all(v >= 0 for v in importance.values())
