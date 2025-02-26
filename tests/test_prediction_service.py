"""
Tests for prediction service.
"""
import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

from app.core.services.prediction import PredictionService
from app.core.models.database import DiaryEntry, Prediction
from app.core.config.test_config import FeatureConfig

@pytest.mark.asyncio
async def test_predict(prediction_service, test_user, test_features):
    """Test prediction generation."""
    # Train model first
    data = []
    for i in range(100):
        features = {
            "sleep_hours": np.random.normal(7, 1),
            "stress_level": np.random.normal(5, 2),
            "weather_pressure": np.random.normal(1013, 5),
            "heart_rate": np.random.normal(70, 10),
            "hormonal_level": np.random.normal(50, 15)
        }
        target = 1 if features["stress_level"] > 7 or features["sleep_hours"] < 6 else 0
        features["target"] = target
        data.append(features)
    
    df = pd.DataFrame(data)
    X = df.drop("target", axis=1)
    y = df["target"]
    
    prediction_service.model_manager.train(X, y)
    
    # Make prediction
    result = await prediction_service.predict(test_user.id, test_features)
    
    # Check result structure
    assert "probability" in result
    assert "prediction" in result
    assert "drift_detected" in result
    assert "feature_importance" in result
    
    # Validate probability
    assert 0 <= result["probability"] <= 1
    
    # Validate prediction
    assert isinstance(result["prediction"], bool)
    
    # Check database entry
    db = prediction_service.db
    prediction = db.query(Prediction)\
        .filter(Prediction.user_id == test_user.id)\
        .order_by(Prediction.created_at.desc())\
        .first()
    assert prediction is not None
    assert isinstance(prediction.probability, float)
    assert isinstance(prediction.prediction, bool)
    assert isinstance(prediction.drift_detected, bool)

@pytest.mark.asyncio
async def test_drift_detection(prediction_service, test_user):
    """Test concept drift detection."""
    # Train model first
    data = []
    for i in range(100):
        features = {
            "sleep_hours": np.random.normal(7, 1),
            "stress_level": np.random.normal(5, 2),
            "weather_pressure": np.random.normal(1013, 5),
            "heart_rate": np.random.normal(70, 10),
            "hormonal_level": np.random.normal(50, 15)
        }
        target = 1 if features["stress_level"] > 7 or features["sleep_hours"] < 6 else 0
        features["target"] = target
        data.append(features)
    
    df = pd.DataFrame(data)
    X = df.drop("target", axis=1)
    y = df["target"]
    
    prediction_service.model_manager.train(X, y)
    
    # Generate data with drift
    dates = pd.date_range(start=datetime.now(), periods=60, freq='D')
    entries = []
    
    for i, date in enumerate(dates):
        # Introduce drift after 30 days
        stress_level = 5 if i < 30 else 8
        
        entry = DiaryEntry(
            user_id=test_user.id,
            created_at=date,
            sleep_hours=7.0,
            stress_level=stress_level,
            weather_pressure=1013.0,
            heart_rate=70,
            hormonal_level=50,
            migraine_occurred=(stress_level > 7)
        )
        entries.append(entry)
    
    # Add entries to database
    db = prediction_service.db
    for entry in entries:
        db.add(entry)
    db.commit()
    
    # Make prediction with latest data
    features = {
        "sleep_hours": 7.0,
        "stress_level": 8,
        "weather_pressure": 1013.0,
        "heart_rate": 70,
        "hormonal_level": 50
    }
    
    # Check drift detection
    drift_detected = prediction_service.detect_drift(features)
    assert isinstance(drift_detected, bool)

@pytest.mark.asyncio
async def test_trigger_detection(prediction_service, test_user, test_features):
    """Test trigger detection and ranking."""
    # Modify features to trigger warnings
    test_features["sleep_hours"] = 5.0  # Low sleep
    test_features["stress_level"] = 9    # High stress
    
    # Update config thresholds
    prediction_service.config.features = {
        "sleep_hours": FeatureConfig(
            mean=7.0,
            std=1.0,
            min_value=6.0,
            max_value=10.0,
            missing_rate=0.0,
            drift_susceptible=True
        ),
        "stress_level": FeatureConfig(
            mean=5.0,
            std=2.0,
            min_value=0.0,
            max_value=8.0,
            missing_rate=0.0,
            drift_susceptible=True
        ),
        "weather_pressure": FeatureConfig(
            mean=1013.0,
            std=5.0,
            min_value=950.0,
            max_value=1050.0,
            missing_rate=0.0,
            drift_susceptible=True
        ),
        "heart_rate": FeatureConfig(
            mean=70.0,
            std=10.0,
            min_value=50.0,
            max_value=120.0,
            missing_rate=0.0,
            drift_susceptible=True
        ),
        "hormonal_level": FeatureConfig(
            mean=50.0,
            std=15.0,
            min_value=20.0,
            max_value=80.0,
            missing_rate=0.0,
            drift_susceptible=True
        )
    }
    
    triggers = prediction_service.detect_triggers(test_features)
    assert len(triggers) > 0
    
    # Verify trigger status
    assert "sleep_hours_low" in triggers
    assert "stress_level_high" in triggers

def test_risk_level_calculation(prediction_service):
    """Test risk level thresholds."""
    assert prediction_service.calculate_risk_level(0.2) == "low"
    assert prediction_service.calculate_risk_level(0.5) == "medium"
    assert prediction_service.calculate_risk_level(0.8) == "high"
