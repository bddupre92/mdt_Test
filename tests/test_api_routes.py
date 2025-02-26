"""
Test suite for API routes.
"""
from datetime import datetime, timedelta
import pytest
from fastapi import status
from fastapi.testclient import TestClient
from app.main import app
from app.core.auth.jwt import create_access_token

def test_predict_endpoint(client, test_user, test_features):
    """Test prediction endpoint."""
    response = client.post(
        "/api/predict",
        json=test_features,
        headers={"Authorization": f"Bearer {create_access_token({'sub': test_user.username})}"}
    )
    
    assert response.status_code == status.HTTP_200_OK
    data = response.json()
    assert "prediction" in data
    assert isinstance(data["prediction"], float)
    assert 0 <= data["prediction"] <= 1

def test_create_diary_entry(client, test_user, test_features):
    """Test diary entry creation."""
    entry_data = {
        "sleep_hours": 6.5,
        "stress_level": 7,
        "weather_pressure": 1013.2,
        "heart_rate": 75,
        "hormonal_level": 65,
        "migraine_occurred": True
    }
    
    response = client.post(
        "/api/diary",
        json=entry_data,
        headers={"Authorization": f"Bearer {create_access_token({'sub': test_user.username})}"}
    )
    
    assert response.status_code == status.HTTP_200_OK
    data = response.json()
    assert "id" in data
    assert isinstance(data["id"], int)
    assert data["user_id"] == test_user.id
    
    # Test that we can get the entry back
    response = client.get(
        f"/api/diary/{data['id']}",
        headers={"Authorization": f"Bearer {create_access_token({'sub': test_user.username})}"}
    )
    
    assert response.status_code == status.HTTP_200_OK
    entry = response.json()
    assert entry["sleep_hours"] == entry_data["sleep_hours"]
    assert entry["stress_level"] == entry_data["stress_level"]
    assert entry["migraine_occurred"] == entry_data["migraine_occurred"]

def test_get_prediction_history(client, test_user):
    """Test prediction history endpoint."""
    response = client.get(
        "/api/predictions",
        headers={"Authorization": f"Bearer {create_access_token({'sub': test_user.username})}"}
    )
    
    assert response.status_code == status.HTTP_200_OK
    data = response.json()
    assert isinstance(data, list)
    
    # If there are predictions, verify their structure
    if data:
        prediction = data[0]
        assert "id" in prediction
        assert "timestamp" in prediction
        assert "prediction" in prediction
        assert "actual" in prediction

@pytest.fixture
def unauthorized_client():
    """Create unauthorized test client."""
    return TestClient(app)

def test_unauthorized_access(unauthorized_client):
    """Test unauthorized access to endpoints."""
    endpoints = [
        ("/api/predict", "POST", {"sleep_hours": 6.5, "stress_level": 7, "weather_pressure": 1013.2, "heart_rate": 75, "hormonal_level": 65}),
        ("/api/diary", "POST", {"sleep_hours": 6.5, "stress_level": 7, "weather_pressure": 1013.2, "heart_rate": 75, "hormonal_level": 65, "migraine_occurred": True}),
        ("/api/predictions", "GET", None)
    ]
    
    for endpoint, method, data in endpoints:
        if method == "GET":
            response = unauthorized_client.get(endpoint)
        else:
            response = unauthorized_client.post(endpoint, json=data)
        
        assert response.status_code == status.HTTP_401_UNAUTHORIZED
        assert response.headers.get("WWW-Authenticate") == "Bearer"
