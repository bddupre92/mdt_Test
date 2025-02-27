"""
Test configuration and fixtures.
"""
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool
from datetime import datetime, timedelta
from tqdm import tqdm
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from app.core.database import Base, get_db
from app.core.models.database import User, DiaryEntry, Prediction
from app.core.security.auth import auth_handler
from app.main import app
from app.api.dependencies import get_current_user
from app.core.services.prediction import PredictionService

SQLALCHEMY_TEST_DATABASE_URL = "sqlite://"  # In-memory database
engine = create_engine(
    SQLALCHEMY_TEST_DATABASE_URL,
    connect_args={"check_same_thread": False},
    poolclass=StaticPool
)
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def override_get_db():
    """Override database session for testing."""
    try:
        db = TestingSessionLocal()
        yield db
    finally:
        db.close()

app.dependency_overrides[get_db] = override_get_db

@pytest.fixture
def client():
    """Create test client."""
    Base.metadata.create_all(bind=engine)
    yield TestClient(app)
    Base.metadata.drop_all(bind=engine)

@pytest.fixture
def test_user(client):
    """Create test user."""
    user_data = {
        "username": "testuser",
        "email": "test@example.com",
        "password": "testpass123"
    }
    db = TestingSessionLocal()
    user = User(
        username=user_data["username"],
        email=user_data["email"],
        hashed_password="hashed_" + user_data["password"]
    )
    db.add(user)
    db.commit()
    db.refresh(user)
    db.close()
    return user

@pytest.fixture
def test_token(test_user):
    """Create test token."""
    return auth_handler.create_access_token({"sub": test_user.username})

@pytest.fixture
def test_features():
    """Sample features for testing predictions."""
    return {
        "sleep_hours": 6.5,
        "stress_level": 7,
        "weather_pressure": 1013.2,
        "heart_rate": 75,
        "hormonal_level": 65,
        "triggers": ["bright_lights", "noise"]
    }

@pytest.fixture
def test_model(test_features):
    """Create test model."""
    X = pd.DataFrame([test_features])
    y = pd.Series([1])
    model = RandomForestClassifier(n_estimators=10, random_state=42)
    model.fit(X, y)
    return model

@pytest.fixture
def prediction_service(db_session, test_model):
    """Create prediction service instance."""
    service = PredictionService(db_session)
    service.model = test_model
    return service

@pytest.fixture
def sample_patient_data():
    """Sample patient data for testing."""
    return pd.DataFrame({
        "sleep_hours": [6.5, 7.0, 8.0],
        "stress_level": [7, 5, 3],
        "weather_pressure": [1013.2, 1012.8, 1014.5],
        "heart_rate": [75, 72, 68],
        "hormonal_level": [65, 60, 55],
        "migraine_occurred": [1, 0, 0]
    })

"""Pytest configuration and custom plugins."""
import pytest
from tqdm import tqdm
import sys
import datetime

class TestProgress:
    def __init__(self):
        self.current = 0
        self.total = 0
        self.bar = None

    def start(self, total):
        self.total = total
        self.current = 0
        self.bar = tqdm(
            total=total,
            desc="Running tests",
            bar_format='{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} tests',
            file=sys.stdout,
            leave=True
        )

    def advance(self):
        if self.bar:
            self.current += 1
            self.bar.update(1)

    def finish(self):
        if self.bar:
            self.bar.close()
            self.bar = None

test_progress = TestProgress()

def pytest_collection_modifyitems(session, config, items):
    """Start progress tracking."""
    if sys.stdout.isatty():
        test_progress.start(len(items))

def pytest_runtest_logfinish(nodeid, location):
    """Update progress after each test."""
    if sys.stdout.isatty():
        test_progress.advance()

def pytest_sessionfinish(session, exitstatus):
    """Clean up progress bar."""
    if sys.stdout.isatty():
        test_progress.finish()
