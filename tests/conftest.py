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

from app.core.models.database import Base, User, DiaryEntry, Prediction
from app.main import app
from app.api.dependencies import get_db
from app.core.auth.jwt import get_current_user
from app.core.services.prediction import PredictionService

# Create test database
SQLALCHEMY_TEST_DATABASE_URL = "sqlite://"  # In-memory database
engine = create_engine(
    SQLALCHEMY_TEST_DATABASE_URL,
    connect_args={"check_same_thread": False},
    poolclass=StaticPool
)
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

@pytest.fixture(scope="function")
def db_session():
    """Create clean database for each test."""
    Base.metadata.create_all(bind=engine)
    db = TestingSessionLocal()
    try:
        yield db
    finally:
        db.close()
        Base.metadata.drop_all(bind=engine)

@pytest.fixture(scope="function")
def test_user(db_session):
    """Create test user."""
    from app.core.services.auth import AuthService
    auth_service = AuthService(db_session)
    user = User(
        email="test@example.com",
        username="testuser",
        hashed_password=auth_service.get_password_hash("testpass123"),
        is_active=True
    )
    db_session.add(user)
    db_session.commit()
    db_session.refresh(user)
    return user

@pytest.fixture(scope="function")
def client(db_session, test_user):
    """Create FastAPI test client."""
    def override_get_db():
        try:
            yield db_session
        finally:
            pass

    def override_get_current_user():
        return test_user

    app.dependency_overrides[get_db] = override_get_db
    app.dependency_overrides[get_current_user] = override_get_current_user
    
    yield TestClient(app)
    
    # Clear overrides after test
    app.dependency_overrides = {}

@pytest.fixture(scope="function")
def test_features():
    """Sample features for testing predictions."""
    return {
        "sleep_hours": 7.5,
        "stress_level": 3,
        "weather_pressure": 1013.25,
        "heart_rate": 75,
        "hormonal_level": 2.5,
        "triggers": ["bright_lights", "noise"]
    }

@pytest.fixture(scope="function")
def test_model(test_features):
    """Create test model."""
    X = pd.DataFrame([test_features])
    y = pd.Series([0])  # No migraine
    model = RandomForestClassifier(n_estimators=10, random_state=42)
    model.fit(X, y)
    return model

@pytest.fixture(scope="function")
def prediction_service(db_session, test_model):
    """Create prediction service instance."""
    service = PredictionService(db_session)
    service.model = test_model
    return service

@pytest.fixture(scope="function")
def sample_patient_data():
    """Sample patient data for testing."""
    return pd.DataFrame({
        "sleep_hours": [7.5, 6.0, 8.0],
        "stress_level": [3, 4, 2],
        "weather_pressure": [1013.25, 1012.0, 1014.5],
        "heart_rate": [75, 80, 70],
        "hormonal_level": [2.5, 3.0, 2.0],
        "migraine_occurred": [0, 1, 0]
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
