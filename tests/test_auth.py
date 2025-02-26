"""
test_auth.py
-----------
Tests for authentication endpoints.
"""
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from app.main import app
from app.core.models.database import User
from app.core.services.auth import AuthService
from app.core.schemas.auth import UserCreate, Token

@pytest.fixture
def auth_service(db_session: Session):
    """Create auth service fixture."""
    return AuthService(db_session)

@pytest.fixture
def test_user(db_session: Session, auth_service: AuthService):
    """Create test user fixture."""
    user = User(
        email="test@example.com",
        username="testuser",
        hashed_password=auth_service.get_password_hash("testpass123")
    )
    db_session.add(user)
    db_session.commit()
    return user

def test_register(client: TestClient, db_session: Session, auth_service: AuthService):
    """Test user registration."""
    response = client.post(
        "/api/auth/register",
        json={
            "email": "newuser@example.com",
            "username": "newuser",
            "password": "testpass123"
        }
    )
    assert response.status_code == 201
    data = response.json()
    assert data["email"] == "newuser@example.com"
    assert data["username"] == "newuser"
    assert "id" in data

def test_login(client: TestClient, test_user: User, auth_service: AuthService, db_session: Session):
    """Test user login."""
    response = client.post(
        "/api/auth/login",
        data={
            "username": "testuser",
            "password": "testpass123"
        }
    )
    assert response.status_code == 200
    data = response.json()
    assert "access_token" in data
    assert data["token_type"] == "bearer"

def test_login_wrong_password(client: TestClient, test_user: User, auth_service: AuthService, db_session: Session):
    """Test login with wrong password."""
    response = client.post(
        "/api/auth/login",
        data={
            "username": "testuser",
            "password": "wrongpass"
        }
    )
    assert response.status_code == 401
    assert response.json()["detail"] == "Incorrect username or password"

def test_register_duplicate_email(client: TestClient, test_user: User, db_session: Session, auth_service: AuthService):
    """Test registration with duplicate email."""
    response = client.post(
        "/api/auth/register",
        json={
            "email": "test@example.com",
            "username": "newuser",
            "password": "testpass123"
        }
    )
    assert response.status_code == 400
    assert "already registered" in response.json()["detail"]
