"""
Application configuration.
"""
from typing import Any, Dict, Optional
from pydantic_settings import BaseSettings
from pydantic import PostgresDsn, Field
import secrets
import os

class Settings(BaseSettings):
    """Application settings."""
    PROJECT_NAME: str = "Migraine Prediction Service"
    VERSION: str = "0.1.0"
    API_V1_STR: str = "/api"
    
    # Security
    SECRET_KEY: str = Field(default_factory=lambda: secrets.token_urlsafe(32))
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    
    # Database
    POSTGRES_SERVER: str = "localhost"
    POSTGRES_USER: str = "postgres"
    POSTGRES_PASSWORD: str = ""
    POSTGRES_DB: str = "app"
    SQLALCHEMY_DATABASE_URI: str = "sqlite:///./test.db"

    class Config:
        """Pydantic config."""
        case_sensitive = True
        env_file = ".env"

# Create settings instance
settings = Settings()
