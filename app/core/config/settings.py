"""
Application settings.
"""
from typing import Optional
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    """Application settings."""
    
    PROJECT_NAME: str = "Migraine Prediction Service"
    VERSION: str = "0.1.0"
    API_V1_STR: str = "/api/v1"
    
    # Database
    DATABASE_URL: Optional[str] = "sqlite:///./app.db"
    
    # JWT
    SECRET_KEY: str = "your-secret-key"  # Change in production
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    
    # CORS
    BACKEND_CORS_ORIGINS: list[str] = ["*"]
    
    # Model settings
    MODEL_PATH: str = "models/migraine_predictor.joblib"
    FEATURE_IMPORTANCE_THRESHOLD: float = 0.05
    
    # Meta-optimization
    META_OPTIMIZER_HISTORY_SIZE: int = 100
    META_OPTIMIZER_BATCH_SIZE: int = 10
    META_OPTIMIZER_LEARNING_RATE: float = 0.01
    
    # Drift detection
    DRIFT_DETECTION_WINDOW: int = 100
    DRIFT_WARNING_THRESHOLD: float = 0.05
    DRIFT_ALERT_THRESHOLD: float = 0.1
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = True

settings = Settings()