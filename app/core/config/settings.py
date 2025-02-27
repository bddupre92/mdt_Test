"""
Application settings.
"""
from typing import Optional, List
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    """Application settings."""
    
    PROJECT_NAME: str = "Migraine Prediction Service"
    VERSION: str = "0.1.0"
    API_V1_STR: str = "/api/v1"
    
    # Database
    DATABASE_URL: Optional[str] = "sqlite:///./app.db"
    
    # Security
    SECRET_KEY: str = "your-secret-key-for-testing-only"  # Change in production
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    
    # CORS
    BACKEND_CORS_ORIGINS: List[str] = ["*"]
    
    # Model settings
    MODEL_PATH: str = "models/migraine_predictor.joblib"
    FEATURE_IMPORTANCE_THRESHOLD: float = 0.05
    
    # Meta-optimization
    META_OPTIMIZER_HISTORY_SIZE: int = 100
    META_OPTIMIZER_BATCH_SIZE: int = 10
    
    # Drift Detection
    DRIFT_WINDOW_SIZE: int = 50
    DRIFT_THRESHOLD: float = 1.8
    DRIFT_SIGNIFICANCE: float = 0.01
    MIN_DRIFT_INTERVAL: int = 40
    EMA_ALPHA: float = 0.3
    
    # Rate Limiting
    RATE_LIMIT_PREDICTION: int = 100  # requests per minute
    RATE_LIMIT_TRAINING: int = 10    # requests per hour
    
    # Monitoring
    ENABLE_METRICS: bool = True
    METRICS_PORT: int = 9090
    
    # API Security
    API_KEY_HEADER: str = "X-API-Key"
    API_KEY_PREFIX: str = "mdt_"

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = True

settings = Settings()