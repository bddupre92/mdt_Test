"""
Configuration for testing and data generation.
"""
from typing import Dict, List, Optional
from pydantic import BaseModel, Field
from datetime import datetime

class FeatureConfig(BaseModel):
    """Configuration for a single feature."""
    mean: float
    std: float
    min_value: float
    max_value: float
    missing_rate: float = 0.0
    drift_susceptible: bool = True

class DriftConfig(BaseModel):
    """Configuration for concept drift simulation."""
    enabled: bool = True
    start_day: Optional[int] = None
    magnitude: float = 0.5
    affected_features: List[str] = []
    pattern: str = "gradual"  # gradual, sudden, recurring

class ModelConfig(BaseModel):
    """Configuration for model parameters."""
    max_depth: Optional[int] = 10
    min_samples_split: int = 2
    n_estimators: int = 100
    random_state: int = 42

class TrainingConfig(BaseModel):
    """Configuration for model training."""
    validation_split: float = 0.2
    cross_validation_folds: int = 5
    performance_threshold: float = 0.7

class DriftDetectionConfig(BaseModel):
    """Configuration for drift detection."""
    window_size: int = 10  # Reduced from 30 to 10
    significance_level: float = 0.1  # Increased from 0.05 to 0.1

class TestConfig(BaseModel):
    """Main configuration for testing environment."""
    
    # Feature configurations
    features: Dict[str, FeatureConfig] = {
        "sleep_hours": FeatureConfig(
            mean=7.0,
            std=1.0,
            min_value=4.0,
            max_value=10.0,
            missing_rate=0.1,
            drift_susceptible=True
        ),
        "stress_level": FeatureConfig(
            mean=5.0,
            std=2.0,
            min_value=1.0,
            max_value=10.0,
            missing_rate=0.15,
            drift_susceptible=True
        ),
        "weather_pressure": FeatureConfig(
            mean=1013.0,
            std=5.0,
            min_value=980.0,
            max_value=1050.0,
            missing_rate=0.05,
            drift_susceptible=True
        ),
        "heart_rate": FeatureConfig(
            mean=75.0,
            std=8.0,
            min_value=50.0,
            max_value=100.0,
            missing_rate=0.2,
            drift_susceptible=False
        ),
        "hormonal_level": FeatureConfig(
            mean=50.0,
            std=15.0,
            min_value=0.0,
            max_value=100.0,
            missing_rate=0.1,
            drift_susceptible=True
        )
    }
    
    # Drift configuration
    drift: DriftConfig = DriftConfig(
        enabled=True,
        start_day=30,
        magnitude=0.5,
        affected_features=["stress_level", "sleep_hours", "weather_pressure", "hormonal_level"],  # Added more drift-susceptible features
        pattern="gradual"
    )
    
    # Model configuration
    model_params: ModelConfig = ModelConfig()
    training_params: TrainingConfig = TrainingConfig()
    drift_params: DriftDetectionConfig = DriftDetectionConfig()
    
    # Default settings
    default_n_patients: int = 5
    default_time_range_days: int = 90
    random_seed: int = 42
    
    class Config:
        """Pydantic config."""
        validate_assignment = True

def load_config() -> TestConfig:
    """Load test configuration."""
    return TestConfig()
