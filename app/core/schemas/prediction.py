"""
Prediction schemas.
"""
from typing import Dict, Optional, Any
from pydantic import BaseModel, Field, ConfigDict
from datetime import datetime

class PredictionRequest(BaseModel):
    """Schema for prediction request."""
    sleep_hours: float = Field(..., description="Hours of sleep", ge=0, le=24)
    stress_level: int = Field(..., description="Stress level (1-10)", ge=1, le=10)
    weather_pressure: float = Field(..., description="Atmospheric pressure (hPa)")
    heart_rate: int = Field(..., description="Heart rate (bpm)", ge=30, le=220)
    hormonal_level: float = Field(..., description="Hormonal level", ge=0, le=100)
    additional_features: Optional[Dict[str, Any]] = Field(default=None, description="Additional features")

class PredictionResponse(BaseModel):
    """Schema for prediction response."""
    prediction: float = Field(..., description="Predicted probability of migraine")
    probability: Optional[float] = Field(None, description="Prediction probability")
    feature_importance: Optional[Dict[str, float]] = Field(None, description="Feature importance scores")
    drift_detected: Optional[bool] = Field(None, description="Whether concept drift was detected")

class PredictionHistoryResponse(BaseModel):
    """Schema for prediction history response."""
    model_config = ConfigDict(from_attributes=True)
    
    id: int = Field(..., description="Prediction ID")
    timestamp: datetime = Field(..., description="Prediction timestamp")
    prediction: bool = Field(..., description="Predicted class")
    actual: Optional[bool] = Field(None, description="Actual class (if available)")
    probability: float = Field(..., description="Prediction probability")
    feature_importance: Optional[Dict[str, float]] = Field(None, description="Feature importance scores")
