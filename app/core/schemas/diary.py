"""
Diary entry schemas.
"""
from typing import Dict, Optional, Any
from pydantic import BaseModel, Field, ConfigDict
from datetime import datetime

class DiaryEntryCreate(BaseModel):
    """Schema for creating a diary entry."""
    sleep_hours: float = Field(..., description="Hours of sleep", ge=0, le=24)
    stress_level: int = Field(..., description="Stress level (1-10)", ge=1, le=10)
    weather_pressure: float = Field(..., description="Atmospheric pressure (hPa)")
    heart_rate: int = Field(..., description="Heart rate (bpm)", ge=30, le=220)
    hormonal_level: float = Field(..., description="Hormonal level", ge=0, le=100)
    migraine_occurred: bool = Field(..., description="Whether a migraine occurred")
    triggers: Optional[Dict[str, Any]] = Field(default=None, description="Identified triggers")

class DiaryEntryResponse(BaseModel):
    """Schema for diary entry response."""
    model_config = ConfigDict(from_attributes=True)
    
    id: int = Field(..., description="Entry ID")
    user_id: int = Field(..., description="User ID")
    created_at: datetime = Field(..., description="Entry creation timestamp")
    sleep_hours: float = Field(..., description="Hours of sleep")
    stress_level: int = Field(..., description="Stress level")
    weather_pressure: float = Field(..., description="Atmospheric pressure")
    heart_rate: int = Field(..., description="Heart rate")
    hormonal_level: float = Field(..., description="Hormonal level")
    migraine_occurred: bool = Field(..., description="Whether a migraine occurred")
    triggers: Optional[Dict[str, Any]] = Field(None, description="Identified triggers")
