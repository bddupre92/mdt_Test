"""Database models for the application."""

from sqlalchemy import Column, Integer, String, Float, Boolean, DateTime, ForeignKey, JSON
from sqlalchemy.orm import relationship
from datetime import datetime

from app.core.database import Base

class User(Base):
    """User model."""
    
    __tablename__ = "users"
    
    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True, index=True)
    email = Column(String, unique=True, index=True)
    hashed_password = Column(String)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    diary_entries = relationship("DiaryEntry", back_populates="user")
    predictions = relationship("Prediction", back_populates="user")

class DiaryEntry(Base):
    """Diary entry model for tracking migraine episodes and triggers."""
    
    __tablename__ = "diary_entries"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    timestamp = Column(DateTime, default=datetime.utcnow)
    migraine_severity = Column(Integer, nullable=True)
    sleep_hours = Column(Float, nullable=True)
    stress_level = Column(Integer, nullable=True)
    weather_data = Column(JSON, nullable=True)
    triggers = Column(JSON, nullable=True)
    notes = Column(String, nullable=True)
    
    user = relationship("User", back_populates="diary_entries")

class Prediction(Base):
    """Prediction model for storing migraine risk predictions."""
    
    __tablename__ = "predictions"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    timestamp = Column(DateTime, default=datetime.utcnow)
    prediction_time = Column(DateTime)
    risk_level = Column(Float)
    confidence = Column(Float)
    features = Column(JSON)
    trigger_factors = Column(JSON, nullable=True)
    
    user = relationship("User", back_populates="predictions") 