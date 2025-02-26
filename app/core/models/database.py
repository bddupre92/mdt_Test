"""
Database models.
"""
from sqlalchemy import Column, Integer, String, Float, Boolean, DateTime, ForeignKey, JSON
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from datetime import datetime

from app.core.database import Base

class User(Base):
    """User model."""
    __tablename__ = "users"
    __table_args__ = {'extend_existing': True}

    id = Column(Integer, primary_key=True, index=True)
    email = Column(String, unique=True, index=True)
    username = Column(String, unique=True, index=True)
    hashed_password = Column(String, nullable=False)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    diary_entries = relationship("DiaryEntry", back_populates="user")
    predictions = relationship("Prediction", back_populates="user")

class DiaryEntry(Base):
    """Diary entry model."""
    __tablename__ = "diary_entries"
    __table_args__ = {'extend_existing': True}

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    sleep_hours = Column(Float)
    stress_level = Column(Integer)
    weather_pressure = Column(Float)
    heart_rate = Column(Integer)
    hormonal_level = Column(Float)
    migraine_occurred = Column(Boolean)
    triggers = Column(JSON)

    user = relationship("User", back_populates="diary_entries")

class Prediction(Base):
    """Prediction model."""
    __tablename__ = "predictions"
    __table_args__ = {'extend_existing': True}

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    features = Column(JSON)
    probability = Column(Float)
    prediction = Column(Boolean)
    actual = Column(Boolean, nullable=True)
    feature_importance = Column(JSON, nullable=True)
    drift_detected = Column(Boolean, default=False)

    user = relationship("User", back_populates="predictions")