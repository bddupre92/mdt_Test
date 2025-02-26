"""
Database model for diary entries.
"""
from datetime import datetime
from sqlalchemy import Column, Integer, Float, Boolean, DateTime, ForeignKey, JSON
from sqlalchemy.orm import relationship

from app.core.models.database import Base

class DiaryEntry(Base):
    __tablename__ = "diary_entries"
    __table_args__ = {'extend_existing': True}

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    sleep_hours = Column(Float, nullable=False)
    stress_level = Column(Integer, nullable=False)
    weather_pressure = Column(Float, nullable=False)
    heart_rate = Column(Integer, nullable=False)
    hormonal_level = Column(Float, nullable=False)
    migraine_occurred = Column(Boolean, nullable=False)
    triggers = Column(JSON)

    # Relationships
    user = relationship("User", back_populates="diary_entries")

    @classmethod
    def get_by_id(cls, entry_id: int) -> "DiaryEntry":
        """Get diary entry by ID."""
        from app.core.models.database import db_session
        return db_session.query(cls).filter(cls.id == entry_id).first()

    def save(self):
        """Save diary entry to database."""
        from app.core.models.database import db_session
        db_session.add(self)
        db_session.commit()
