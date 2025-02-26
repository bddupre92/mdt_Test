"""
Database model for prediction history.
"""
from datetime import datetime
from sqlalchemy import Column, Integer, Float, JSON, DateTime, ForeignKey, Boolean
from sqlalchemy.orm import relationship

from app.core.models.database import Base

class Prediction(Base):
    __tablename__ = "predictions"
    __table_args__ = {'extend_existing': True}

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    probability = Column(Float, nullable=False)
    prediction = Column(Float, nullable=False)  # Changed to Float for probability
    features = Column(JSON, nullable=False)
    feature_importance = Column(JSON)
    drift_detected = Column(Boolean, default=False)

    # Relationships
    user = relationship("User", back_populates="predictions")

    @classmethod
    def get_by_user_id(cls, user_id: int) -> list["Prediction"]:
        """Get prediction history for user."""
        from app.core.models.database import db_session
        return (
            db_session.query(cls)
            .filter(cls.user_id == user_id)
            .order_by(cls.created_at.desc())
            .all()
        )

    def save(self):
        """Save prediction to database."""
        from app.core.models.database import db_session
        db_session.add(self)
        db_session.commit()
