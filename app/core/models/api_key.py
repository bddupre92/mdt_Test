"""
API key model.
"""
from datetime import datetime
from typing import List
from sqlalchemy import Column, Integer, String, DateTime, Boolean, JSON
from app.core.db import Base

class APIKey(Base):
    """API key model."""
    __tablename__ = "api_keys"
    
    id = Column(Integer, primary_key=True, index=True)
    key = Column(String, unique=True, index=True, nullable=False)
    name = Column(String, nullable=False)
    scopes = Column(JSON, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    expires_at = Column(DateTime, nullable=True)
    is_active = Column(Boolean, default=True)
    created_by = Column(String, nullable=False)
    last_used_at = Column(DateTime, nullable=True)
    use_count = Column(Integer, default=0)
