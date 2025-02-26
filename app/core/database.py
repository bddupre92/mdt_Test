"""
Database configuration.
"""
from typing import Generator
from sqlalchemy import create_engine, MetaData
from sqlalchemy.orm import declarative_base, Session, sessionmaker

from app.core.config.settings import Settings

# Create a new MetaData instance
metadata = MetaData()

# Create Base with the new MetaData
Base = declarative_base()

def get_db() -> Generator[Session, None, None]:
    """Get database session."""
    settings = Settings()
    engine = create_engine(settings.DATABASE_URL or "sqlite:///./test.db")
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
