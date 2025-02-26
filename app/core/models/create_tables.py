"""
Create database tables.
"""
from sqlalchemy import create_engine
from database import Base

def create_tables():
    """Create all database tables."""
    SQLALCHEMY_DATABASE_URL = "sqlite:///./test.db"
    engine = create_engine(SQLALCHEMY_DATABASE_URL)
    Base.metadata.create_all(bind=engine)

if __name__ == "__main__":
    create_tables()
