#!/usr/bin/env python3
"""
Script to initialize the database and create tables.
"""
import sys
import os
from pathlib import Path
import logging
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

# Add the project root to the Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.core.models.database import Base
from app.core.config.settings import Settings

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def init_db():
    """
    Initialize the database and create tables.
    
    Returns:
        True if database initialization is successful, False otherwise
    """
    try:
        # Set up database connection
        settings = Settings()
        engine = create_engine(settings.DATABASE_URL or "sqlite:///./test.db")
        
        # Create all tables
        Base.metadata.create_all(bind=engine)
        
        logger.info("Database tables created successfully")
        return True
        
    except Exception as e:
        logger.error(f"Error initializing database: {str(e)}")
        return False

def main():
    """Initialize the database."""
    success = init_db()
    
    if success:
        logger.info("Database initialization successful")
    else:
        logger.error("Database initialization failed")

if __name__ == "__main__":
    main()
