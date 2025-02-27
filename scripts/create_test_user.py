#!/usr/bin/env python3
"""
Script to create a test user for accessing the dashboard.
"""
import sys
import os
from pathlib import Path
import logging
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

# Add the project root to the Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.core.models.database import User
from app.core.services.auth import AuthService
from app.core.config.settings import Settings

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def create_test_user(username="testuser", password="testpassword", email="test@example.com"):
    """
    Create a test user if it doesn't exist.
    
    Args:
        username: Username for the test user
        password: Password for the test user
        email: Email for the test user
        
    Returns:
        True if user creation is successful or user already exists, False otherwise
    """
    try:
        # Set up database connection
        settings = Settings()
        engine = create_engine(settings.DATABASE_URL or "sqlite:///./test.db")
        SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
        db = SessionLocal()
        
        # Check if user already exists
        auth_service = AuthService(db)
        existing_user = db.query(User).filter(User.username == username).first()
        
        if existing_user:
            logger.info(f"User {username} already exists")
            return True
            
        # Create user
        try:
            new_user = auth_service.create_user(
                email=email,
                username=username,
                password=password
            )
            logger.info(f"Created user {username} with ID {new_user.id}")
            return True
        except ValueError as e:
            logger.warning(f"Could not create user: {str(e)}")
            return False
        
    except Exception as e:
        logger.error(f"Error creating test user: {str(e)}")
        return False
        
    finally:
        db.close()

def main():
    """Create a test user with default credentials."""
    success = create_test_user()
    
    if success:
        logger.info("Test user creation successful")
        logger.info("You can now log in with username 'testuser' and password 'testpassword'")
    else:
        logger.error("Test user creation failed")

if __name__ == "__main__":
    main()
