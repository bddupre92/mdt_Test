"""
Database initialization and migration utilities.
"""
import logging
from sqlalchemy.orm import Session

from ..models.database import Base, User
from ..config.settings import Settings
from ...api.dependencies import engine

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def init_db(db: Session) -> None:
    """Initialize database with required tables and initial data."""
    # Create tables
    Base.metadata.create_all(bind=engine)
    
    # Check if we need to create initial data
    if db.query(User).first():
        return
        
    logger.info("Creating initial data")
    
    # Create test user if in development
    settings = Settings()
    if settings.DEBUG:
        test_user = User(
            email="test@example.com",
            password_hash="$2b$12$LQv3c1yqBWVHxkd0LHAkCOYz6TtxMQJqhN8/LewKyNiGpJ4vZKkOS",  # password: test123
            name="Test User"
        )
        db.add(test_user)
        db.commit()
        logger.info("Created test user")
    
def get_db():
    """Get database session."""
    db = Session(engine)
    try:
        yield db
    finally:
        db.close()

if __name__ == "__main__":
    # Initialize database when script is run directly
    db = next(get_db())
    init_db(db)
