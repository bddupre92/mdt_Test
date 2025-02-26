"""
Initialize database.
"""
from sqlalchemy import create_engine
from app.core.models.database import Base
from app.core.config.settings import Settings

settings = Settings()

def init_db() -> None:
    """Initialize database."""
    database_url = settings.DATABASE_URL or "sqlite:///./test.db"
    engine = create_engine(database_url)
    Base.metadata.create_all(bind=engine)

if __name__ == "__main__":
    init_db()
