from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from app.models.user import User

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

async def get_current_user(token: str = Depends(oauth2_scheme)) -> User:
    """Simplified auth for demo purposes."""
    # For demo, just return a mock user
    return User(
        id=1,
        username="demo_user",
        email="demo@example.com"
    )
