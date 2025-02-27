"""
Authentication routes.
"""
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy.orm import Session

from app.core.database import get_db
from app.core.models.database import User
from app.core.security.auth import auth_handler
from app.core.services.auth import AuthService
from app.core.schemas.auth import Token, UserCreate, UserResponse

router = APIRouter()

@router.post("/login", response_model=Token)
async def login(
    form_data: OAuth2PasswordRequestForm = Depends(),
    db: Session = Depends(get_db)
) -> Token:
    """Login user and return access token."""
    auth_service = AuthService(db)
    user = auth_service.authenticate_user(form_data.username, form_data.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    access_token = auth_handler.create_access_token(
        data={"sub": user.username}
    )
    return Token(access_token=access_token, token_type="bearer")

@router.post("/register", response_model=UserResponse)
async def register(
    user_data: UserCreate,
    db: Session = Depends(get_db)
) -> UserResponse:
    """Register new user."""
    auth_service = AuthService(db)
    user = auth_service.get_user_by_username(user_data.username)
    if user:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Username already registered"
        )
    return auth_service.create_user(user_data)
