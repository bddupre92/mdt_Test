"""
JWT authentication utilities (DEPRECATED).

This module is deprecated and will be removed in a future version.
Please use app.core.security.auth.auth_handler instead.
"""
import warnings
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, Union

from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from sqlalchemy.orm import Session

from app.core.database import get_db
from app.core.models.database import User
from app.core.security.auth import auth_handler

def _deprecation_warning(function_name: str) -> None:
    """Show deprecation warning."""
    warnings.warn(
        f"{function_name} is deprecated. Use auth_handler.{function_name} instead.",
        DeprecationWarning,
        stacklevel=2
    )

async def create_access_token(
    data: Dict[str, Any],
    expires_delta: Optional[timedelta] = None
) -> str:
    """
    Create JWT access token.
    
    DEPRECATED: Use auth_handler.create_access_token instead.
    """
    _deprecation_warning("create_access_token")
    return await auth_handler.create_access_token(data, expires_delta)

async def verify_token(token: str) -> Dict[str, Any]:
    """
    Verify JWT token.
    
    DEPRECATED: Use auth_handler.verify_token instead.
    """
    _deprecation_warning("verify_token")
    return await auth_handler.verify_token(token)

async def get_current_user(
    token: str = Depends(OAuth2PasswordBearer(tokenUrl="token")),
    db: Session = Depends(get_db)
) -> Dict[str, Any]:
    """
    Get current user from token.
    
    DEPRECATED: Use auth_handler.get_current_user instead.
    """
    _deprecation_warning("get_current_user")
    return await auth_handler.get_current_user(token, db)

def create_test_token(username: str = "testuser") -> str:
    """
    Create test JWT token.
    
    DEPRECATED: Use auth_handler.create_access_token instead.
    """
    _deprecation_warning("create_test_token")
    return auth_handler.create_access_token({"sub": username})
