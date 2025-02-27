"""
OAuth2 authentication handler for the MDT API.
"""
from datetime import datetime, timedelta
from typing import Dict, Any, Optional
import jwt
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from pydantic import BaseModel

from app.core.config.settings import settings

class Token(BaseModel):
    """Token model."""
    access_token: str
    token_type: str = "bearer"

class TokenData(BaseModel):
    """Token data model."""
    username: Optional[str] = None
    scopes: list[str] = []

class OAuth2Handler:
    """OAuth2 authentication handler."""
    
    def __init__(self):
        """Initialize OAuth2 handler."""
        self.secret_key = settings.SECRET_KEY
        self.algorithm = settings.ALGORITHM
        self.token_expire_minutes = settings.ACCESS_TOKEN_EXPIRE_MINUTES
        self.oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")
        
    def create_access_token(
        self,
        data: Dict[str, Any],
        expires_delta: Optional[timedelta] = None
    ) -> Token:
        """Create access token."""
        to_encode = data.copy()
        
        if expires_delta:
            expire = datetime.utcnow() + expires_delta
        else:
            expire = datetime.utcnow() + timedelta(
                minutes=self.token_expire_minutes
            )
            
        to_encode.update({"exp": expire})
        encoded_jwt = jwt.encode(
            to_encode,
            self.secret_key,
            algorithm=self.algorithm
        )
        
        return Token(access_token=encoded_jwt)
        
    async def get_current_user(
        self,
        token: str = Depends(OAuth2PasswordBearer(tokenUrl="token"))
    ) -> TokenData:
        """Get current user from token."""
        credentials_exception = HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )
        
        try:
            payload = jwt.decode(
                token,
                self.secret_key,
                algorithms=[self.algorithm]
            )
            username: str = payload.get("sub")
            if username is None:
                raise credentials_exception
                
            token_scopes = payload.get("scopes", [])
            token_data = TokenData(
                username=username,
                scopes=token_scopes
            )
            
        except jwt.PyJWTError:
            raise credentials_exception
            
        return token_data
        
    async def verify_token_scope(
        self,
        token_data: TokenData,
        required_scope: str
    ) -> bool:
        """Verify token has required scope."""
        return required_scope in token_data.scopes

oauth2_handler = OAuth2Handler()
