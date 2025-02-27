"""
Unified authentication module for the MDT API.
"""
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, Union
from fastapi import Depends, HTTPException, Security, status
from fastapi.security import OAuth2PasswordBearer, APIKeyHeader
from sqlalchemy.orm import Session
import jwt

from app.core.db import get_db
from app.core.models.database import User
from app.core.models.api_key import APIKey
from app.core.config.settings import settings

class AuthHandler:
    """Unified authentication handler."""
    
    def __init__(self):
        """Initialize auth handler."""
        self.secret_key = settings.SECRET_KEY
        self.algorithm = settings.ALGORITHM
        self.token_expire_minutes = settings.ACCESS_TOKEN_EXPIRE_MINUTES
        self.oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")
        self.api_key_header = APIKeyHeader(
            name=settings.API_KEY_HEADER,
            auto_error=True
        )
        
    def create_access_token(
        self,
        data: Dict[str, Any],
        expires_delta: Optional[timedelta] = None
    ) -> str:
        """Create JWT access token."""
        to_encode = data.copy()
        
        if expires_delta:
            expire = datetime.utcnow() + expires_delta
        else:
            expire = datetime.utcnow() + timedelta(
                minutes=self.token_expire_minutes
            )
            
        to_encode.update({"exp": expire})
        return jwt.encode(
            to_encode,
            self.secret_key,
            algorithm=self.algorithm
        )
        
    async def verify_token(
        self,
        token: str = Depends(OAuth2PasswordBearer(tokenUrl="token"))
    ) -> Dict[str, Any]:
        """Verify JWT token."""
        try:
            payload = jwt.decode(
                token,
                self.secret_key,
                algorithms=[self.algorithm]
            )
            username: str = payload.get("sub")
            if username is None:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Could not validate credentials",
                    headers={"WWW-Authenticate": "Bearer"},
                )
            return payload
        except jwt.PyJWTError:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Could not validate credentials",
                headers={"WWW-Authenticate": "Bearer"},
            )
            
    async def verify_api_key(
        self,
        api_key: str = Security(APIKeyHeader(name="X-API-Key")),
        db: Session = Depends(get_db)
    ) -> APIKey:
        """Verify API key."""
        db_key = db.query(APIKey).filter(
            APIKey.key == api_key,
            APIKey.is_active == True
        ).first()
        
        if not db_key:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid API key"
            )
            
        if db_key.expires_at and db_key.expires_at < datetime.utcnow():
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="API key has expired"
            )
            
        # Update usage stats
        db_key.last_used_at = datetime.utcnow()
        db_key.use_count += 1
        db.commit()
        
        return db_key
        
    async def get_current_user(
        self,
        security_token: Union[str, None] = Security(OAuth2PasswordBearer(tokenUrl="token")),
        db: Session = Depends(get_db)
    ) -> User:
        """Get current user from either JWT token or API key."""
        try:
            # First try JWT token
            payload = await self.verify_token(security_token)
            username = payload.get("sub")
            user = db.query(User).filter(User.username == username).first()
            if not user:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="User not found",
                    headers={"WWW-Authenticate": "Bearer"},
                )
            return user
        except HTTPException:
            # If JWT fails, try API key
            try:
                api_key = await self.verify_api_key(security_token, db)
                user = db.query(User).filter(
                    User.username == api_key.created_by
                ).first()
                if not user:
                    raise HTTPException(
                        status_code=status.HTTP_401_UNAUTHORIZED,
                        detail="API key user not found",
                        headers={"WWW-Authenticate": "Bearer"},
                    )
                return user
            except HTTPException:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Invalid authentication credentials",
                    headers={"WWW-Authenticate": "Bearer"},
                )

auth_handler = AuthHandler()
