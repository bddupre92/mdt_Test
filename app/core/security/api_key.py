"""
API key management.
"""
from datetime import datetime, timedelta
from typing import List, Optional
from uuid import uuid4
from fastapi import Depends, HTTPException, Security, status
from fastapi.security import APIKeyHeader
from sqlalchemy.orm import Session

from app.core.db import get_db
from app.core.models.api_key import APIKey
from app.core.config.settings import settings

class APIKeyManager:
    """API key management."""
    
    def __init__(self):
        """Initialize API key manager."""
        self.api_key_header = APIKeyHeader(
            name=settings.API_KEY_HEADER,
            auto_error=True
        )
    
    async def create_key(
        self,
        created_by: str,
        scopes: List[str],
        expires_in_days: Optional[int] = None,
        db: Session = Depends(get_db)
    ) -> APIKey:
        """Create new API key."""
        key = str(uuid4())
        expires_at = None
        
        if expires_in_days:
            expires_at = datetime.utcnow() + timedelta(days=expires_in_days)
            
        api_key = APIKey(
            key=key,
            created_by=created_by,
            scopes=scopes,
            expires_at=expires_at,
            is_active=True
        )
        
        db.add(api_key)
        db.commit()
        db.refresh(api_key)
        
        return api_key
    
    async def verify_key(
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
    
    async def revoke_key(
        self,
        api_key: str,
        db: Session = Depends(get_db)
    ) -> None:
        """Revoke API key."""
        db_key = db.query(APIKey).filter(APIKey.key == api_key).first()
        
        if not db_key:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="API key not found"
            )
            
        db_key.is_active = False
        db_key.revoked_at = datetime.utcnow()
        db.commit()

api_key_manager = APIKeyManager()
