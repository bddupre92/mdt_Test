"""
Authentication middleware.
"""
from fastapi import Request, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.responses import Response
from typing import Optional

from app.core.auth.jwt import verify_token

class AuthMiddleware(BaseHTTPMiddleware):
    """Authentication middleware."""
    def __init__(self, app, exclude_paths: Optional[list] = None):
        super().__init__(app)
        self.exclude_paths = exclude_paths or []
        self.security = HTTPBearer(auto_error=False)  # Make bearer token optional

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        """Process request through middleware."""
        # Skip authentication for excluded paths
        if any(request.url.path.startswith(path) for path in self.exclude_paths):
            return await call_next(request)

        try:
            # Get token from header
            auth = await self.security(request)
            if not auth:
                # For development, allow requests without auth
                return await call_next(request)

            # Verify token if provided
            token = auth.credentials
            if not verify_token(token):
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Invalid token"
                )
            
            return await call_next(request)
            
        except HTTPException as e:
            raise e
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail=str(e)
            )
