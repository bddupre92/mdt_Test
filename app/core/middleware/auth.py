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
        self.security = HTTPBearer()

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        """Process request through middleware."""
        # Skip authentication for excluded paths
        if any(request.url.path.startswith(path) for path in self.exclude_paths):
            return await call_next(request)

        try:
            # Get token from header
            auth = await self.security(request)
            if not auth:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Not authenticated",
                    headers={"WWW-Authenticate": "Bearer"},
                )

            # Verify token
            token_data = verify_token(auth.credentials)
            if not token_data:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Invalid token",
                    headers={"WWW-Authenticate": "Bearer"},
                )

            # Add user data to request state
            request.state.user = token_data
            return await call_next(request)

        except HTTPException as e:
            return Response(
                status_code=status.HTTP_401_UNAUTHORIZED,
                content=str(e.detail),
                headers={"WWW-Authenticate": "Bearer"},
            )
