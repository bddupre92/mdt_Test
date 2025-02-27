"""
Rate limiting middleware for the MDT API.
"""
from datetime import datetime, timedelta
from typing import Dict, Tuple, Optional
import asyncio
from fastapi import Request, HTTPException, status
from pydantic import BaseModel

class RateLimitConfig(BaseModel):
    """Rate limit configuration."""
    requests: int
    window_seconds: int
    
class RateLimit:
    def __init__(self, requests: int, window: int):
        self.requests = requests
        self.window = window
        self.tokens = requests
        self.last_update = datetime.utcnow()

class RateLimiter:
    def __init__(self):
        """Initialize rate limiter with default configs."""
        self.rate_limits: Dict[str, Dict[str, RateLimit]] = {}
        self.lock = asyncio.Lock()
        
        # Default rate limit configurations
        self.default_limits = {
            "prediction": RateLimitConfig(requests=100, window_seconds=60),  # 100 requests per minute
            "training": RateLimitConfig(requests=10, window_seconds=3600),   # 10 requests per hour
            "default": RateLimitConfig(requests=1000, window_seconds=3600)   # 1000 requests per hour
        }
        
    def _get_client_identifier(self, request: Request) -> str:
        """Get unique client identifier from request."""
        # Prioritize API key if present
        api_key = request.headers.get("X-API-Key")
        if api_key:
            return f"api_key:{api_key}"
            
        # Fall back to IP address
        forwarded = request.headers.get("X-Forwarded-For")
        if forwarded:
            return f"ip:{forwarded.split(',')[0]}"
        return f"ip:{request.client.host}"
        
    def _get_endpoint_type(self, request: Request) -> str:
        """Determine endpoint type from request path."""
        path = request.url.path.lower()
        if "/predict" in path:
            return "prediction"
        if "/train" in path:
            return "training"
        return "default"
        
    async def is_rate_limited(self, request: Request) -> Tuple[bool, Optional[str]]:
        """Check if request should be rate limited."""
        client_id = self._get_client_identifier(request)
        endpoint_type = self._get_endpoint_type(request)
        
        async with self.lock:
            # Initialize rate limits for client if not exists
            if client_id not in self.rate_limits:
                self.rate_limits[client_id] = {}
                
            if endpoint_type not in self.rate_limits[client_id]:
                config = self.default_limits[endpoint_type]
                self.rate_limits[client_id][endpoint_type] = RateLimit(
                    config.requests,
                    config.window_seconds
                )
                
            limit = self.rate_limits[client_id][endpoint_type]
            now = datetime.utcnow()
            time_passed = (now - limit.last_update).total_seconds()
            
            # Replenish tokens based on time passed
            tokens_to_add = int(time_passed * (limit.requests / limit.window))
            limit.tokens = min(limit.requests, limit.tokens + tokens_to_add)
            limit.last_update = now
            
            if limit.tokens > 0:
                limit.tokens -= 1
                return False, None
                
            # Calculate time until next token is available
            next_token_time = limit.window / limit.requests
            retry_after = int(next_token_time)
            
            return True, str(retry_after)
            
    async def rate_limit_middleware(self, request: Request):
        """Middleware to enforce rate limiting."""
        is_limited, retry_after = await self.is_rate_limited(request)
        
        if is_limited:
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail="Rate limit exceeded",
                headers={"Retry-After": retry_after}
            )

rate_limiter = RateLimiter()
