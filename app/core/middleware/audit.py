"""
Audit logging middleware for the MDT API.
"""
import json
import logging
from datetime import datetime
from typing import Dict, Any
from fastapi import Request
import uuid

class AuditLogger:
    def __init__(self):
        """Initialize audit logger."""
        self.logger = logging.getLogger("audit")
        self.logger.setLevel(logging.INFO)
        
        # Add file handler for audit logs
        handler = logging.FileHandler("audit.log")
        formatter = logging.Formatter(
            '{"timestamp": "%(asctime)s", "level": "%(levelname)s", '
            '"request_id": "%(request_id)s", "message": %(message)s}'
        )
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
        
    async def log_request(self, request: Request) -> str:
        """Log incoming request details."""
        request_id = str(uuid.uuid4())
        
        # Extract basic request info
        log_data = {
            "request_id": request_id,
            "method": request.method,
            "url": str(request.url),
            "client_ip": request.client.host,
            "user_agent": request.headers.get("user-agent"),
        }
        
        # Add authentication info if present
        auth_type = None
        if "Authorization" in request.headers:
            auth_type = "bearer"
        elif "X-API-Key" in request.headers:
            auth_type = "api_key"
        if auth_type:
            log_data["auth_type"] = auth_type
        
        # Log request
        self.logger.info(
            json.dumps(log_data),
            extra={"request_id": request_id}
        )
        
        return request_id
        
    async def log_response(
        self,
        request_id: str,
        status_code: int,
        processing_time: float
    ) -> None:
        """Log response details."""
        log_data = {
            "status_code": status_code,
            "processing_time_ms": round(processing_time * 1000, 2)
        }
        
        self.logger.info(
            json.dumps(log_data),
            extra={"request_id": request_id}
        )
        
    async def log_error(
        self,
        request_id: str,
        error: Exception,
        status_code: int
    ) -> None:
        """Log error details."""
        log_data = {
            "error_type": error.__class__.__name__,
            "error_message": str(error),
            "status_code": status_code
        }
        
        self.logger.error(
            json.dumps(log_data),
            extra={"request_id": request_id}
        )
        
    async def audit_middleware(self, request: Request, call_next):
        """Middleware to audit requests and responses."""
        start_time = datetime.utcnow()
        request_id = await self.log_request(request)
        
        try:
            response = await call_next(request)
            
            # Log response
            processing_time = (datetime.utcnow() - start_time).total_seconds()
            await self.log_response(
                request_id,
                response.status_code,
                processing_time
            )
            
            return response
            
        except Exception as e:
            # Log error
            await self.log_error(request_id, e, 500)
            raise

audit_logger = AuditLogger()
