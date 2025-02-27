"""
Request validation middleware for the MDT API.
"""
from typing import Dict, Any, Optional
from fastapi import Request, HTTPException, status
import json
from pydantic import ValidationError, BaseModel

class RequestValidator:
    def __init__(self):
        """Initialize request validator with default rules."""
        self.max_payload_size = 10 * 1024 * 1024  # 10MB
        self.allowed_content_types = [
            "application/json",
            "multipart/form-data",
            "application/x-www-form-urlencoded"
        ]
        
    async def validate_content_type(self, request: Request) -> None:
        """Validate request content type."""
        content_type = request.headers.get("content-type", "").lower()
        if not any(allowed in content_type for allowed in self.allowed_content_types):
            raise HTTPException(
                status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
                detail=f"Content type must be one of: {', '.join(self.allowed_content_types)}"
            )
            
    async def validate_payload_size(self, request: Request) -> None:
        """Validate request payload size."""
        content_length = request.headers.get("content-length")
        if content_length and int(content_length) > self.max_payload_size:
            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail=f"Payload size exceeds maximum allowed size of {self.max_payload_size} bytes"
            )
            
    async def validate_json_format(self, body: Dict[str, Any]) -> None:
        """Validate JSON format and structure."""
        try:
            # Ensure it can be serialized
            json.dumps(body)
        except (TypeError, ValueError) as e:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid JSON format: {str(e)}"
            )
            
    async def validate_feature_values(self, data: Dict[str, Any]) -> None:
        """Validate feature values for prediction requests."""
        if "features" not in data:
            return
            
        features = data["features"]
        for feature, value in features.items():
            # Validate numeric features
            if feature in ["temperature", "pressure", "stress_level", "sleep_hours"]:
                if not isinstance(value, (int, float)):
                    raise HTTPException(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        detail=f"Feature '{feature}' must be numeric"
                    )
                    
            # Validate range constraints
            if feature == "sleep_hours" and (value < 0 or value > 24):
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="sleep_hours must be between 0 and 24"
                )
            elif feature == "stress_level" and (value < 0 or value > 10):
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="stress_level must be between 0 and 10"
                )
                
    async def sanitize_input(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Sanitize input data."""
        sanitized = {}
        for key, value in data.items():
            # Remove any HTML tags
            if isinstance(value, str):
                from html import escape
                sanitized[key] = escape(value)
            # Recursively sanitize nested dictionaries
            elif isinstance(value, dict):
                sanitized[key] = await self.sanitize_input(value)
            # Handle lists
            elif isinstance(value, list):
                sanitized[key] = [
                    await self.sanitize_input(item) if isinstance(item, dict)
                    else escape(item) if isinstance(item, str)
                    else item
                    for item in value
                ]
            else:
                sanitized[key] = value
        return sanitized
        
    async def validation_middleware(self, request: Request):
        """Middleware to validate incoming requests."""
        # Validate content type
        await self.validate_content_type(request)
        
        # Validate payload size
        await self.validate_payload_size(request)
        
        # For POST/PUT requests, validate body
        if request.method in ["POST", "PUT"]:
            try:
                body = await request.json()
                
                # Validate JSON format
                await self.validate_json_format(body)
                
                # Validate feature values for prediction requests
                if "/predict" in request.url.path:
                    await self.validate_feature_values(body)
                    
                # Sanitize input
                sanitized_body = await self.sanitize_input(body)
                
                # Update request with sanitized body
                setattr(request, "state", {"validated_data": sanitized_body})
                
            except json.JSONDecodeError:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Invalid JSON format"
                )
            except ValidationError as e:
                raise HTTPException(
                    status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                    detail=str(e)
                )

request_validator = RequestValidator()
