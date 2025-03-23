"""Authentication handler for JWT tokens."""

import jwt
from datetime import datetime, timedelta

class AuthHandler:
    """Authentication handler for JWT tokens."""
    
    def __init__(self):
        self.secret_key = "test_secret_key"
        
    def create_access_token(self, data: dict, expires_delta: timedelta = None):
        """Create JWT access token."""
        to_encode = data.copy()
        if expires_delta:
            expire = datetime.utcnow() + expires_delta
        else:
            expire = datetime.utcnow() + timedelta(minutes=15)
        to_encode.update({"exp": expire})
        encoded_jwt = jwt.encode(to_encode, self.secret_key, algorithm="HS256")
        return encoded_jwt
        
    def verify_token(self, token: str):
        """Verify JWT token."""
        try:
            payload = jwt.decode(token, self.secret_key, algorithms=["HS256"])
            return payload
        except jwt.PyJWTError:
            return None

auth_handler = AuthHandler() 