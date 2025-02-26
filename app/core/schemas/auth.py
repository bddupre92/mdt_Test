"""
Authentication schemas.
"""
from pydantic import BaseModel, EmailStr

class Token(BaseModel):
    """Token schema."""
    access_token: str
    token_type: str

class TokenData(BaseModel):
    """Token data schema."""
    username: str | None = None

class UserBase(BaseModel):
    """Base user schema."""
    email: EmailStr
    username: str

class UserCreate(UserBase):
    """User creation schema."""
    password: str

class UserResponse(UserBase):
    """User response schema."""
    id: int
    is_active: bool

    class Config:
        """Pydantic config."""
        from_attributes = True
