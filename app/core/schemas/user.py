"""
User schemas.
"""
from typing import Optional
from pydantic import BaseModel, Field, EmailStr

class UserBase(BaseModel):
    """Base user schema."""
    email: EmailStr = Field(..., description="User email")
    username: str = Field(..., description="Username")
    is_active: bool = Field(default=True, description="Whether the user is active")

class UserCreate(UserBase):
    """Schema for creating a user."""
    password: str = Field(..., description="User password", min_length=8)

class UserUpdate(BaseModel):
    """Schema for updating a user."""
    email: Optional[EmailStr] = Field(None, description="User email")
    username: Optional[str] = Field(None, description="Username")
    password: Optional[str] = Field(None, description="User password", min_length=8)

class UserInDB(UserBase):
    """Schema for user in database."""
    id: int = Field(..., description="User ID")
    hashed_password: str = Field(..., description="Hashed password")

class UserResponse(UserBase):
    """Schema for user response."""
    id: int = Field(..., description="User ID")
