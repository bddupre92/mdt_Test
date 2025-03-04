"""
API routes.
"""
from fastapi import APIRouter

from .prediction import router as prediction_router
from .auth import router as auth_router

router = APIRouter()

# Include routers
router.include_router(auth_router, prefix="/auth", tags=["auth"])
router.include_router(prediction_router, tags=["prediction"])