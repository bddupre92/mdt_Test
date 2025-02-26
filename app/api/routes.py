"""
API routes for the migraine prediction service.
"""
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import HTTPBearer
from typing import List, Dict, Any
from datetime import datetime
from sqlalchemy.orm import Session
import traceback

from app.core.auth.jwt import get_current_user
from app.core.models.database import User, DiaryEntry, get_db
from app.core.schemas.diary import DiaryEntryCreate, DiaryEntryResponse
from app.core.schemas.prediction import PredictionRequest, PredictionResponse, PredictionHistoryResponse
from app.core.services.prediction import PredictionService

router = APIRouter()
security = HTTPBearer(auto_error=True)

@router.post("/predict")
async def predict(
    request: PredictionRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
) -> Dict[str, float]:
    """Generate migraine prediction for user."""
    try:
        prediction_service = PredictionService(db)
        result = await prediction_service.predict(
            user_id=current_user.id,
            features=request.dict(exclude={'additional_features'})
        )
        return {
            "prediction": result["prediction"]
        }
    except ValueError as ve:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(ve)
        )
    except Exception as e:
        print(f"Prediction error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to generate prediction"
        )

@router.post("/diary", response_model=DiaryEntryResponse)
async def create_diary_entry(
    entry: DiaryEntryCreate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
) -> DiaryEntryResponse:
    """Create a new diary entry."""
    try:
        # Validate entry data
        if entry.sleep_hours < 0 or entry.sleep_hours > 24:
            raise ValueError("Sleep hours must be between 0 and 24")
        if entry.stress_level < 0 or entry.stress_level > 10:
            raise ValueError("Stress level must be between 0 and 10")
        if entry.heart_rate < 30 or entry.heart_rate > 200:
            raise ValueError("Heart rate must be between 30 and 200")
        if entry.hormonal_level < 0 or entry.hormonal_level > 100:
            raise ValueError("Hormonal level must be between 0 and 100")
            
        # Create diary entry
        diary_entry = DiaryEntry(
            user_id=current_user.id,
            created_at=datetime.utcnow(),
            sleep_hours=entry.sleep_hours,
            stress_level=entry.stress_level,
            weather_pressure=entry.weather_pressure,
            heart_rate=entry.heart_rate,
            hormonal_level=entry.hormonal_level,
            migraine_occurred=entry.migraine_occurred
        )
        
        # Add to database
        db.add(diary_entry)
        db.commit()
        db.refresh(diary_entry)
        
        # Convert to response model
        return DiaryEntryResponse(
            id=diary_entry.id,
            user_id=diary_entry.user_id,
            created_at=diary_entry.created_at,
            sleep_hours=diary_entry.sleep_hours,
            stress_level=diary_entry.stress_level,
            weather_pressure=diary_entry.weather_pressure,
            heart_rate=diary_entry.heart_rate,
            hormonal_level=diary_entry.hormonal_level,
            migraine_occurred=diary_entry.migraine_occurred
        )
    except ValueError as ve:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(ve)
        )
    except Exception as e:
        db.rollback()
        print(f"Error creating diary entry: {str(e)}")
        print(f"Traceback: {traceback.format_exc()}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to create diary entry"
        )

@router.get("/predictions", response_model=List[PredictionHistoryResponse])
async def get_prediction_history(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
) -> List[PredictionHistoryResponse]:
    """Get prediction history for user."""
    try:
        prediction_service = PredictionService(db)
        history = prediction_service.get_history(current_user.id)
        return [
            PredictionHistoryResponse(
                id=h["id"],
                timestamp=h["timestamp"],
                prediction=h["prediction"],
                actual=None,
                features=h["features"]
            )
            for h in history
        ]
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )

@router.get("/diary/{entry_id}", response_model=DiaryEntryResponse)
async def get_diary_entry(
    entry_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
) -> DiaryEntryResponse:
    """Get a specific diary entry."""
    entry = db.query(DiaryEntry).filter(
        DiaryEntry.id == entry_id,
        DiaryEntry.user_id == current_user.id
    ).first()
    
    if not entry:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Entry not found"
        )
    return DiaryEntryResponse.from_orm(entry)
