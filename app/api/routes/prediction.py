"""
API routes for the migraine prediction service.
"""
from fastapi import APIRouter, Depends, HTTPException, status
from typing import List, Dict, Any
from datetime import datetime, timezone
from sqlalchemy.orm import Session

from app.core.database import get_db
from app.core.models.database import User, DiaryEntry, Prediction
from app.core.schemas.diary import DiaryEntryCreate, DiaryEntryResponse
from app.core.schemas.prediction import PredictionRequest, PredictionResponse, PredictionHistoryResponse
from app.core.services.prediction import PredictionService
from app.api.dependencies import get_current_user

router = APIRouter()

@router.post("/predict", response_model=PredictionResponse)
async def predict(
    request: PredictionRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
) -> PredictionResponse:
    """Generate migraine prediction for user."""
    try:
        prediction_service = PredictionService(db)
        result = await prediction_service.predict(
            user_id=current_user.id,
            features=request.model_dump(exclude={'additional_features'})
        )
        return PredictionResponse(
            prediction=result["prediction"],
            probability=result.get("probability"),
            feature_importance=result.get("feature_importance"),
            drift_detected=result.get("drift_detected", False)
        )
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
    """Create new diary entry."""
    try:
        db_entry = DiaryEntry(
            user_id=current_user.id,
            created_at=datetime.now(timezone.utc),
            **entry.model_dump()
        )
        db.add(db_entry)
        db.commit()
        db.refresh(db_entry)
        return DiaryEntryResponse.model_validate(db_entry)
    except Exception as e:
        db.rollback()
        print(f"Failed to create diary entry: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to create diary entry"
        )

@router.get("/diary/{entry_id}", response_model=DiaryEntryResponse)
async def get_diary_entry(
    entry_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
) -> DiaryEntryResponse:
    """Get diary entry by ID."""
    entry = db.query(DiaryEntry).filter(
        DiaryEntry.id == entry_id,
        DiaryEntry.user_id == current_user.id
    ).first()
    if not entry:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Diary entry not found"
        )
    return DiaryEntryResponse.model_validate(entry)

@router.get("/predictions", response_model=List[PredictionHistoryResponse])
async def get_prediction_history(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
) -> List[PredictionHistoryResponse]:
    """Get prediction history for user."""
    predictions = db.query(Prediction).filter(
        Prediction.user_id == current_user.id
    ).order_by(Prediction.created_at.desc()).all()
    
    return [
        PredictionHistoryResponse(
            id=p.id,
            timestamp=p.created_at,
            prediction=p.prediction,
            actual=p.actual,
            probability=p.probability,
            feature_importance=p.feature_importance
        ) for p in predictions
    ]