"""Prediction service for migraine risk assessment."""

from datetime import datetime, timedelta
import pandas as pd
from sqlalchemy.orm import Session
from typing import Dict, Any, List, Optional

from app.core.models.database import User, DiaryEntry, Prediction

class PredictionService:
    """Service for managing migraine predictions."""
    
    def __init__(self, db: Session):
        """Initialize the prediction service."""
        self.db = db
        self.model = None  # Will be set in tests
    
    def predict_risk(self, user_id: int, features: Dict[str, Any]) -> Dict[str, Any]:
        """Predict migraine risk based on provided features."""
        if self.model is None:
            # Return mock prediction if no model is available
            return {
                "risk_level": 0.4,
                "confidence": 0.8,
                "prediction_time": datetime.now() + timedelta(hours=24),
                "trigger_factors": []
            }
            
        # Convert features to DataFrame for prediction
        features_df = pd.DataFrame([features])
        
        # Make prediction
        risk_level = self.model.predict_proba(features_df)[0][1]
        
        # Extract potential triggers
        trigger_importance = {}
        if hasattr(self.model, 'feature_importances_'):
            for feature, importance in zip(features.keys(), self.model.feature_importances_):
                trigger_importance[feature] = float(importance)
        
        # Sort triggers by importance
        trigger_factors = [
            {"factor": factor, "importance": importance}
            for factor, importance in sorted(
                trigger_importance.items(),
                key=lambda x: x[1],
                reverse=True
            )[:3]  # Top 3 triggers
        ]
        
        prediction_result = {
            "risk_level": float(risk_level),
            "confidence": 0.8,  # Mock confidence value
            "prediction_time": datetime.now() + timedelta(hours=24),
            "trigger_factors": trigger_factors
        }
        
        # Save prediction to database
        self._save_prediction(user_id, prediction_result, features)
        
        return prediction_result
    
    def _save_prediction(self, user_id: int, prediction: Dict[str, Any], features: Dict[str, Any]) -> None:
        """Save prediction to database."""
        db_prediction = Prediction(
            user_id=user_id,
            timestamp=datetime.now(),
            prediction_time=prediction["prediction_time"],
            risk_level=prediction["risk_level"],
            confidence=prediction["confidence"],
            features=features,
            trigger_factors=prediction["trigger_factors"]
        )
        self.db.add(db_prediction)
        self.db.commit()
        self.db.refresh(db_prediction)
        
    def get_user_predictions(self, user_id: int, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent predictions for a user."""
        predictions = self.db.query(Prediction).filter(
            Prediction.user_id == user_id
        ).order_by(Prediction.timestamp.desc()).limit(limit).all()
        
        return [
            {
                "id": p.id,
                "timestamp": p.timestamp,
                "prediction_time": p.prediction_time,
                "risk_level": p.risk_level,
                "confidence": p.confidence,
                "trigger_factors": p.trigger_factors
            }
            for p in predictions
        ] 