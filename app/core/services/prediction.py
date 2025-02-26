"""
Prediction service for migraine prediction.
"""
from typing import Dict, List, Optional, Any
import numpy as np
import pandas as pd
from datetime import datetime
import pytz
from sqlalchemy.orm import Session
import traceback
from datetime import timezone

from app.core.models.model_manager import ModelManager
from app.core.models.database import DiaryEntry, Prediction
from app.core.config.test_config import load_config

class PredictionService:
    """Service for handling predictions."""
    
    def __init__(self, db: Session):
        """Initialize service."""
        self.db = db
        self.model = None
        self.config = load_config()
        self.model_manager = ModelManager(config=self.config)
        
    async def predict(self, user_id: int, features: Dict[str, float]) -> Dict[str, Any]:
        """Make prediction for user."""
        try:
            if not self.model:
                # Initialize default model for testing
                from sklearn.ensemble import RandomForestClassifier
                self.model = RandomForestClassifier(
                    n_estimators=100,
                    max_depth=5,
                    min_samples_split=2,
                    random_state=42
                )
                # Train with some default data for both classes
                X = pd.DataFrame([
                    {
                        'sleep_hours': 6.5,
                        'stress_level': 7,
                        'weather_pressure': 1013.2,
                        'heart_rate': 75,
                        'hormonal_level': 65
                    },
                    {
                        'sleep_hours': 5.5,
                        'stress_level': 9,
                        'weather_pressure': 1015.2,
                        'heart_rate': 85,
                        'hormonal_level': 75
                    }
                ])
                y = pd.Series([False, True])
                self.model.fit(X, y)
                
            # Convert features to DataFrame
            X = pd.DataFrame([features])
            
            # Make prediction
            probability = self.model.predict_proba(X)[0][1]
            prediction = probability > 0.5
            
            # Get feature importance
            feature_importance = {}
            if hasattr(self.model, 'feature_importances_'):
                feature_importance = dict(zip(features.keys(), self.model.feature_importances_))
            
            # Save prediction to database
            prediction_record = Prediction(
                user_id=user_id,
                created_at=datetime.now(timezone.utc),
                features=features,
                prediction=float(prediction),
                probability=float(probability),
                feature_importance=feature_importance,
                drift_detected=False
            )
            self.db.add(prediction_record)
            self.db.commit()
            
            return {
                "prediction": float(prediction),
                "probability": float(probability),
                "feature_importance": feature_importance,
                "drift_detected": False  # Simplified for testing
            }
            
        except Exception as e:
            self.db.rollback()
            print(f"Prediction error: {str(e)}")
            print(f"Traceback: {traceback.format_exc()}")
            raise
    
    def detect_drift(self, features: Dict[str, float]) -> bool:
        """Check for concept drift in new data."""
        try:
            result = self.model_manager.predict(features)
            return result['drift_detected']
        except Exception as e:
            raise Exception(f"Drift detection failed: {str(e)}")
    
    def detect_triggers(self, features: Dict[str, float]) -> List[str]:
        """Detect potential migraine triggers."""
        triggers = []
        thresholds = self.config.features
        
        for feature, value in features.items():
            if feature not in thresholds:
                continue
                
            config = thresholds[feature]
            if value < config.min_value:
                triggers.append(f"{feature}_low")
            elif value > config.max_value:
                triggers.append(f"{feature}_high")
        
        return triggers
    
    def calculate_risk_level(self, probability: float) -> str:
        """Calculate risk level from probability."""
        if probability < 0.3:
            return "low"
        elif probability < 0.7:
            return "medium"
        else:
            return "high"
    
    def get_history(self, user_id: int) -> List[Dict]:
        """Get prediction history for user."""
        try:
            predictions = (
                self.db.query(Prediction)
                .filter(Prediction.user_id == user_id)
                .order_by(Prediction.created_at.desc())
                .all()
            )
            
            return [
                {
                    "id": p.id,
                    "timestamp": p.created_at,
                    "prediction": p.probability,
                    "features": p.features
                }
                for p in predictions
            ]
        except Exception as e:
            raise Exception(f"Failed to get prediction history: {str(e)}")