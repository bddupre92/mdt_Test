"""
Service for handling migraine predictions.
"""
from typing import Dict, Any, List
from datetime import datetime
import numpy as np
from app.core.models.prediction import PredictionHistory
from app.core.models.model_manager import ModelManager

class PredictionService:
    def __init__(self):
        """Initialize prediction service."""
        self.model_manager = ModelManager()
    
    def predict(self, features: Dict[str, Any], user_id: int) -> float:
        """Generate prediction for given features."""
        try:
            # Preprocess features
            processed_features = self._preprocess_features(features)
            
            # Get prediction
            prediction = self.model_manager.predict(processed_features)
            
            # Save prediction to history
            self._save_prediction(user_id, prediction, features)
            
            return float(prediction)
        except Exception as e:
            raise Exception(f"Prediction failed: {str(e)}")
    
    def get_history(self, user_id: int) -> List[Dict[str, Any]]:
        """Get prediction history for user."""
        try:
            history = PredictionHistory.get_by_user_id(user_id)
            return [
                {
                    "id": h.id,
                    "timestamp": h.timestamp,
                    "prediction": h.prediction,
                    "actual": h.actual,
                    "features": h.features
                }
                for h in history
            ]
        except Exception as e:
            raise Exception(f"Failed to get prediction history: {str(e)}")
    
    def _preprocess_features(self, features: Dict[str, Any]) -> np.ndarray:
        """Preprocess features for prediction."""
        required_features = [
            'sleep_hours', 'stress_level', 'weather_pressure',
            'heart_rate', 'hormonal_level'
        ]
        
        # Ensure all required features are present
        for feature in required_features:
            if feature not in features and feature != 'hormonal_level':
                raise ValueError(f"Missing required feature: {feature}")
        
        # Convert to numpy array in correct order
        feature_array = np.array([
            features['sleep_hours'],
            features['stress_level'],
            features['weather_pressure'],
            features['heart_rate'],
            features.get('hormonal_level', 0)  # Default to 0 if not provided
        ])
        
        return feature_array.reshape(1, -1)
    
    def _save_prediction(self, user_id: int, prediction: float, features: Dict[str, Any]):
        """Save prediction to history."""
        history = PredictionHistory(
            user_id=user_id,
            timestamp=datetime.utcnow(),
            prediction=float(prediction),
            features=features
        )
        history.save()
