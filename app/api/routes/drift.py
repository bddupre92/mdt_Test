"""
API routes for drift detection and analysis.
"""
from fastapi import APIRouter, Depends, HTTPException, status
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
from sqlalchemy.orm import Session
import numpy as np
import pandas as pd
import logging

from app.core.models.database import get_db, User, Prediction
from app.core.auth.jwt import get_current_user
from app.core.data.drift import DriftDetector

logger = logging.getLogger(__name__)

router = APIRouter()

@router.get("/history")
async def get_drift_history(
    days: int = 30,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
) -> Dict[str, Any]:
    """
    Get drift detection history for visualization.
    
    Args:
        days: Number of days to look back
        
    Returns:
        Dictionary with drift history data
    """
    try:
        # Calculate date range
        end_date = datetime.utcnow()
        start_date = end_date - timedelta(days=days)
        
        # Get predictions with drift information
        predictions = db.query(Prediction).filter(
            Prediction.user_id == current_user.id,
            Prediction.created_at >= start_date,
            Prediction.created_at <= end_date
        ).order_by(Prediction.created_at).all()
        
        if not predictions:
            return {
                "timestamps": [],
                "severities": [],
                "trends": [],
                "feature_drifts": {},
                "total_drifts": 0,
                "average_severity": 0.0,
                "last_detection": None,
                "current_drift_detected": False,
                "recent_events": []
            }
        
        # Extract data for visualization
        timestamps = [p.created_at.strftime("%Y-%m-%d %H:%M") for p in predictions]
        severities = []
        drift_detected = []
        feature_drifts = {}
        recent_events = []
        
        for p in predictions:
            # Extract severity from feature_importance if available
            severity = 0.0
            if hasattr(p, "drift_severity") and p.drift_severity is not None:
                severity = p.drift_severity
            elif hasattr(p, "feature_importance") and p.feature_importance:
                # Use max feature importance as a proxy for drift severity
                severity = max(p.feature_importance.values()) if p.feature_importance else 0.0
            
            severities.append(severity)
            drift_detected.append(p.drift_detected)
            
            # Track feature drifts
            if p.drift_detected and hasattr(p, "feature_importance") and p.feature_importance:
                for feature, importance in p.feature_importance.items():
                    if importance > 0.3:  # Threshold for considering a feature as drifting
                        feature_drifts[feature] = feature_drifts.get(feature, 0) + 1
                        
                # Add to recent events if drift detected
                if p.drift_detected:
                    # Find the feature with highest importance
                    max_feature = max(p.feature_importance.items(), key=lambda x: x[1])[0] if p.feature_importance else None
                    
                    recent_events.append({
                        "timestamp": p.created_at.strftime("%Y-%m-%d %H:%M"),
                        "severity": severity,
                        "feature": max_feature,
                        "drift_type": "distribution" if severity > 0.5 else "trend"
                    })
        
        # Calculate trend using simple moving average
        window_size = min(10, len(severities))
        trends = []
        
        if window_size > 0:
            for i in range(len(severities)):
                if i < window_size - 1:
                    trends.append(0)  # Not enough data points yet
                else:
                    window = severities[i-window_size+1:i+1]
                    trends.append(np.mean(window))
        
        # Calculate summary statistics
        total_drifts = sum(drift_detected)
        average_severity = np.mean(severities) if severities else 0.0
        
        # Find last detection
        last_detection = None
        if total_drifts > 0:
            for i in range(len(drift_detected) - 1, -1, -1):
                if drift_detected[i]:
                    last_detection = timestamps[i]
                    break
        
        # Determine if drift is currently detected
        current_drift_detected = drift_detected[-1] if drift_detected else False
        
        # Limit recent events to last 10
        recent_events = sorted(recent_events, key=lambda x: x["timestamp"], reverse=True)[:10]
        
        return {
            "timestamps": timestamps,
            "severities": severities,
            "trends": trends,
            "feature_drifts": feature_drifts,
            "total_drifts": total_drifts,
            "average_severity": float(average_severity),
            "last_detection": last_detection,
            "current_drift_detected": current_drift_detected,
            "recent_events": recent_events
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve drift history: {str(e)}"
        )

@router.get("/status")
async def get_drift_status(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
) -> Dict[str, Any]:
    """
    Get current drift status.
    
    Returns:
        Dictionary with drift status information
    """
    try:
        # Get most recent prediction
        prediction = db.query(Prediction).filter(
            Prediction.user_id == current_user.id
        ).order_by(Prediction.created_at.desc()).first()
        
        if not prediction:
            return {
                "drift_detected": False,
                "severity": 0.0,
                "timestamp": None
            }
        
        # Extract severity
        severity = 0.0
        if hasattr(prediction, "drift_severity") and prediction.drift_severity is not None:
            severity = prediction.drift_severity
        elif hasattr(prediction, "feature_importance") and prediction.feature_importance:
            # Use max feature importance as a proxy for drift severity
            severity = max(prediction.feature_importance.values()) if prediction.feature_importance else 0.0
        
        return {
            "drift_detected": prediction.drift_detected,
            "severity": float(severity),
            "timestamp": prediction.created_at.strftime("%Y-%m-%d %H:%M") if prediction.created_at else None
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve drift status: {str(e)}"
        )

@router.get("/visualization")
async def get_drift_visualization(
    days: int = 30,
    window_size: int = 50,
    threshold: float = 0.01,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
) -> Dict[str, Any]:
    """
    Get drift visualization data.
    
    Args:
        days: Number of days to look back
        window_size: Window size for drift detection
        threshold: Significance level for drift detection
        
    Returns:
        Dictionary with visualization data
    """
    logger.info(f"Generating drift visualization for user {current_user.id}, days={days}, window_size={window_size}, threshold={threshold:.3e}")
    
    try:
        # Calculate date range
        end_date = datetime.utcnow()
        start_date = end_date - timedelta(days=days)
        logger.debug(f"Date range: {start_date.isoformat()} to {end_date.isoformat()}")
        
        # Get predictions
        predictions = db.query(Prediction).filter(
            Prediction.user_id == current_user.id,
            Prediction.created_at >= start_date,
            Prediction.created_at <= end_date
        ).order_by(Prediction.created_at).all()
        
        logger.debug(f"Retrieved {len(predictions)} predictions for visualization")
        
        if not predictions:
            logger.info(f"No predictions found for user {current_user.id} in the specified date range")
            return {
                "timestamps": [],
                "severities": [],
                "drift_points": [],
                "feature_drifts": {}
            }
        
        # Extract features from predictions
        features = []
        timestamps = []
        
        for pred in predictions:
            # Extract features from prediction data
            feature_dict = pred.features
            if feature_dict:
                features.append(feature_dict)
                timestamps.append(pred.created_at.timestamp())
        
        if not features:
            logger.warning(f"No feature data found in predictions for user {current_user.id}")
            return {
                "timestamps": [],
                "severities": [],
                "drift_points": [],
                "feature_drifts": {}
            }
            
        # Convert to DataFrame
        df = pd.DataFrame(features)
        logger.debug(f"Feature data shape: {df.shape}, columns: {list(df.columns)}")
        
        # Log data statistics
        numeric_stats = df.describe().transpose()
        logger.debug(f"Data statistics: mean={numeric_stats['mean'].to_dict()}, std={numeric_stats['std'].to_dict()}, min={numeric_stats['min'].to_dict()}, max={numeric_stats['max'].to_dict()}")
        
        # Run drift detection
        logger.info(f"Initializing drift detector with window_size={window_size}, significance_level={threshold:.3e}")
        detector = DriftDetector(window_size=window_size, significance_level=threshold)
        
        # Filter out non-numeric columns
        numeric_cols = df.select_dtypes(include=["number"]).columns
        df_numeric = df[numeric_cols]
        logger.debug(f"Filtered to {len(numeric_cols)} numeric columns: {list(numeric_cols)}")
        
        # Set reference window
        reference_data = df_numeric.iloc[:min(window_size, len(df_numeric))]
        logger.info(f"Initializing reference window with {len(reference_data)} samples")
        detector.initialize_reference(reference_data)
        
        # Process data and detect drift
        drift_points = []
        drift_severities = []
        feature_drifts = {col: 0 for col in numeric_cols}
        
        # Process data in sliding windows
        for i in range(window_size, len(df_numeric), max(1, window_size//4)):
            end_idx = min(i + window_size//2, len(df_numeric))
            if end_idx <= i:
                continue
                
            current_window = df_numeric.iloc[i:end_idx]
            
            if len(current_window) < 5:  # Minimum samples to detect drift
                logger.debug(f"Skipping window at index {i} with only {len(current_window)} samples (minimum 5 required)")
                continue
                
            logger.debug(f"Processing window at index {i} with {len(current_window)} samples")
            drift_results = detector.detect_drift(current_window)
            
            # Calculate overall severity as max of feature severities
            severity = 0.0
            drift_detected = False
            
            for result in drift_results:
                if result.detected:
                    drift_detected = True
                    feature_drifts[result.feature] += 1
                    logger.info(f"Drift detected in feature '{result.feature}' at index {i} with severity {result.severity:.3f}, p-value {result.p_value:.3e}, KS statistic {result.statistic:.3f}")
                    if result.severity and result.severity > severity:
                        severity = result.severity
                else:
                    logger.debug(f"No drift detected in feature '{result.feature}' at index {i}, p-value {result.p_value:.3e}, KS statistic {result.statistic:.3f}")
            
            if i < len(timestamps):
                drift_severities.append(severity)
                
                if drift_detected:
                    drift_points.append(i)
                    logger.info(f"Overall drift detected at index {i} with max severity {severity:.3f}")
        
        # Log summary statistics
        total_drifts = len(drift_points)
        avg_severity = sum(drift_severities) / len(drift_severities) if drift_severities else 0
        max_severity = max(drift_severities) if drift_severities else 0
        
        logger.info(f"Drift detection summary: total_drifts={total_drifts}, avg_severity={avg_severity:.3f}, max_severity={max_severity:.3f}")
        logger.debug(f"Feature drift counts: {feature_drifts}")
        
        return {
            "timestamps": timestamps[:len(drift_severities)],
            "severities": drift_severities,
            "drift_points": drift_points,
            "feature_drifts": feature_drifts
        }
        
    except Exception as e:
        logger.error(f"Error generating drift visualization: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error generating drift visualization: {str(e)}"
        )
