# api/routers/prediction.py
from fastapi import APIRouter, BackgroundTasks, HTTPException, Depends, File, UploadFile
from typing import Dict, List, Any, Optional, Union
import logging
import uuid
import time
import json
import numpy as np
from datetime import datetime
from pydantic import BaseModel, Field

# Create router
router = APIRouter()

# Logger
logger = logging.getLogger(__name__)

# Models for request/response
class PhysiologicalDataPoint(BaseModel):
    timestamp: str
    ecg: Optional[List[float]] = None
    hrv: Optional[List[float]] = None
    eeg: Optional[List[float]] = None
    gsr: Optional[float] = None
    temperature: Optional[float] = None
    steps: Optional[int] = None
    sleep_quality: Optional[float] = None

class EnvironmentalDataPoint(BaseModel):
    timestamp: str
    temperature: Optional[float] = None
    humidity: Optional[float] = None
    pressure: Optional[float] = None
    light_intensity: Optional[float] = None
    noise_level: Optional[float] = None
    air_quality: Optional[float] = None

class BehavioralDataPoint(BaseModel):
    timestamp: str
    activity_type: Optional[str] = None
    stress_level: Optional[int] = None
    medication_taken: Optional[List[str]] = None
    food_consumed: Optional[List[str]] = None
    hydration_level: Optional[float] = None
    symptom_rating: Optional[int] = None

class PhysiologicalDataBatch(BaseModel):
    patient_id: str
    data_points: List[PhysiologicalDataPoint]

class EnvironmentalDataBatch(BaseModel):
    patient_id: str
    data_points: List[EnvironmentalDataPoint]

class BehavioralDataBatch(BaseModel):
    patient_id: str
    data_points: List[BehavioralDataPoint]

class PredictionRequest(BaseModel):
    physiological_data: Dict[str, List[float]]
    environmental_data: Optional[Dict[str, List[float]]] = None
    behavioral_data: Optional[Dict[str, List[float]]] = None
    timestamps: List[str]
    patient_id: Optional[str] = None

class TriggerInfo(BaseModel):
    trigger: str
    confidence: float
    severity: float

class PredictionResult(BaseModel):
    prediction_id: str
    risk_score: float
    confidence: float
    timeframe: str
    potential_triggers: List[TriggerInfo]
    recommendations: List[str]
    timestamp: str

class PatientTriggerProfile(BaseModel):
    patient_id: str
    triggers: List[Dict[str, Any]]
    last_updated: str

# In-memory storage for predictions and patient data
prediction_results = {}
patient_triggers = {}
patient_data = {}

# Physiological data processing functions
def preprocess_ecg(ecg_data: List[float], sampling_rate: int = 250) -> Dict[str, Any]:
    """
    Preprocess ECG data and extract features.
    """
    if not ecg_data or len(ecg_data) < sampling_rate * 2:  # Need at least 2 seconds of data
        return {"quality": 0, "features": {}}
    
    try:
        # Basic preprocessing (in real implementation, use specialized libraries)
        # 1. Remove baseline wander with high-pass filter
        # 2. Remove noise with low-pass filter
        # 3. Detect R-peaks 
        # 4. Calculate heart rate and HRV features
        
        # Simulate preprocessing with dummy implementation
        mean_val = np.mean(ecg_data)
        std_val = np.std(ecg_data)
        max_val = np.max(ecg_data)
        min_val = np.min(ecg_data)
        
        # Simulate R-peak detection and HRV calculation
        simulated_hr = 60 + 20 * np.random.random()  # Simulate HR between 60-80 bpm
        
        features = {
            "mean": float(mean_val),
            "std": float(std_val),
            "range": float(max_val - min_val),
            "heart_rate": float(simulated_hr),
            "sdnn": float(50 + 20 * np.random.random()),  # Simulate SDNN (HRV feature)
            "rmssd": float(30 + 15 * np.random.random()),  # Simulate RMSSD (HRV feature)
        }
        
        return {
            "quality": 0.8,  # Simulate quality assessment
            "features": features
        }
    except Exception as e:
        logger.error(f"Error preprocessing ECG data: {str(e)}")
        return {"quality": 0, "features": {}}

def preprocess_eeg(eeg_data: List[float], sampling_rate: int = 250) -> Dict[str, Any]:
    """
    Preprocess EEG data and extract features.
    """
    if not eeg_data or len(eeg_data) < sampling_rate * 2:
        return {"quality": 0, "features": {}}
    
    try:
        # Basic preprocessing (in real implementation, use specialized libraries)
        # 1. Filter the data to remove artifacts
        # 2. Extract frequency bands (delta, theta, alpha, beta, gamma)
        # 3. Calculate band powers
        
        # Simulate preprocessing with dummy implementation
        mean_val = np.mean(eeg_data)
        std_val = np.std(eeg_data)
        
        # Simulate band powers
        features = {
            "mean": float(mean_val),
            "std": float(std_val),
            "delta_power": float(10 + 5 * np.random.random()),
            "theta_power": float(5 + 3 * np.random.random()),
            "alpha_power": float(8 + 4 * np.random.random()),
            "beta_power": float(4 + 2 * np.random.random()),
            "gamma_power": float(2 + 1 * np.random.random()),
        }
        
        return {
            "quality": 0.7,  # Simulate quality assessment
            "features": features
        }
    except Exception as e:
        logger.error(f"Error preprocessing EEG data: {str(e)}")
        return {"quality": 0, "features": {}}

def preprocess_gsr(gsr_value: float) -> Dict[str, Any]:
    """
    Process Galvanic Skin Response (GSR) data.
    """
    if gsr_value is None:
        return {"quality": 0, "features": {}}
    
    try:
        # For GSR, we might normalize the value or calculate rate of change
        features = {
            "value": float(gsr_value),
            "normalized": float(max(0, min(1, gsr_value / 20.0)))  # Assuming max GSR is around 20
        }
        
        return {
            "quality": 0.9,  # Simulate quality assessment
            "features": features
        }
    except Exception as e:
        logger.error(f"Error preprocessing GSR data: {str(e)}")
        return {"quality": 0, "features": {}}

# Endpoints for data ingestion
@router.post("/data/physiological", tags=["data"])
async def ingest_physiological_data(data_batch: PhysiologicalDataBatch):
    """
    Ingest and process a batch of physiological data for a patient.
    """
    patient_id = data_batch.patient_id
    logger.info(f"Ingesting physiological data for patient {patient_id}: {len(data_batch.data_points)} points")
    
    # Initialize patient data if not exists
    if patient_id not in patient_data:
        patient_data[patient_id] = {
            "physiological": [],
            "environmental": [],
            "behavioral": [],
            "processed_features": []
        }
    
    # Process each data point
    processed_points = []
    for point in data_batch.data_points:
        processed_point = {
            "timestamp": point.timestamp,
            "raw": {k: v for k, v in point.dict().items() if k != "timestamp"},
            "processed": {}
        }
        
        # Process each signal type if present
        if point.ecg:
            processed_point["processed"]["ecg"] = preprocess_ecg(point.ecg)
        
        if point.eeg:
            processed_point["processed"]["eeg"] = preprocess_eeg(point.eeg)
        
        if point.gsr is not None:
            processed_point["processed"]["gsr"] = preprocess_gsr(point.gsr)
        
        # Other signal types would have similar processing
        
        processed_points.append(processed_point)
    
    # Store processed data
    patient_data[patient_id]["physiological"].extend(processed_points)
    
    # Only keep most recent data (e.g., last 24 hours)
    max_data_points = 8640  # Assuming data every 10 seconds for 24 hours
    if len(patient_data[patient_id]["physiological"]) > max_data_points:
        patient_data[patient_id]["physiological"] = patient_data[patient_id]["physiological"][-max_data_points:]
    
    return {"status": "success", "processed_count": len(processed_points)}

@router.post("/data/environmental", tags=["data"])
async def ingest_environmental_data(data_batch: EnvironmentalDataBatch):
    """
    Ingest environmental data for a patient.
    """
    patient_id = data_batch.patient_id
    logger.info(f"Ingesting environmental data for patient {patient_id}: {len(data_batch.data_points)} points")
    
    if patient_id not in patient_data:
        patient_data[patient_id] = {
            "physiological": [],
            "environmental": [],
            "behavioral": [],
            "processed_features": []
        }
    
    # Store environmental data
    for point in data_batch.data_points:
        patient_data[patient_id]["environmental"].append(point.dict())
    
    # Only keep most recent data
    max_data_points = 1440  # Assuming data every minute for 24 hours
    if len(patient_data[patient_id]["environmental"]) > max_data_points:
        patient_data[patient_id]["environmental"] = patient_data[patient_id]["environmental"][-max_data_points:]
    
    return {"status": "success", "processed_count": len(data_batch.data_points)}

@router.post("/data/behavioral", tags=["data"])
async def ingest_behavioral_data(data_batch: BehavioralDataBatch):
    """
    Ingest behavioral data for a patient.
    """
    patient_id = data_batch.patient_id
    logger.info(f"Ingesting behavioral data for patient {patient_id}: {len(data_batch.data_points)} points")
    
    if patient_id not in patient_data:
        patient_data[patient_id] = {
            "physiological": [],
            "environmental": [],
            "behavioral": [],
            "processed_features": []
        }
    
    # Store behavioral data
    for point in data_batch.data_points:
        patient_data[patient_id]["behavioral"].append(point.dict())
    
    # Only keep most recent data
    max_data_points = 100  # Behavioral data is typically less frequent
    if len(patient_data[patient_id]["behavioral"]) > max_data_points:
        patient_data[patient_id]["behavioral"] = patient_data[patient_id]["behavioral"][-max_data_points:]
    
    return {"status": "success", "processed_count": len(data_batch.data_points)}

# Prediction endpoints
@router.post("/risk", response_model=PredictionResult)
async def predict_risk(request: PredictionRequest):
    """
    Predict migraine risk based on physiological, environmental, and behavioral data.
    """
    logger.info(f"Received prediction request for patient {request.patient_id}")
    
    # Generate prediction ID
    prediction_id = str(uuid.uuid4())
    
    try:
        # In a real implementation, this would call the digital twin model for prediction
        # For now, we'll simulate a prediction
        
        # Simulate risk score (0-1 scale)
        risk_score = round(0.3 + 0.4 * np.random.random(), 2)
        
        # Higher confidence for extreme scores, lower for middle range
        confidence = round(0.7 + 0.2 * abs(risk_score - 0.5) * 2, 2)
        
        # Simulate timeframe
        timeframe = "next 24 hours"
        
        # Simulate potential triggers
        triggers = [
            {"trigger": "sleep deprivation", "confidence": round(0.6 + 0.3 * np.random.random(), 2), "severity": 3},
            {"trigger": "weather change", "confidence": round(0.4 + 0.3 * np.random.random(), 2), "severity": 2},
            {"trigger": "stress", "confidence": round(0.5 + 0.4 * np.random.random(), 2), "severity": 4}
        ]
        
        # Generate recommendations based on triggers
        recommendations = []
        for trigger in triggers:
            if trigger["confidence"] > 0.6:
                if trigger["trigger"] == "sleep deprivation":
                    recommendations.append("Prioritize getting 7-8 hours of sleep tonight")
                elif trigger["trigger"] == "weather change":
                    recommendations.append("Stay hydrated and consider indoor activities")
                elif trigger["trigger"] == "stress":
                    recommendations.append("Practice 15 minutes of meditation or deep breathing")
        
        # Add general recommendation if none specific
        if not recommendations:
            recommendations.append("Monitor symptoms and maintain regular routine")
        
        # Create prediction result
        result = {
            "prediction_id": prediction_id,
            "risk_score": risk_score,
            "confidence": confidence,
            "timeframe": timeframe,
            "potential_triggers": triggers,
            "recommendations": recommendations,
            "timestamp": datetime.now().isoformat()
        }
        
        # Store prediction
        prediction_results[prediction_id] = result
        
        # Update patient triggers (in a real implementation, this would be more sophisticated)
        if request.patient_id:
            if request.patient_id not in patient_triggers:
                patient_triggers[request.patient_id] = {
                    "patient_id": request.patient_id,
                    "triggers": [],
                    "last_updated": datetime.now().isoformat()
                }
            
            # Update trigger profile
            existing_triggers = {t["trigger"]: t for t in patient_triggers[request.patient_id]["triggers"]}
            
            for trigger in triggers:
                trigger_name = trigger["trigger"]
                if trigger_name in existing_triggers:
                    # Update existing trigger
                    existing_triggers[trigger_name]["confidence"] = (
                        existing_triggers[trigger_name]["confidence"] * 0.7 + trigger["confidence"] * 0.3
                    )
                    existing_triggers[trigger_name]["occurrences"] += 1
                    existing_triggers[trigger_name]["last_detected"] = datetime.now().isoformat()
                else:
                    # Add new trigger
                    existing_triggers[trigger_name] = {
                        "trigger": trigger_name,
                        "confidence": trigger["confidence"],
                        "occurrences": 1,
                        "last_detected": datetime.now().isoformat()
                    }
            
            # Update patient triggers
            patient_triggers[request.patient_id]["triggers"] = list(existing_triggers.values())
            patient_triggers[request.patient_id]["last_updated"] = datetime.now().isoformat()
        
        return result
    
    except Exception as e:
        logger.error(f"Error processing prediction request: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error generating prediction: {str(e)}")

@router.get("/triggers/{patient_id}", response_model=List[Dict[str, Any]])
async def get_patient_triggers(patient_id: str):
    """
    Get the trigger profile for a specific patient.
    """
    if patient_id not in patient_triggers:
        return []
    
    # Sort triggers by confidence (descending)
    triggers = sorted(
        patient_triggers[patient_id]["triggers"],
        key=lambda x: x["confidence"],
        reverse=True
    )
    
    return triggers

@router.get("/data/{patient_id}", response_model=Dict[str, Any])
async def get_patient_data_summary(patient_id: str):
    """
    Get a summary of the data available for a patient.
    """
    if patient_id not in patient_data:
        raise HTTPException(status_code=404, detail=f"No data found for patient {patient_id}")
    
    # Create a summary of available data
    data = patient_data[patient_id]
    
    return {
        "patient_id": patient_id,
        "physiological_count": len(data["physiological"]),
        "environmental_count": len(data["environmental"]),
        "behavioral_count": len(data["behavioral"]),
        "data_period": {
            "start": data["physiological"][0]["timestamp"] if data["physiological"] else None,
            "end": data["physiological"][-1]["timestamp"] if data["physiological"] else None
        }
    }