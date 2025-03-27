"""
patient_memory.py
----------------
Implementation of patient-specific memory for the MetaLearner
to enable personalized adaptations that persist across sessions.
"""

import os
import json
import logging
import datetime
from typing import Dict, Any, Optional

# Configure logging
logger = logging.getLogger(__name__)

class PatientMemory:
    """
    Manages patient-specific adaptation memory for the MetaLearner.
    
    This class stores and retrieves patient-specific information including:
    - Expert weight adjustments
    - Historical data quality
    - Past drift detection results
    - Performance metrics
    
    The memory persists between sessions via a simple JSON file storage.
    """
    
    def __init__(self, storage_dir: str = None):
        """
        Initialize the PatientMemory with optional storage directory.
        
        Args:
            storage_dir: Directory for storing patient memory files 
                         (defaults to data/patient_memory)
        """
        self.storage_dir = storage_dir or os.path.join('data', 'patient_memory')
        self._ensure_storage_dir()
        self.current_patient_id = None
        self.memory = {}
        
    def _ensure_storage_dir(self):
        """Ensure the storage directory exists."""
        os.makedirs(self.storage_dir, exist_ok=True)
        
    def _get_memory_path(self, patient_id: str) -> str:
        """Get the file path for a patient's memory file."""
        return os.path.join(self.storage_dir, f"patient_{patient_id}.json")
        
    def select_patient(self, patient_id: str) -> Dict[str, Any]:
        """
        Load a patient's memory data and set as current patient.
        
        Args:
            patient_id: Unique identifier for the patient
            
        Returns:
            Dictionary containing the patient's memory data
        """
        self.current_patient_id = patient_id
        memory_path = self._get_memory_path(patient_id)
        
        # If patient file exists, load it
        if os.path.exists(memory_path):
            try:
                with open(memory_path, 'r') as f:
                    self.memory = json.load(f)
                logger.info(f"Loaded memory for patient {patient_id}")
            except Exception as e:
                logger.error(f"Error loading memory for patient {patient_id}: {str(e)}")
                self.memory = self._initialize_memory(patient_id)
        else:
            # Create new memory for this patient
            self.memory = self._initialize_memory(patient_id)
            self._save_current_memory()
            
        return self.memory
        
    def _initialize_memory(self, patient_id: str) -> Dict[str, Any]:
        """Create an initial memory structure for a new patient."""
        return {
            "patient_id": patient_id,
            "created_at": datetime.datetime.now().isoformat(),
            "last_updated": datetime.datetime.now().isoformat(),
            "expert_weights": {},
            "domain_quality_history": {
                "physiological": [],
                "behavioral": [],
                "environmental": []
            },
            "drift_history": {
                "physiological": [],
                "behavioral": [],
                "environmental": []
            },
            "performance_history": [],
            "expert_specialty_weights": {}
        }
        
    def _save_current_memory(self) -> bool:
        """Save the current patient's memory to disk."""
        if not self.current_patient_id:
            logger.warning("No patient currently selected")
            return False
            
        # Update timestamp before saving
        self.memory["last_updated"] = datetime.datetime.now().isoformat()
        
        try:
            memory_path = self._get_memory_path(self.current_patient_id)
            with open(memory_path, 'w') as f:
                json.dump(self.memory, f, indent=2)
            logger.info(f"Saved memory for patient {self.current_patient_id}")
            return True
        except Exception as e:
            logger.error(f"Error saving memory for patient {self.current_patient_id}: {str(e)}")
            return False
            
    def update_expert_weights(self, expert_weights: Dict[int, float]) -> bool:
        """
        Update stored expert weights for the current patient.
        
        Args:
            expert_weights: Dictionary mapping expert IDs to weights
            
        Returns:
            True if update was successful, False otherwise
        """
        if not self.current_patient_id:
            logger.warning("No patient currently selected")
            return False
            
        self.memory["expert_weights"] = expert_weights
        return self._save_current_memory()
        
    def get_expert_weights(self) -> Dict[int, float]:
        """
        Get stored expert weights for the current patient.
        
        Returns:
            Dictionary mapping expert IDs to weights
        """
        if not self.current_patient_id:
            logger.warning("No patient currently selected")
            return {}
            
        return self.memory.get("expert_weights", {})
        
    def store_domain_quality(self, domain: str, quality_score: float, timestamp: Optional[str] = None) -> bool:
        """
        Store domain quality information for the current patient.
        
        Args:
            domain: Data domain (physiological, behavioral, environmental)
            quality_score: Quality score (0-1)
            timestamp: Optional timestamp (ISO format string)
            
        Returns:
            True if update was successful, False otherwise
        """
        if not self.current_patient_id:
            logger.warning("No patient currently selected")
            return False
            
        timestamp = timestamp or datetime.datetime.now().isoformat()
        
        if domain not in self.memory["domain_quality_history"]:
            self.memory["domain_quality_history"][domain] = []
            
        self.memory["domain_quality_history"][domain].append({
            "timestamp": timestamp,
            "score": quality_score
        })
        
        return self._save_current_memory()
        
    def store_drift_event(self, domain: str, drift_detected: bool, 
                          drift_score: float, p_value: float,
                          timestamp: Optional[str] = None) -> bool:
        """
        Store drift detection information for the current patient.
        
        Args:
            domain: Data domain (physiological, behavioral, environmental)
            drift_detected: Whether drift was detected
            drift_score: Drift detection score
            p_value: P-value from statistical test
            timestamp: Optional timestamp (ISO format string)
            
        Returns:
            True if update was successful, False otherwise
        """
        if not self.current_patient_id:
            logger.warning("No patient currently selected")
            return False
            
        timestamp = timestamp or datetime.datetime.now().isoformat()
        
        if domain not in self.memory["drift_history"]:
            self.memory["drift_history"][domain] = []
            
        self.memory["drift_history"][domain].append({
            "timestamp": timestamp,
            "detected": drift_detected,
            "score": drift_score,
            "p_value": p_value
        })
        
        return self._save_current_memory()
        
    def update_specialty_weights(self, specialty_weights: Dict[str, float]) -> bool:
        """
        Update specialty weights for the current patient.
        
        Args:
            specialty_weights: Dictionary mapping specialties to weights
            
        Returns:
            True if update was successful, False otherwise
        """
        if not self.current_patient_id:
            logger.warning("No patient currently selected")
            return False
            
        self.memory["expert_specialty_weights"] = specialty_weights
        return self._save_current_memory()
        
    def get_specialty_weights(self) -> Dict[str, float]:
        """
        Get specialty weights for the current patient.
        
        Returns:
            Dictionary mapping specialties to weights
        """
        if not self.current_patient_id:
            logger.warning("No patient currently selected")
            return {}
            
        return self.memory.get("expert_specialty_weights", {})
        
    def get_domain_quality_history(self, domain: str, limit: int = 10) -> list:
        """
        Get historical quality data for a domain.
        
        Args:
            domain: Data domain (physiological, behavioral, environmental)
            limit: Maximum number of historical entries to return
            
        Returns:
            List of historical quality entries, most recent first
        """
        if not self.current_patient_id:
            logger.warning("No patient currently selected")
            return []
            
        history = self.memory.get("domain_quality_history", {}).get(domain, [])
        return sorted(history, key=lambda x: x["timestamp"], reverse=True)[:limit]
        
    def get_drift_history(self, domain: str, limit: int = 10) -> list:
        """
        Get historical drift data for a domain.
        
        Args:
            domain: Data domain (physiological, behavioral, environmental)
            limit: Maximum number of historical entries to return
            
        Returns:
            List of historical drift entries, most recent first
        """
        if not self.current_patient_id:
            logger.warning("No patient currently selected")
            return []
            
        history = self.memory.get("drift_history", {}).get(domain, [])
        return sorted(history, key=lambda x: x["timestamp"], reverse=True)[:limit]
        
    def store_performance(self, score: float, details: Dict[str, Any] = None,
                         timestamp: Optional[str] = None) -> bool:
        """
        Store performance information for the current patient.
        
        Args:
            score: Overall performance score
            details: Additional performance details
            timestamp: Optional timestamp (ISO format string)
            
        Returns:
            True if update was successful, False otherwise
        """
        if not self.current_patient_id:
            logger.warning("No patient currently selected")
            return False
            
        timestamp = timestamp or datetime.datetime.now().isoformat()
        
        self.memory["performance_history"].append({
            "timestamp": timestamp,
            "score": score,
            "details": details or {}
        })
        
        return self._save_current_memory()
        
    def get_performance_history(self, limit: int = 10) -> list:
        """
        Get historical performance data.
        
        Args:
            limit: Maximum number of historical entries to return
            
        Returns:
            List of historical performance entries, most recent first
        """
        if not self.current_patient_id:
            logger.warning("No patient currently selected")
            return []
            
        history = self.memory.get("performance_history", [])
        return sorted(history, key=lambda x: x["timestamp"], reverse=True)[:limit]
        
    def clear_patient_data(self, patient_id: str) -> bool:
        """
        Clear all memory data for a specific patient.
        
        Args:
            patient_id: Unique identifier for the patient
            
        Returns:
            True if deletion was successful, False otherwise
        """
        memory_path = self._get_memory_path(patient_id)
        
        if os.path.exists(memory_path):
            try:
                os.remove(memory_path)
                logger.info(f"Cleared memory for patient {patient_id}")
                
                # Reset current memory if it was the current patient
                if self.current_patient_id == patient_id:
                    self.memory = self._initialize_memory(patient_id)
                    
                return True
            except Exception as e:
                logger.error(f"Error clearing memory for patient {patient_id}: {str(e)}")
                return False
        else:
            logger.warning(f"No memory file found for patient {patient_id}")
            return False
