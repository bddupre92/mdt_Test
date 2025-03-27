#!/usr/bin/env python
"""
Test script for patient memory integration in the MetaLearner.
This demonstrates the patient-specific adaptations through memory persistence.
"""

import os
import logging
import numpy as np
from meta.meta_learner import MetaLearner
from meta.patient_memory import PatientMemory

# Setup basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('patient_memory_test')

# Define a simple expert class for testing
class ExpertModel:
    def __init__(self, specialty="general", performance=0.75):
        self.specialty = specialty
        self.performance = performance
        self.weights = {}
    
    def predict(self, X):
        return np.ones(len(X)) * self.performance

# Create a temporary directory for patient memory files
def setup_test_env():
    memory_dir = os.path.join(os.getcwd(), "patient_memory_test")
    os.makedirs(memory_dir, exist_ok=True)
    return memory_dir

def test_patient_specific_adaptations():
    """Test patient-specific adaptations through memory persistence."""
    memory_dir = setup_test_env()
    logger.info(f"Using memory directory: {memory_dir}")
    
    # Initialize MetaLearner with patient memory enabled
    meta_learner = MetaLearner(
        method='bayesian',
        drift_detector=None,
        quality_impact=0.5,
        drift_impact=0.3,
        memory_storage_dir=memory_dir,
        enable_personalization=True
    )
    
    # Register experts with different specialties
    meta_learner.register_expert("expert1", ExpertModel(specialty="physiological", performance=0.8))
    meta_learner.register_expert("expert2", ExpertModel(specialty="behavioral", performance=0.7))
    meta_learner.register_expert("expert3", ExpertModel(specialty="environmental", performance=0.6))
    meta_learner.register_expert("expert4", ExpertModel(specialty="general", performance=0.5))
    
    # Test with first patient
    patient1_id = "patient001"
    logger.info(f"Testing with patient: {patient1_id}")
    
    # Set patient and confirm memory was created
    patient1_memory = meta_learner.set_patient(patient1_id)
    logger.info(f"Initial patient memory: {patient1_memory}")
    
    # Set patient preference for physiological data (important for this patient)
    meta_learner.update_patient_specialty_preference("physiological", 1.5)
    
    # First prediction with quality metrics
    context = {
        "has_physiological": True,
        "has_behavioral": True,
        "has_environmental": True,
        "quality_metrics": {
            "physiological": 0.85,  # High quality physiological data
            "behavioral": 0.6,      # Medium quality behavioral data
            "environmental": 0.4,   # Lower quality environmental data
        },
        "patient_id": patient1_id
    }
    
    # Get expert weights
    weights_p1_first = meta_learner.predict_weights(context)
    logger.info(f"Patient 1 (first prediction) - Expert weights: {weights_p1_first}")
    
    # Track performance for experts
    meta_learner.track_performance("expert1", 0.9, {"prediction_type": "classification"})
    meta_learner.track_performance("expert2", 0.7, {"prediction_type": "classification"})
    
    # Test with second patient (different preferences)
    patient2_id = "patient002"
    logger.info(f"\nTesting with patient: {patient2_id}")
    
    # Set patient and confirm memory was created
    patient2_memory = meta_learner.set_patient(patient2_id)
    logger.info(f"Initial patient memory: {patient2_memory}")
    
    # Set patient preference for behavioral data (important for this patient)
    meta_learner.update_patient_specialty_preference("behavioral", 1.8)
    
    # First prediction with quality metrics (different from patient 1)
    context = {
        "has_physiological": True,
        "has_behavioral": True,
        "has_environmental": True,
        "quality_metrics": {
            "physiological": 0.6,   # Medium quality physiological data
            "behavioral": 0.9,      # High quality behavioral data 
            "environmental": 0.5,   # Medium quality environmental data
        },
        "patient_id": patient2_id
    }
    
    # Get expert weights
    weights_p2_first = meta_learner.predict_weights(context)
    logger.info(f"Patient 2 (first prediction) - Expert weights: {weights_p2_first}")
    
    # Now switch back to first patient (should load saved preferences)
    logger.info(f"\nSwitching back to patient: {patient1_id}")
    meta_learner.set_patient(patient1_id)
    
    # Get weights again (should reflect saved preferences)
    weights_p1_second = meta_learner.predict_weights(context)
    logger.info(f"Patient 1 (second prediction) - Expert weights: {weights_p1_second}")
    
    # Show history data for patient 1
    p1_history = meta_learner.get_patient_history()
    logger.info(f"Patient 1 history data:\n{p1_history}")
    
    # Test clearing patient data
    logger.info(f"\nClearing data for patient: {patient1_id}")
    meta_learner.clear_patient_data(patient1_id)
    meta_learner.set_patient(patient1_id)
    p1_history_after_clear = meta_learner.get_patient_history()
    logger.info(f"Patient 1 history after clearing: {p1_history_after_clear}")
    
    logger.info("Patient memory test completed successfully")

if __name__ == "__main__":
    test_patient_specific_adaptations()
