"""
Enhanced Data Support for MoE Validation Framework

This module provides support for using enhanced synthetic data with the MoE validation framework.
"""

import os
import sys
import json
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_enhanced_data_config(config_path: str) -> Dict[str, Any]:
    """
    Load enhanced data configuration
    
    Parameters:
    -----------
    config_path : str
        Path to enhanced data configuration file
        
    Returns:
    --------
    Dict[str, Any]
        Enhanced data configuration
    """
    # Ensure the path exists
    if not os.path.exists(config_path):
        logger.error(f"Enhanced data configuration file not found at {config_path}")
        return {}
    
    # Load the configuration
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    logger.info(f"Loaded enhanced data configuration from {config_path}")
    return config

def load_data_pointers(pointers_path: str) -> Dict[str, Any]:
    """
    Load data pointers file
    
    Parameters:
    -----------
    pointers_path : str
        Path to data pointers file
        
    Returns:
    --------
    Dict[str, Any]
        Data pointers
    """
    # Ensure the path exists
    if not os.path.exists(pointers_path):
        logger.error(f"Data pointers file not found at {pointers_path}")
        return {}
    
    # Load the data pointers
    with open(pointers_path, 'r') as f:
        pointers = json.load(f)
    
    logger.info(f"Loaded data pointers for {len(pointers.get('patients', []))} patients")
    return pointers

def load_patient_data(patient_path: str) -> Dict[str, Any]:
    """
    Load patient data from enhanced synthetic dataset
    
    Parameters:
    -----------
    patient_path : str
        Path to patient directory
        
    Returns:
    --------
    Dict[str, Any]
        Patient data
    """
    # Ensure the path exists
    patient_path = Path(patient_path)
    if not patient_path.exists():
        logger.error(f"Patient directory not found at {patient_path}")
        return {}
    
    # Load patient data
    patient_data_path = patient_path / 'patient_data.json'
    if not patient_data_path.exists():
        logger.error(f"Patient data file not found at {patient_data_path}")
        return {}
    
    with open(patient_data_path, 'r') as f:
        patient_data = json.load(f)
    
    # Load demographics
    demographics_path = patient_path / 'demographics.json'
    if demographics_path.exists():
        with open(demographics_path, 'r') as f:
            demographics = json.load(f)
        patient_data['demographics'] = demographics
    
    # Load drift metadata
    drift_path = patient_path / 'drift_metadata.json'
    if drift_path.exists():
        with open(drift_path, 'r') as f:
            drift_metadata = json.load(f)
        patient_data['drift_metadata'] = drift_metadata
    
    # Load evaluation metrics
    eval_path = patient_path / 'evaluation_metrics.json'
    if eval_path.exists():
        with open(eval_path, 'r') as f:
            evaluation = json.load(f)
        patient_data['evaluation'] = evaluation
    
    return patient_data

def get_visualization_paths(patient_id: str, config: Dict[str, Any]) -> Dict[str, str]:
    """
    Get paths to visualization files for a patient
    
    Parameters:
    -----------
    patient_id : str
        Patient ID
    config : Dict[str, Any]
        Enhanced data configuration
        
    Returns:
    --------
    Dict[str, str]
        Paths to visualization files
    """
    vis_dir = Path(config.get('visualization_dir', './results/moe_validation/visualizations'))
    
    return {
        'drift_analysis': str(vis_dir / f"{patient_id}_drift_analysis.png"),
        'feature_importance': str(vis_dir / f"{patient_id}_feature_importance.png"),
        'timeseries': str(vis_dir / f"{patient_id}_timeseries.png")
    }
