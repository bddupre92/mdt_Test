#!/usr/bin/env python3
"""
Prepare Enhanced Synthetic Data for MoE Validation Framework

This script prepares the enhanced synthetic data for use with the MoE validation framework.
It copies relevant data to the MoE validation results directory and creates configuration
files that link the enhanced data to the validation framework.
"""

import os
import sys
import json
import shutil
import logging
from pathlib import Path
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import the enhanced synthetic data generator
from utils.enhanced_synthetic_data import EnhancedPatientDataGenerator

def prepare_enhanced_data(args=None):
    """
    Prepare enhanced synthetic data for MoE validation
    
    Parameters:
    -----------
    args : dict, optional
        Dictionary of arguments
        
    Returns:
    --------
    dict
        Dictionary with paths to prepared data
    """
    # Default arguments
    if args is None:
        args = {}
        
    # Parse arguments
    enhanced_data_dir = args.get('enhanced_data_dir', './test_data/enhanced_output')
    moe_results_dir = args.get('results_dir', './results/moe_validation')
    drift_type = args.get('drift_type', 'gradual')
    
    # Ensure directories exist
    enhanced_data_path = Path(enhanced_data_dir)
    moe_results_path = Path(moe_results_dir)
    moe_results_path.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Preparing enhanced data from {enhanced_data_path} for MoE validation")
    
    # Find all patient directories
    patient_dirs = [d for d in enhanced_data_path.glob("patient_*") if d.is_dir()]
    
    if not patient_dirs:
        logger.error(f"No patient data found in {enhanced_data_path}")
        return {
            'success': False,
            'message': f"No patient data found in {enhanced_data_path}"
        }
    
    # Create data pointers file
    data_pointers = {
        'timestamp': datetime.now().isoformat(),
        'patients': [],
        'drift_type': drift_type,
        'source_path': str(enhanced_data_path),
        'validation_timestamp': datetime.now().isoformat()
    }
    
    for patient_dir in patient_dirs:
        patient_id = patient_dir.name
        
        # Check for required files
        required_files = ['patient_data.json', 'demographics.json', 'drift_metadata.json']
        if all((patient_dir / file).exists() for file in required_files):
            # Read drift metadata to determine if this patient has the requested drift type
            with open(patient_dir / 'drift_metadata.json', 'r') as f:
                drift_metadata = json.load(f)
                
            patient_drift_type = drift_metadata.get('drift_type', 'none')
            
            # If no specific drift type is requested or this patient matches the requested type
            if drift_type == 'all' or patient_drift_type == drift_type:
                # Add patient to data pointers
                data_pointers['patients'].append({
                    'patient_id': patient_id,
                    'path': str(patient_dir),
                    'drift_type': patient_drift_type,
                    'has_evaluation': (patient_dir / 'evaluation_metrics.json').exists(),
                    'has_visualization': (patient_dir / 'drift_analysis.png').exists() or 
                                        (patient_dir / 'feature_importance.png').exists() or
                                        (patient_dir / 'timeseries_visualization.png').exists()
                })
    
    # Save data pointers file
    data_pointers_path = moe_results_path / 'enhanced_data_pointers.json'
    with open(data_pointers_path, 'w') as f:
        json.dump(data_pointers, f, indent=2)
    
    logger.info(f"Created data pointers file with {len(data_pointers['patients'])} patients")
    
    # Create symbolic links to visualization files for the interactive report
    vis_dir = moe_results_path / 'visualizations'
    vis_dir.mkdir(exist_ok=True)
    
    # First, clean the visualization directory to avoid issues with existing files
    try:
        for item in vis_dir.iterdir():
            if item.is_file() or item.is_symlink():
                try:
                    item.unlink()
                    logger.info(f"Removed existing file/symlink: {item}")
                except Exception as e:
                    logger.warning(f"Could not remove file {item}: {e}")
    except Exception as e:
        logger.warning(f"Error cleaning visualization directory: {e}")
    
    for patient in data_pointers['patients']:
        patient_path = Path(patient['path'])
        patient_id = patient['patient_id']
        
        # Link drift analysis visualization
        if (patient_path / 'drift_analysis.png').exists():
            target = vis_dir / f"{patient_id}_drift_analysis.png"
            # Remove existing target if it exists
            if target.exists():
                target.unlink()
                
            if os.name == 'nt':  # Windows
                shutil.copy2(str(patient_path / 'drift_analysis.png'), str(target))
            else:  # Unix/Linux/Mac
                # Create a symlink at the target location that points to the source file
                # os.symlink(src, dst) - src is the file the link points to, dst is the link itself
                os.symlink(str(patient_path / 'drift_analysis.png'), str(target))
        
        # Link feature importance visualization
        if (patient_path / 'feature_importance.png').exists():
            target = vis_dir / f"{patient_id}_feature_importance.png"
            # Remove existing target if it exists
            if target.exists():
                target.unlink()
                
            if os.name == 'nt':  # Windows
                shutil.copy2(str(patient_path / 'feature_importance.png'), str(target))
            else:  # Unix/Linux/Mac
                # Create a symlink at the target location that points to the source file
                os.symlink(str(patient_path / 'feature_importance.png'), str(target))
        
        # Link timeseries visualization
        if (patient_path / 'timeseries_visualization.png').exists():
            target = vis_dir / f"{patient_id}_timeseries.png"
            # Remove existing target if it exists
            if target.exists():
                target.unlink()
                
            if os.name == 'nt':  # Windows
                shutil.copy2(str(patient_path / 'timeseries_visualization.png'), str(target))
            else:  # Unix/Linux/Mac
                # Create a symlink at the target location that points to the source file
                os.symlink(str(patient_path / 'timeseries_visualization.png'), str(target))
    
    # Create a config file for the MoE validation framework
    config = {
        'data_source': 'enhanced_synthetic',
        'data_pointers_file': str(data_pointers_path),
        'timestamp': datetime.now().isoformat(),
        'drift_type': drift_type,
        'visualization_dir': str(vis_dir),
        'num_patients': len(data_pointers['patients']),
        'enable_interactive': True
    }
    
    config_path = moe_results_path / 'enhanced_validation_config.json'
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    logger.info(f"Created configuration file at {config_path}")
    
    return {
        'success': True,
        'message': f"Successfully prepared {len(data_pointers['patients'])} patients for MoE validation",
        'data_pointers_path': str(data_pointers_path),
        'config_path': str(config_path),
        'visualization_dir': str(vis_dir)
    }

if __name__ == "__main__":
    # Parse command-line arguments
    import argparse
    
    parser = argparse.ArgumentParser(description="Prepare enhanced synthetic data for MoE validation")
    parser.add_argument("--enhanced-data-dir", default="./test_data/enhanced_output",
                        help="Directory containing enhanced synthetic data")
    parser.add_argument("--results-dir", default="./results/moe_validation",
                        help="Directory to store MoE validation results")
    parser.add_argument("--drift-type", default="all",
                        choices=["none", "sudden", "gradual", "recurring", "all"],
                        help="Type of drift to include in the validation")
    
    args = parser.parse_args()
    
    # Prepare enhanced data
    result = prepare_enhanced_data(vars(args))
    
    if result['success']:
        print(f"\nSuccess! {result['message']}")
        print(f"Data pointers file: {result['data_pointers_path']}")
        print(f"Configuration file: {result['config_path']}")
        print(f"Visualization directory: {result['visualization_dir']}")
        print("\nTo run the MoE validation with this data, use:")
        print(f"python main_v2.py validate-moe --interactive --enhanced-data-config {result['config_path']}")
    else:
        print(f"\nError: {result['message']}")
        sys.exit(1)
