"""
Expert Training Workflow Factory Module

This module provides factory functions for retrieving appropriate training
workflows based on expert types.
"""

import logging
from typing import Dict, Any, Optional

from .base_workflow import ExpertTrainingWorkflow
from .physiological import PhysiologicalTrainingWorkflow
from .behavioral import BehavioralTrainingWorkflow
from .environmental import EnvironmentalTrainingWorkflow
from .medication import MedicationTrainingWorkflow

# Configure logging
logger = logging.getLogger(__name__)

def get_expert_training_workflow(expert_type: str, config: Optional[Dict[str, Any]] = None, 
                                 verbose: bool = False) -> ExpertTrainingWorkflow:
    """
    Get the appropriate training workflow for the specified expert type.
    
    Args:
        expert_type: Type of expert ('physiological', 'behavioral', etc.)
        config: Configuration parameters for the training workflow
        verbose: Whether to display detailed logs during training
        
    Returns:
        Appropriate expert training workflow instance
    """
    # Normalize expert type to lowercase
    expert_type = expert_type.lower()
    
    # Create the appropriate workflow based on expert type
    if expert_type == 'physiological':
        return PhysiologicalTrainingWorkflow(config=config, verbose=verbose)
    elif expert_type == 'behavioral':
        return BehavioralTrainingWorkflow(config=config, verbose=verbose)
    elif expert_type == 'environmental':
        return EnvironmentalTrainingWorkflow(config=config, verbose=verbose)
    elif expert_type in ['medication', 'medication_history']:
        return MedicationTrainingWorkflow(config=config, verbose=verbose)
    else:
        logger.warning(f"Unknown expert type: {expert_type}. Using base workflow.")
        return ExpertTrainingWorkflow(config=config, verbose=verbose)
