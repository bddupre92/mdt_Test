"""
Expert Training Workflows Module

DEPRECATED: This module has been refactored into a modular package structure.
Please import from moe_framework.workflow.training instead.

This file is maintained for backward compatibility but will be removed in a future version.
"""

import warnings
from typing import Dict, List, Any, Optional, Callable

# Issue deprecation warning
warnings.warn(
    "The expert_training_workflows module is deprecated. "
    "Please import from moe_framework.workflow.training instead.",
    DeprecationWarning, 
    stacklevel=2
)

# Import from new modular package for backward compatibility
from .training import (
    ExpertTrainingWorkflow,
    PhysiologicalTrainingWorkflow,
    BehavioralTrainingWorkflow,
    EnvironmentalTrainingWorkflow,
    MedicationTrainingWorkflow,
    get_expert_training_workflow
)

# Re-export all components to maintain backward compatibility
__all__ = [
    'ExpertTrainingWorkflow',
    'PhysiologicalTrainingWorkflow',
    'BehavioralTrainingWorkflow',
    'EnvironmentalTrainingWorkflow',
    'MedicationTrainingWorkflow',
    'get_expert_training_workflow'
]
