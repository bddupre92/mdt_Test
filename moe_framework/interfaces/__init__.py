"""
Interfaces for the Mixture of Experts (MoE) framework.

This package provides the core interfaces and abstract base classes that define
the contract between different components of the MoE system, ensuring consistent
integration and interaction patterns.
"""

from .base import (
    Configurable, 
    Persistable,
    DataStructure,
    PatientContext,
    PredictionResult,
    SystemState
)

from .expert import (
    ExpertModel,
    ExpertRegistry
)

from .gating import (
    GatingNetwork,
    QualityAwareGating,
    DriftAwareGating
)

from .pipeline import (
    Pipeline,
    TrainingPipeline,
    CheckpointingPipeline
)

from .optimizer import (
    Optimizer,
    ExpertSpecificOptimizer,
    OptimizerFactory
)

__all__ = [
    # Base interfaces
    'Configurable', 
    'Persistable',
    'DataStructure',
    
    # Data structures
    'PatientContext',
    'PredictionResult',
    'SystemState',
    
    # Expert interfaces
    'ExpertModel',
    'ExpertRegistry',
    
    # Gating interfaces
    'GatingNetwork',
    'QualityAwareGating',
    'DriftAwareGating',
    
    # Pipeline interfaces
    'Pipeline',
    'TrainingPipeline',
    'CheckpointingPipeline',
    
    # Optimizer interfaces
    'Optimizer',
    'ExpertSpecificOptimizer',
    'OptimizerFactory'
]
