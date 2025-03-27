"""
Integration module for the Mixture of Experts (MoE) framework.

This module provides components that connect expert models and gating networks
to create a cohesive prediction system.
"""

from .integration_layer import (
    IntegrationLayer,
    WeightedAverageIntegration,
    AdaptiveIntegration
)

from .event_system import (
    EventManager,
    Event,
    EventListener,
    MoEEventTypes
)

from .pipeline_connector import (
    IntegrationConnector
)

__all__ = [
    # Integration layers
    'IntegrationLayer',
    'WeightedAverageIntegration',
    'AdaptiveIntegration',
    
    # Event system
    'EventManager',
    'Event',
    'EventListener',
    'MoEEventTypes',
    
    # Connectors
    'IntegrationConnector'
]
