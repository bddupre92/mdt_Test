"""
Pipeline connector for the MoE framework integration layer.

This module provides connectors to integrate the existing pipeline components
with the integration layer, event system, and state management functionality.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple, Union
import os

import numpy as np

from moe_framework.interfaces.base import PatientContext, SystemState
from moe_framework.interfaces.expert import ExpertModel
from moe_framework.interfaces.gating import GatingNetwork
from moe_framework.interfaces.pipeline import Pipeline

from moe_framework.integration.integration_layer import IntegrationLayer, WeightedAverageIntegration
from moe_framework.integration.event_system import EventManager, Event, MoEEventTypes
from moe_framework.persistence.state_manager import StateManager, FileSystemStateManager

logger = logging.getLogger(__name__)


class IntegrationConnector:
    """
    Connector class that bridges the integration layer with existing pipeline components.
    
    This class serves as an adapter between the MoEPipeline and the integration
    components, facilitating event-based communication and state management.
    """
    
    def __init__(
        self,
        integration_layer=None,
        event_manager=None,
        state_manager=None,
        config: Optional[Dict[str, Any]] = None
    ):
        """Initialize the integration connector.
        
        Args:
            integration_layer: Optional custom integration layer
            event_manager: Optional custom event manager
            state_manager: Optional custom state manager
            config: Optional configuration dictionary
        """
        self.config = config or {}
        
        # Initialize event manager if not provided
        if event_manager is None:
            self.event_manager = EventManager()
        else:
            self.event_manager = event_manager
            
        # Initialize state manager if not provided
        if state_manager is None:
            state_config = self.config.get('state_management', {})
            base_dir = state_config.get('checkpoint_dir', os.path.join(os.getcwd(), 'checkpoints'))
            self.state_manager = FileSystemStateManager(base_dir=base_dir)
        else:
            self.state_manager = state_manager
            
        # Initialize integration layer if not provided
        if integration_layer is None:
            integration_config = self.config.get('integration', {})
            self.integration_layer = WeightedAverageIntegration(
                config=integration_config
            )
        else:
            self.integration_layer = integration_layer
        
        # Register for relevant events
        self._register_event_handlers()
        
    def _register_event_handlers(self):
        """Register internal event handlers."""
        # This could be extended to register actual handlers
        pass
        
    def integrate_predictions(
        self,
        expert_outputs: Dict[str, np.ndarray],
        weights: Dict[str, float],
        context: Optional[PatientContext] = None
    ) -> np.ndarray:
        """
        Integrate predictions using the integration layer.
        
        Args:
            expert_outputs: Dictionary mapping expert names to prediction outputs
            weights: Dictionary mapping expert names to weights
            context: Optional patient context
            
        Returns:
            Integrated prediction
        """
        # Emit event before integration
        self.event_manager.emit_event(
            Event(
                MoEEventTypes.EXPERT_PREDICTION_COMPLETED,
                {"expert_count": len(expert_outputs), "context": context}
            )
        )
        
        # Perform integration
        result = self.integration_layer.integrate(
            expert_outputs=expert_outputs,
            weights=weights,
            context=context
        )
        
        # Emit event after integration
        self.event_manager.emit_event(
            Event(
                MoEEventTypes.INTEGRATION_COMPLETED,
                {
                    "result_shape": result.shape if hasattr(result, 'shape') else None,
                    "experts_used": list(expert_outputs.keys()),
                    "context": context
                }
            )
        )
        
        return result
        
    def save_pipeline_state(self, pipeline_state: SystemState, path: str) -> bool:
        """
        Save pipeline state using the state manager.
        
        Args:
            pipeline_state: System state to save
            path: Path where to save the state
            
        Returns:
            Success status
        """
        success = self.state_manager.save_state(pipeline_state, path)
        
        if success:
            self.event_manager.emit_event(
                Event(
                    MoEEventTypes.CHECKPOINT_SAVED,
                    {"path": path, "state_version": pipeline_state.version_info["version"]}
                )
            )
            
        return success
        
    def load_pipeline_state(self, path: str) -> Optional[SystemState]:
        """
        Load pipeline state using the state manager.
        
        Args:
            path: Path from which to load the state
            
        Returns:
            Loaded system state, or None if loading failed
        """
        state = self.state_manager.load_state(path)
        
        if state:
            self.event_manager.emit_event(
                Event(
                    MoEEventTypes.CHECKPOINT_LOADED,
                    {"path": path, "state_version": state.version_info["version"]}
                )
            )
            
        return state
        
    def get_available_checkpoints(self, base_dir: str = "") -> List[Dict[str, Any]]:
        """
        Get available checkpoints with metadata.
        
        Args:
            base_dir: Base directory to search
            
        Returns:
            List of checkpoint information dictionaries
        """
        checkpoint_paths = self.state_manager.list_checkpoints(base_dir)
        
        checkpoint_info = []
        for path in checkpoint_paths:
            metadata = self.state_manager.get_metadata(path)
            checkpoint_info.append({
                "path": path,
                "metadata": metadata
            })
            
        return checkpoint_info
