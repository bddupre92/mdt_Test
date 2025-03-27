"""
Persistence module for the Mixture of Experts (MoE) framework.

This module provides components for saving and loading model state,
checkpointing long-running operations, and managing pipeline state.
"""

from .state_manager import (
    StateManager,
    FileSystemStateManager
)

__all__ = [
    'StateManager',
    'FileSystemStateManager'
]
