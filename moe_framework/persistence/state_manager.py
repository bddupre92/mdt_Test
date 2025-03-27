"""
State management implementations for the MoE framework.

This module provides implementations for managing pipeline state,
supporting checkpointing and resumable operations.
"""

import json
import logging
import os
import pickle
import time
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from moe_framework.interfaces.base import SystemState, Persistable

logger = logging.getLogger(__name__)


class StateManager(ABC):
    """
    Abstract base class for pipeline state management.
    
    State managers handle saving and loading system state,
    enabling checkpoint and resume functionality.
    """
    
    @abstractmethod
    def save_state(self, state: SystemState, path: str) -> bool:
        """
        Save the current system state to the specified path.
        
        Args:
            state: System state to save
            path: Where to save the state
            
        Returns:
            Success status
        """
        pass
    
    @abstractmethod
    def load_state(self, path: str) -> Optional[SystemState]:
        """
        Load system state from the specified path.
        
        Args:
            path: Path from which to load state
            
        Returns:
            Loaded system state, or None if loading failed
        """
        pass
    
    @abstractmethod
    def list_checkpoints(self, base_dir: str) -> List[str]:
        """
        List available checkpoints in the specified directory.
        
        Args:
            base_dir: Directory to search for checkpoints
            
        Returns:
            List of checkpoint paths
        """
        pass
    
    @abstractmethod
    def get_metadata(self, checkpoint_path: str) -> Dict[str, Any]:
        """
        Get metadata about a specific checkpoint.
        
        Args:
            checkpoint_path: Path to the checkpoint
            
        Returns:
            Dictionary of metadata about the checkpoint
        """
        pass


class FileSystemStateManager(StateManager):
    """
    File system-based state manager implementation.
    
    This implementation saves and loads state objects to/from
    the file system, using a structured directory format.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None, **kwargs):
        """
        Initialize the file system state manager.
        
        Args:
            config: Configuration dictionary with storage settings
            **kwargs: Additional arguments for backward compatibility
                      - checkpoint_dir: Base directory for checkpoints
                      - max_checkpoints: Maximum number of checkpoints to keep
        """
        self.config = config or {}
        
        # Handle both config dict and direct parameters for backward compatibility
        if 'checkpoint_dir' in kwargs:
            self.base_dir = kwargs['checkpoint_dir']
            self.config['base_dir'] = self.base_dir
        else:
            self.base_dir = self.config.get('base_dir', 'checkpoints')
            
        self.max_checkpoints = kwargs.get('max_checkpoints', self.config.get('max_checkpoints', 5))
        self.config['max_checkpoints'] = self.max_checkpoints
        
        # Environment-specific settings
        env = os.environ.get('MOE_ENV', 'dev')
        if not os.path.isabs(self.base_dir):  # Only apply env subfolder if not absolute path
            self.base_dir = os.path.join(self.base_dir, env)
        
        # Create base directory if it doesn't exist
        os.makedirs(self.base_dir, exist_ok=True)
        
    def save_state(self, state: Dict[str, Any]) -> str:
        """
        Save the system state to a file.
        
        Args:
            state: State dictionary to save
            
        Returns:
            Full path to the saved checkpoint file
        """
        # Generate checkpoint path with timestamp and microseconds for uniqueness
        # This is especially important for tests where checkpoints are created in quick succession
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        microseconds = datetime.now().microsecond
        checkpoint_name = f"checkpoint_{timestamp}_{microseconds:06d}.json"
        checkpoint_path = os.path.join(self.base_dir, checkpoint_name)
        
        # Test JSON serialization before writing to file to ensure TypeError is raised
        # for non-serializable objects
        try:
            json.dumps(state)
        except (TypeError, OverflowError) as e:
            logger.error(f"State is not JSON serializable: {e}")
            raise
            
        try:
            # Write to file
            with open(checkpoint_path, 'w') as f:
                json.dump(state, f, indent=2)
            
            logger.info(f"Saved checkpoint to {checkpoint_path}")
            
            # Enforce max_checkpoints limit
            self._enforce_max_checkpoints()
            
            return checkpoint_path
        except Exception as e:
            logger.error(f"Error saving state: {e}")
            return ""
    
    def load_state(self, checkpoint_path: str) -> Dict[str, Any]:
        """
        Load a state from a checkpoint file.
        
        Args:
            checkpoint_path: Path to the checkpoint file
            
        Returns:
            The loaded state dictionary
        """
        try:
            with open(checkpoint_path, 'r') as f:
                state = json.load(f)
            return state
        except FileNotFoundError:
            logger.error(f"Checkpoint not found: {checkpoint_path}")
            raise
        except json.JSONDecodeError as e:
            logger.error(f"Error decoding checkpoint: {e}")
            raise
    
    def _enforce_max_checkpoints(self) -> None:
        """
        Enforce the maximum number of checkpoints by deleting the oldest ones.
        """
        # List all checkpoints
        checkpoints = self.list_checkpoints()
        
        # Delete oldest checkpoints if over the limit
        if len(checkpoints) > self.max_checkpoints:
            # Sort by creation time (oldest first)
            checkpoints.sort(key=os.path.getctime)
            
            # Debug logging
            logger.debug(f"Found {len(checkpoints)} checkpoints, keeping only {self.max_checkpoints}")
            logger.debug(f"Checkpoints found: {checkpoints}")
            logger.debug(f"Removing checkpoints: {checkpoints[:-self.max_checkpoints]}")
            
            # Delete oldest checkpoints, keeping only the most recent max_checkpoints
            for checkpoint in checkpoints[:-self.max_checkpoints]:
                logger.debug(f"Removing old checkpoint: {checkpoint}")
                os.remove(checkpoint)  # Direct file removal for checkpoints
    
    def list_checkpoints(self) -> List[str]:
        """
        List all available checkpoints.
        
        Returns:
            List of checkpoint file paths
        """
        if not os.path.exists(self.base_dir):
            return []
            
        checkpoints = []
        for file in os.listdir(self.base_dir):
            if file.startswith('checkpoint_') and file.endswith('.json'):
                checkpoints.append(os.path.join(self.base_dir, file))
                
        return checkpoints
    
    def get_latest_checkpoint(self) -> Optional[str]:
        """
        Get the latest checkpoint based on creation time.
        
        Returns:
            Path to the latest checkpoint or None if no checkpoints exist
        """
        checkpoints = self.list_checkpoints()
        if not checkpoints:
            return None
            
        # Return the most recently created checkpoint
        return max(checkpoints, key=os.path.getctime)
    
    def delete_checkpoint(self, checkpoint_path: str) -> bool:
        """
        Delete a specific checkpoint.
        
        Args:
            checkpoint_path: Path to the checkpoint to delete
            
        Returns:
            True if deletion was successful, False otherwise
        """
        try:
            if os.path.exists(checkpoint_path):
                os.remove(checkpoint_path)
                logger.info(f"Deleted checkpoint: {checkpoint_path}")
                return True
            return False
        except Exception as e:
            logger.error(f"Error deleting checkpoint: {e}")
            return False
            
    def delete_all_checkpoints(self) -> bool:
        """
        Delete all checkpoints.
        
        Returns:
            True if all deletions were successful, False otherwise
        """
        success = True
        for checkpoint in self.list_checkpoints():
            if not self.delete_checkpoint(checkpoint):
                success = False
        return success
        
    def get_metadata(self, checkpoint_path: str) -> Dict[str, Any]:
        """
        Get metadata for a checkpoint.
        
        Args:
            checkpoint_path: Path to the checkpoint
            
        Returns:
            Dictionary with checkpoint metadata
        """
        full_path = os.path.join(self.base_dir, checkpoint_path)
        metadata_path = os.path.join(full_path, 'metadata.json')
        
        if not os.path.exists(metadata_path):
            return {'error': 'Metadata file not found'}
            
        try:
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
                
            # Add file system metadata
            state_path = os.path.join(full_path, 'state.pkl')
            if os.path.exists(state_path):
                metadata['file_size'] = os.path.getsize(state_path)
                metadata['last_modified'] = datetime.fromtimestamp(
                    os.path.getmtime(state_path)
                ).isoformat()
                
            return metadata
            
        except Exception as e:
            logger.error(f"Error reading metadata: {str(e)}")
            return {'error': str(e)}
