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
    
    def __init__(self, base_dir: Optional[str] = None):
        """
        Initialize the file system state manager.
        
        Args:
            base_dir: Base directory for checkpoints
        """
        self.base_dir = base_dir or os.path.join(os.getcwd(), 'checkpoints')
        os.makedirs(self.base_dir, exist_ok=True)
        
    def save_state(self, state_data: Dict[str, Any], path: Optional[str] = None) -> bool:
        """
        Save the system state to a file.
        
        Args:
            state_data: Dictionary containing state data
            path: Optional path to save the state file
            
        Returns:
            Success flag
        """
        try:
            if path is None:
                # Generate default path if none provided
                timestamp = int(time.time())
                path = os.path.join(self.base_dir, f"state_{timestamp}.json")
                
            # Ensure directory exists
            os.makedirs(os.path.dirname(path), exist_ok=True)
            
            # Save state to file
            with open(path, 'w') as f:
                json.dump(state_data, f, indent=2)
                
            return True
            
        except Exception as e:
            logger.error(f"Error saving state: {str(e)}")
            return False
    
    def load_state(self, path: str) -> Optional[Dict[str, Any]]:
        """
        Load a state from a checkpoint file.
        
        Args:
            path: Path to the checkpoint file
            
        Returns:
            The loaded state dictionary
        """
        try:
            with open(path, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error loading state: {str(e)}")
            return None
    
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
